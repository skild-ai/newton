# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nesterov sweep history for CUDA PGS owners."""

from dataclasses import dataclass

import warp as wp

from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT, WARP_DETERMINISM_OFF

wp.set_module_options(WARP_DETERMINISM_OFF)


@dataclass(frozen=True)
class SweepField:
    """A lane-strided vector in the sweep's own coordinates."""

    value: str  # CUDA lvalue, indexed by i
    count: str
    capacity: int
    impulse: bool = False


def warp_momentum(
    enabled: bool,
    fields: list[SweepField],
    *,
    lanes: int = 32,
    iteration: str = "iteration",
    restart: str = "false",
) -> tuple[str, str]:
    """Emit register-local history and the update before a sweep.

    The impulse displacement determines the gradient restart. Velocity, residuals
    and other affine images of the impulses take the same extrapolation in their
    native coordinates. No extra response traversal or shared memory is needed.
    """
    if not enabled:
        return "", ""
    declarations = ["int momentum_age = 0;"]
    products = []
    updates = []
    for k, field in enumerate(fields):
        slots = (field.capacity + lanes - 1) // lanes
        previous, extrap = f"previous_{k}", f"extrap_{k}"
        declarations.append(f"float {previous}[{slots}];")
        loop = f"for (int i = lane, j = 0; i < {field.count}; i += {lanes}, ++j)"
        if field.impulse:
            declarations.append(f"float {extrap}[{slots}];")
            products.append(f"{loop} dot += ({field.value} - {previous}[j] - {extrap}[j]) * {extrap}[j];")
        updates.append(f"""{loop} {{
            const float x = {field.value};
            const float y = beta != 0.0f ? x + beta * (x - {previous}[j]) : x;
            {previous}[j] = x;
            {f"{extrap}[j] = y - x;" if field.impulse else ""}
            if (beta != 0.0f) {field.value} = y;
        }}""")
    step = f"""
        float beta = 0.0f;
        if ({iteration} > 0) {{
            float dot = 0.0f;
            {chr(10).join(products)}
            for (int d = {lanes // 2}; d; d >>= 1) dot += __shfl_xor_sync(MASK, dot, d, {lanes});
            momentum_age = dot < 0.0f || ({restart}) ? 0 : momentum_age + 1;
            if (momentum_age > 1) beta = static_cast<float>(momentum_age - 1) / (momentum_age + 2);
        }}
        {chr(10).join(updates)}
        __syncwarp(MASK);
"""
    return "\n".join(declarations), step


@wp.struct
class SplitSweep:
    """State shared across the launches of a phased or propagation sweep."""

    dense_count: wp.array[int]
    free_count: wp.array[int]
    prop_count: wp.array[int]
    dense_type: wp.array2d[int]
    free_type: wp.array2d[int]
    prop_type: wp.array2d[int]
    dense: wp.array2d[float]
    free: wp.array2d[float]
    prop: wp.array2d[float]
    dofs: wp.array2d[int]
    velocity: wp.array[float]
    bodies: wp.array[int]
    body_start: wp.array[int]
    twist: wp.array2d[float]
    prev_twist: wp.array2d[float]
    prev_velocity: wp.array[float]
    prev: wp.array2d[float]
    extrap: wp.array2d[float]
    age: wp.array[int]
    correction_norm_sq: wp.array[float]
    clamp: wp.array2d[float]


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
#else
    return x;
#endif
""")
def _warp_sum(x: float) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
""")
def _warp_sync(): ...


@wp.func
def _sweep_progress(
    world: int,
    lane: int,
    rows: wp.array2d[float],
    kinds: wp.array2d[int],
    count: wp.array[int],
    offset: int,
    h: SplitSweep,
):
    dot = float(0.0)
    norm = float(0.0)
    released = float(0.0)
    displacement = float(0.0)
    for i in range(lane, wp.min(count[world], rows.shape[1]), 32):
        e = h.extrap[world, offset + i]
        x = rows[world, i]
        prev = h.prev[world, offset + i]
        correction = x - prev - e
        dot += correction * e
        norm += correction * correction
        displacement += (x - prev) * (x - prev)
        # Extrapolating a newly released unilateral row would reapply it in tension.
        kind = kinds[world, i]
        if (kind == PGS_CONSTRAINT_TYPE_CONTACT or kind == PGS_CONSTRAINT_TYPE_JOINT_LIMIT) and x == 0.0 and prev > 0.0:
            released = 1.0
    return wp.vec4(dot, norm, released, displacement)


@wp.func
def _update_rows(
    world: int, lane: int, rows: wp.array2d[float], count: wp.array[int], offset: int, beta: float, h: SplitSweep
):
    for i in range(lane, wp.min(count[world], rows.shape[1]), 32):
        x = rows[world, i]
        e = float(0.0)
        if beta != 0.0:
            y = x + beta * (x - h.prev[world, offset + i])
            rows[world, i] = y
            e = y - x
        h.prev[world, offset + i] = x
        h.extrap[world, offset + i] = e


@wp.func
def split_momentum(world: int, lane: int, iteration: int, friction_transition: bool, h: SplitSweep):
    """Advance the complete iterate once, before its first row phase, including cached tree twists."""
    free_offset = h.dense.shape[1]
    prop_offset = free_offset + h.free.shape[1]
    age = int(0)
    beta = float(0.0)
    if iteration > 0:
        progress = _sweep_progress(world, lane, h.dense, h.dense_type, h.dense_count, 0, h)
        progress += _sweep_progress(world, lane, h.free, h.free_type, h.free_count, free_offset, h)
        progress += _sweep_progress(world, lane, h.prop, h.prop_type, h.prop_count, prop_offset, h)
        dot = _warp_sum(progress[0])
        norm = _warp_sum(progress[1])
        released = _warp_sum(progress[2])
        displacement = _warp_sum(progress[3])
        contracting = iteration == 1 or norm <= h.correction_norm_sq[world]
        if dot >= 0.0 and contracting and released == 0.0 and not friction_transition and h.clamp[world, 0] == 0.0:
            age = h.age[world] + 1
            if age > 1:
                beta = float(age - 1) / float(age + 2)
                if displacement > 0.0:
                    # Keep the extrapolated impulse within the size of the last projected correction.
                    beta = wp.min(beta, wp.sqrt(norm / displacement))
        _warp_sync()
        if lane == 0:
            h.correction_norm_sq[world] = norm
    _warp_sync()
    if lane == 0:
        h.age[world] = age
        h.clamp[world, 0] = 0.0
    _update_rows(world, lane, h.dense, h.dense_count, 0, beta, h)
    _update_rows(world, lane, h.free, h.free_count, free_offset, beta, h)
    _update_rows(world, lane, h.prop, h.prop_count, prop_offset, beta, h)
    for i in range(lane, h.dofs.shape[1], 32):
        dof = h.dofs[world, i]
        if dof >= 0:
            v = h.velocity[dof]
            if beta != 0.0:
                h.velocity[dof] = v + beta * (v - h.prev_velocity[dof])
            h.prev_velocity[dof] = v
    start = h.body_start[world]
    for i in range(lane, 6 * (h.body_start[world + 1] - start), 32):
        body = h.bodies[start + i // 6]
        axis = i % 6
        v = h.twist[body, axis]
        if beta != 0.0:
            h.twist[body, axis] = v + beta * (v - h.prev_twist[body, axis])
        h.prev_twist[body, axis] = v
    _warp_sync()


@wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
def advance_split_sweep(histories: wp.array[SplitSweep], iteration: int, friction_transition: bool):
    world, lane = wp.tid()
    split_momentum(world, lane, iteration, friction_transition, histories[0])
