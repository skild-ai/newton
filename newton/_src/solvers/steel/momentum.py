# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nesterov sweep history for CUDA PGS owners."""

from dataclasses import dataclass

import warp as wp

from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
    WARP_DETERMINISM_OFF,
)

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
    row_gamma: str = "",
) -> tuple[str, str]:
    """Emit register-local history and the update before a sweep.

    The impulse displacement determines the gradient restart. Velocity, residuals
    and other affine images of the impulses take the same extrapolation in their
    native coordinates. No extra response traversal or shared memory is needed.
    """
    if not enabled:
        return "", ""
    declarations = ["int momentum_age = 0;"]
    if row_gamma:
        declarations.append(f"""bool momentum_soft = false;
            for (int i = lane; i < {fields[0].count}; i += {lanes}) momentum_soft |= {row_gamma} > 0.0f;
            momentum_soft = __any_sync(MASK, momentum_soft);""")
    products = []
    updates = []
    for k, field in enumerate(fields):
        slots = (field.capacity + lanes - 1) // lanes
        previous, extrap = f"previous_{k}", f"extrap_{k}"
        declarations.append(f"float {previous}[{slots}];")
        loop = f"for (int i = lane, j = 0; i < {field.count}; i += {lanes}, ++j)"
        if field.impulse:
            declarations.append(f"float {extrap}[{slots}];")
            products.append(f"""{loop} {{
                const float move = {field.value} - {previous}[j];
                const float correction = move - {extrap}[j];
                dot += correction * {extrap}[j];
                {"norm += correction * correction; displacement += move * move;" if row_gamma else ""}
            }}""")
        updates.append(f"""{loop} {{
            const float x = {field.value};
            const float y = beta != 0.0f ? x + beta * (x - {previous}[j]) : x;
            {previous}[j] = x;
            {f"{extrap}[j] = y - x;" if field.impulse else ""}
            if (beta != 0.0f) {field.value} = y;
        }}""")
    bound = ""
    if row_gamma:
        bound = f"""if (momentum_soft) {{
                for (int d = {lanes // 2}; d; d >>= 1) {{
                    norm += __shfl_xor_sync(MASK, norm, d, {lanes});
                    displacement += __shfl_xor_sync(MASK, displacement, d, {lanes});
                }}
                if (displacement > 0.0f) beta = fminf(beta, sqrtf(norm / displacement));
            }}"""
    step = f"""
        float beta = 0.0f;
        if ({iteration} > 0) {{
            float dot = 0.0f;
            {"float norm = 0.0f, displacement = 0.0f;" if row_gamma else ""}
            {chr(10).join(products)}
            for (int d = {lanes // 2}; d; d >>= 1) dot += __shfl_xor_sync(MASK, dot, d, {lanes});
            momentum_age = dot < 0.0f || ({restart}) ? 0 : momentum_age + 1;
            if (momentum_age > 1) beta = static_cast<float>(momentum_age - 1) / (momentum_age + 2);
            {bound}
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
    propagation: int
    torsion: int
    prop_torsion_head: wp.array[int]
    warmstart: int
    initial_norm_sq: wp.array[float]
    fixed_normal: int
    compliance: int
    dense_gamma: wp.array2d[float]
    free_gamma: wp.array2d[float]
    prop_gamma: wp.array2d[float]
    response: wp.array3d[float]
    free_a: wp.array2d[int]
    free_b: wp.array2d[int]
    response_a: wp.array3d[float]
    response_b: wp.array3d[float]
    clamp: wp.array2d[float]


_WARP_SUM = """
#if defined(__CUDA_ARCH__)
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
#endif
    return x;
"""


@wp.func_native(_WARP_SUM)
def _warp_sum(x: float) -> float: ...


@wp.func_native(_WARP_SUM)
def _warp_sum_fp64(x: wp.float64) -> float: ...


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
    # Color capacity changes the packing order. Round the world sum once
    # before choosing beta so a different packing gives the same lookahead.
    dot = wp.float64(0.0)
    norm = wp.float64(0.0)
    released = wp.float64(0.0)
    displacement = wp.float64(0.0)
    for i in range(lane, wp.min(count[world], rows.shape[1]), 32):
        e = h.extrap[world, offset + i]
        x = rows[world, i]
        prev = h.prev[world, offset + i]
        correction = x - prev - e
        dot += wp.float64(correction * e)
        norm += wp.float64(correction * correction)
        displacement += wp.float64((x - prev) * (x - prev))
        # Extrapolating a newly released unilateral row would reapply it in tension.
        kind = kinds[world, i]
        if (kind == PGS_CONSTRAINT_TYPE_CONTACT or kind == PGS_CONSTRAINT_TYPE_JOINT_LIMIT) and x == 0.0 and prev > 0.0:
            released = wp.float64(1.0)
    return wp.vec4d(dot, norm, released, displacement)


@wp.func
def _impulse_norm_sq(world: int, lane: int, count: wp.array[int], rows: wp.array2d[float]):
    norm = wp.float64(0.0)
    for i in range(lane, wp.min(count[world], rows.shape[1]), 32):
        x = rows[world, i]
        norm += wp.float64(x * x)
    return norm


@wp.func
def _has_soft_row(world: int, lane: int, count: wp.array[int], gamma: wp.array2d[float]):
    soft = int(0)
    for i in range(lane, wp.min(count[world], gamma.shape[1]), 32):
        if gamma[world, i] > 0.0:
            soft = 1
    return soft


@wp.func
def _update_rows(
    world: int,
    lane: int,
    rows: wp.array2d[float],
    kinds: wp.array2d[int],
    count: wp.array[int],
    offset: int,
    beta: float,
    h: SplitSweep,
):
    for i in range(lane, wp.min(count[world], rows.shape[1]), 32):
        x = rows[world, i]
        e = float(0.0)
        if beta != 0.0 and kinds[world, i] != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
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
    if iteration == 0 and h.warmstart != 0:
        carried = _impulse_norm_sq(world, lane, h.dense_count, h.dense)
        carried += _impulse_norm_sq(world, lane, h.free_count, h.free)
        carried += _impulse_norm_sq(world, lane, h.prop_count, h.prop)
        carried_norm = _warp_sum_fp64(carried)
        if lane == 0:
            h.initial_norm_sq[world] = carried_norm
    if iteration > 0:
        progress = _sweep_progress(world, lane, h.dense, h.dense_type, h.dense_count, 0, h)
        progress += _sweep_progress(world, lane, h.free, h.free_type, h.free_count, free_offset, h)
        progress += _sweep_progress(world, lane, h.prop, h.prop_type, h.prop_count, prop_offset, h)
        dot = _warp_sum_fp64(progress[0])
        norm = _warp_sum_fp64(progress[1])
        released = _warp_sum_fp64(progress[2])
        displacement = _warp_sum_fp64(progress[3])
        contracting = iteration == 1 or norm <= h.correction_norm_sq[world]
        restart = friction_transition or dot < 0.0
        if h.propagation != 0:
            restart = restart or h.clamp[world, 0] != 0.0
        if h.propagation != 0 and h.fixed_normal != 0:
            restart = restart or released != 0.0
        torsion = False
        if h.torsion != 0:
            # Torsion rows occur on either articulated contact route. Shape
            # filtering can leave a world with ordinary contacts only.
            present = int(0)
            for i in range(lane, wp.min(h.dense_count[world], h.dense.shape[1]), 32):
                if h.dense_type[world, i] == PGS_CONSTRAINT_TYPE_TORSION:
                    present = 1
            # Propagation reserves inert spin slots before grouping. The list
            # contains only groups admitted to the current constraint solve.
            if world < h.prop_torsion_head.shape[0] and h.prop_torsion_head[world] >= 0:
                present = 1
            torsion = _warp_sum(float(present)) > 0.0
            restart = restart or (torsion and (not contracting or released != 0.0))
        if not restart:
            age = h.age[world] + 1
            if age > 1:
                beta = float(age - 1) / float(age + 2)
                if h.warmstart != 0 and h.initial_norm_sq[world] > 0.0:
                    # A nearly satisfied warm start already carries the contact
                    # load. Weight the lookahead by the new correction's share
                    # of the squared impulse, to avoid re-exciting the grasp.
                    beta *= norm / (norm + h.initial_norm_sq[world])
                bounded = torsion or (h.propagation != 0 and h.fixed_normal != 0)
                if h.compliance != 0:
                    soft = _has_soft_row(world, lane, h.dense_count, h.dense_gamma)
                    soft += _has_soft_row(world, lane, h.free_count, h.free_gamma)
                    soft += _has_soft_row(world, lane, h.prop_count, h.prop_gamma)
                    bounded = bounded or _warp_sum(float(soft)) > 0.0
                if displacement > 0.0 and bounded:
                    # Keep the extrapolated impulse within the size of the last projected correction.
                    beta = wp.min(beta, wp.sqrt(norm / displacement))
        _warp_sync()
        if lane == 0:
            h.correction_norm_sq[world] = norm
    _warp_sync()
    if lane == 0:
        h.age[world] = age
        h.clamp[world, 0] = 0.0
    _update_rows(world, lane, h.dense, h.dense_type, h.dense_count, 0, beta, h)
    _update_rows(world, lane, h.free, h.free_type, h.free_count, free_offset, beta, h)
    _update_rows(world, lane, h.prop, h.prop_type, h.prop_count, prop_offset, beta, h)
    _warp_sync()
    if h.propagation == 0:
        # Stateless clamps contribute to velocity independently of stored
        # impulses. Apply only the extrapolated row impulses on phased sweeps.
        if beta != 0.0:
            for i in range(lane, h.dofs.shape[1], 32):
                dof = h.dofs[world, i]
                if dof >= 0:
                    v = h.velocity[dof]
                    for row in range(wp.min(h.dense_count[world], h.dense.shape[1])):
                        e = h.extrap[world, row]
                        if e != 0.0:
                            v += h.response[world, row, i] * e
                    h.velocity[dof] = v
            _warp_sync()
            # Each lane owns one component of every free body, including
            # coincident endpoints. Apply both responses in row order.
            if lane < 6:
                for row in range(wp.min(h.free_count[world], h.free.shape[1])):
                    e = h.extrap[world, free_offset + row]
                    if e != 0.0:
                        first = h.free_a[world, row]
                        second = h.free_b[world, row]
                        if first >= 0:
                            dof = h.dofs[world, first + lane]
                            h.velocity[dof] += e * h.response_a[world, row, lane]
                        if second >= 0:
                            dof = h.dofs[world, second + lane]
                            h.velocity[dof] += e * h.response_b[world, row, lane]
            _warp_sync()
        return
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
