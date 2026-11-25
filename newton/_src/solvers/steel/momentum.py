# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nesterov sweep history for CUDA PGS owners."""

from dataclasses import dataclass
from functools import cache

import warp as wp

from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
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
    history: wp.array2d[wp.vec2]
    age: wp.array[int]
    correction_norm_sq: wp.array[float]
    prop_torsion_head: wp.array[int]
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
def _warp_sum_double(x: wp.float64) -> wp.float64: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
""")
def _warp_sync(): ...


_Progress = wp.types.vector(length=5, dtype=wp.float64)


@wp.func
def _sweep_progress(
    world: int,
    lane: int,
    stride: int,
    rows: wp.array2d[float],
    kinds: wp.array2d[int],
    count: wp.array[int],
    offset: int,
    h: SplitSweep,
    measure_load: bool,
):
    # Color capacity changes the packing order. Round the world sum once
    # before choosing beta so a different packing gives the same lookahead.
    load = wp.float64(0.0)
    dot = wp.float64(0.0)
    norm = wp.float64(0.0)
    released = wp.float64(0.0)
    displacement = wp.float64(0.0)
    for i in range(lane, wp.min(count[world], rows.shape[1]), stride):
        old = h.history[world, offset + i]
        x = rows[world, i]
        prev, e = old[0], old[1]
        correction = x - prev - e
        dot += wp.float64(correction * e)
        norm += wp.float64(correction * correction)
        displacement += wp.float64((x - prev) * (x - prev))
        # Extrapolating a newly released unilateral row would reapply it in tension.
        if measure_load:
            kind = kinds[world, i]
            if kind == PGS_CONSTRAINT_TYPE_CONTACT or kind == PGS_CONSTRAINT_TYPE_FRICTION:
                load += wp.float64(x * x)
        if x == 0.0 and prev > 0.0:
            kind = kinds[world, i]
            if kind == PGS_CONSTRAINT_TYPE_CONTACT or kind == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                released = wp.float64(1.0)
    return _Progress(dot, norm, released, displacement, load)


@wp.func
def _has_soft_row(world: int, lane: int, stride: int, count: wp.array[int], gamma: wp.array2d[float]):
    soft = int(0)
    for i in range(lane, wp.min(count[world], gamma.shape[1]), stride):
        if gamma[world, i] > 0.0:
            soft = 1
    return soft


@wp.func
def _update_rows(
    world: int,
    lane: int,
    stride: int,
    rows: wp.array2d[float],
    kinds: wp.array2d[int],
    count: wp.array[int],
    offset: int,
    beta: float,
    has_velocity_limits: bool,
    h: SplitSweep,
):
    for i in range(lane, wp.min(count[world], rows.shape[1]), stride):
        x = rows[world, i]
        e = float(0.0)
        if beta != 0.0 and (not has_velocity_limits or kinds[world, i] != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT):
            y = x + beta * (x - h.history[world, offset + i][0])
            rows[world, i] = y
            e = y - x
        h.history[world, offset + i] = wp.vec2(x, e)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
""")
def _block_sync(): ...


@cache
def _get_block_progress(warps: int):
    """Sum one progress vector across all warps of a world's fused solve."""

    @wp.func_native(
        """
    #if defined(__CUDA_ARCH__)
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        __shared__ double partial[WARPS][5];
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            for (int d = 16; d; d >>= 1) x[i] += __shfl_xor_sync(0xffffffff, x[i], d);
            if (lane == 0) partial[warp][i] = x[i];
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            double sum = 0;
            for (int w = 0; w < blockDim.x / 32; ++w) sum += partial[w][i];
            x[i] = sum;
        }
        __syncthreads();
    #endif
        return x;
    """.replace("WARPS", str(warps))
    )
    def _block_progress(x: _Progress) -> _Progress: ...

    return _block_progress


@wp.func
def _warp_progress(x: _Progress):
    return _Progress(
        _warp_sum_double(x[0]),
        _warp_sum_double(x[1]),
        _warp_sum_double(x[2]),
        _warp_sum_double(x[3]),
        _warp_sum_double(x[4]),
    )


@dataclass(frozen=True)
class SplitSweepOptions:
    """Sweep features fixed when the solver selects its CUDA kernels."""

    propagation: bool
    torsion: bool
    warmstart: bool
    fixed_normal: bool
    compliance: bool
    velocity_limits: bool


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __syncthreads_or(x);
#else
    return x;
#endif
""")
def _block_any(x: int) -> int: ...


@wp.func
def _released(
    world: int,
    lane: int,
    stride: int,
    rows: wp.array2d[float],
    kinds: wp.array2d[int],
    count: wp.array[int],
    offset: int,
    h: SplitSweep,
):
    hit = int(0)
    for i in range(lane, wp.min(count[world], rows.shape[1]), stride):
        if rows[world, i] == 0.0 and h.history[world, offset + i][0] > 0.0:
            kind = kinds[world, i]
            if kind == PGS_CONSTRAINT_TYPE_CONTACT or kind == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                hit = 1
    return hit


@cache
def get_split_momentum(warps: int, options: SplitSweepOptions):
    """Use the owner block to advance the iterate before its first row phase."""
    measure_load = options.warmstart or (options.propagation and options.fixed_normal)
    release_only = (
        warps > 1 and options.fixed_normal and not (options.torsion or options.warmstart or options.compliance)
    )
    stride = 32 * warps
    sync = _warp_sync if warps == 1 else _block_sync
    reduce_progress = _warp_progress if warps == 1 else _get_block_progress(warps)

    @wp.func(module="unique")
    def advance(world: int, lane: int, iteration: int, previous_age: int, friction_transition: bool, h: SplitSweep):
        """Advance the complete iterate once, before its first row phase, including cached tree twists."""
        free_offset = h.dense.shape[1]
        prop_offset = free_offset + h.free.shape[1]
        age = int(0)
        beta = float(0.0)
        if iteration > 0:
            if wp.static(release_only) and previous_age == 0:
                # The recovery sweep cannot extrapolate yet. Only a newly
                # released row can prevent the age counter from advancing.
                released_row = _released(world, lane, stride, h.dense, h.dense_type, h.dense_count, 0, h)
                released_row |= _released(world, lane, stride, h.free, h.free_type, h.free_count, free_offset, h)
                released_row |= _released(world, lane, stride, h.prop, h.prop_type, h.prop_count, prop_offset, h)
                dot, norm, released, displacement, load = (
                    float(0.0),
                    float(0.0),
                    float(_block_any(released_row)),
                    float(0.0),
                    float(0.0),
                )
            else:
                progress = _sweep_progress(
                    world, lane, stride, h.dense, h.dense_type, h.dense_count, 0, h, measure_load
                )
                progress += _sweep_progress(
                    world, lane, stride, h.free, h.free_type, h.free_count, free_offset, h, measure_load
                )
                progress += _sweep_progress(
                    world, lane, stride, h.prop, h.prop_type, h.prop_count, prop_offset, h, measure_load
                )
                total = reduce_progress(progress)
                dot, norm, released, displacement = float(total[0]), float(total[1]), float(total[2]), float(total[3])
                load = float(total[4])
            restart = friction_transition or dot < 0.0
            if wp.static(options.propagation and options.velocity_limits):
                restart = restart or h.clamp[world, 0] != 0.0
            if wp.static(options.propagation) and wp.static(options.fixed_normal):
                restart = restart or released != 0.0
            torsion = False
            if wp.static(options.torsion):
                # Torsion rows occur on either articulated contact route. Shape
                # filtering can leave a world with ordinary contacts only.
                present = int(0)
                for i in range(lane, wp.min(h.dense_count[world], h.dense.shape[1]), stride):
                    if h.dense_type[world, i] == PGS_CONSTRAINT_TYPE_TORSION:
                        present = 1
                # Propagation reserves inert spin slots before grouping. The list
                # contains only groups admitted to the current constraint solve.
                if world < h.prop_torsion_head.shape[0] and h.prop_torsion_head[world] >= 0:
                    present = 1
                contracting = iteration == 1 or norm <= h.correction_norm_sq[world]
                torsion = reduce_progress(_Progress(wp.float64(present), 0.0, 0.0, 0.0, 0.0))[0] > 0.0
                restart = restart or (torsion and (not contracting or released != 0.0))
            if not restart:
                age = previous_age + 1
                if age > 1:
                    beta = float(age - 1) / float(age + 2)
                    has_soft_contacts = False
                    if wp.static(options.compliance):
                        soft = _has_soft_row(world, lane, stride, h.dense_count, h.dense_gamma)
                        soft += _has_soft_row(world, lane, stride, h.free_count, h.free_gamma)
                        soft += _has_soft_row(world, lane, stride, h.prop_count, h.prop_gamma)
                        has_soft_contacts = reduce_progress(_Progress(wp.float64(soft), 0.0, 0.0, 0.0, 0.0))[0] > 0.0
                    if wp.static(measure_load) and load > 0.0:
                        if wp.static(options.warmstart):
                            # A matching contact may gain load after the seed is applied.
                            beta *= norm / (norm + load)
                        elif not has_soft_contacts:
                            # Limit lookahead as corrections become small relative to the
                            # supported load, including before a projectile leaves its cup.
                            beta = wp.min(beta, wp.sqrt(norm / load))
                    bounded = (
                        torsion
                        or (wp.static(options.propagation) and wp.static(options.fixed_normal))
                        or has_soft_contacts
                    )
                    if displacement > 0.0 and bounded:
                        # Keep the extrapolated impulse within the size of the last projected correction.
                        beta = wp.min(beta, wp.sqrt(norm / displacement))
            if wp.static(options.torsion):
                sync()
                if lane == 0:
                    h.correction_norm_sq[world] = norm
        if wp.static(options.propagation and options.velocity_limits):
            sync()
            if lane == 0:
                h.clamp[world, 0] = 0.0
        _update_rows(world, lane, stride, h.dense, h.dense_type, h.dense_count, 0, beta, True, h)
        _update_rows(world, lane, stride, h.free, h.free_type, h.free_count, free_offset, beta, True, h)
        _update_rows(world, lane, stride, h.prop, h.prop_type, h.prop_count, prop_offset, beta, False, h)
        if wp.static(not options.propagation):
            sync()
            # Stateless clamps contribute to velocity independently of stored
            # impulses. Apply only the extrapolated row impulses on phased sweeps.
            if beta != 0.0:
                for i in range(lane, h.dofs.shape[1], stride):
                    dof = h.dofs[world, i]
                    if dof >= 0:
                        v = h.velocity[dof]
                        for row in range(wp.min(h.dense_count[world], h.dense.shape[1])):
                            e = h.history[world, row][1]
                            if e != 0.0:
                                v += h.response[world, row, i] * e
                        h.velocity[dof] = v
                sync()
                # Each lane owns one component of every free body, including
                # coincident endpoints. Apply both responses in row order.
                if lane < 6:
                    for row in range(wp.min(h.free_count[world], h.free.shape[1])):
                        e = h.history[world, free_offset + row][1]
                        if e != 0.0:
                            first = h.free_a[world, row]
                            second = h.free_b[world, row]
                            if first >= 0:
                                dof = h.dofs[world, first + lane]
                                h.velocity[dof] += e * h.response_a[world, row, lane]
                            if second >= 0:
                                dof = h.dofs[world, second + lane]
                                h.velocity[dof] += e * h.response_b[world, row, lane]
                sync()
            return age
        # After a restart the next sweep has zero lookahead too. No later
        # extrapolation can read this iteration's velocity or twist snapshot.
        if age > 0:
            for i in range(lane, h.dofs.shape[1], stride):
                dof = h.dofs[world, i]
                if dof >= 0:
                    v = h.velocity[dof]
                    if beta != 0.0:
                        h.velocity[dof] = v + beta * (v - h.prev_velocity[dof])
                    h.prev_velocity[dof] = v
            start = h.body_start[world]
            for i in range(lane, 6 * (h.body_start[world + 1] - start), stride):
                body = h.bodies[start + i // 6]
                axis = i % 6
                v = h.twist[body, axis]
                if beta != 0.0:
                    h.twist[body, axis] = v + beta * (v - h.prev_twist[body, axis])
                h.prev_twist[body, axis] = v
        sync()
        return age

    return advance


@cache
def get_advance_split_sweep(options: SplitSweepOptions) -> wp.Kernel:
    advance = get_split_momentum(1, options)

    @wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
    def advance_split_sweep(histories: wp.array[SplitSweep], iteration: int, friction_transition: bool):
        world, lane = wp.tid()
        age = int(0)
        if iteration > 0:
            age = histories[0].age[world]
        age = advance(world, lane, iteration, age, friction_transition, histories[0])
        if lane == 0:
            histories[0].age[world] = age

    return advance_split_sweep
