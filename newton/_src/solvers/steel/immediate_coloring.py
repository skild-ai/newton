# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA contact sweeps colored by immediate articulated response."""

import functools

import warp as wp

from .friction import friction_pair_candidate
from .friction_patches import WARP_DETERMINISM_OFF, patch_normal_load
from .kernels import contact_restitution_fires

wp.set_module_options(WARP_DETERMINISM_OFF)


@wp.struct
class _Rows:
    packed_sides: bool
    J: wp.array3d[float]
    Y: wp.array3d[float]
    Ja: wp.array3d[float]
    Jb: wp.array3d[float]
    Ya: wp.array3d[float]
    Yb: wp.array3d[float]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    mu: wp.array2d[float]
    weight: wp.array2d[float]
    parent: wp.array2d[int]
    impulse: wp.array2d[float]


@wp.kernel
def _collect(
    count: wp.array[int],
    path: wp.array[int],
    world: wp.array[int],
    slot: wp.array[int],
    size: wp.array[int],
    art_a: wp.array[int],
    art_b: wp.array[int],
    dense_capacity: int,
    units: wp.array2d[int],
    sides: wp.array2d[wp.vec2i],
):
    c = wp.tid()
    if c >= count[0] or slot[c] < 0 or path[c] < 0:
        return
    i = slot[c]
    if path[c] == 1:
        i += dense_capacity
    w = world[c]
    units[w, i] = size[c]
    sides[w, i] = wp.vec2i(art_a[c], art_b[c])


@wp.kernel
def _color(
    units: wp.array2d[int],
    sides: wp.array2d[wp.vec2i],
    response: wp.array[int],
    dense_count: wp.array[int],
    free_count: wp.array[int],
    dense_capacity: int,
    serial: bool,
    last: wp.array[int],
    colors: wp.array2d[int],
    offsets: wp.array2d[int],
    cursor: wp.array2d[int],
    order: wp.array2d[int],
    num_colors: wp.array[int],
):
    # Keep the serial order of conflicting contacts (including patch-load dependencies).
    # Disjoint articulations commute even if their contacts touch the same ground.
    w = wp.tid()
    dense = wp.min(dense_count[w], dense_capacity)
    free = wp.min(free_count[w], units.shape[1] - dense_capacity)
    n = int(0)
    for row in range(dense + free):
        i = row if row < dense else dense_capacity + row - dense
        if units[w, i] == 0:
            continue
        ab = sides[w, i]
        color = int(0)
        for end in range(2):
            a = ab[end]
            if a >= 0 and response[a] > 0:
                color = wp.max(color, last[a])
        for end in range(2):
            a = ab[end]
            if a >= 0 and response[a] > 0:
                last[a] = color + 1
        if serial:
            color = n
        colors[w, i] = color
        cursor[w, color] += 1
        n = wp.max(n, color + 1)
    total = int(0)
    for color in range(n):
        offsets[w, color] = total
        total += cursor[w, color]
        cursor[w, color] = offsets[w, color]
    offsets[w, n] = total
    num_colors[w] = n
    for row in range(dense + free):
        i = row if row < dense else dense_capacity + row - dense
        if units[w, i] > 0:
            c = colors[w, i]
            order[w, cursor[w, c]] = i
            cursor[w, c] += 1


@wp.func
def _coeff(rows: _Rows, free: bool, w: int, r: int, end: int, off: int, k: int):
    if free:
        if end == 0:
            return rows.Ja[w, r, k], rows.Ya[w, r, k]
        return rows.Jb[w, r, k], rows.Yb[w, r, k]
    if rows.packed_sides:
        return rows.J[end, r, k], rows.Y[end, r, k]
    return rows.J[w, r, off + k], rows.Y[w, r, off + k]


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    for (int shift = 16; shift; shift >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, shift);
    return __shfl_sync(0xffffffffu, value, 0);
#else
    return value;
#endif
""")
def _warp_sum(value: float) -> float:
    """Sum a row dot product and distribute it to the warp."""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
""")
def _warp_sync():
    """Order impulse and velocity writes within one constraint unit."""


@wp.func
def _visit(
    rows: _Rows,
    free: bool,
    w: int,
    row: int,
    pair: bool,
    ab: wp.vec2i,
    starts: wp.array[int],
    offsets: wp.array[int],
    sizes: wp.array[int],
    v: wp.array[float],
    apply: bool,
    delta: wp.vec2,
    lane: int,
):
    result = wp.vec3()
    for end in range(2):
        a = ab[end]
        if a < 0 or (end == 1 and a == ab[0]):
            continue
        for k in range(lane, sizes[a], 32):
            j, y = _coeff(rows, free, w, row, end, offsets[a], k)
            other_j = float(0.0)
            other_y = float(0.0)
            if pair:
                other_j, other_y = _coeff(rows, free, w, row + 1, end, offsets[a], k)
            d = starts[a] + k
            if apply:
                v[d] += y * delta[0] + other_y * delta[1]
            else:
                velocity = v[d]
                result += wp.vec3(j * velocity, other_j * velocity, j * other_y)
    if apply:
        _warp_sync()
    else:
        first = _warp_sum(result[0])
        if pair:
            result = wp.vec3(first, _warp_sum(result[1]), _warp_sum(result[2]))
        else:
            result = wp.vec3(first, 0.0, 0.0)
    return result


@wp.func
def _solve(
    rows: _Rows,
    free: bool,
    world: int,
    normal: int,
    size: int,
    ab: wp.vec2i,
    starts: wp.array[int],
    offsets: wp.array[int],
    sizes: wp.array[int],
    v: wp.array[float],
    omega: float,
    regularize: bool,
    friction: bool,
    lane: int,
):
    diagonal = rows.diag[world, normal]
    if diagonal > 0.0:
        if not free:
            diagonal = 1.0 / diagonal
        jv = _visit(rows, free, world, normal, False, ab, starts, offsets, sizes, v, False, wp.vec2(), lane)[0]
        old = rows.impulse[world, normal]
        weight = float(1.0)
        if regularize:
            weight = rows.weight[world, normal]
        step = -diagonal * (jv + rows.rhs[world, normal]) * weight - (1.0 - weight) * old
        updated = wp.max(0.0, old + omega * step)
        _warp_sync()
        if lane == 0:
            rows.impulse[world, normal] = updated
        _visit(rows, free, world, normal, False, ab, starts, offsets, sizes, v, True, wp.vec2(updated - old, 0.0), lane)
    if size >= 3 and friction:
        t = normal + 1
        measure = _visit(rows, free, world, t, True, ab, starts, offsets, sizes, v, False, wp.vec2(), lane)
        old_pair = wp.vec2(rows.impulse[world, t], rows.impulse[world, t + 1])
        residual = wp.vec2(measure[0] + rows.rhs[world, t], measure[1] + rows.rhs[world, t + 1])
        radius = wp.max(0.0, rows.mu[world, t] * patch_normal_load(rows.parent, rows.impulse, world, normal))
        a = rows.diag[world, t]
        b = rows.diag[world, t + 1]
        if free:
            a = 1.0 / a if a > 0.0 else 0.0
            b = 1.0 / b if b > 0.0 else 0.0
        updated_pair = friction_pair_candidate(a, measure[2], b, residual, old_pair, radius, omega)
        mag = wp.length(updated_pair)
        if mag > radius:
            updated_pair *= radius / mag
        _warp_sync()
        if lane == 0:
            rows.impulse[world, t] = updated_pair[0]
            rows.impulse[world, t + 1] = updated_pair[1]
        _visit(rows, free, world, t, True, ab, starts, offsets, sizes, v, True, updated_pair - old_pair, lane)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
""")
def _barrier():
    """Publish disjoint generalized-velocity writes before the next color."""


_THREADS = 256


@wp.kernel(enable_backward=False)
def _sweep(
    dense: _Rows,
    free: _Rows,
    units: wp.array2d[int],
    sides: wp.array2d[wp.vec2i],
    order: wp.array2d[int],
    offsets: wp.array2d[int],
    counts: wp.array[int],
    art_start: wp.array[int],
    art_offset: wp.array[int],
    response: wp.array[int],
    dense_capacity: int,
    omega: float,
    regularize: bool,
    friction_start: int,
    iterations: int,
    v: wp.array[float],
):
    w, thread = wp.tid()
    lane = thread % 32
    for iteration in range(iterations):
        for color in range(counts[w]):
            u = offsets[w, color] + thread // 32
            while u < offsets[w, color + 1]:
                i = order[w, u]
                if i < dense_capacity:
                    _solve(
                        dense,
                        False,
                        w,
                        i,
                        units[w, i],
                        sides[w, i],
                        art_start,
                        art_offset,
                        response,
                        v,
                        omega,
                        regularize,
                        iteration >= friction_start,
                        lane,
                    )
                else:
                    _solve(
                        free,
                        True,
                        w,
                        i - dense_capacity,
                        units[w, i],
                        sides[w, i],
                        art_start,
                        art_offset,
                        response,
                        v,
                        omega,
                        regularize,
                        iteration >= friction_start,
                        lane,
                    )
                u += _THREADS // 32
            _barrier()


_SHARED_DOFS = 11776  # 46 KB of float velocities per block, within the 48 KB static limit with the kernel's own scratch


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ float s_v[11776];
    if (op == 0) return s_v[i];
    if (op == 1) { s_v[i] = x; return x; }
    s_v[i] += x;
    return s_v[i];
#else
    return 0.0f;
#endif
""")
def _shared_v(op: int, i: int, x: float) -> float:
    """The block's world velocities in shared memory: 0 reads, 1 writes, 2 adds."""


@functools.cache
def sweep_kernel(lane_steps: int):
    """The sweep with the world's velocities in shared memory and each lane's DOF loop unrolled to ``lane_steps``.

    Same arithmetic and order as :func:`_sweep`; the velocities stay in the block's shared memory between
    units instead of global memory, and each lane issues all its coefficient loads of a row at once.
    """

    coeffs = wp.types.vector(length=lane_steps, dtype=float)

    @wp.func
    def load_side(rows: _Rows, free: bool, w: int, row: int, end: int, base: int, size: int, lane: int):
        """One lane's coefficients of a row on one side: all loads issued before any is used."""
        j = coeffs()
        y = coeffs()
        for step in range(wp.static(lane_steps)):
            k = lane + 32 * step
            if k < size:
                jk, yk = _coeff(rows, free, w, row, end, base, k)
                j[step] = jk
                y[step] = yk
        return j, y

    @wp.func
    def dot_v(j: coeffs, base: int, size: int, lane: int):
        acc = float(0.0)
        for step in range(wp.static(lane_steps)):
            k = lane + 32 * step
            if k < size:
                acc += j[step] * _shared_v(0, base + k, 0.0)
        return acc

    @wp.func
    def dot_c(j: coeffs, y: coeffs, size: int, lane: int):
        acc = float(0.0)
        for step in range(wp.static(lane_steps)):
            if lane + 32 * step < size:
                acc += j[step] * y[step]
        return acc

    @wp.func
    def add_v(y0: coeffs, y1: coeffs, d0: float, d1: float, base: int, size: int, lane: int):
        for step in range(wp.static(lane_steps)):
            k = lane + 32 * step
            if k < size:
                _shared_v(2, base + k, y0[step] * d0 + y1[step] * d1)

    @wp.func
    def solve(
        rows: _Rows,
        free: bool,
        world: int,
        normal: int,
        size: int,
        ab: wp.vec2i,
        offsets: wp.array[int],
        sizes: wp.array[int],
        omega: float,
        regularize: bool,
        friction: bool,
        lane: int,
    ):
        # Everything the unit reads before it changes anything: scalars and both sides' coefficients.
        t = normal + 1
        pair = size >= 3 and friction
        diagonal = rows.diag[world, normal]
        rhs_n = rows.rhs[world, normal]
        old = rows.impulse[world, normal]
        weight = float(1.0)
        if regularize:
            weight = rows.weight[world, normal]
        rhs_t = float(0.0)
        rhs_u = float(0.0)
        old_pair = wp.vec2()
        mu = float(0.0)
        a_t = float(0.0)
        b_t = float(0.0)
        parent = int(-1)
        if pair:
            rhs_t = rows.rhs[world, t]
            rhs_u = rows.rhs[world, t + 1]
            old_pair = wp.vec2(rows.impulse[world, t], rows.impulse[world, t + 1])
            mu = rows.mu[world, t]
            a_t = rows.diag[world, t]
            b_t = rows.diag[world, t + 1]
            parent = rows.parent[world, normal]
        a0 = ab[0]
        a1 = ab[1]
        on0 = a0 >= 0
        on1 = a1 >= 0 and a1 != a0
        base0 = int(0)
        size0 = int(0)
        base1 = int(0)
        size1 = int(0)
        if on0:
            base0 = offsets[a0]
            size0 = sizes[a0]
        if on1:
            base1 = offsets[a1]
            size1 = sizes[a1]
        jn0, yn0 = load_side(rows, free, world, normal, 0, base0, size0, lane)
        jn1, yn1 = load_side(rows, free, world, normal, 1, base1, size1, lane)
        jt0 = coeffs()
        yt0 = coeffs()
        ju0 = coeffs()
        yu0 = coeffs()
        jt1 = coeffs()
        yt1 = coeffs()
        ju1 = coeffs()
        yu1 = coeffs()
        if pair:
            jt0, yt0 = load_side(rows, free, world, t, 0, base0, size0, lane)
            ju0, yu0 = load_side(rows, free, world, t + 1, 0, base0, size0, lane)
            jt1, yt1 = load_side(rows, free, world, t, 1, base1, size1, lane)
            ju1, yu1 = load_side(rows, free, world, t + 1, 1, base1, size1, lane)
        updated = old
        if diagonal > 0.0:
            if not free:
                diagonal = 1.0 / diagonal
            jv = float(0.0)
            if on0:
                jv += dot_v(jn0, base0, size0, lane)
            if on1:
                jv += dot_v(jn1, base1, size1, lane)
            jv = _warp_sum(jv)
            step = -diagonal * (jv + rhs_n) * weight - (1.0 - weight) * old
            updated = wp.max(0.0, old + omega * step)
            _warp_sync()
            if lane == 0:
                rows.impulse[world, normal] = updated
            dn = updated - old
            if on0:
                add_v(yn0, yn0, dn, 0.0, base0, size0, lane)
            if on1:
                add_v(yn1, yn1, dn, 0.0, base1, size1, lane)
            _warp_sync()
        if pair:
            m0 = float(0.0)
            m1 = float(0.0)
            m2 = float(0.0)
            if on0:
                m0 += dot_v(jt0, base0, size0, lane)
                m1 += dot_v(ju0, base0, size0, lane)
                m2 += dot_c(jt0, yu0, size0, lane)
            if on1:
                m0 += dot_v(jt1, base1, size1, lane)
                m1 += dot_v(ju1, base1, size1, lane)
                m2 += dot_c(jt1, yu1, size1, lane)
            measure = wp.vec3(_warp_sum(m0), _warp_sum(m1), _warp_sum(m2))
            residual = wp.vec2(measure[0] + rhs_t, measure[1] + rhs_u)
            load = updated
            if parent >= 0:
                load = patch_normal_load(rows.parent, rows.impulse, world, normal)
            radius = wp.max(0.0, mu * load)
            if free:
                a_t = 1.0 / a_t if a_t > 0.0 else 0.0
                b_t = 1.0 / b_t if b_t > 0.0 else 0.0
            updated_pair = friction_pair_candidate(a_t, measure[2], b_t, residual, old_pair, radius, omega)
            mag = wp.length(updated_pair)
            if mag > radius:
                updated_pair *= radius / mag
            _warp_sync()
            if lane == 0:
                rows.impulse[world, t] = updated_pair[0]
                rows.impulse[world, t + 1] = updated_pair[1]
            dp = updated_pair - old_pair
            if on0:
                add_v(yt0, yu0, dp[0], dp[1], base0, size0, lane)
            if on1:
                add_v(yt1, yu1, dp[0], dp[1], base1, size1, lane)
            _warp_sync()

    def sweep_shared(
        dense: _Rows,
        free: _Rows,
        units: wp.array2d[int],
        sides: wp.array2d[wp.vec2i],
        order: wp.array2d[int],
        offsets: wp.array2d[int],
        counts: wp.array[int],
        art_offset: wp.array[int],
        response: wp.array[int],
        world_dofs: wp.array[int],
        world_dof_indices: wp.array2d[int],
        dense_capacity: int,
        omega: float,
        regularize: bool,
        friction_start: int,
        iterations: int,
        v: wp.array[float],
    ):
        w, thread = wp.tid()
        lane = thread % 32
        n = world_dofs[w]
        for i in range(thread, n, _THREADS):
            _shared_v(1, i, v[world_dof_indices[w, i]])
        _barrier()
        for iteration in range(iterations):
            for color in range(counts[w]):
                u = offsets[w, color] + thread // 32
                while u < offsets[w, color + 1]:
                    i = order[w, u]
                    if i < dense_capacity:
                        solve(dense, False, w, i, units[w, i], sides[w, i], art_offset, response, omega, regularize,
                              iteration >= friction_start, lane)  # fmt: skip
                    else:
                        solve(free, True, w, i - dense_capacity, units[w, i], sides[w, i], art_offset, response,
                              omega, regularize, iteration >= friction_start, lane)  # fmt: skip
                    u += _THREADS // 32
                _barrier()
        for i in range(thread, n, _THREADS):
            v[world_dof_indices[w, i]] = _shared_v(0, i, 0.0)

    sweep_shared.__name__ = f"immediate_sweep_shared_{lane_steps}"
    sweep_shared.__qualname__ = sweep_shared.__name__
    return wp.kernel(enable_backward=False, module="unique")(sweep_shared)


class ImmediateColoring:
    def __init__(self, solver):
        self.device = solver.device
        self.worlds = solver._world_count
        self.dense_capacity = solver._articulation_constraint_max
        capacity = self.dense_capacity + solver._mf_impulses.shape[1]
        shape = (self.worlds, capacity)
        self.units = wp.zeros(shape, dtype=int, device=self.device)
        self.sides = wp.empty(shape, dtype=wp.vec2i, device=self.device)
        self.colors = wp.empty_like(self.units)
        self.order = wp.empty_like(self.units)
        self.cursor = wp.empty_like(self.units)
        self.offsets = wp.empty((self.worlds, capacity + 1), dtype=int, device=self.device)
        self.last = wp.empty(solver.model.articulation_count, dtype=int, device=self.device)
        self.count = wp.empty(self.worlds, dtype=int, device=self.device)
        # Worlds whose velocities fit shared memory sweep there, each lane unrolling its share of the widest row.
        widest = int(solver._articulation_response_dof_count.numpy().max(initial=0))
        self.lane_steps = max((widest + 31) // 32, 1)
        self.shared = (
            0 < solver._max_world_dofs <= _SHARED_DOFS
            and getattr(solver, "_world_dof_indices", None) is not None
            and bool(solver._kernel_overrides.get("immediate_shared_sweep", True))
        )

    def prepare(self, solver, contacts):
        self.units.zero_()
        self.cursor.zero_()
        self.last.zero_()
        if contacts is not None and contacts.rigid_contact_max:
            wp.launch(
                _collect,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    solver._contact_path,
                    solver._contact_world,
                    solver._contact_slot,
                    solver._contact_slots_needed,
                    solver._contact_art_a,
                    solver._contact_art_b,
                    self.dense_capacity,
                    self.units,
                    self.sides,
                ],
                device=self.device,
            )
        wp.launch(
            _color,
            dim=self.worlds,
            inputs=[
                self.units,
                self.sides,
                solver._articulation_response_dof_count,
                solver._constraint_count,
                solver._mf_constraint_count,
                self.dense_capacity,
                solver._solve_mode == "serial",
                self.last,
                self.colors,
                self.offsets,
                self.cursor,
                self.order,
                self.count,
            ],
            device=self.device,
        )

    def solve(self, solver, rhs, iterations, regularize, friction_start):
        dense, free = _Rows(), _Rows()
        dense.packed_sides = solver._packed_rod_size is not None
        if dense.packed_sides:
            size = solver._packed_rod_size
            dense.J, dense.Y = solver._J_by_size[size], solver._Y_by_size[size]
        else:
            dense.J, dense.Y = solver._J_world, solver._Y_world
        dense.rhs, dense.diag = rhs, solver._diag
        dense.mu, dense.weight = solver._row_mu, solver._row_w
        dense.parent, dense.impulse = solver._row_parent, solver._impulses
        free.Ja, free.Jb = solver._mf_J_a, solver._mf_J_b
        free.Ya, free.Yb = solver._mf_MiJt_a, solver._mf_MiJt_b
        free.rhs, free.diag = solver._immediate_mf_rhs, solver._mf_eff_mass_inv
        free.mu, free.weight = solver._mf_row_mu, solver._mf_row_w
        free.parent, free.impulse = solver._mf_row_parent, solver._mf_impulses
        if self.shared:
            # The world's velocities fit the block's shared memory: the units read and write them there.
            wp.launch(
                sweep_kernel(self.lane_steps),
                dim=(self.worlds, _THREADS),
                block_dim=_THREADS,
                inputs=[
                    dense,
                    free,
                    self.units,
                    self.sides,
                    self.order,
                    self.offsets,
                    self.count,
                    solver._articulation_world_dof_offset,
                    solver._articulation_response_dof_count,
                    solver._world_dof_count,
                    solver._world_dof_indices,
                    self.dense_capacity,
                    solver.relaxation,
                    regularize,
                    friction_start,
                    iterations,
                    solver._v_out,
                ],
                device=self.device,
            )
            return
        wp.launch(
            _sweep,
            dim=(self.worlds, _THREADS),
            block_dim=_THREADS,
            inputs=[
                dense,
                free,
                self.units,
                self.sides,
                self.order,
                self.offsets,
                self.count,
                solver._articulation_dof_start,
                solver._articulation_world_dof_offset,
                solver._articulation_response_dof_count,
                self.dense_capacity,
                solver.relaxation,
                regularize,
                friction_start,
                iterations,
                solver._v_out,
            ],
            device=self.device,
        )


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    const int n = factor.shape[1];
    const int pitch = jac.shape[1];
    const int base = (end * pitch + row) * n;
    const float *L = factor.data + group * n * n;
    float *y = response.data + base;
    const float *j = jac.data + base;
    unsigned mask = 0xffffffff;
    // Forward and backward triangular solves. One warp owns an endpoint
    // and publishes each pivot to the other lanes before they consume it.
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int k = lane; k < i; k += 32) sum += L[i*n+k] * y[k];
        for (int k = 16; k; k >>= 1) sum += __shfl_down_sync(mask,sum,k);
        if (!lane) y[i] = (j[i] - sum) / L[i*n+i];
        __syncwarp(mask);
    }
    for (int i = n-1; i >= 0; --i) {
        float sum = 0.0f;
        for (int k = i+1+lane; k < n; k += 32) sum += L[k*n+i] * y[k];
        for (int k = 16; k; k >>= 1) sum += __shfl_down_sync(mask,sum,k);
        if (!lane) y[i] = (y[i] - sum) / L[i*n+i];
        __syncwarp(mask);
    }
    float diag = 0.0f;
    for (int k = lane; k < n; k += 32) diag += j[k]*y[k];
    for (int k = 16; k; k >>= 1) diag += __shfl_down_sync(mask,diag,k);
    if (!lane) atomicAdd(diagonal.data + row,diag);
#endif
""")
def _factor_response(
    lane: int,
    end: int,
    row: int,
    group: int,
    jac: wp.array3d[float],
    factor: wp.array3d[float],
    response: wp.array3d[float],
    diagonal: wp.array2d[float],
):
    """Apply one cable's augmented factor to one side of a constraint row."""


@wp.kernel(enable_backward=False)
def packed_responses(
    workers: int,
    count: wp.array[int],
    paths: wp.array[int],
    slots: wp.array[int],
    widths: wp.array[int],
    side_a: wp.array[int],
    side_b: wp.array[int],
    groups: wp.array[int],
    jac: wp.array3d[float],
    factor: wp.array3d[float],
    response: wp.array3d[float],
    diagonal: wp.array2d[float],
):
    worker, unit, lane = wp.tid()
    for contact in range(worker, wp.min(count[0], paths.shape[0]), workers):
        if paths[contact] != 0 or slots[contact] < 0:
            continue
        side = unit % 2
        row = unit // 2
        art = side_a[contact]
        if side == 1:
            art = side_b[contact]
            if art == side_a[contact]:
                continue
        if art >= 0 and row < widths[contact]:
            _factor_response(lane, side, slots[contact] + row, groups[art], jac, factor, response, diagonal)


@wp.kernel
def packed_restitution(
    count: wp.array[int],
    path: wp.array[int],
    slot: wp.array[int],
    side_a: wp.array[int],
    side_b: wp.array[int],
    starts: wp.array[int],
    jac: wp.array3d[float],
    speed: wp.array[float],
    gap: wp.array2d[float],
    target: wp.array2d[float],
    bounce: wp.array2d[float],
    dt: float,
    regularized: bool,
    rhs: wp.array2d[float],
    weight: wp.array2d[float],
):
    c = wp.tid()
    if c >= count[0] or path[c] != 0:
        return
    r = slot[c]
    if r < 0 or bounce[0, r] <= 0.0:
        return
    incident = -target[0, r]
    for end in range(2):
        a = side_a[c] if end == 0 else side_b[c]
        if a >= 0 and (end == 0 or a != side_a[c]):
            for k in range(jac.shape[2]):
                incident += jac[end, r, k] * speed[starts[a] + k]
    if contact_restitution_fires(gap[0, r], incident, dt):
        rhs[0, r] = -target[0, r] - bounce[0, r]
        if regularized:
            weight[0, r] = 1.0
