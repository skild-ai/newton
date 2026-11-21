# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serial propagation sweep of one world per warp over contact units, with the world's twists in shared memory.

The serial sweep visits a world's rows in order. Here a contact's normal row and its sliding pair are one unit:
the unit's rows are staged in shared memory a chunk at a time, butterflies reduce the unit's products (each
row's residual against the twists, in the row sweep's order, and the pair's coupling), and every lane solves the
unit. The twists and deferred impulses of the world's bodies live in shared memory for the sweep. Other rows (rod
closures, contacts without a pair) take one row at a time on the same twists. The Gauss-Seidel order and update
are those of the row sweep.
"""

import functools

import warp as wp

from .friction import FRICTION_PAIR_CUDA
from .friction_patches import WARP_DETERMINISM_OFF
from .kernels import PGS_CONSTRAINT_TYPE_CONNECT, PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION

wp.set_module_options(WARP_DETERMINISM_OFF)

# Rows a world stages in shared memory at a time, and the shared memory a world takes: its bodies' twists and
# impulses (12 words each) and a chunk of rows (33 words each).
CHUNK_ROWS = 96
_STATIC_SHARED_BYTES = 48 * 1024


def _world_bytes(max_bodies: int) -> int:
    return 4 * (12 * max(max_bodies, 1) + 33 * CHUNK_ROWS)


def fits(max_bodies: int) -> bool:
    """Whether a world's responding bodies fit the sweep's shared memory."""
    return 0 < max_bodies and _world_bytes(max_bodies) <= _STATIC_SHARED_BYTES


def worlds_per_block(max_bodies: int) -> int:
    """Worlds (warps) per block within the static shared memory."""
    return max(1, min(4, _STATIC_SHARED_BYTES // _world_bytes(max_bodies)))


_SOURCE = (
    r"""
#if defined(__CUDA_ARCH__)
"""
    + FRICTION_PAIR_CUDA
    + r"""
    const int lane = threadIdx.x & 31;
    const int wib = threadIdx.x >> 5;
    const int world = tile * SU_WPB + wib;
    __shared__ float s_qd_all[SU_WPB * SU_MB * 6];
    __shared__ float s_imp_all[SU_WPB * SU_MB * 6];
    if (world >= world_count) return;
    float* s_qd = s_qd_all + wib * SU_MB * 6;
    float* s_imp = s_imp_all + wib * SU_MB * 6;
    int m = propagation_constraint_count.data[world];
    if (m > SU_M) m = SU_M;
    const int world_base = world * SU_M;
    int nb = propagation_body_count.data[world];
    if (nb > SU_MB) nb = SU_MB;
    for (int e = lane; e < nb * 6; e += 32) {
        const int body = propagation_body_list.data[world * SU_MB + e / 6];
        s_qd[e] = propagation_body_qd.data[body * 6 + e % 6];
        s_imp[e] = 0.0f;
    }
    __syncwarp();

    // Rows are staged in shared memory a chunk at a time with asynchronous copies, so one memory latency serves
    // a chunk of units instead of each unit waiting on its own loads. Per row: J (A then B), M^-1 J^T (A then B),
    // eff, rhs, impulse, mu, w; and type, parent, and the two bodies' twist slots.
    __shared__ float sf_all[SU_WPB * SU_CH * 29];
    __shared__ int si_all[SU_WPB * SU_CH * 4];
    float* sf = sf_all + wib * SU_CH * 29;
    int* si = si_all + wib * SU_CH * 4;
    auto cpa = [](void* dst, const void* src) {
        const unsigned dst_s = (unsigned)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst_s), "l"(src));
    };
    int c0 = 0, c1 = 0;
    auto stage = [&](int start) {
        __syncwarp();
        const int cnt = min(SU_CH, m - start);
        for (int e = lane; e < cnt * 12; e += 32) {
            const int r = e / 12, c = e - r * 12;
            const int off = world_base + start + r;
            cpa(&sf[r * 29 + c], (c < 6 ? propagation_J_a.data : propagation_J_b.data) + off * 6 + (c < 6 ? c : c - 6));
            cpa(&sf[r * 29 + 12 + c],
                (c < 6 ? propagation_MiJt_a.data : propagation_MiJt_b.data) + off * 6 + (c < 6 ? c : c - 6));
        }
        for (int e = lane; e < cnt; e += 32) {
            const int off = world_base + start + e;
            cpa(&sf[e * 29 + 24], propagation_eff_mass_inv.data + off);
            cpa(&sf[e * 29 + 25], propagation_rhs.data + off);
            cpa(&sf[e * 29 + 26], propagation_impulses.data + off);
            cpa(&sf[e * 29 + 27], propagation_row_mu.data + off);
            if (row_w_full) cpa(&sf[e * 29 + 28], propagation_row_w.data + off);
            else sf[e * 29 + 28] = 1.0f;
            cpa(&si[e * 4 + 0], propagation_row_type.data + off);
            cpa(&si[e * 4 + 1], propagation_row_parent.data + off);
            cpa(&si[e * 4 + 2], propagation_row_la.data + off);
            cpa(&si[e * 4 + 3], propagation_row_lb.data + off);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncwarp();
        c0 = start;
        c1 = start + cnt;
    };

    // Lanes 0-5 own body A's six components of a row, lanes 6-11 body B's.
    const bool side_a = lane < 6;
    const int k = side_a ? lane : lane - 6;
    const bool active_lane = lane < 12;
    auto reduce = [&](float v) {
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        // Lanes 0-15 now hold the 16-lane sum; broadcast lane 0's.
        return __shfl_sync(0xffffffffu, v, 0);
    };

    // The row sweep accumulates a sibling row's residual and a pair's coupling in lane 0, body A's then body B's
    // component k for k = 0..5; these do the same, so the two sweeps round alike.
    auto seq_dot = [&](float x, float y, float acc, bool has_a, bool has_b) {
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            const float xa = __shfl_sync(0xffffffffu, x, c), ya = __shfl_sync(0xffffffffu, y, c);
            const float xb = __shfl_sync(0xffffffffu, x, 6 + c), yb = __shfl_sync(0xffffffffu, y, 6 + c);
            if (has_a) acc = fmaf(xa, ya, acc);
            if (has_b) acc = fmaf(xb, yb, acc);
        }
        return acc;
    };

    // A window of three rows from i: a unit (contact and sliding pair) when it is one, else row i (and its
    // pair sibling at i + 1). Every load is indexed by row, so the next window loads while this one solves.
    struct Win {
        int t0, t1, t2, p1, la, lb, pn;
        float eff[3], rhs[3], imp[3], mu1, w;
        float j[3], mi[3];
    };
    auto load = [&](int i, Win& o) {
        if (i + 3 > c1 && c1 < m) stage(i);
        const int r = i - c0;
        const bool r1 = i + 1 < m, r2 = i + 2 < m;
        o.t0 = si[r * 4];
        o.t1 = r1 ? si[(r + 1) * 4] : -1;
        o.t2 = r2 ? si[(r + 2) * 4] : -1;
        o.p1 = r1 ? si[(r + 1) * 4 + 1] : -1;
        o.pn = si[r * 4 + 1];
        o.la = si[r * 4 + 2];
        o.lb = si[r * 4 + 3];
        o.mu1 = r1 ? sf[(r + 1) * 29 + 27] : 0.0f;
        o.w = sf[r * 29 + 28];
        #pragma unroll
        for (int q = 0; q < 3; ++q) {
            const bool ok = i + q < m;
            o.eff[q] = ok ? sf[(r + q) * 29 + 24] : 0.0f;
            o.rhs[q] = ok ? sf[(r + q) * 29 + 25] : 0.0f;
            o.imp[q] = ok ? sf[(r + q) * 29 + 26] : 0.0f;
            o.j[q] = (ok && active_lane) ? sf[(r + q) * 29 + lane] : 0.0f;
            o.mi[q] = (ok && active_lane) ? sf[(r + q) * 29 + 12 + lane] : 0.0f;
        }
    };

    int i = 0;
    Win cur;
    while (i < m) {
        load(i, cur);
        const bool unit = cur.t0 == SU_CONTACT && cur.t1 == SU_FRICTION && cur.p1 == i && cur.t2 == SU_FRICTION;
        const bool pair_first = !unit && cur.t0 == SU_FRICTION && cur.t1 >= 0 && i == cur.pn + 1;
        const int next = i + (unit ? 3 : (pair_first && global_iter >= friction_start_iteration ? 2 : 1));
        const int off = world_base + i;
        const int my_slot = side_a ? cur.la : cur.lb;
        const bool my_on = active_lane && my_slot >= 0;
        const float v = my_on ? s_qd[my_slot * 6 + k] : 0.0f;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
        if (unit) {
            // The row sweep's order: the normal row, then the pair against the twists the normal impulse moved.
            const float s_n = reduce(cur.j[0] * v);
            const bool has_a = cur.la >= 0, has_b = cur.lb >= 0;
            const float c_12 = seq_dot(cur.j[1], cur.mi[2], 0.0f, has_a, has_b);
            float new_n = cur.imp[0];
            if (cur.eff[0] > 0.0f) {
                new_n = cur.imp[0] + omega * (-(s_n + cur.rhs[0]) * cur.eff[0] * cur.w - (1.0f - cur.w) * cur.imp[0]);
                if (new_n < 0.0f) new_n = 0.0f;
            }
            d0 = new_n - cur.imp[0];
            const float v1 = v + cur.mi[0] * d0;
            float new_1 = 0.0f, new_2 = 0.0f;
            if (global_iter >= friction_start_iteration) {
                const float s_1 = reduce(cur.j[1] * v1);
                const float r_2 = seq_dot(cur.j[2], v1, cur.rhs[2], has_a, has_b);
                float lambda_n = new_n;
                for (int p = cur.pn; p >= 0 && p != i; p = propagation_row_parent.data[world_base + p])
                    lambda_n += propagation_impulses.data[world_base + p];
                const float radius = fmaxf(cur.mu1 * lambda_n, 0.0f);
                const float2 pair = friction_pair_candidate(cur.eff[1] > 0.0f ? 1.0f / cur.eff[1] : 0.0f, c_12,
                    cur.eff[2] > 0.0f ? 1.0f / cur.eff[2] : 0.0f, s_1 + cur.rhs[1], r_2, cur.imp[1],
                    cur.imp[2], radius, omega);
                const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = mag > radius ? radius / mag : 1.0f;
                new_1 = pair.x * scale;
                new_2 = pair.y * scale;
                d1 = new_1 - cur.imp[1];
                d2 = new_2 - cur.imp[2];
            }
            if (lane == 0) {
                propagation_impulses.data[off] = new_n;
                propagation_impulses.data[off + 1] = new_1;
                propagation_impulses.data[off + 2] = new_2;
            }
        } else if (pair_first && global_iter >= friction_start_iteration) {
            // A sliding pair outside a unit: its first row solves both of its rows.
            const float s0 = reduce(cur.j[0] * v);
            const float r1 = seq_dot(cur.j[1], v, cur.rhs[1], cur.la >= 0, cur.lb >= 0);
            const float cross = seq_dot(cur.j[0], cur.mi[1], 0.0f, cur.la >= 0, cur.lb >= 0);
            const int parent = cur.pn;
            float lambda_n = propagation_impulses.data[world_base + parent];
            for (int p = propagation_row_parent.data[world_base + parent]; p >= 0 && p != parent;
                 p = propagation_row_parent.data[world_base + p])
                lambda_n += propagation_impulses.data[world_base + p];
            const float radius = fmaxf(propagation_row_mu.data[off] * lambda_n, 0.0f);
            const float2 pair = friction_pair_candidate(cur.eff[0] > 0.0f ? 1.0f / cur.eff[0] : 0.0f, cross,
                cur.eff[1] > 0.0f ? 1.0f / cur.eff[1] : 0.0f, s0 + cur.rhs[0], r1, cur.imp[0], cur.imp[1],
                radius, omega);
            const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
            const float scale = mag > radius ? radius / mag : 1.0f;
            d0 = pair.x * scale - cur.imp[0];
            d1 = pair.y * scale - cur.imp[1];
            if (lane == 0) {
                propagation_impulses.data[off] = cur.imp[0] + d0;
                propagation_impulses.data[off + 1] = cur.imp[1] + d1;
            }
        } else if (cur.t0 == SU_FRICTION) {
            if (global_iter < friction_start_iteration && lane == 0) propagation_impulses.data[off] = 0.0f;
        } else if (cur.eff[0] > 0.0f) {
            // A contact without a pair, or a rod closure.
            const float s0 = reduce(cur.j[0] * v);
            const float wr = (cur.t0 == SU_CONTACT && !regularize) ? 1.0f : cur.w;
            float nw = cur.imp[0] + omega * (-(s0 + cur.rhs[0]) * cur.eff[0] * wr - (1.0f - wr) * cur.imp[0]);
            if (cur.t0 == SU_CONTACT && nw < 0.0f) nw = 0.0f;
            d0 = nw - cur.imp[0];
            if (lane == 0) propagation_impulses.data[off] = nw;
        }
        if (my_on && (d0 != 0.0f || d1 != 0.0f || d2 != 0.0f)) {
            // Added one row at a time, as the row sweep does.
            float q = v, im = s_imp[my_slot * 6 + k];
            if (d0 != 0.0f) { q += cur.mi[0] * d0; im += cur.j[0] * d0; }
            if (d1 != 0.0f) { q += cur.mi[1] * d1; im += cur.j[1] * d1; }
            if (d2 != 0.0f) { q += cur.mi[2] * d2; im += cur.j[2] * d2; }
            s_qd[my_slot * 6 + k] = q;
            s_imp[my_slot * 6 + k] = im;
        }
        __syncwarp();
        i = next;
    }

    for (int e = lane; e < nb * 6; e += 32) {
        const int body = propagation_body_list.data[world * SU_MB + e / 6];
        propagation_body_qd.data[body * 6 + e % 6] = s_qd[e];
        propagation_body_impulses.data[body * 6 + e % 6] += s_imp[e];
    }
#endif
"""
)


@wp.kernel(enable_backward=False)
def fill_row_slots(
    propagation_constraint_count: wp.array[int],
    propagation_max_constraints: int,
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_body_local_slot: wp.array[int],
    # outputs
    propagation_row_la: wp.array2d[int],
    propagation_row_lb: wp.array2d[int],
):
    """Write each row's two bodies' shared-memory slots, so the sweep's row loads do not wait on its bodies."""
    world, i = wp.tid()
    if i >= wp.min(propagation_constraint_count[world], propagation_max_constraints):
        return
    ba = propagation_body_a[world, i]
    bb = propagation_body_b[world, i]
    la = int(-1)
    lb = int(-1)
    if ba >= 0:
        la = propagation_body_local_slot[ba]
    if bb >= 0:
        lb = propagation_body_local_slot[bb]
    propagation_row_la[world, i] = la
    propagation_row_lb[world, i] = lb


@functools.cache
def sweep_kernel(max_rows: int, max_bodies: int) -> wp.Kernel:
    """Kernel running one serial unit sweep of every world; launch ``launch_dim`` tiles of ``block_dim`` threads."""
    wpb = worlds_per_block(max_bodies)
    defines = (
        f"#define SU_WPB {wpb}\n#define SU_MB {max(max_bodies, 1)}\n#define SU_M {max_rows}\n"
        f"#define SU_CONTACT {int(PGS_CONSTRAINT_TYPE_CONTACT)}\n#define SU_FRICTION {int(PGS_CONSTRAINT_TYPE_FRICTION)}\n"
        f"#define SU_CONNECT {int(PGS_CONSTRAINT_TYPE_CONNECT)}\n#define SU_CH {CHUNK_ROWS}\n"
    )
    source = _SOURCE.replace("#if defined(__CUDA_ARCH__)\n", "#if defined(__CUDA_ARCH__)\n" + defines, 1)

    @wp.func_native(source)
    def serial_units_native(
        tile: int,
        world_count: int,
        propagation_constraint_count: wp.array[int],
        propagation_body_count: wp.array[int],
        propagation_body_list: wp.array2d[int],
        propagation_body_local_slot: wp.array[int],
        propagation_row_la: wp.array2d[int],
        propagation_row_lb: wp.array2d[int],
        propagation_body_a: wp.array2d[int],
        propagation_body_b: wp.array2d[int],
        propagation_MiJt_a: wp.array3d[float],
        propagation_MiJt_b: wp.array3d[float],
        propagation_J_a: wp.array3d[float],
        propagation_J_b: wp.array3d[float],
        propagation_eff_mass_inv: wp.array2d[float],
        propagation_rhs: wp.array2d[float],
        propagation_row_type: wp.array2d[int],
        propagation_row_parent: wp.array2d[int],
        propagation_row_mu: wp.array2d[float],
        propagation_row_w: wp.array2d[float],
        row_w_full: int,
        omega: float,
        regularize: int,
        friction_start_iteration: int,
        global_iter: int,
        propagation_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        propagation_body_impulses: wp.array2d[float],
    ): ...

    def serial_units(  # pragma: no cover - compiled by wp.kernel
        world_count: int,
        propagation_constraint_count: wp.array[int],
        propagation_body_count: wp.array[int],
        propagation_body_list: wp.array2d[int],
        propagation_body_local_slot: wp.array[int],
        propagation_row_la: wp.array2d[int],
        propagation_row_lb: wp.array2d[int],
        propagation_body_a: wp.array2d[int],
        propagation_body_b: wp.array2d[int],
        propagation_MiJt_a: wp.array3d[float],
        propagation_MiJt_b: wp.array3d[float],
        propagation_J_a: wp.array3d[float],
        propagation_J_b: wp.array3d[float],
        propagation_eff_mass_inv: wp.array2d[float],
        propagation_rhs: wp.array2d[float],
        propagation_row_type: wp.array2d[int],
        propagation_row_parent: wp.array2d[int],
        propagation_row_mu: wp.array2d[float],
        propagation_row_w: wp.array2d[float],
        row_w_full: int,
        omega: float,
        regularize: int,
        friction_start_iteration: int,
        global_iter: int,
        propagation_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        propagation_body_impulses: wp.array2d[float],
    ):
        tile, _lane = wp.tid()
        serial_units_native(
            tile,
            world_count,
            propagation_constraint_count,
            propagation_body_count,
            propagation_body_list,
            propagation_body_local_slot,
            propagation_row_la,
            propagation_row_lb,
            propagation_body_a,
            propagation_body_b,
            propagation_MiJt_a,
            propagation_MiJt_b,
            propagation_J_a,
            propagation_J_b,
            propagation_eff_mass_inv,
            propagation_rhs,
            propagation_row_type,
            propagation_row_parent,
            propagation_row_mu,
            propagation_row_w,
            row_w_full,
            omega,
            regularize,
            friction_start_iteration,
            global_iter,
            propagation_impulses,
            propagation_body_qd,
            propagation_body_impulses,
        )

    name = f"pgs_solve_propagation_serial_units_{max_rows}_mb{max_bodies}_w{wpb}"
    serial_units.__name__ = name
    serial_units.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(serial_units)
