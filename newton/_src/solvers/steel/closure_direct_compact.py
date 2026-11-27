# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact contact rows for the sparse direct closure solve (``closure_direct_contacts="elim"``).

The contact rows' responses ``Y = K^-1 J^T`` come from the closure factor, so a contact impulse moves the velocity
along the closures' (regularized) constraint manifold; the closure rows are not swept again. Only the contact and
friction rows are stored, in a compact layout:

- ``J`` on its support. A contact row on body ``b`` is nonzero only on ``b``'s DOF root path, so it is stored by depth
  from the path's deepest DOF ``h``: ``cJ[w, r, s]`` is ``J`` at the ``s``-th ancestor of ``h`` (``s < na[h]``). The
  row's head ``chead[w, r]`` packs ``h + 1`` per side (16 bits each); side b is stored at depth offset ``KZ`` only
  when two bodies of the articulation can collide (self-contact), so ground contacts cost ``KZ`` floats per row.
- ``Y`` dense over the world's DOFs, ``cY[w, r, :]``: ``K^-1`` couples every DOF of a loop-connected articulation
  (the measured fill of a contact response is all of them).

Rows are indexed by ``r = slot - cbase[w]``, the slot past the world's closure rows, with capacity the contact-row
budget ``M_D`` minus the closure rows. Every offset is 64-bit. Row order is Steel's ordered (sorted) contact rows,
so it is deterministic; the kernels here use no atomics on the data path.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from .friction import FRICTION_PAIR_CUDA, contact_tangent_basis
from .friction_patches import WARP_DETERMINISM_OFF, FrictionPatches
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    _contact_points_world,
    contact_restitution_fires,
    contact_row_points,
)

wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})

# status entries written here (DirectSparseBuffers.status): rows past the compact capacity, self-contact sides
# dropped because the layout stores one side, rows of other types in the contact range.
STATUS_COMPACT_OVERFLOW = 4
STATUS_COMPACT_SIDE = 5
STATUS_COMPACT_TYPE = 6


def self_contacts_possible(solver, art: int) -> bool:
    """Whether two shapes on different bodies of articulation ``art`` can collide (groups and filter pairs)."""
    model = solver.model
    if model.shape_count == 0:
        return False
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    group = model.shape_collision_group.numpy()
    body_art = solver._body_to_articulation.numpy()
    on_body = shape_body >= 0
    sel = np.flatnonzero(
        on_body
        & (body_art[np.maximum(shape_body, 0)] == art)
        & ((flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0)
        & (group != 0)
    )
    if len(sel) < 2:
        return False
    g, b = group[sel].astype(np.int64), shape_body[sel]
    # Distinct groups (or a positive group shared) are the only candidates; one negative group never self-collides.
    if len(np.unique(g)) == 1 and g[0] < 0:
        return False
    i, j = np.triu_indices(len(sel), 1)
    gi, gj = g[i], g[j]
    collide = np.where(gi > 0, (gi == gj) | (gj < 0), np.where(gj > 0, True, gi != gj)) & (b[i] != b[j])
    pairs = np.stack([sel[i[collide]], sel[j[collide]]], axis=1).astype(np.int64)
    if len(pairs) == 0:
        return False
    filtered = model.shape_collision_filter_pairs.mask_pairs(pairs)
    return bool((~filtered).any())


@wp.func
def _compact_side_walk(
    world: int,
    rc: int,
    offset: int,
    body: int,
    sign: float,
    point_world: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
    dof_start: int,
    n_dofs: int,
    head: int,
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    na: wp.array[int],
    cJ: wp.array3d[float],
):
    """Write one side of a row by depth along the root path of its head DOF ``head`` (local)."""
    point_rel = point_world - origin
    depth = na[head]
    curr_joint = body_to_joint[body]
    while curr_joint >= 0:
        for global_dof in range(joint_qd_start[curr_joint], joint_qd_start[curr_joint + 1]):
            local = global_dof - dof_start
            if local >= 0 and local < n_dofs:
                S = joint_S_s[global_dof]
                v = wp.vec3(S[0], S[1], S[2]) + wp.cross(wp.vec3(S[3], S[4], S[5]), point_rel)
                cJ[world, rc, offset + depth - na[local]] = sign * wp.dot(direction, v)
        curr_joint = joint_ancestor[curr_joint]


@wp.func
def _compact_head(
    body: int,
    dof_start: int,
    n_dofs: int,
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
):
    """Deepest local DOF on the root path of ``body`` (the last DOF of the first joint with DOFs), or -1."""
    curr_joint = body_to_joint[body]
    while curr_joint >= 0:
        last = joint_qd_start[curr_joint + 1] - 1 - dof_start
        if joint_qd_start[curr_joint + 1] > joint_qd_start[curr_joint] and last >= 0 and last < n_dofs:
            return last
        curr_joint = joint_ancestor[curr_joint]
    return -1


@wp.kernel
def ds_compact_contact_rows(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    friction_patches: FrictionPatches,
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
    cbase: wp.array[int],
    na: wp.array[int],
    n_dofs: int,
    capacity: int,
    side_stride: int,
    two_sided: int,
    # outputs
    cJ: wp.array3d[float],
    chead: wp.array2d[int],
    status: wp.array[int],
):
    """Build each dense contact's rows in the compact layout: ``J`` by depth along each side's root path."""
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(wp.tid(), total_contacts, total_num_threads):
        if contact_path[c] != 0:
            continue
        slot = contact_slot[c]
        if slot < 0:
            continue
        world = contact_world[c]
        rc0 = slot - cbase[world]
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        normal = -contact_normal[c]
        shape_a = contact_shape0[c]
        shape_b = contact_shape1[c]
        body_a = -1
        body_b = -1
        if shape_a >= 0:
            body_a = shape_body[shape_a]
        if shape_b >= 0:
            body_b = shape_body[shape_b]
        point_a_world, point_b_world = _contact_points_world(
            c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
        )
        t0, t1 = contact_tangent_basis(normal)
        normal_point_a, normal_point_b = contact_row_points(
            c, 0, point_a_world, point_b_world, contact_shared_anchor, contact_friction_shared_anchor, friction_patches
        )
        friction_point_a, friction_point_b = contact_row_points(
            c, 1, point_a_world, point_b_world, contact_shared_anchor, contact_friction_shared_anchor, friction_patches
        )
        # Heads of both sides (local DOFs); a side off the articulation or without DOFs stores nothing.
        head_a = int(-1)
        head_b = int(-1)
        if art_a >= 0 and body_a >= 0:
            head_a = _compact_head(body_a, art_dof_start[art_a], n_dofs, body_to_joint, joint_ancestor, joint_qd_start)
        if art_b >= 0 and body_b >= 0:
            head_b = _compact_head(body_b, art_dof_start[art_b], n_dofs, body_to_joint, joint_ancestor, joint_qd_start)
        if head_a >= 0 and head_b >= 0 and two_sided == 0:
            # The layout was sized for one side per row (no self-collisions expected): drop side b, flag it.
            wp.atomic_add(status, STATUS_COMPACT_SIDE, 1)
            head_b = -1
        row_count = contact_slots_needed[c]
        for row in range(row_count):
            rc = rc0 + row
            if rc < 0 or rc >= capacity:
                wp.atomic_add(status, STATUS_COMPACT_OVERFLOW, 1)
                continue
            direction = normal
            pa = normal_point_a
            pb = normal_point_b
            if row == 1:
                direction = t0
                pa = friction_point_a
                pb = friction_point_b
            elif row == 2:
                direction = t1
                pa = friction_point_a
                pb = friction_point_b
            stored = int(0)
            packed = int(0)
            if head_a >= 0:
                _compact_side_walk(
                    world,
                    rc,
                    0,
                    body_a,
                    1.0,
                    pa,
                    articulation_origin[art_a],
                    direction,
                    art_dof_start[art_a],
                    n_dofs,
                    head_a,
                    body_to_joint,
                    joint_ancestor,
                    joint_qd_start,
                    joint_S_s,
                    na,
                    cJ,
                )
                packed = head_a + 1
                stored = 1
            if head_b >= 0:
                _compact_side_walk(
                    world,
                    rc,
                    stored * side_stride,
                    body_b,
                    -1.0,
                    pb,
                    articulation_origin[art_b],
                    direction,
                    art_dof_start[art_b],
                    n_dofs,
                    head_b,
                    body_to_joint,
                    joint_ancestor,
                    joint_qd_start,
                    joint_S_s,
                    na,
                    cJ,
                )
                packed = packed | ((head_b + 1) << (16 * stored))
            chead[world, rc] = packed


@wp.func
def compact_row_dot(
    world: int,
    rc: int,
    chead: wp.array2d[int],
    cJ: wp.array3d[float],
    ancestors: wp.array2d[int],
    na: wp.array[int],
    side_stride: int,
    world_dof_indices: wp.array2d[int],
    velocity: wp.array[float],
):
    """Dot one compact row with a generalized velocity (global DOF indexing)."""
    packed = chead[world, rc]
    out = float(0.0)
    for side in range(2):
        head = ((packed >> (16 * side)) & 0xFFFF) - 1
        if head < 0:
            continue
        for s in range(na[head]):
            gd = world_dof_indices[world, ancestors[head, s]]
            if gd >= 0:
                out += cJ[world, rc, side * side_stride + s] * velocity[gd]
    return out


@wp.kernel
def ds_compact_contact_restitution(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    cbase: wp.array[int],
    capacity: int,
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_rebound_velocity: wp.array2d[float],
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    chead: wp.array2d[int],
    cJ: wp.array3d[float],
    ancestors: wp.array2d[int],
    na: wp.array[int],
    side_stride: int,
    dt: float,
    write_row_w: int,
    # in/out
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Compact twin of ``apply_world_contact_restitution``: an impacting contact's bias becomes its rebound target."""
    world, rc = wp.tid()
    i = cbase[world] + rc
    if i >= wp.min(world_constraint_count[world], max_constraints):
        return
    if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
        return
    rebound_velocity = world_row_rebound_velocity[world, i]
    if rebound_velocity <= 0.0:
        return
    target_vel = world_target_velocity[world, i]
    relative_incident = (
        compact_row_dot(world, rc, chead, cJ, ancestors, na, side_stride, world_dof_indices, world_incident_velocity)
        - target_vel
    )
    if contact_restitution_fires(world_phi[world, i], relative_incident, dt):
        world_rhs[world, i] = -target_vel - rebound_velocity
        if write_row_w != 0:
            world_row_w[world, i] = 1.0


@cache
def get_compact_pgs_kernel(D: int, C: int, KS: int, KZ: int, NA: int, M_D: int, SIDES: int) -> wp.Kernel:
    """Projected Gauss-Seidel over a world's compact contact rows, one warp per world.

    The update is the dense matrix-free sweep's for contact and friction rows (``_get_pgs_solve_mf_gs_kernel`` with
    ``friction_mode="fixed_normal"``): relaxed, regularized normal rows projected onto ``lambda >= 0``; each contact's
    tangent pair solved on the Coulomb disk of its (patch-pooled) normal load at its first tangent row; an exactly
    stationary sweep ends the launch once friction is on. ``J v`` and ``J_i Y_j`` run on the row's support, ``v += Y
    dlambda`` over the world's DOFs with the next row's ``Y`` prefetched into registers.
    """
    E = (D + 31) // 32
    CONTACT = int(PGS_CONSTRAINT_TYPE_CONTACT)
    FRICTION = int(PGS_CONSTRAINT_TYPE_FRICTION)

    def lanes(template: str) -> str:
        return "\n".join(
            template.replace("{k}", str(k)).replace("{d}", f"lane + {k * 32}" if k else "lane") for k in range(E)
        )

    decl = lanes("    float pre_y{k} = 0.0f;")
    prefetch_first = lanes("        if ({d} < " + str(D) + ") pre_y{k} = Yw[{d}];")
    consume = lanes("            const float cur_y{k} = pre_y{k};")
    prefetch_next = lanes("                if ({d} < " + str(D) + ") pre_y{k} = Yn[{d}];")
    v_update = lanes("                if ({d} < " + str(D) + ") s_v[{d}] += cur_y{k} * delta_impulse;")
    side_loop = "\n".join(
        f"""        {{
            const int head = ((packed >> {16 * k}) & 0xFFFF) - 1;
            if (head >= 0) {{
                const int n = na.data[head];
                const int* apath = &ancestors.data[(size_t)head * {NA}];
                for (int s = lane; s < n; s += 32) acc += Jr[{k * KZ} + s] * vec[apath[s]];
            }}
        }}"""
        for k in range(SIDES)
    )
    snippet = f"""
#if defined(__CUDA_ARCH__)
{FRICTION_PAIR_CUDA}
    auto warp_sum = [&](float value) {{
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
        return value;
    }};
    const unsigned MASK = 0xFFFFFFFF;
    const int lane = threadIdx.x;
    const int cb = cbase.data[world];
    int nc = min(constraint_count.data[world], {M_D}) - cb;
    if (nc > {C}) nc = {C};
    if (nc <= 0) return;
    const size_t wrow = (size_t)world * {M_D} + (size_t)cb;
    const float* Jw = &cJ.data[(size_t)world * {C * KS}];
    const float* Yw = &cY.data[(size_t)world * {C} * {D}];
    const int* Hw = &chead.data[(size_t)world * {C}];
    const int* dofi = &world_dof_indices.data[(size_t)world * {D}];
    __shared__ float s_v[{D}];
    __shared__ float s_lam[{C}];
    // Low byte: row type; above it, the compact parent row + 1 (0: none).
    __shared__ int s_meta[{C}];
    for (int d = lane; d < {D}; d += 32) {{
        const int gd = dofi[d];
        s_v[d] = gd >= 0 ? v_out.data[gd] : 0.0f;
    }}
    for (int r = lane; r < nc; r += 32) {{
        s_lam[r] = impulses.data[wrow + r];
        const int t = row_type.data[wrow + r];
        const int p = row_parent.data[wrow + r];
        const int pr = p >= cb && p - cb < nc ? p - cb : -1;
        s_meta[r] = (t & 0xFF) | ((pr + 1) << 8);
    }}
    __syncwarp();
    // J_r . vec on the row's support (vec indexed by local DOF).
    auto jdot = [&](int r, const float* vec) {{
        const int packed = Hw[r];
        const float* Jr = Jw + (size_t)r * {KS};
        float acc = 0.0f;
{side_loop}
        return warp_sum(acc);
    }};
{decl}
    for (int iter = 0; iter < iterations; iter++) {{
        int iteration_changed = 0;
        {{
{prefetch_first}
        }}
        for (int i = 0; i < nc; i++) {{
{consume}
            if (i + 1 < nc) {{
                const float* Yn = Yw + (size_t)(i + 1) * {D};
{prefetch_next}
            }}
            const int meta = s_meta[i];
            const int row_type_i = meta & 0xFF;
            const int parent_idx = (meta >> 8) - 1;
            if (row_type_i != {CONTACT} && row_type_i != {FRICTION}) continue;
            if (row_type_i == {FRICTION} && iter < friction_start_iteration) {{
                s_lam[i] = 0.0f;
                __syncwarp();
                continue;
            }}
            if (row_type_i == {FRICTION} && i != parent_idx + 1) continue;
            const float denom = diag.data[wrow + i];
            if (denom <= 0.0f && row_type_i != {FRICTION}) continue;
            const float jv = jdot(i, s_v);
            const float old_impulse = s_lam[i];
            const float residual = jv + rhs.data[wrow + i];
            const float w = regularize != 0 ? row_w.data[wrow + i] : 1.0f;
            const float delta = denom > 0.0f ? -residual / denom * w - (1.0f - w) * old_impulse : 0.0f;
            float new_impulse = old_impulse + omega * delta;
            if (row_type_i == {FRICTION}) {{
                // The first tangent row solves both tangents of its contact on the disk of the pooled normal load.
                const int sib = parent_idx + 2;
                float lambda_n = s_lam[parent_idx];
                for (int patch_row = (s_meta[parent_idx] >> 8) - 1; patch_row >= 0 && patch_row != parent_idx;
                     patch_row = (s_meta[patch_row] >> 8) - 1)
                    lambda_n += s_lam[patch_row];
                const float radius = fmaxf(row_mu.data[wrow + i] * lambda_n, 0.0f);
                const float* Ys = Yw + (size_t)sib * {D};
                // J_i . Y_sib needs Y_sib on row i's support only (local DOF indexing).
                const float sibling_residual = jdot(sib, s_v) + rhs.data[wrow + sib];
                const float cross = jdot(i, Ys);
                const float2 pair = friction_pair_candidate(denom, cross, diag.data[wrow + sib], residual,
                    sibling_residual, old_impulse, s_lam[sib], radius, omega);
                const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = mag > radius ? radius / mag : 1.0f;
                new_impulse = pair.x * scale;
                const float sib_delta = pair.y * scale - s_lam[sib];
                __syncwarp();
                s_lam[sib] = pair.y * scale;
                if (sib_delta != 0.0f) {{
                    iteration_changed = 1;
                    for (int d = lane; d < {D}; d += 32) s_v[d] += Ys[d] * sib_delta;
                }}
            }} else if (new_impulse < 0.0f) {{
                new_impulse = 0.0f;
            }}
            const float delta_impulse = new_impulse - old_impulse;
            __syncwarp();
            s_lam[i] = new_impulse;
            if (delta_impulse != 0.0f) {{
                iteration_changed = 1;
{v_update}
            }}
            __syncwarp();
        }}
        if (iter >= friction_start_iteration && __ballot_sync(MASK, iteration_changed != 0) == 0u) break;
    }}
    for (int d = lane; d < {D}; d += 32) {{
        const int gd = dofi[d];
        if (gd >= 0) v_out.data[gd] = s_v[d];
    }}
    for (int r = lane; r < nc; r += 32) impulses.data[wrow + r] = s_lam[r];
#endif
"""

    @wp.func_native(snippet)
    def compact_pgs_native(
        world: int,
        constraint_count: wp.array[int],
        cbase: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs: wp.array2d[float],
        diag: wp.array2d[float],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        row_w: wp.array2d[float],
        chead: wp.array2d[int],
        cJ: wp.array3d[float],
        cY: wp.array3d[float],
        ancestors: wp.array2d[int],
        na: wp.array[int],
        iterations: int,
        omega: float,
        regularize: int,
        friction_start_iteration: int,
        v_out: wp.array[float],
        impulses: wp.array2d[float],
    ): ...

    def compact_pgs(
        constraint_count: wp.array[int],
        cbase: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs: wp.array2d[float],
        diag: wp.array2d[float],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        row_w: wp.array2d[float],
        chead: wp.array2d[int],
        cJ: wp.array3d[float],
        cY: wp.array3d[float],
        ancestors: wp.array2d[int],
        na: wp.array[int],
        iterations: int,
        omega: float,
        regularize: int,
        friction_start_iteration: int,
        v_out: wp.array[float],
        impulses: wp.array2d[float],
    ):
        world, _lane = wp.tid()
        compact_pgs_native(
            world,
            constraint_count,
            cbase,
            world_dof_indices,
            rhs,
            diag,
            row_type,
            row_parent,
            row_mu,
            row_w,
            chead,
            cJ,
            cY,
            ancestors,
            na,
            iterations,
            omega,
            regularize,
            friction_start_iteration,
            v_out,
            impulses,
        )

    name = f"closure_ds_compact_pgs_{D}_{C}_{KS}_{KZ}_{NA}_{M_D}_{SIDES}"
    compact_pgs.__name__ = name
    compact_pgs.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(compact_pgs)


def launch_rows(solver, state_in, state_aug, contacts, contact_build_threads: int) -> None:
    """Build the step's contact rows in the compact layout (after the contact slots are assigned)."""
    buf = solver._closure_direct_sparse
    model = solver.model
    wp.launch(
        ds_compact_contact_rows,
        dim=contact_build_threads,
        inputs=[
            contacts.rigid_contact_count,
            contact_build_threads,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            solver._contact_world,
            solver._contact_slot,
            solver._contact_art_a,
            solver._contact_art_b,
            solver._contact_path,
            solver._contact_slots_needed,
            solver._articulation_dof_start,
            solver._articulation_origin,
            solver._body_to_joint,
            model.joint_ancestor,
            model.joint_qd_start,
            state_aug._joint_S_s,
            model.shape_body,
            state_in.body_q,
            solver._friction_patches.view,
            *solver._contact_anchor_args,
            buf.cbase,
            buf.na,
            buf.D,
            buf.C,
            buf.KZ,
            int(buf.sides == 2),
        ],
        outputs=[buf.cJ, buf.chead, buf.status],
        device=model.device,
    )


def launch_restitution(solver, dt: float) -> None:
    """Rebound targets of the impacting compact contact rows."""
    buf = solver._closure_direct_sparse
    wp.launch(
        ds_compact_contact_restitution,
        dim=(buf.W, buf.C),
        inputs=[
            solver._constraint_count,
            solver._articulation_constraint_max,
            buf.cbase,
            buf.C,
            solver._phi,
            solver._row_type,
            solver._target_velocity,
            solver._row_rebound_velocity,
            solver._v_hat,
            solver._world_dof_indices,
            buf.chead,
            buf.cJ,
            buf.ancestors,
            buf.na,
            buf.KZ,
            float(dt),
            int(solver._regularization_enabled),
        ],
        outputs=[solver._rhs, solver._row_w],
        device=solver.model.device,
    )


def launch_sweeps(solver, rhs: wp.array, iterations: int, friction_start_iteration: int) -> None:
    """Contact PGS on the compact rows after the sparse direct closure solve."""
    buf = solver._closure_direct_sparse
    kernel = get_compact_pgs_kernel(buf.D, buf.C, buf.KS, buf.KZ, buf.NA, buf.M_D, buf.sides)
    wp.launch_tiled(
        kernel,
        dim=[buf.W],
        inputs=[
            solver._constraint_count,
            buf.cbase,
            solver._world_dof_indices,
            rhs,
            solver._diag,
            solver._row_type,
            solver._row_parent,
            solver._row_mu,
            solver._row_w,
            buf.chead,
            buf.cJ,
            buf.cY,
            buf.ancestors,
            buf.na,
            int(iterations),
            float(solver._relaxation),
            int(solver._regularization_enabled),
            int(min(friction_start_iteration, iterations)),
        ],
        outputs=[solver._v_out, solver._impulses],
        block_dim=32,
        device=solver.model.device,
    )
