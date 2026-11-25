# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse batched direct closure solve (experimental, kernel override ``closure_direct_factor="sparse"``).

Replaces the serial sweep over the loop-closure rows by one regularized direct solve per substep:
``K dv = -J^T W r0`` with ``K = H + J^T W J``, solved once plus ``iterations - 1`` primal refinements. ``W``
weights each row by ``1 / (eps * sum_d J_d^2 / H_dd)``, a diagonal estimate of the row's inverse effective mass
scaled by ``closure_direct_eps``, and ``r0`` is the rows' velocity error plus bias.
Everything the solve touches is sparse and its pattern is fixed by the joint tree and the closures, so the pattern is
analysed once on the host (``closure_direct_symbolic.SparseCholeskyPlan``: minimum-degree order, supernodes, levels)
and each world runs one thread block that keeps its whole factor in shared memory:

- ``geometry`` / ``rows``: the loop-closure rows are built straight on their static support (the DOFs between the
  two bodies, at most a few dozen), one warp per closure; no dense ``J`` exists. Gaps and lever arms of closures
  inside one articulation come from root-relative body positions.
- ``mass``: ``H`` in the ancestor layout ``Ha[d, s]`` (CRBA plus armature and implicit drive terms), only on mass
  updates.
- ``solve``: one block per world. (1) Row weights, bias and ``r0`` per closure row; the rows are scaled by
  ``sqrt(w)`` in place. (2) ``P K P^T`` gathered straight into the packed storage of ``L`` and ``g0``. (3) A
  left-looking supernodal Cholesky level by level of the supernodal elimination tree (a warp per supernode; the
  whole block on single-supernode levels, such as the dense root separator). (4) Forward/backward solves and
  refinements. (5) ``v`` and the impulses.

With ``closure_direct_contacts="elim"`` the same block then forms each contact row's response ``Y = K^-1 J^T``
through the factor (``closure_direct_compact.py`` builds and sweeps those rows).

Valid with one articulation per world (identical articulations) and loop-closure (connect) rows as the only
articulation rows besides contacts. Deterministic (no atomics on the data path).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import warp as wp

from .closure_direct_symbolic import SparseCholeskyPlan
from .friction_patches import WARP_DETERMINISM_OFF
from .kernels import (
    CONNECT_MAX_ROWS,
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    _closure_anchor_parts,
    _connect_rotation_rows,
    _walk_has_free_dof,
)

# Steel orders its own data paths; Warp's deterministic lowering would only add cost (see kernels.py).
wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})

SOLVE_THREADS = 512
WHOLE_W = 16  # a level that is a single supernode at least this wide runs with the whole block
ROWS_WPB = 4  # closures (warps) per block of the row builder
GEOM_FLOATS = 8  # per closure row: e (3), phi, target, angular, unused (2)
CLOSURE_FLOATS = 8  # per closure: rel_a (3), rel_b (3), prescribed, valid
# Dynamic shared memory per block: the opt-in limit of sm_86/sm_89/sm_120 (99 KiB; sm_80/sm_90 allow more) less the
# 2 KiB Warp reserves statically in every kernel.
SMEM_LIMIT = 97 * 1024


def default_solve_threads(n_dofs: int) -> int:
    """Threads of the per-world solve block: the DOF count rounded up to a power of two, 32 to ``SOLVE_THREADS``.

    A small tree has too little parallel work per level for a wide block, whose barriers then dominate; one
    block per world also caps the worlds resident per SM, so a narrow block raises throughput at scale.
    """
    threads = 32
    while threads < n_dofs and threads < SOLVE_THREADS:
        threads *= 2
    return threads


def _set_dynamic_smem(kernel: wp.Kernel, nbytes: int) -> wp.Kernel:
    """Make Warp reserve ``nbytes`` of dynamic shared memory for a native-snippet kernel."""
    kernel.adj.get_total_required_shared = lambda: int(nbytes)
    return kernel


# ---------------------------------------------------------------------------------------------------------------
# Host-side structure
# ---------------------------------------------------------------------------------------------------------------


class DirectSparseBuffers:
    """Static plan tables of the group's articulation and the per-world buffers of the sparse direct solve."""

    def __init__(self, solver):
        model = solver.model
        device = model.device
        if not solver._jy_world_aliased:
            raise NotImplementedError("closure_direct_factor='sparse' needs one articulation per world")
        if int(getattr(model, "constraint_mimic_count", 0) or 0):
            raise NotImplementedError("closure_direct_factor='sparse' does not build mimic rows")
        size = solver._size_groups[0]
        W = solver._world_count
        M_D = solver._articulation_constraint_max
        D = size
        self.size, self.W, self.M_D, self.D = size, W, M_D, D
        art_start = model.articulation_start.numpy()
        joint_end = solver._articulation_joint_end.numpy()
        qd_start = model.joint_qd_start.numpy()
        ancestor_of = model.joint_ancestor.numpy()
        dof_start = solver._articulation_dof_start.numpy()
        g2a = solver._group_to_art[size].numpy()
        if len(g2a) != W:
            raise NotImplementedError("closure_direct_factor='sparse': group and world counts differ")
        art_world = solver._art_to_world.numpy()
        if not np.array_equal(art_world[g2a], np.arange(W)):
            raise NotImplementedError("closure_direct_factor='sparse': group index must equal the world index")
        a0 = int(g2a[0])
        j0, j1, d0 = int(art_start[a0]), int(joint_end[a0]), int(dof_start[a0])

        # DOF tree of the articulation (local DOFs, parents before children).
        parent = np.full(D, -1, dtype=np.int64)
        joint_of = np.full(D, -1, dtype=np.int64)
        last_dof = {}
        for j in range(j0, j1):
            q0, q1 = int(qd_start[j]) - d0, int(qd_start[j + 1]) - d0
            a = int(ancestor_of[j])
            pd = -1
            while a >= j0:
                if a in last_dof:
                    pd = last_dof[a]
                    break
                a = int(ancestor_of[a])
            for q in range(q0, q1):
                parent[q] = pd if q == q0 else q - 1
                joint_of[q] = j
            if q1 > q0:
                last_dof[j] = q1 - 1
            elif pd >= 0:
                last_dof[j] = pd
        if int((joint_of >= 0).sum()) != D or np.any(parent >= np.arange(D)):
            raise NotImplementedError("closure_direct_factor='sparse': unexpected DOF layout")
        rel = (ancestor_of[j0:j1] - j0).astype(np.int64)
        sample = g2a if W <= 64 else g2a[np.linspace(0, W - 1, 64).astype(int)]
        for art in sample:
            ja, jb = int(art_start[art]), int(joint_end[art])
            other = (ancestor_of[ja:jb] - ja).astype(np.int64)
            if jb - ja != len(rel) or not np.array_equal(np.where(other < 0, -1, other), np.where(rel < 0, -1, rel)):
                raise NotImplementedError("closure_direct_factor='sparse' needs identical articulations")

        # Closures: contiguous per world, the same local list in every world.
        nk = solver._connect_count
        if nk == 0:
            raise NotImplementedError("closure_direct_factor='sparse' needs loop closures")
        c_art = solver._connect_art_np
        if np.any(np.diff(c_art) < 0):
            raise NotImplementedError("closure_direct_factor='sparse': closures must be grouped by articulation")
        c_world = art_world[c_art]
        counts = np.bincount(c_world, minlength=W)
        KL = int(counts[0])
        if np.any(counts != KL):
            raise NotImplementedError("closure_direct_factor='sparse': every world needs the same closures")
        self.KL = KL
        world_closure_start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
        closure_local = (np.arange(nk) - world_closure_start[c_world]).astype(np.int32)
        body_to_joint = solver._body_to_joint.numpy()
        kin_mask = solver._kinematic_dof_mask.numpy()
        body_p = solver._connect_body_p_np
        body_c = solver._connect_body_c.numpy()
        prescribed = solver._connect_parent_prescribed_np
        rows_np = solver._connect_rows_np

        def path(body):
            dofs = []
            curr = int(body_to_joint[body]) if body >= 0 else -1
            while curr != -1:
                dofs.extend(range(int(qd_start[curr]), int(qd_start[curr + 1])))
                curr = int(ancestor_of[curr])
            return dofs

        def local_closure(k):
            art = int(c_art[k])
            dsa = int(dof_start[art])
            pp = [] if prescribed[k] else path(int(body_p[k]))
            pc = path(int(body_c[k]))
            mid = (not prescribed[k]) and any(kin_mask[d] == 0 for d in pp) and any(kin_mask[d] == 0 for d in pc)
            sp, sc = {d - dsa for d in pp}, {d - dsa for d in pc}
            sup = sorted(sp ^ sc) if mid else sorted(sp | sc)
            side = [(1 if d in sp else 0) | (2 if d in sc else 0) for d in sup]
            return sup, side, mid, int(rows_np[k])

        locs = [local_closure(k) for k in range(KL)]
        check_worlds = [W - 1] if W <= 1 else sorted({1, W // 2, W - 1})
        for w in check_worlds:
            for kl in range(KL):
                k = int(world_closure_start[w]) + kl
                if local_closure(k) != locs[kl]:
                    raise NotImplementedError("closure_direct_factor='sparse': closures differ between worlds")
        supports = [np.asarray(s, dtype=np.int64) for s, _, _, _ in locs]
        self.plan = plan = SparseCholeskyPlan(parent, supports)
        KJ = max(1, plan.KJ)
        if KJ > 64:
            raise NotImplementedError(f"closure_direct_factor='sparse': a closure touches {KJ} DOFs (max 64)")
        if KL >= 4096:
            raise NotImplementedError("closure_direct_factor='sparse': at most 4095 closures per world")
        self.KJ = KJ
        self.MAXR = int(rows_np[:KL].max())
        sup_dof = np.full((KL, KJ), 0, dtype=np.int32)
        sup_side = np.zeros((KL, KJ), dtype=np.int32)
        sup_count = np.zeros(KL, dtype=np.int32)
        for kl, (s, side, _, _) in enumerate(locs):
            sup_count[kl] = len(s)
            sup_dof[kl, : len(s)] = s
            sup_side[kl, : len(s)] = side
        self.mid = np.array([m for _, _, m, _ in locs], dtype=bool)
        self.mid_local = wp.array(self.mid.astype(np.int32), dtype=wp.int32, device=device)

        def i32(x):
            return wp.array(np.ascontiguousarray(np.asarray(x, dtype=np.int32)), dtype=wp.int32, device=device)

        self.closure_local = i32(closure_local)
        self.world_closure_start = i32(world_closure_start)
        self.sup_dof = i32(sup_dof.reshape(-1))
        self.sup_side = i32(sup_side.reshape(-1))
        self.sup_count = i32(sup_count)
        self.rows_local = i32(rows_np[:KL])
        self.ancestors = i32(plan.ancestors)
        self.ancestors_flat = i32(plan.ancestors.reshape(-1))
        self.na = i32(plan.na)
        self.NA = plan.NA
        self.dof_joint = i32(np.maximum(joint_of - j0, 0))
        # Plan tables.
        p = plan
        self.t_perm = i32(p.perm)
        self.t_iperm = i32(p.iperm)
        self.t_col_off = i32(p.col_off)
        self.t_rowidx = i32(p.rowidx)
        self.t_sn = i32(np.stack([p.sn_c0[:-1], p.sn_w, p.sn_q, p.upd_start[:-1], p.upd_start[1:]], axis=1).reshape(-1))
        self.t_lvl_sn = i32(p.lvl_sn)
        self.t_lvl_start = i32(p.lvl_start)
        self.t_pairs = i32(p.pairs.reshape(-1) if len(p.pairs) else np.zeros(4))
        self.t_rel = i32(p.rel if len(p.rel) else np.zeros(1))
        self.t_row_ptr = i32(p.row_ptr)
        self.t_row_ent = i32(p.row_ent if len(p.row_ent) else np.zeros(1))
        self.t_row_col = i32(p.row_col if len(p.row_col) else np.zeros(1))
        self.t_hidx = i32(p.hidx)
        self.t_asm_ptr = i32(p.asm_ptr)
        self.t_asm_list = i32(p.asm_list if len(p.asm_list) else np.zeros(1))
        cnt = np.diff(p.asm_ptr)
        first = (
            np.where(cnt > 0, p.asm_list[np.minimum(p.asm_ptr[:-1], max(len(p.asm_list) - 1, 0))], 0)
            if len(p.asm_list)
            else np.zeros(p.nnz)
        )
        second = (
            np.where(cnt > 1, p.asm_list[np.minimum(p.asm_ptr[:-1] + 1, max(len(p.asm_list) - 1, 0))], 0)
            if len(p.asm_list)
            else np.zeros(p.nnz)
        )
        if cnt.max(initial=0) > 255 or p.asm_ptr[-1] >= (1 << 23):
            raise NotImplementedError("closure_direct_factor='sparse': assembly table too large")
        self.t_asm_rec = i32(np.stack([p.hidx, first, second, (p.asm_ptr[:-1] << 8) | cnt], axis=1).reshape(-1))
        self.t_g_ptr = i32(p.g_ptr)
        self.t_g_list = i32(p.g_list if len(p.g_list) else np.zeros(1))
        self.t_h_ptr = i32(p.h_ptr)
        self.t_h_list = i32(p.h_list.reshape(-1))
        self.t_ct_ptr = i32(p.ct_ptr)
        self.t_ct_rec = i32(p.ct_rec.reshape(-1) if len(p.ct_rec) else np.zeros(4))
        self.t_row_split = i32(p.row_split)
        self.t_sc_ptr = i32(p.sc_ptr)
        self.t_sc_list = i32(p.sc_list if len(p.sc_list) else np.zeros(1))
        self.t_sc_rec = i32(p.sc_rec.reshape(-1) if len(p.sc_rec) else np.zeros(4))
        if np.any(p.sn_w[p.pairs[:, 0]] > 255) if len(p.pairs) else False:
            raise NotImplementedError("closure_direct_factor='sparse': a descendant supernode is wider than 255")
        self.NNZ, self.S, self.NLEV = p.nnz, p.S, p.nlev
        # Whole-block levels (one supernode of width >= WHOLE_W): the widest row count and tile count.
        whole = []
        for lev in range(p.nlev):
            sns = p.lvl_sn[p.lvl_start[lev] : p.lvl_start[lev + 1]]
            if len(sns) == 1 and int(p.sn_w[sns[0]]) >= WHOLE_W:
                whole.extend(int(x) for x in sns)
        self.QW = max([int(p.sn_q[x]) for x in whole], default=1)
        self.TILES = max(
            [sum((int(p.sn_q[x]) + 3) // 4 - ct for ct in range((int(p.sn_w[x]) + 3) // 4)) for x in whole], default=1
        )
        self.threads = int(solver._kernel_overrides.get("closure_ds_threads", default_solve_threads(D)))
        if self.threads < 32 or self.threads > 1024 or self.threads % 32:
            raise ValueError("closure_ds_threads must be a multiple of 32 between 32 and 1024")
        # Shared memory: 4 vectors (g0, x, rhs scratch, sum), the 32 x 33 diagonal block and pivots, L's column
        # offsets and L's storage; the dense update buffers reuse x .. sD (3 x BC columns of QW rows) when they fit.
        vec_floats = 4 * D + 32 * 33 + 32 + D + 1
        work = 3 * D + 32 * 33
        self.smem_L = (p.nnz + vec_floats) * 4 <= SMEM_LIMIT
        bc = int(solver._kernel_overrides.get("closure_ds_bc", 4))
        if self.smem_L:
            self.BC = max(1, min(bc, work // (3 * self.QW)))
            self.wb_in_vec = True
            self.smem_bytes = (p.nnz + vec_floats) * 4
        else:
            self.BC = bc
            self.wb_in_vec = False
            self.smem_bytes = (vec_floats + 3 * self.BC * self.QW) * 4
        # Per-world buffers.
        self.Ha = wp.zeros((W, D, plan.NA), dtype=wp.float32, device=device)
        self.Jw = wp.zeros((W, KL * self.MAXR * KJ), dtype=wp.float32, device=device)
        self.qrow = wp.zeros((W, KL * self.MAXR), dtype=wp.float32, device=device)
        self.swrow = wp.zeros((W, KL * self.MAXR), dtype=wp.float32, device=device)
        self.Lg = wp.zeros((1, 1) if self.smem_L else (W, p.nnz), dtype=wp.float32, device=device)
        self.geom_row = wp.zeros((nk * CONNECT_MAX_ROWS, GEOM_FLOATS), dtype=wp.float32, device=device)
        self.geom_closure = wp.zeros((nk, CLOSURE_FLOATS), dtype=wp.float32, device=device)
        # [non-positive pivots, anchor-mode mismatches, unused, unused, contact rows past the capacity, self-contact
        #  sides dropped, contact-range rows of other types]
        self.status = wp.zeros(7, dtype=wp.int32, device=device)
        # Contact rows (closure_direct_contacts="elim"): their responses K^-1 J^T come from the closure factor.
        self.cmode = 1 if solver._closure_ds_contacts == "elim" else 0
        # Compact contact rows (closure_direct_compact.py): J by depth along each side's root path (KZ floats per side),
        # Y dense over the DOFs, both indexed by the contact row past the world's closure rows; capacity: the
        # contact-row budget M_D minus the closure rows.
        self.compact = self.cmode == 1
        self.KZ = int(plan.NA)
        self.sides = 1
        self.C = 1
        if self.compact:
            from .closure_direct_compact import self_contacts_possible  # noqa: PLC0415

            self.sides = 2 if self_contacts_possible(solver, a0) else 1
            self.C = max(1, M_D - int(rows_np[:KL].sum()))
        self.KS = self.KZ * self.sides
        self.cJ = wp.zeros((W, self.C, self.KS) if self.compact else (1, 1, 1), dtype=wp.float32, device=device)
        self.cY = wp.zeros((W, self.C, D) if self.compact else (1, 1, 1), dtype=wp.float32, device=device)
        self.chead = wp.zeros((W, self.C) if self.compact else (1, 1), dtype=wp.int32, device=device)
        self.cbase = wp.zeros(W if self.compact else 1, dtype=wp.int32, device=device)
        # Contact responses: one warp per row, each warp keeping its vector in shared memory after the solve's own
        # storage (column-oriented triangular solves that skip zero entries), when that fits; else the whole block
        # solves one row at a time.
        self.yoff = self.smem_bytes // 4
        self.yw = 0
        if self.cmode:
            nw = self.threads // 32
            yw = min(nw, (SMEM_LIMIT - self.smem_bytes) // (4 * D))
            yw = min(yw, int(solver._kernel_overrides.get("closure_ds_contact_warps", nw)))
            if yw >= 2:
                self.yw = yw
                self.smem_bytes += yw * D * 4
        # Compact layout: YR rows per warp share each pass over L (closure_ds_contact_rows_per_warp). Default: 3 when L
        # is in global memory (its loads dominate), else 2 (more rows raise register pressure in the whole kernel), as
        # long as 8 warps keep their vectors in shared memory. RTX PRO 6000 / 3090: the response phase falls 31% on
        # the P38 drop (2 rows) and 53% on the flower (3 rows).
        self.yr = 1
        if self.compact and self.yw:
            nw = self.threads // 32
            avail = (SMEM_LIMIT - self.yoff * 4) // (4 * D)
            want = solver._kernel_overrides.get("closure_ds_contact_rows_per_warp")
            choices = [int(want)] if want is not None else [2 if self.smem_L else 3, 2]
            for r in choices:
                yw = min(nw, avail // r)
                if r > 1 and yw >= (1 if want is not None else 8):
                    self.yr, self.yw = r, yw
                    self.smem_bytes = (self.yoff + yw * r * D) * 4
                    break
        # Per-world phase times [ns] (rows, factor, solves) and world 0's level stamps, with ``closure_ds_timing``.
        self.timing_enabled = bool(solver._kernel_overrides.get("closure_ds_timing", False))
        self.timing = wp.zeros(max(W * 4, 160) if self.timing_enabled else 1, dtype=wp.int64, device=device)
        self.info = {
            "n": D,
            "nnz": p.nnz,
            "supernodes": p.S,
            "levels": p.nlev,
            "KJ": KJ,
            "KL": KL,
            "NA": plan.NA,
            "flops": p.flops,
            "smem_L": self.smem_L,
            "smem_bytes": self.smem_bytes,
            "asm": int(len(p.asm_list)),
            "pairs": int(len(p.pairs)),
            "midpoint": int(self.mid.sum()),
            "compact_rows": self.C if self.compact else 0,
            "compact_sides": self.sides if self.compact else 0,
            "y_warps": self.yw,
            "y_rows_per_warp": self.yr,
        }


# ---------------------------------------------------------------------------------------------------------------
# Mass updates: CRBA into the ancestor layout plus the implicit drive terms
# ---------------------------------------------------------------------------------------------------------------


@wp.kernel
def ds_crba_fill(
    D: int,
    NA: int,
    group_to_art: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    dof_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    R_group: wp.array2d[float],
    # outputs
    Ha: wp.array3d[float],
):
    """Column ``c`` of the joint-space mass matrix (with armature): its entries on the DOF's root path."""
    tid = wp.tid()
    idx = tid // D
    c = tid - idx * D
    art = group_to_art[idx]
    if mass_update_mask[art] == 0:
        return
    j0 = articulation_start[art]
    dof0 = articulation_dof_start[art]
    pivot = j0 + dof_joint[c]
    F = body_I_c[joint_child[pivot]] * joint_S_s[dof0 + c]
    s = int(0)
    curr = pivot
    top = c
    while curr >= j0 and s < NA:
        q0 = joint_qd_start[curr] - dof0
        r = top
        while r >= q0 and s < NA:
            val = wp.dot(joint_S_s[dof0 + r], F)
            if r == c:
                val += R_group[idx, c]
            Ha[idx, c, s] = val
            s += 1
            r -= 1
        curr = joint_ancestor[curr]
        if curr >= j0:
            top = joint_qd_start[curr + 1] - 1 - dof0


@wp.kernel
def ds_add_drive_diagonal(
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    max_dofs: int,
    mass_update_mask: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    # outputs
    Ha: wp.array3d[float],
):
    """Add the implicit drive terms ``h kd + h^2 ke`` to the diagonal."""
    idx = wp.tid()
    art = group_to_art[idx]
    if mass_update_mask[art] == 0:
        return
    dof0 = articulation_dof_start[art]
    for i in range(row_counts[art]):
        K = row_K[art * max_dofs + i]
        if K > 0.0:
            Ha[idx, row_dof_index[art * max_dofs + i] - dof0, 0] += K


def refresh_mass(solver, state_aug) -> None:
    """Rebuild the ancestor-layout mass matrices of the articulations due for a mass update."""
    buf = solver._closure_direct_sparse
    model = solver.model
    size = buf.size
    wp.launch(
        ds_crba_fill,
        dim=buf.W * buf.D,
        inputs=[
            buf.D,
            buf.NA,
            solver._group_to_art[size],
            model.articulation_start,
            solver._articulation_dof_start,
            solver._mass_update_mask,
            buf.dof_joint,
            model.joint_ancestor,
            model.joint_child,
            model.joint_qd_start,
            state_aug._joint_S_s,
            solver._body_I_c,
            solver._R_by_size[size],
        ],
        outputs=[buf.Ha],
        device=model.device,
    )
    if solver._drive_mode == "augmented":
        wp.launch(
            ds_add_drive_diagonal,
            dim=buf.W,
            inputs=[
                solver._group_to_art[size],
                solver._articulation_dof_start,
                solver._articulation_max_dofs,
                solver._mass_update_mask,
                solver._aug_row_counts,
                solver._aug_row_dof_index,
                solver._aug_row_K,
            ],
            outputs=[buf.Ha],
            device=model.device,
        )


# ---------------------------------------------------------------------------------------------------------------
# Rows: geometry (one thread per closure row), then one warp per closure builds the normalized rows on the
# closure's static support (lane t owns support entries t, t + 32).
# ---------------------------------------------------------------------------------------------------------------


@wp.kernel
def ds_connect_geometry(
    articulation_dof_start: wp.array[int],
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    art_group_idx: wp.array[int],
    connect_slot: wp.array[int],
    connect_art: wp.array[int],
    connect_joint: wp.array[int],
    connect_body_p: wp.array[int],
    connect_body_c: wp.array[int],
    connect_rows: wp.array[int],
    joint_axis: wp.array[wp.vec3],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    connect_parent_prescribed: wp.array[int],
    kinematic_dof_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    closure_local: wp.array[int],
    mid_local: wp.array[int],
    body_rel_t: wp.array[wp.vec3],
    articulation_ref_t: wp.array[wp.vec3],
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    # outputs
    geom_row: wp.array2d[float],
    geom_closure: wp.array2d[float],
    connect_at_midpoint: wp.array[int],
):
    """Directions, raw position errors and targets of one closure row (as ``populate_connect_J_for_size``).

    ``geom_closure[k, 7]`` is 0 for an inactive closure, 1 when the anchor mode matches the static support
    (``mid_local``), 2 otherwise (the rows kernel counts those in ``status[1]``). Each anchor is split into the
    articulation's reference point ``articulation_ref_t`` and a small offset, from the root-relative positions
    ``body_rel_t`` (see :func:`~newton._src.solvers.steel.kernels.eval_body_relative_poses`), as in
    ``populate_connect_J_for_size``.
    """
    tid = wp.tid()
    k = tid // CONNECT_MAX_ROWS
    r = tid - k * CONNECT_MAX_ROWS
    if r >= connect_rows[k]:
        return
    valid = int(1)
    art = connect_art[k]
    group_idx = art_group_idx[art]
    if group_idx < 0 or group_idx >= group_to_art.shape[0]:
        valid = 0
    elif group_to_art[group_idx] != art:
        valid = 0
    elif connect_slot[k] < 0:
        valid = 0
    if valid == 0:
        if r == 0:
            geom_closure[k, 7] = 0.0
        return
    body_p = connect_body_p[k]
    body_c = connect_body_c[k]
    prescribed = connect_parent_prescribed[k] != 0
    joint = connect_joint[k]
    anchor_p = wp.transform_get_translation(joint_X_p[joint])
    v_a = wp.vec3(0.0, 0.0, 0.0)
    if body_p >= 0 and prescribed:
        twist = body_qd[body_p]
        lever = wp.quat_rotate(wp.transform_get_rotation(body_q[body_p]), anchor_p - body_com[body_p])
        v_a = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), lever)
    # Gap and lever arms from the articulation's reference point, which do not carry the float32 rounding of the
    # articulation's world position. The child is always in the articulation, the parent unless it is prescribed.
    frame_t = articulation_ref_t[art]
    root = joint_child[articulation_start[art]]
    origin = body_rel_t[root] + wp.quat_rotate(wp.transform_get_rotation(body_q[root]), body_com[root])
    big_a, small_a = _closure_anchor_parts(
        art, body_p, not prescribed, anchor_p, body_q, body_rel_t, articulation_ref_t
    )
    big_b, small_b = _closure_anchor_parts(
        art, body_c, True, wp.transform_get_translation(joint_X_c[joint]), body_q, body_rel_t, articulation_ref_t
    )
    rel_a = (big_a - frame_t) + small_a - origin
    rel_b = (big_b - frame_t) + small_b - origin
    delta = (big_a - big_b) + (small_a - small_b)
    at_midpoint = (
        not prescribed
        and _walk_has_free_dof(body_p, body_to_joint, joint_ancestor, joint_qd_start, kinematic_dof_mask)
        and _walk_has_free_dof(body_c, body_to_joint, joint_ancestor, joint_qd_start, kinematic_dof_mask)
    )
    if at_midpoint:
        rel_a = rel_a - 0.5 * delta
        rel_b = rel_a
    e = wp.vec3(0.0, 0.0, 0.0)
    phi = float(0.0)
    target = float(0.0)
    angular = float(0.0)
    if r < 3:
        e[r] = 1.0
        phi = delta[r]
        if prescribed:
            target = -wp.dot(e, v_a)
    else:
        dirs, phis = _connect_rotation_rows(
            connect_rows[k], body_p, body_c, joint, joint_axis, joint_qd_start, joint_X_p, joint_X_c, body_q
        )
        i = r - 3
        e = dirs[i]
        phi = phis[i]
        angular = 1.0
        if prescribed and body_p >= 0:
            target = -wp.dot(e, wp.spatial_bottom(body_qd[body_p]))
    row = k * CONNECT_MAX_ROWS + r
    geom_row[row, 0] = e[0]
    geom_row[row, 1] = e[1]
    geom_row[row, 2] = e[2]
    geom_row[row, 3] = phi
    geom_row[row, 4] = target
    geom_row[row, 5] = angular
    if r == 0:
        connect_at_midpoint[k] = int(at_midpoint)
        geom_closure[k, 0] = rel_a[0]
        geom_closure[k, 1] = rel_a[1]
        geom_closure[k, 2] = rel_a[2]
        geom_closure[k, 3] = rel_b[0]
        geom_closure[k, 4] = rel_b[1]
        geom_closure[k, 5] = rel_b[2]
        geom_closure[k, 6] = float(int(prescribed))
        match = int(at_midpoint) == mid_local[closure_local[k]]
        geom_closure[k, 7] = 1.0 if match else 2.0


@cache
def get_ds_rows_kernel(KJ: int, MAXR: int, M_D: int) -> wp.Kernel:
    WPB = ROWS_WPB
    SL = (KJ + 31) // 32
    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const int k = blk * {WPB} + (threadIdx.x >> 5);
    if (k >= n_closures) return;
    if (geom_closure.data[k * {CLOSURE_FLOATS} + 7] == 0.0f) return;
    const int rows = connect_rows.data[k];
    const int art = connect_art.data[k];
    const int world = art_to_world.data[art];
    const int kl = closure_local.data[k];
    const int dof0 = articulation_dof_start.data[art];
    const int base = connect_slot.data[k];
    const int ns = sup_count.data[kl];
    const float* gc = &geom_closure.data[k * {CLOSURE_FLOATS}];
    float lin[{SL}][3], ca[{SL}][3], cb[{SL}][3], ang[{SL}][3];
    int side[{SL}];
    #pragma unroll
    for (int sl = 0; sl < {SL}; sl++) {{
        const int a = lane + 32 * sl;
        side[sl] = 0;
        for (int q = 0; q < 3; q++) {{ lin[sl][q] = 0.0f; ang[sl][q] = 0.0f; }}
        if (a < ns) {{
            const int d = sup_dof.data[kl * {KJ} + a];
            side[sl] = sup_side.data[kl * {KJ} + a];
            const float* Sd = (const float*)&joint_S_s.data[dof0 + d];
            lin[sl][0] = Sd[0]; lin[sl][1] = Sd[1]; lin[sl][2] = Sd[2];
            ang[sl][0] = Sd[3]; ang[sl][1] = Sd[4]; ang[sl][2] = Sd[5];
        }}
        ca[sl][0] = ang[sl][1] * gc[2] - ang[sl][2] * gc[1];
        ca[sl][1] = ang[sl][2] * gc[0] - ang[sl][0] * gc[2];
        ca[sl][2] = ang[sl][0] * gc[1] - ang[sl][1] * gc[0];
        cb[sl][0] = ang[sl][1] * gc[5] - ang[sl][2] * gc[4];
        cb[sl][1] = ang[sl][2] * gc[3] - ang[sl][0] * gc[5];
        cb[sl][2] = ang[sl][0] * gc[4] - ang[sl][1] * gc[3];
    }}
    float* Jrow = &Jw.data[((size_t)world * {MAXR} * n_local + (size_t)kl * {MAXR}) * {KJ}];
    for (int r = 0; r < {MAXR}; r++) {{
        float v[{SL}];
        float s = 0.0f;
        const float* g = &geom_row.data[(size_t)(k * {CONNECT_MAX_ROWS} + r) * {GEOM_FLOATS}];
        const bool live = r < rows;
        const float e0 = live ? g[0] : 0.0f, e1 = live ? g[1] : 0.0f, e2 = live ? g[2] : 0.0f;
        const bool angular = live && g[5] != 0.0f;
        #pragma unroll
        for (int sl = 0; sl < {SL}; sl++) {{
            float vp, vc;
            if (angular) {{
                vp = e0 * ang[sl][0] + e1 * ang[sl][1] + e2 * ang[sl][2];
                vc = vp;
            }} else {{
                vp = e0 * (lin[sl][0] + ca[sl][0]) + e1 * (lin[sl][1] + ca[sl][1]) + e2 * (lin[sl][2] + ca[sl][2]);
                vc = e0 * (lin[sl][0] + cb[sl][0]) + e1 * (lin[sl][1] + cb[sl][1]) + e2 * (lin[sl][2] + cb[sl][2]);
            }}
            float x = 0.0f;
            if (side[sl] & 1) x += vp;
            if (side[sl] & 2) x -= vc;
            v[sl] = x;
            s += x * x;
        }}
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFF, s, o);
        const float inv = s > 1.0e-8f ? 1.0f / sqrtf(s) : 0.0f;
        #pragma unroll
        for (int sl = 0; sl < {SL}; sl++) {{
            const int a = lane + 32 * sl;
            if (a < {KJ}) Jrow[r * {KJ} + a] = v[sl] * inv;
        }}
        if (live && lane == 0) {{
            const size_t wr = (size_t)world * {M_D} + base + r;
            world_row_type.data[wr] = {PGS_CONSTRAINT_TYPE_CONNECT};
            world_row_parent.data[wr] = -1;
            world_row_mu.data[wr] = 0.0f;
            world_phi.data[wr] = g[3] * inv;
            world_target_velocity.data[wr] = g[4] * inv;
            connect_row_inv_norm.data[k * {CONNECT_MAX_ROWS} + r] = inv;
        }}
    }}
    if (lane == 0 && gc[7] != 1.0f) atomicAdd(&status.data[1], 1);
#endif
"""

    @wp.func_native(snippet)
    def ds_rows_native(
        blk: int,
        n_closures: int,
        n_local: int,
        connect_rows: wp.array[int],
        connect_art: wp.array[int],
        connect_slot: wp.array[int],
        art_to_world: wp.array[int],
        closure_local: wp.array[int],
        articulation_dof_start: wp.array[int],
        sup_count: wp.array[int],
        sup_dof: wp.array[int],
        sup_side: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        geom_row: wp.array2d[float],
        geom_closure: wp.array2d[float],
        status: wp.array[int],
        Jw: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_phi: wp.array2d[float],
        world_target_velocity: wp.array2d[float],
        connect_row_inv_norm: wp.array2d[float],
    ): ...

    def ds_rows(
        n_closures: int,
        n_local: int,
        connect_rows: wp.array[int],
        connect_art: wp.array[int],
        connect_slot: wp.array[int],
        art_to_world: wp.array[int],
        closure_local: wp.array[int],
        articulation_dof_start: wp.array[int],
        sup_count: wp.array[int],
        sup_dof: wp.array[int],
        sup_side: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        geom_row: wp.array2d[float],
        geom_closure: wp.array2d[float],
        status: wp.array[int],
        Jw: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_phi: wp.array2d[float],
        world_target_velocity: wp.array2d[float],
        connect_row_inv_norm: wp.array2d[float],
    ):
        blk, _lane = wp.tid()
        ds_rows_native(
            blk,
            n_closures,
            n_local,
            connect_rows,
            connect_art,
            connect_slot,
            art_to_world,
            closure_local,
            articulation_dof_start,
            sup_count,
            sup_dof,
            sup_side,
            joint_S_s,
            geom_row,
            geom_closure,
            status,
            Jw,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_phi,
            world_target_velocity,
            connect_row_inv_norm,
        )

    name = f"closure_ds_rows_{KJ}_{MAXR}_{M_D}"
    ds_rows.__name__ = name
    ds_rows.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(ds_rows)


def launch_connect_rows(solver, state_in, state_aug) -> None:
    """Build the connect rows on their static supports (replaces ``populate_connect_J_for_size``)."""
    buf = solver._closure_direct_sparse
    model = solver.model
    nk = solver._connect_count
    size = buf.size
    wp.launch(
        ds_connect_geometry,
        dim=nk * CONNECT_MAX_ROWS,
        inputs=[
            solver._articulation_dof_start,
            solver._art_to_world,
            solver._group_to_art[size],
            solver._art_group_idx,
            solver._connect_slot,
            solver._connect_art,
            solver._connect_joint,
            solver._connect_body_p,
            solver._connect_body_c,
            solver._connect_rows,
            model.joint_axis,
            model.joint_X_p,
            model.joint_X_c,
            solver._connect_parent_prescribed,
            solver._kinematic_dof_mask,
            state_in.body_q,
            state_in.body_qd,
            model.body_com,
            solver._body_to_joint,
            model.joint_ancestor,
            model.joint_qd_start,
            buf.closure_local,
            buf.mid_local,
            solver._body_rel_t,
            solver._articulation_ref_t,
            model.articulation_start,
            model.joint_child,
        ],
        outputs=[buf.geom_row, buf.geom_closure, solver._connect_at_midpoint],
        device=model.device,
    )
    wp.launch_tiled(
        get_ds_rows_kernel(buf.KJ, buf.MAXR, buf.M_D),
        dim=[(nk + ROWS_WPB - 1) // ROWS_WPB],
        inputs=[
            nk,
            buf.KL,
            solver._connect_rows,
            solver._connect_art,
            solver._connect_slot,
            solver._art_to_world,
            buf.closure_local,
            solver._articulation_dof_start,
            buf.sup_count,
            buf.sup_dof,
            buf.sup_side,
            state_aug._joint_S_s,
            buf.geom_row,
            buf.geom_closure,
            buf.status,
        ],
        outputs=[
            buf.Jw,
            solver._row_type,
            solver._row_parent,
            solver._row_mu,
            solver._phi,
            solver._target_velocity,
            solver._connect_row_inv_norm,
        ],
        block_dim=32 * ROWS_WPB,
        device=model.device,
    )


# ---------------------------------------------------------------------------------------------------------------
# Solve: one block per world (rows, assembly, supernodal factor, solves, v and impulses)
# ---------------------------------------------------------------------------------------------------------------

_WARP_SUM = "for (int o = 16; o > 0; o >>= 1) {v} += __shfl_xor_sync(0xFFFFFFFF, {v}, o); {v} = __shfl_sync(0xFFFFFFFF, {v}, 0);"


@cache
def get_ds_solve_kernel(
    D: int,
    NNZ: int,
    NLEV: int,
    KL: int,
    KJ: int,
    MAXR: int,
    NA: int,
    M_D: int,
    USE_SMEM: bool,
    NT: int = SOLVE_THREADS,
    QW: int = 1,
    TILES: int = 1,
    BC: int = 4,
    WB_IN_VEC: bool = True,
    TIMING: bool = False,
    CMODE: int = 0,
    YW: int = 0,
    YOFF: int = 0,
    C: int = 1,
    KS: int = 1,
    KZ: int = 1,
    SIDES: int = 1,
    YR: int = 1,
) -> wp.Kernel:
    TPT = max(1, (TILES + NT - 1) // NT)
    SLOT = NT // BC
    DZ = (528 + NT - 1) // NT  # lower-triangle entries of a 32 x 32 block per thread
    SOLVE_WHOLE = 4  # solves: levels with at most this many supernodes run them one at a time with the whole block
    WHOLE_W = globals()["WHOLE_W"]
    NW = NT // 32
    SL = (KJ + 31) // 32
    CF = CLOSURE_FLOATS

    def wsum(v):
        return _WARP_SUM.format(v=v)

    CONTACT = int(PGS_CONSTRAINT_TYPE_CONTACT)
    FRICTION = int(PGS_CONSTRAINT_TYPE_FRICTION)
    # Contact rows (CMODE 1, "elim"): their responses Y = K^-1 J^T through the closure factor, after the closure solve.
    contact_common = f"""
    const int ccnt = min(constraint_count.data[world], {M_D});
    const float* Jc = &Jg.data[(size_t)world * {C * KS}];
"""
    contact_post = ""

    def multi_row_responses(head: str) -> str:
        """Compact layout, YR rows per warp: every pass over L serves YR rows (one load of L and its row index per
        entry), the column loops and reductions are shared; YW warps stay in flight to hide latency."""
        R = YR
        ks = range(R)

        def each(fmt: str, sep: str = " ") -> str:
            return sep.join(fmt.format(k=k) for k in ks)

        def sides_k(k: int, body: str) -> str:
            return "\n".join(
                f"""                if (ok{k}) {{
                    const int head = ((hd{k} >> {16 * s}) & 0xFFFF) - 1;
                    if (head >= 0) {{
                        const int n = na_arr.data[head];
                        const int* apath = &ancestors.data[(size_t)head * {NA}];
                        const float* Js = Jr{k} + {s * KZ};
{body.replace("xw", f"xw{k}").replace("part", f"part{k}")}
                    }}
                }}
                __syncwarp();"""
                for s in range(SIDES)
            )

        scatter = "\n".join(
            sides_k(k, "                        for (int s = lane; s < n; s += 32) xw[iperm.data[apath[s]]] += Js[s];")
            for k in ks
        )
        dot = "\n".join(
            sides_k(
                k,
                "                        for (int s = lane; s < n; s += 32) part += Js[s] * xw[iperm.data[apath[s]]];",
            )
            for k in ks
        )
        red = "".join(
            f"for (int o = 16; o > 0; o >>= 1) {{ {each('acc{k} += __shfl_xor_sync(0xFFFFFFFF, acc{k}, o);')} }}"
            for _ in [0]
        )
        redp = "".join(
            f"for (int o = 16; o > 0; o >>= 1) {{ {each('part{k} += __shfl_xor_sync(0xFFFFFFFF, part{k}, o);')} }}"
            for _ in [0]
        )
        return f"""
    {{{head}
        if (warp < {YW}) {{
            {each("float* xw{k} = ds_dyn + " + str(YOFF) + " + (warp * " + str(R) + " + {k}) * " + str(D) + ";")}
            for (int g = warp; g * {R} < nc; g += {YW}) {{
                {each("const int r{k} = g * " + str(R) + " + {k};")}
                {each("const int t{k} = r{k} < nc ? row_type.data[wrow + cb + r{k}] : -1;")}
                {each("const bool ok{k} = t{k} == " + str(CONTACT) + " || t{k} == " + str(FRICTION) + ";")}
                if (lane == 0) {{ {each("if (r{k} < nc && !ok{k}) atomicAdd(&status.data[6], 1);")} }}
                {each("const float* Jr{k} = Jc + (size_t)(ok{k} ? r{k} : 0) * " + str(KS) + ";")}
                {each("const int hd{k} = ok{k} ? Hc[r{k}] : 0;")}
                for (int d = lane; d < {D}; d += 32) {{ {each("xw{k}[d] = 0.0f;")} }}
                __syncwarp();
{scatter}
                // Forward L z = P J^T for the group's rows at once; columns zero in every row are skipped.
                for (int j = 0; j < {D}; j++) {{
                    {each("float x{k} = xw{k}[j];")}
                    if ({each("x{k} == 0.0f", " && ")}) continue;
                    const int o = co[j], o1 = co[j + 1];
                    {each("x{k} = x{k} / L[o];")}
                    __syncwarp();
                    if (lane == 0) {{ {each("xw{k}[j] = x{k};")} }}
                    for (int q = o + 1 + lane; q < o1; q += 32) {{
                        const float l = L[q];
                        const int i = ri[q];
                        {each("xw{k}[i] -= l * x{k};")}
                    }}
                    __syncwarp();
                }}
                // Backward L^T y = z.
                for (int j = {D} - 1; j >= 0; j--) {{
                    const int o = co[j], o1 = co[j + 1];
                    {each("float acc{k} = 0.0f;")}
                    for (int q = o + 1 + lane; q < o1; q += 32) {{
                        const float l = L[q];
                        const int i = ri[q];
                        {each("acc{k} += l * xw{k}[i];")}
                    }}
                    {red}
                    if (lane == 0) {{ {each("xw{k}[j] = (xw{k}[j] - acc{k}) / L[o];")} }}
                    __syncwarp();
                }}
                for (int d = lane; d < {D}; d += 32) {{
                    const int id = iperm.data[d];
                    {each("if (ok{k}) Yc[(size_t)r{k} * " + str(D) + " + d] = xw{k}[id];")}
                }}
                {each("float part{k} = 0.0f;")}
{dot}
                {redp}
                if (lane == 0) {{ {each("if (ok{k}) cdiag.data[wrow + cb + r{k}] += part{k};")} }}
                __syncwarp();
            }}
        }}
        __syncthreads();
    }}
"""

    def compact_responses() -> str:
        """Compact layout: Y_r = K^-1 J_r^T and diag_r += J_r Y_r for the world's contact and friction rows.

        J_r is scattered from its support (by depth along each side's root path) and J_r Y_r runs on the support.
        """

        def sides(body: str) -> str:
            return "\n".join(
                f"""                {{
                    const int head = ((hd >> {16 * k}) & 0xFFFF) - 1;
                    if (head >= 0) {{
                        const int n = na_arr.data[head];
                        const int* apath = &ancestors.data[(size_t)head * {NA}];
                        const float* Js = Jr + {k * KZ};
{body}
                    }}
                    __syncwarp();
                }}"""
                for k in range(SIDES)
            )

        scatter_warp = sides(
            "                        for (int s = lane; s < n; s += 32) xw[iperm.data[apath[s]]] += Js[s];"
        )
        dot_warp = sides(
            "                        for (int s = lane; s < n; s += 32) part += Js[s] * xw[iperm.data[apath[s]]];"
        )
        head = f"""
    const int cb = cbase.data[world];
    int nc = ccnt - cb;
    if (nc > {C}) nc = {C};
    float* Yc = &Yg.data[(size_t)world * {C} * {D}];
    const int* Hc = &chead.data[(size_t)world * {C}];
"""
        if YW > 0 and YR > 1:
            return multi_row_responses(head)
        if YW > 0:
            return f"""
    {{{head}
        float* xw = ds_dyn + {YOFF} + warp * {D};
        if (warp < {YW}) {{
            for (int r = warp; r < nc; r += {YW}) {{
                const int t = row_type.data[wrow + cb + r];
                if (t != {CONTACT} && t != {FRICTION}) {{
                    if (lane == 0) atomicAdd(&status.data[6], 1);
                    continue;
                }}
                const float* Jr = Jc + (size_t)r * {KS};
                const int hd = Hc[r];
                for (int d = lane; d < {D}; d += 32) xw[d] = 0.0f;
                __syncwarp();
{scatter_warp}
                // Forward L z = P J^T, column by column; columns still zero are skipped.
                for (int j = 0; j < {D}; j++) {{
                    float xj = xw[j];
                    if (xj == 0.0f) continue;
                    const int o = co[j], o1 = co[j + 1];
                    xj /= L[o];
                    __syncwarp();
                    if (lane == 0) xw[j] = xj;
                    for (int q = o + 1 + lane; q < o1; q += 32) xw[ri[q]] -= L[q] * xj;
                    __syncwarp();
                }}
                // Backward L^T y = z.
                for (int j = {D} - 1; j >= 0; j--) {{
                    const int o = co[j], o1 = co[j + 1];
                    float acc = 0.0f;
                    for (int q = o + 1 + lane; q < o1; q += 32) acc += L[q] * xw[ri[q]];
                    {wsum("acc")}
                    if (lane == 0) xw[j] = (xw[j] - acc) / L[o];
                    __syncwarp();
                }}
                for (int d = lane; d < {D}; d += 32) Yc[(size_t)r * {D} + d] = xw[iperm.data[d]];
                float part = 0.0f;
{dot_warp}
                {wsum("part")}
                if (lane == 0) cdiag.data[wrow + cb + r] += part;
                __syncwarp();
            }}
        }}
        __syncthreads();
    }}
"""
        scatter_block = "\n".join(
            f"""            {{
                const int head = ((hd >> {16 * k}) & 0xFFFF) - 1;
                if (head >= 0) {{
                    const int n = na_arr.data[head];
                    const int* apath = &ancestors.data[(size_t)head * {NA}];
                    for (int s = tid; s < n; s += {NT}) x[iperm.data[apath[s]]] += Jr[{k * KZ} + s];
                }}
                __syncthreads();
            }}"""
            for k in range(SIDES)
        )
        dot_block = "\n".join(
            f"""            {{
                const int head = ((hd >> {16 * k}) & 0xFFFF) - 1;
                if (head >= 0) {{
                    const int n = na_arr.data[head];
                    const int* apath = &ancestors.data[(size_t)head * {NA}];
                    for (int s = tid; s < n; s += {NT}) part += Jr[{k * KZ} + s] * x[iperm.data[apath[s]]];
                }}
            }}"""
            for k in range(SIDES)
        )
        return f"""
    {{{head}
        for (int r = 0; r < nc; r++) {{
            const int t = row_type.data[wrow + cb + r];
            if (t != {CONTACT} && t != {FRICTION}) {{
                if (tid == 0) atomicAdd(&status.data[6], 1);
                continue;
            }}
            const float* Jr = Jc + (size_t)r * {KS};
            const int hd = Hc[r];
            for (int i = tid; i < {D}; i += {NT}) x[i] = 0.0f;
            __syncthreads();
{scatter_block}
            trisolve();
            for (int i = tid; i < {D}; i += {NT}) Yc[(size_t)r * {D} + perm.data[i]] = x[i];
            float part = 0.0f;
{dot_block}
            {wsum("part")}
            if (lane == 0) sInv[warp] = part;
            __syncthreads();
            if (tid == 0) {{
                float sum = 0.0f;
                for (int z = 0; z < {NW}; z++) sum += sInv[z];
                cdiag.data[wrow + cb + r] += sum;
            }}
            __syncthreads();
        }}
    }}
"""

    if CMODE == 1:
        contact_post = (
            contact_common
            + """
    // Contact responses through the closure factor: Y_r = K^-1 J_r^T, diag_r += J_r Y_r (the factor stays in L).
"""
            + compact_responses()
        )

    snippet = f"""
#undef DS_TIMING
#define DS_TIMING {int(TIMING)}
#if defined(__CUDA_ARCH__)
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    extern __shared__ float ds_dyn[];
    float* g = ds_dyn;            // [D] g0 = -J^T W r0 (new order)
    float* x = g + {D};           // [D] solve vector (new order)
    float* tv = x + {D};          // [D] scratch
    float* sm = tv + {D};         // [D] sum of the solves (new order)
    float* sD = sm + {D};         // [32 x 33] diagonal block of a block-wide panel
    float* sInv = sD + 32 * 33;   // [32]
    int* co = (int*)(sInv + 32);  // [D + 1] column offsets of L's storage
    float* L = {"(float*)(co + " + str(D + 1) + ")" if USE_SMEM else "&Lg.data[(size_t)world * " + str(NNZ) + "]"};
    // Dense column buffers of the whole-block updates: x .. sD while factoring (or their own region).
    float* wbuf = {"x" if WB_IN_VEC else "(float*)(co + " + str(D + 1) + ") + " + (str(NNZ) if USE_SMEM else "0")};
#if DS_TIMING
    unsigned long long t0, t1, t2, t3;
#endif
#if DS_TIMING
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
#endif
    const size_t wrow = (size_t)world * {M_D};
    float* Jb = &Jw.data[(size_t)world * {KL * MAXR * KJ}];
    float* Q = &qrow.data[(size_t)world * {KL * MAXR}];
    float* SW = &swrow.data[(size_t)world * {KL * MAXR}];
    const float* Hw = &Ha.data[(size_t)world * {D * NA}];
    const int* dofi = &world_dof_indices.data[(size_t)world * {D}];
    const int kbase = world_closure_start.data[world];
    for (int i = tid; i <= {D}; i += {NT}) co[i] = col_off.data[i];
    const int* ri = rowidx.data;
    const int* sn = sn_info.data;
    int dr[{DZ}], dc[{DZ}];  // this thread's entries of the 32 x 32 lower triangle (528 in all)
    #pragma unroll
    for (int z = 0; z < {DZ}; z++) {{
        const int e = tid + z * {NT};
        int r = (int)((sqrtf(8.0f * e + 1.0f) - 1.0f) * 0.5f);
        while (r * (r + 1) / 2 > e) r--;
        while ((r + 1) * (r + 2) / 2 <= e) r++;
        dr[z] = e < 528 ? r : 32;
        dc[z] = e < 528 ? e - r * (r + 1) / 2 : 0;
    }}
    __syncthreads();

    // 1. Rows: weights, bias, r0 = J v_hat + b; rows scaled by sqrt(w) in place.
    for (int kl = warp; kl < {KL}; kl += {NW}) {{
        const int k = kbase + kl;
        const int slot = connect_slot.data[k];
        const bool valid = slot >= 0 && geom_closure.data[(size_t)k * {CF} + 7] != 0.0f;
        const int rows = valid ? rows_local.data[kl] : 0;
        const int ns = sup_count.data[kl];
        float* Jk = Jb + (size_t)kl * {MAXR * KJ};
        float vh[{SL}], hd[{SL}], q0[{SL}];
        #pragma unroll
        for (int sl = 0; sl < {SL}; sl++) {{
            const int a = lane + 32 * sl;
            vh[sl] = 0.0f; hd[sl] = 0.0f; q0[sl] = 0.0f;
            if (a < ns) {{
                const int d = sup_dof.data[kl * {KJ} + a];
                const int gd = dofi[d];
                hd[sl] = Hw[d * {NA}];
                if (gd >= 0) {{
                    vh[sl] = v_out.data[gd];
                    if (bias_mode == 1) q0[sl] = qd0.data[gd];
                }}
            }}
        }}
        for (int r = 0; r < {MAXR}; r++) {{
            float jr[{SL}];
            float jv = 0.0f, js = 0.0f, j0 = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < {SL}; sl++) {{
                const int a = lane + 32 * sl;
                jr[sl] = a < ns ? Jk[r * {KJ} + a] : 0.0f;
                jv += jr[sl] * vh[sl];
                if (hd[sl] > 0.0f) js += jr[sl] * jr[sl] / hd[sl];
                j0 += jr[sl] * q0[sl];
            }}
            {wsum("jv")}
            {wsum("js")}
            {wsum("j0")}
            float sw = 0.0f, q = 0.0f;
            if (r < rows) {{
                float b = rhs.data[wrow + slot + r];
                float e = eps;
                if (bias_mode == 1) {{
                    const float f = 1.0f + h_dt * spring_b + h_dt * h_dt * spring_k;
                    b = -target_velocity.data[wrow + slot + r] + (h_dt * spring_k * phi.data[wrow + slot + r] - j0) / f;
                    e = eps / f;
                }}
                const float w = js > 0.0f ? 1.0f / (e * js) : 0.0f;
                sw = sqrtf(w);
                q = sw * (jv + b);
            }}
            #pragma unroll
            for (int sl = 0; sl < {SL}; sl++) {{
                const int a = lane + 32 * sl;
                if (a < {KJ}) Jk[r * {KJ} + a] = jr[sl] * sw;
            }}
            if (lane == 0) {{ Q[kl * {MAXR} + r] = q; SW[kl * {MAXR} + r] = sw; }}
        }}
    }}
    __syncthreads();
#if DS_TIMING
    if (tid == 0 && world == 0) {{ unsigned long long tq; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tq)); timing.data[100] = (long long)(tq - t0); }}
#endif

    // Forward and backward solves with L (new order), in place on x; every thread of the block calls it.
    auto trisolve = [&]() {{
        // Forward L y = b: per supernode, gather the descendants' terms, then its own triangle.
        for (int lev = 0; lev < {NLEV}; lev++) {{
            const int l0 = lvl_start.data[lev], l1 = lvl_start.data[lev + 1];
            const bool whole = l1 - l0 <= {SOLVE_WHOLE};
            for (int li = l0 + (whole ? 0 : warp); li < l1; li += (whole ? 1 : {NW})) {{
                const int s = lvl_sn.data[li];
                const int c0 = sn[5 * s], w = sn[5 * s + 1];
                for (int tt = (whole ? warp : 0); tt < w; tt += (whole ? {NW} : 1)) {{
                    const int col = c0 + tt;
                    float acc = 0.0f;
                    const int e1 = row_split.data[col];
                    for (int e = row_ptr.data[col] + lane; e < e1; e += 32) acc += L[row_ent.data[e]] * x[row_col.data[e]];
                    {wsum("acc")}
                    if (lane == 0) x[col] -= acc;
                }}
                if (whole) __syncthreads(); else __syncwarp();
                for (int kb = 0; kb < w; kb += 32) {{
                    const int nb = min(32, w - kb);
                    if (!whole || warp == 0) {{
                        float xr = lane < nb ? x[c0 + kb + lane] : 0.0f;
                        for (int j = 0; j < nb; j++) {{
                            const int oj = co[c0 + kb + j] - j;
                            if (lane == j) xr /= L[oj + j];
                            const float xj = __shfl_sync(0xFFFFFFFF, xr, j);
                            if (lane > j && lane < nb) xr -= L[oj + lane] * xj;
                        }}
                        if (lane < nb) x[c0 + kb + lane] = xr;
                    }}
                    if (whole) __syncthreads(); else __syncwarp();
                    if (kb + nb < w) {{
                        const int first = kb + nb + (whole ? tid : lane);
                        for (int u = first; u < w; u += (whole ? {NT} : 32)) {{
                            float acc = 0.0f;
                            for (int j = 0; j < nb; j++) acc += L[co[c0 + kb + j] + u - (kb + j)] * x[c0 + kb + j];
                            x[c0 + u] -= acc;
                        }}
                        if (whole) __syncthreads(); else __syncwarp();
                    }}
                }}
            }}
            __syncthreads();
        }}
        // Backward L^T x = y: per supernode, gather the ancestors' terms, then its own triangle from the bottom.
        for (int lev = {NLEV} - 1; lev >= 0; lev--) {{
            const int l0 = lvl_start.data[lev], l1 = lvl_start.data[lev + 1];
            const bool whole = l1 - l0 <= {SOLVE_WHOLE};
            for (int li = l0 + (whole ? 0 : warp); li < l1; li += (whole ? 1 : {NW})) {{
                const int s = lvl_sn.data[li];
                const int c0 = sn[5 * s], w = sn[5 * s + 1], q = sn[5 * s + 2];
                const int rbase = co[c0] + w;  // the rows below the supernode, R_s[w:], as stored in column c0
                for (int tt = (whole ? warp : 0); tt < w; tt += (whole ? {NW} : 1)) {{
                    const int o = co[c0 + tt] + (w - tt);
                    float acc = 0.0f;
                    for (int e = lane; e < q - w; e += 32) acc += L[o + e] * x[ri[rbase + e]];
                    {wsum("acc")}
                    if (lane == 0) x[c0 + tt] -= acc;
                }}
                if (whole) __syncthreads(); else __syncwarp();
                for (int kb = ((w - 1) / 32) * 32; kb >= 0; kb -= 32) {{
                    const int nb = min(32, w - kb);
                    if (!whole || warp == 0) {{
                        float xr = lane < nb ? x[c0 + kb + lane] : 0.0f;
                        const int ol = co[c0 + kb + (lane < nb ? lane : 0)] - (lane < nb ? lane : 0);
                        for (int j = nb - 1; j >= 0; j--) {{
                            if (lane == j) xr /= L[ol + j];
                            const float xj = __shfl_sync(0xFFFFFFFF, xr, j);
                            if (lane < j) xr -= L[ol + j] * xj;
                        }}
                        if (lane < nb) x[c0 + kb + lane] = xr;
                    }}
                    if (whole) __syncthreads(); else __syncwarp();
                    if (kb > 0) {{
                        const int first = whole ? tid : lane;
                        for (int t = first; t < kb; t += (whole ? {NT} : 32)) {{
                            float acc = 0.0f;
                            const int obase = co[c0 + t] - t;
                            for (int j = 0; j < nb; j++) acc += L[obase + kb + j] * x[c0 + kb + j];
                            x[c0 + t] -= acc;
                        }}
                        if (whole) __syncthreads(); else __syncwarp();
                    }}
                }}
            }}
            __syncthreads();
        }}
    }};
    const int nsolve = iterations > 0 ? iterations : 1;
    // 2. Assembly of P K P^T into L's storage, and g0.
    {{
        // Record per entry: (H index or -1, first two contributions, (start << 8) | count); two entries in flight.
        const int4* arec = (const int4*)asm_rec.data;
        auto contrib = [&](int code) -> float {{
            const float* Jk = Jb + (size_t)(code >> 12) * {MAXR * KJ};
            const int a = (code >> 6) & 63, b = code & 63;
            float v = 0.0f;
            #pragma unroll
            for (int r = 0; r < {MAXR}; r++) v += Jk[r * {KJ} + a] * Jk[r * {KJ} + b];
            return v;
        }};
        for (int e0 = tid; e0 < {NNZ}; e0 += 2 * {NT}) {{
            const int e1 = e0 + {NT};
            const int4 ra = arec[e0];
            const int4 rb = e1 < {NNZ} ? arec[e1] : make_int4(-1, 0, 0, 0);
            const int na = ra.w & 255, nb2 = rb.w & 255;
            float va = ra.x >= 0 ? Hw[ra.x] : 0.0f;
            float vb = rb.x >= 0 ? Hw[rb.x] : 0.0f;
            if (na > 0) va += contrib(ra.y);
            if (nb2 > 0) vb += contrib(rb.y);
            if (na > 1) va += contrib(ra.z);
            if (nb2 > 1) vb += contrib(rb.z);
            for (int c = 2; c < na; c++) va += contrib(asm_list.data[(ra.w >> 8) + c]);
            for (int c = 2; c < nb2; c++) vb += contrib(asm_list.data[(rb.w >> 8) + c]);
            L[e0] = va;
            if (e1 < {NNZ}) L[e1] = vb;
        }}
    }}
    for (int i = tid; i < {D}; i += {NT}) {{
        float s = 0.0f;
        const int c1 = g_ptr.data[i + 1];
        for (int c = g_ptr.data[i]; c < c1; c++) {{
            const int code = g_list.data[c];
            const int k = code >> 6, a = code & 63;
            #pragma unroll
            for (int r = 0; r < {MAXR}; r++) s += Jb[((size_t)k * {MAXR} + r) * {KJ} + a] * Q[k * {MAXR} + r];
        }}
        g[i] = -s;
        sm[i] = 0.0f;
    }}
    __syncthreads();
#if DS_TIMING
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
#endif

    // 3. Left-looking supernodal Cholesky, one height level of the supernodal elimination tree at a time.
    //    A level that is one wide supernode (such as the dense root separator) runs with the whole block: a dense
    //    update from its descendants' columns, then panels of 32. Other levels spread their (supernode, column)
    //    update tasks over the warps, then factor each supernode within a warp.
    for (int lev = 0; lev < {NLEV}; lev++) {{
        const int l0 = lvl_start.data[lev], l1 = lvl_start.data[lev + 1];
        const bool whole = l1 - l0 == 1 && sn[5 * lvl_sn.data[l0] + 1] >= {WHOLE_W};
        if (!whole) {{
            // Updates: (supernode, column) tasks of the level spread over all warps; then a warp per supernode
            // factors its columns.
            int task = 0;
            for (int li = l0; li < l1; li++) {{
                const int s = lvl_sn.data[li];
                const int c0 = sn[5 * s], w = sn[5 * s + 1];
                for (int tc = 0; tc < w; tc++, task++) {{
                    if (task % {NW} != warp) continue;
                    const int col = c0 + tc;
                    const int obase = co[col] - tc;
                    const int i0 = ct_ptr.data[col], i1 = ct_ptr.data[col + 1];
                const int4* recs = (const int4*)ct_rec.data;
                for (int ib = i0; ib < i1; ib += 32) {{
                    const int nit = min(32, i1 - ib);
                    const int4 mine = lane < nit ? recs[ib + lane] : make_int4(0, 0, 0, 0);
                    // rel of item t (first chunk), two items in flight.
                    auto first_rel = [&](int t) -> int {{
                        const int z = __shfl_sync(0xFFFFFFFF, mine.z, t), pjt = __shfl_sync(0xFFFFFFFF, mine.w, t);
                        const int qt = __shfl_sync(0xFFFFFFFF, mine.y, t) >> 16;
                        return (t < nit && pjt + lane < qt) ? rel.data[z + pjt + lane] : 0;
                    }};
                    int rl0 = first_rel(0);
                    int rl1 = nit > 1 ? first_rel(1) : 0;
                    for (int t = 0; t < nit; t++) {{
                        const int rl2 = t + 2 < nit ? first_rel(t + 2) : 0;
                        const int c0d = __shfl_sync(0xFFFFFFFF, mine.x, t);
                        const int wq = __shfl_sync(0xFFFFFFFF, mine.y, t);
                        const int rb = __shfl_sync(0xFFFFFFFF, mine.z, t);
                        const int pj = __shfl_sync(0xFFFFFFFF, mine.w, t);
                        const int wd = wq & 0xFFFF, qd = wq >> 16;
                        for (int pb = pj; pb < qd; pb += 32) {{
                            const int pi = pb + lane;
                            const bool act = pi < qd;
                            const int tgt = act ? obase + (pb == pj ? rl0 : rel.data[rb + pi]) : 0;
                            float acc = 0.0f;
                            for (int cb = 0; cb < wd; cb += 32) {{
                                const int myoc = cb + lane < wd ? co[c0d + cb + lane] - (cb + lane) : 0;
                                const int nc = min(32, wd - cb);
                                for (int c = 0; c < nc; c++) {{
                                    const int oc = __shfl_sync(0xFFFFFFFF, myoc, c);
                                    acc += L[oc + (act ? pi : pj)] * L[oc + pj];
                                }}
                            }}
                            if (act) L[tgt] -= acc;
                        }}
                        rl0 = rl1;
                        rl1 = rl2;
                    }}
                }}
                }}
            }}
            __syncthreads();
            for (int li = l0 + warp; li < l1; li += {NW}) {{
                const int s = lvl_sn.data[li];
                const int c0 = sn[5 * s], w = sn[5 * s + 1], q = sn[5 * s + 2];
                for (int tt = 0; tt < w; tt++) {{
                    const int o = co[c0 + tt];
                    float piv = L[o];
                    if (!(piv > 0.0f)) {{ if (lane == 0) atomicAdd(&status.data[0], 1); piv = 1.0e-30f; }}
                    const float dg = sqrtf(piv);
                    const float id = 1.0f / dg;
                    __syncwarp();
                    if (lane == 0) L[o] = dg;
                    for (int u = tt + 1 + lane; u < q; u += 32) L[o + u - tt] *= id;
                    __syncwarp();
                    for (int t2 = tt + 1; t2 < w; t2++) {{
                        const float f = L[o + t2 - tt];
                        const int o2 = co[c0 + t2] - t2;
                        for (int u = t2 + lane; u < q; u += 32) L[o2 + u] -= L[o + u - tt] * f;
                    }}
                    __syncwarp();
                }}
            }}
            __syncthreads();
#if DS_TIMING
            if (tid == 0 && world == 0) {{ unsigned long long tq; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tq)); timing.data[5 + 2 * lev] = (long long)tq; }}
#endif
            continue;
        }}
        {{
            const int li = l0;
            const int s = lvl_sn.data[li];
            const int c0 = sn[5 * s], w = sn[5 * s + 1], q = sn[5 * s + 2];
            {{
                // Dense update: each descendant column's part on R_s is scattered into a dense q-vector
                // (three rotating buffers of BC columns), every thread keeps 4 x 4 tiles of the trapezoid.
                const int nct = (w + 3) >> 2, nrt = (q + 3) >> 2;
                int tct[{TPT}], trt[{TPT}];
                #pragma unroll
                for (int z = 0; z < {TPT}; z++) {{
                    int idx = tid + z * {NT};
                    int ct = 0;
                    while (ct < nct && idx >= nrt - ct) {{ idx -= nrt - ct; ct++; }}
                    tct[z] = ct < nct ? ct : -1;
                    trt[z] = ct < nct ? ct + idx : 0;
                }}
                float acc[{TPT}][16];
                #pragma unroll
                for (int z = 0; z < {TPT}; z++)
                    #pragma unroll
                    for (int k = 0; k < 16; k++) acc[z][k] = 0.0f;
                const int s0 = sc_ptr.data[s], ncols = sc_ptr.data[s + 1] - s0;
                const int nbat = (ncols + {BC} - 1) / {BC};
                const int4* srec = (const int4*)sc_rec.data + s0;
                // Scatter slot of this thread: column my_jb of each batch, rows p0 + my_po (+ SLOT ...).
                const int my_jb = tid / {SLOT}, my_po = tid % {SLOT};
                auto buf = [&](int b) -> float* {{ return wbuf + (size_t)(b % 3) * {BC} * q; }};
                auto zero = [&](int b) {{ float* B = buf(b); for (int i = tid; i < {BC} * q; i += {NT}) B[i] = 0.0f; }};
                auto rec_of = [&](int b) -> int4 {{
                    const int jj = b * {BC} + my_jb;
                    return (b < nbat && my_jb < {BC} && jj < ncols) ? srec[jj] : make_int4(0, 0, 0, 0);
                }};
                auto rel_of = [&](int4 r) -> int {{ return my_po + r.y < r.z ? rel.data[r.w + r.y + my_po] : -1; }};
                auto scatter = [&](int b, int4 r, int rv) {{
                    if (my_jb >= {BC}) return;
                    float* B = buf(b) + my_jb * q;
                    if (rv >= 0) B[rv] = L[r.x + r.y + my_po];
                    for (int p = r.y + my_po + {SLOT}; p < r.z; p += {SLOT}) B[rel.data[r.w + p]] = L[r.x + p];
                }};
                if (nbat > 0) {{
                    int4 r1 = rec_of(0), r2 = rec_of(1), r3 = rec_of(2);
                    int v1 = rel_of(r1), v2 = rel_of(r2);
                    zero(0);
                    if (nbat > 1) zero(1);
                    __syncthreads();
                    scatter(0, r1, v1);
                    __syncthreads();
                    r1 = r2; v1 = v2; r2 = r3;
                    for (int b = 0; b < nbat; b++) {{
                        const int4 r4 = rec_of(b + 3);
                        const int v3 = rel_of(r2);
                        const float* B = buf(b);
                        const int nj = min({BC}, ncols - b * {BC});
                        #pragma unroll
                        for (int z = 0; z < {TPT}; z++) {{
                            if (tct[z] < 0) continue;
                            const int tb = 4 * tct[z], ub = 4 * trt[z];
                            for (int jb = 0; jb < nj; jb++) {{
                                const float* col = B + jb * q;
                                float vu[4], vt[4];
                                #pragma unroll
                                for (int k = 0; k < 4; k++) {{ vu[k] = ub + k < q ? col[ub + k] : 0.0f; vt[k] = tb + k < q ? col[tb + k] : 0.0f; }}
                                #pragma unroll
                                for (int a = 0; a < 4; a++)
                                    #pragma unroll
                                    for (int k = 0; k < 4; k++) acc[z][a * 4 + k] += vu[a] * vt[k];
                            }}
                        }}
                        if (b + 1 < nbat) scatter(b + 1, r1, v1);
                        if (b + 2 < nbat) zero(b + 2);
                        __syncthreads();
                        r1 = r2; v1 = v3; r2 = r4;
                    }}
                    #pragma unroll
                    for (int z = 0; z < {TPT}; z++) {{
                        if (tct[z] < 0) continue;
                        const int tb = 4 * tct[z], ub = 4 * trt[z];
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {{
                            const int t = tb + k;
                            if (t >= w) continue;
                            const int ob = co[c0 + t] - t;
                            #pragma unroll
                            for (int a = 0; a < 4; a++) {{
                                const int u = ub + a;
                                if (u < q && u >= t) L[ob + u] -= acc[z][a * 4 + k];
                            }}
                        }}
                    }}
                }}
            }}
            __syncthreads();
#if DS_TIMING
            if (tid == 0 && world == 0) {{ unsigned long long tq; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tq)); timing.data[4 + 2 * lev] = (long long)tq; }}
#endif
            // Whole-block supernode: right-looking panels of 32 columns.
            for (int k0 = 0; k0 < w; k0 += 32) {{
                const int nb = min(32, w - k0);
                // Diagonal block: right-looking rank-1 steps on an unscaled copy, the whole block (528 entries).
                for (int e = tid; e < 528; e += {NT}) {{
                    const int r = dr[e / {NT}], c = dc[e / {NT}];
                    if (r < nb) sD[r * 33 + c] = L[co[c0 + k0 + c] + r - c];
                }}
                __syncthreads();
                for (int j = 0; j < nb; j++) {{
                    float pv = sD[j * 33 + j];
                    if (!(pv > 0.0f)) pv = 1.0e-30f;
                    const float ip = 1.0f / pv;
                    #pragma unroll
                    for (int z = 0; z < {DZ}; z++) {{
                        const int r = dr[z], c = dc[z];
                        if (c > j && r < nb) sD[r * 33 + c] -= sD[r * 33 + j] * sD[c * 33 + j] * ip;
                    }}
                    __syncthreads();
                }}
                for (int e = tid; e < 528; e += {NT}) {{
                    const int r = dr[e / {NT}], c = dc[e / {NT}];
                    if (r < nb) {{
                        float pv = sD[c * 33 + c];
                        if (!(pv > 0.0f)) {{ if (r == c) atomicAdd(&status.data[0], 1); pv = 1.0e-30f; }}
                        const float dg = sqrtf(pv);
                        const float v = r == c ? dg : sD[r * 33 + c] / dg;
                        L[co[c0 + k0 + c] + r - c] = v;
                    }}
                }}
                __syncthreads();
                for (int e = tid; e < 528; e += {NT}) {{
                    const int r = dr[e / {NT}], c = dc[e / {NT}];
                    if (r < nb) sD[r * 33 + c] = L[co[c0 + k0 + c] + r - c];
                    if (r == c && r < nb) sInv[r] = 1.0f / L[co[c0 + k0 + c]];
                }}
                __syncthreads();
#if DS_TIMING
                if (tid == 0 && world == 0 && w >= 64) {{ unsigned long long tq; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tq)); timing.data[110 + 3 * (k0 / 32) + 0] = (long long)tq; }}
#endif
                // Panel rows below the diagonal block: x L_kk^T = row, one row per thread in registers.
                for (int u = k0 + nb + tid; u < q; u += {NT}) {{
                    float xr[32];
                    #pragma unroll
                    for (int j = 0; j < 32; j++) xr[j] = j < nb ? L[co[c0 + k0 + j] + u - (k0 + j)] : 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {{
                        if (j < nb) {{
                            float a0 = xr[j], a1 = 0.0f;
                            #pragma unroll
                            for (int l = 0; l < j; l += 2) {{
                                a0 -= xr[l] * sD[j * 33 + l];
                                if (l + 1 < j) a1 -= xr[l + 1] * sD[j * 33 + l + 1];
                            }}
                            xr[j] = (a0 + a1) * sInv[j];
                        }}
                    }}
                    #pragma unroll
                    for (int j = 0; j < 32; j++) if (j < nb) L[co[c0 + k0 + j] + u - (k0 + j)] = xr[j];
                }}
                __syncthreads();
#if DS_TIMING
                if (tid == 0 && world == 0 && w >= 64) {{ unsigned long long tq; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tq)); timing.data[110 + 3 * (k0 / 32) + 1] = (long long)tq; }}
#endif
                // Trailing update of the supernode's remaining columns: 4 x 4 register tiles.
                {{
                    const int cb0 = k0 + nb;
                    const int nct = (w - cb0 + 3) >> 2, nrt = (q - cb0 + 3) >> 2;
                    for (int idx = tid; idx < nct * nrt; idx += {NT}) {{
                        const int ct = idx / nrt, rt = idx - ct * nrt;
                        if (rt < ct) continue;
                        const int tcb = cb0 + 4 * ct, ub = cb0 + 4 * rt;
                        int tci[4], ui[4];
                        #pragma unroll
                        for (int z = 0; z < 4; z++) {{ tci[z] = min(tcb + z, w - 1); ui[z] = min(ub + z, q - 1); }}
                        float acc[4][4];
                        #pragma unroll
                        for (int a = 0; a < 4; a++)
                            #pragma unroll
                            for (int b = 0; b < 4; b++) acc[a][b] = 0.0f;
                        for (int c = 0; c < nb; c++) {{
                            const int oc = co[c0 + k0 + c] - (k0 + c);
                            float vu[4], vt[4];
                            #pragma unroll
                            for (int z = 0; z < 4; z++) {{ vu[z] = L[oc + ui[z]]; vt[z] = L[oc + tci[z]]; }}
                            #pragma unroll
                            for (int a = 0; a < 4; a++)
                                #pragma unroll
                                for (int b = 0; b < 4; b++) acc[a][b] += vu[a] * vt[b];
                        }}
                        #pragma unroll
                        for (int b = 0; b < 4; b++) {{
                            const int tc = tcb + b;
                            if (tc >= w) continue;
                            const int ob = co[c0 + tc] - tc;
                            #pragma unroll
                            for (int a = 0; a < 4; a++) {{
                                const int u = ub + a;
                                if (u < q && u >= tc) L[ob + u] -= acc[a][b];
                            }}
                        }}
                    }}
                }}
                __syncthreads();
#if DS_TIMING
                if (tid == 0 && world == 0 && w >= 64) {{ unsigned long long tq; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tq)); timing.data[110 + 3 * (k0 / 32) + 2] = (long long)tq; }}
#endif
            }}
        }}
        __syncthreads();
#if DS_TIMING
        if (tid == 0 && world == 0) {{ unsigned long long tq; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tq)); timing.data[5 + 2 * lev] = (long long)tq; }}
#endif
    }}
#if DS_TIMING
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t2));
#endif

    // 4. Solves: the first gives the regularized solution, each further one is a primal refinement.
    for (int i = tid; i < {D}; i += {NT}) {{ x[i] = g[i]; sm[i] = 0.0f; }}
    __syncthreads();
    for (int it = 0; it < nsolve; it++) {{
        if (it > 0) {{
            for (int i = tid; i < {D}; i += {NT}) {{
                float s = g[i];
                const int c1 = h_ptr.data[i + 1];
                for (int c = h_ptr.data[i]; c < c1; c++) s += Hw[h_list.data[2 * c]] * x[h_list.data[2 * c + 1]];
                tv[i] = s;
            }}
            __syncthreads();
            for (int i = tid; i < {D}; i += {NT}) x[i] = tv[i];
            __syncthreads();
        }}
        trisolve();
        for (int i = tid; i < {D}; i += {NT}) sm[i] += x[i];
        __syncthreads();
    }}

    // 5. Velocity and impulses lam = -W (k r0 + J sum_k d_k).
    for (int i = tid; i < {D}; i += {NT}) {{
        const int gd = dofi[perm.data[i]];
        if (gd >= 0) v_out.data[gd] += x[i];
    }}
    for (int kl = warp; kl < {KL}; kl += {NW}) {{
        const int k = kbase + kl;
        const int slot = connect_slot.data[k];
        const bool valid = slot >= 0 && geom_closure.data[(size_t)k * {CF} + 7] != 0.0f;
        if (!valid) continue;
        const int rows = rows_local.data[kl];
        const int ns = sup_count.data[kl];
        const float* Jk = Jb + (size_t)kl * {MAXR * KJ};
        float sv[{SL}];
        #pragma unroll
        for (int sl = 0; sl < {SL}; sl++) {{
            const int a = lane + 32 * sl;
            sv[sl] = a < ns ? sm[iperm.data[sup_dof.data[kl * {KJ} + a]]] : 0.0f;
        }}
        for (int r = 0; r < rows; r++) {{
            float dot = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < {SL}; sl++) {{
                const int a = lane + 32 * sl;
                if (a < ns) dot += Jk[r * {KJ} + a] * sv[sl];
            }}
            {wsum("dot")}
            if (lane == 0)
                impulses.data[wrow + slot + r] = -SW[kl * {MAXR} + r] * ((float)nsolve * Q[kl * {MAXR} + r] + dot);
        }}
    }}
#if DS_TIMING
    unsigned long long tc;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tc));
#endif
{contact_post}
#if DS_TIMING
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t3));
    if (tid == 0 && world > 0) {{
        timing.data[world * 4 + 0] = (long long)(t1 - t0);
        timing.data[world * 4 + 1] = (long long)(t2 - t1);
        timing.data[world * 4 + 2] = (long long)(t3 - t2);
        timing.data[world * 4 + 3] = (long long)(t3 - tc);  // contact responses
    }}
    if (tid == 0 && world == 0) {{
        timing.data[0] = (long long)(t1 - t0);
        timing.data[1] = (long long)(t2 - t1);
        timing.data[2] = (long long)(t3 - t2);
        timing.data[3] = (long long)t1;
    }}
#endif
#endif
"""

    @wp.func_native(snippet)
    def ds_solve_native(
        world: int,
        world_closure_start: wp.array[int],
        connect_slot: wp.array[int],
        geom_closure: wp.array2d[float],
        rows_local: wp.array[int],
        sup_count: wp.array[int],
        sup_dof: wp.array[int],
        world_dof_indices: wp.array2d[int],
        Ha: wp.array3d[float],
        rhs: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        qd0: wp.array[float],
        eps: float,
        bias_mode: int,
        h_dt: float,
        spring_k: float,
        spring_b: float,
        iterations: int,
        perm: wp.array[int],
        iperm: wp.array[int],
        col_off: wp.array[int],
        rowidx: wp.array[int],
        sn_info: wp.array[int],
        lvl_sn: wp.array[int],
        lvl_start: wp.array[int],
        pairs: wp.array[int],
        rel: wp.array[int],
        row_ptr: wp.array[int],
        row_ent: wp.array[int],
        row_col: wp.array[int],
        hidx: wp.array[int],
        asm_ptr: wp.array[int],
        asm_list: wp.array[int],
        asm_rec: wp.array[int],
        g_ptr: wp.array[int],
        g_list: wp.array[int],
        h_ptr: wp.array[int],
        h_list: wp.array[int],
        ct_ptr: wp.array[int],
        ct_rec: wp.array[int],
        row_split: wp.array[int],
        sc_ptr: wp.array[int],
        sc_list: wp.array[int],
        sc_rec: wp.array[int],
        Jw: wp.array2d[float],
        qrow: wp.array2d[float],
        swrow: wp.array2d[float],
        Lg: wp.array2d[float],
        status: wp.array[int],
        timing: wp.array[wp.int64],
        constraint_count: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        Jg: wp.array3d[float],
        Yg: wp.array3d[float],
        cbase: wp.array[int],
        chead: wp.array2d[int],
        cdiag: wp.array2d[float],
        ancestors: wp.array[int],
        na_arr: wp.array[int],
        v_out: wp.array[float],
        impulses: wp.array2d[float],
    ): ...

    def ds_solve(
        world_closure_start: wp.array[int],
        connect_slot: wp.array[int],
        geom_closure: wp.array2d[float],
        rows_local: wp.array[int],
        sup_count: wp.array[int],
        sup_dof: wp.array[int],
        world_dof_indices: wp.array2d[int],
        Ha: wp.array3d[float],
        rhs: wp.array2d[float],
        phi: wp.array2d[float],
        target_velocity: wp.array2d[float],
        qd0: wp.array[float],
        eps: float,
        bias_mode: int,
        h_dt: float,
        spring_k: float,
        spring_b: float,
        iterations: int,
        perm: wp.array[int],
        iperm: wp.array[int],
        col_off: wp.array[int],
        rowidx: wp.array[int],
        sn_info: wp.array[int],
        lvl_sn: wp.array[int],
        lvl_start: wp.array[int],
        pairs: wp.array[int],
        rel: wp.array[int],
        row_ptr: wp.array[int],
        row_ent: wp.array[int],
        row_col: wp.array[int],
        hidx: wp.array[int],
        asm_ptr: wp.array[int],
        asm_list: wp.array[int],
        asm_rec: wp.array[int],
        g_ptr: wp.array[int],
        g_list: wp.array[int],
        h_ptr: wp.array[int],
        h_list: wp.array[int],
        ct_ptr: wp.array[int],
        ct_rec: wp.array[int],
        row_split: wp.array[int],
        sc_ptr: wp.array[int],
        sc_list: wp.array[int],
        sc_rec: wp.array[int],
        Jw: wp.array2d[float],
        qrow: wp.array2d[float],
        swrow: wp.array2d[float],
        Lg: wp.array2d[float],
        status: wp.array[int],
        timing: wp.array[wp.int64],
        constraint_count: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        row_mu: wp.array2d[float],
        Jg: wp.array3d[float],
        Yg: wp.array3d[float],
        cbase: wp.array[int],
        chead: wp.array2d[int],
        cdiag: wp.array2d[float],
        ancestors: wp.array[int],
        na_arr: wp.array[int],
        v_out: wp.array[float],
        impulses: wp.array2d[float],
    ):
        world, _lane = wp.tid()
        ds_solve_native(
            world,
            world_closure_start,
            connect_slot,
            geom_closure,
            rows_local,
            sup_count,
            sup_dof,
            world_dof_indices,
            Ha,
            rhs,
            phi,
            target_velocity,
            qd0,
            eps,
            bias_mode,
            h_dt,
            spring_k,
            spring_b,
            iterations,
            perm,
            iperm,
            col_off,
            rowidx,
            sn_info,
            lvl_sn,
            lvl_start,
            pairs,
            rel,
            row_ptr,
            row_ent,
            row_col,
            hidx,
            asm_ptr,
            asm_list,
            asm_rec,
            g_ptr,
            g_list,
            h_ptr,
            h_list,
            ct_ptr,
            ct_rec,
            row_split,
            sc_ptr,
            sc_list,
            sc_rec,
            Jw,
            qrow,
            swrow,
            Lg,
            status,
            timing,
            constraint_count,
            row_type,
            row_parent,
            row_mu,
            Jg,
            Yg,
            cbase,
            chead,
            cdiag,
            ancestors,
            na_arr,
            v_out,
            impulses,
        )

    name = (
        f"closure_ds_solve_{D}_{NNZ}_{KL}_{KJ}_{MAXR}_{'s' if USE_SMEM else 'g'}_{NT}_{BC}_c{CMODE}w{YW}"
        + (f"_cc{C}_{KS}_{SIDES}_r{YR}" if CMODE == 1 else "")
        + ("_t" if TIMING else "")
    )
    ds_solve.__name__ = name
    ds_solve.__qualname__ = name
    # One block per SM (shared memory bound), so the block may use the whole register file.
    k = wp.kernel(
        enable_backward=False,
        module="unique",
        module_options=WARP_DETERMINISM_OFF,
        cuda_max_registers=min(255, 65536 // NT),
    )(ds_solve)
    return k


def solve_smem_bytes(buf) -> int:
    return buf.smem_bytes


def launch_solve(solver, rhs: wp.array, iterations: int, dt: float) -> None:
    """Rows, assembly, factor, solves, velocity and impulses: one block per world."""
    buf = solver._closure_direct_sparse
    device = solver.model.device
    tau = solver._closure_direct_timeconst
    zeta = solver._closure_direct_dampratio
    spring_k = 1.0 / (tau * tau * zeta * zeta)
    spring_b = 2.0 / tau
    qd0 = solver._closure_direct_qd0 if solver._closure_direct_qd0 is not None else solver._v_out
    kernel = _set_dynamic_smem(
        get_ds_solve_kernel(
            buf.D,
            buf.NNZ,
            buf.NLEV,
            buf.KL,
            buf.KJ,
            buf.MAXR,
            buf.NA,
            buf.M_D,
            buf.smem_L,
            buf.threads,
            buf.QW,
            buf.TILES,
            buf.BC,
            buf.wb_in_vec,
            buf.timing_enabled,
            buf.cmode,
            buf.yw,
            buf.yoff,
            buf.C,
            buf.KS,
            buf.KZ,
            buf.sides,
            buf.yr,
        ),
        buf.smem_bytes,
    )
    wp.launch_tiled(
        kernel,
        dim=[buf.W],
        inputs=[
            buf.world_closure_start,
            solver._connect_slot,
            buf.geom_closure,
            buf.rows_local,
            buf.sup_count,
            buf.sup_dof,
            solver._world_dof_indices,
            buf.Ha,
            rhs,
            solver._phi,
            solver._target_velocity,
            qd0,
            float(solver._closure_direct_eps),
            int(solver._closure_direct_bias == "implicit"),
            float(dt),
            float(spring_k),
            float(spring_b),
            int(iterations),
            buf.t_perm,
            buf.t_iperm,
            buf.t_col_off,
            buf.t_rowidx,
            buf.t_sn,
            buf.t_lvl_sn,
            buf.t_lvl_start,
            buf.t_pairs,
            buf.t_rel,
            buf.t_row_ptr,
            buf.t_row_ent,
            buf.t_row_col,
            buf.t_hidx,
            buf.t_asm_ptr,
            buf.t_asm_list,
            buf.t_asm_rec,
            buf.t_g_ptr,
            buf.t_g_list,
            buf.t_h_ptr,
            buf.t_h_list,
            buf.t_ct_ptr,
            buf.t_ct_rec,
            buf.t_row_split,
            buf.t_sc_ptr,
            buf.t_sc_list,
            buf.t_sc_rec,
            buf.Jw,
            buf.qrow,
            buf.swrow,
            buf.Lg,
            buf.status,
            buf.timing,
            solver._constraint_count,
            solver._row_type,
            solver._row_parent,
            solver._row_mu,
            buf.cJ,
            buf.cY,
            buf.cbase,
            buf.chead,
            solver._diag,
            buf.ancestors_flat,
            buf.na,
        ],
        outputs=[solver._v_out, solver._impulses],
        block_dim=buf.threads,
        device=device,
    )
