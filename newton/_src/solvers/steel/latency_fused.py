# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Latency mode of the colored propagation solve: the PGS iteration loop as one launch.

Each iteration of ``SolverSteel._launch_propagation_solve`` launches about a dozen kernels: row phases 3
and 4, the tree and free-body twist refreshes, the cache snapshot, the colored sweep, the cached-response
GEMV, the tree impulse propagation, the free-body flush and row phase 5. With few worlds each launch holds
a handful of warps, so a substep is a long chain of small launches. Every pass reads and writes only its
own world, so one block per world runs the whole loop in a single launch.

The block's warps take fixed roles and call the device functions of the per-pass kernels (each factory
exposes ``kernel.native``):

* warp 0: row phases 3, 4 and 5, the cache snapshot and the GEMV;
* warp 1 + i: tree group ``i``, refresh and propagation of the world's articulations of that group, one
  after another;
* the next warp, with rod trees: the chain-shaped rod trees' refresh and propagation (``rod_tree``'s warp scans),
  one chain after another;
* the last warp: the free-rigid twist refresh and flush;
* every warp: the momentum update and the colored sweep, where a color wider than one warp spreads over the block.

Phased and per-pass sweeps use the same momentum update in a one-warp launch.
Block barriers order the stages as the launch chain does. The device functions read the data written in
the loop through plain loads after a barrier, so the results match the chain bit for bit.

The structs fixed at setup (cache, trees, free bodies) reach the kernel through one-element device arrays,
which keeps its parameters under 4 KB: some drivers crash capturing a launch with more into a CUDA graph.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp

from .kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
    WARP_DETERMINISM_OFF,
    _response_apply,
    contact_response,
    contact_unit_row,
    momentum_beta,
    momentum_step_native,
    next_momentum_age,
    propagation_momentum_apply_row,
    propagation_row_unit,
)

# Tree groups one fused kernel can carry, one warp each.
MAX_TREE_SLOTS = 4

# Fields in the order of ``SolverSteel._mf_gs_inputs``, which fills them.
ROW_FIELDS = (
    "constraint_count",
    "world_dof_indices",
    "rhs",
    "diag",
    "impulses",
    "J_world",
    "Y_world",
    "row_type",
    "row_parent",
    "row_mu",
    "drive_target_vel_bias",
    "drive_vel_multiplier",
    "drive_impulse_multiplier",
    "drive_max_impulse",
    "drive_vel_limit",
    "velocity_limit_slot",
    "velocity_limit_sign",
    "mf_constraint_count",
    "mf_contact_rows_end",
    "mf_meta",
    "mf_impulses",
    "mf_J_a",
    "mf_J_b",
    "mf_MiJt_a",
    "mf_MiJt_b",
    "mf_row_mu",
    "row_w",
    "mf_row_w",
    "torsion_group",
    "torsion_row_radius",
    "dense_gamma",
    "mf_gamma",
    "momentum_scratch",
    "split_previous_velocity",
)

# Fields in the order of ``SolverSteel._colored_sweep_inputs``, which fills them.
COLORED_FIELDS = (
    "constraint_count",
    "body_a",
    "body_b",
    "MiJt_a",
    "MiJt_b",
    "J_a",
    "J_b",
    "eff_mass_inv",
    "rhs",
    "row_w",
    "row_type",
    "row_parent",
    "row_mu",
    "torsion_first",
    "torsion_next",
    "torsion_row_radius",
    "compliance_gamma",
    "row_order",
    "color_offsets",
    "body_list",
    "body_count",
    "body_local_slot",
    "unit_meta",
    "unit_friction_setup",
    "unit_payload",
    "ring_overflow",
    "contact_units",
    "contact_responses",
    "body_response_split",
)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _block_sync(): ...


@wp.struct
class RowArgs:
    constraint_count: wp.array[int]
    world_dof_indices: wp.array2d[int]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    impulses: wp.array2d[float]
    J_world: wp.array3d[float]
    Y_world: wp.array3d[float]
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    row_mu: wp.array2d[float]
    drive_target_vel_bias: wp.array2d[float]
    drive_vel_multiplier: wp.array2d[float]
    drive_impulse_multiplier: wp.array2d[float]
    drive_max_impulse: wp.array2d[float]
    drive_vel_limit: wp.array2d[float]
    velocity_limit_slot: wp.array[int]
    velocity_limit_sign: wp.array[float]
    mf_constraint_count: wp.array[int]
    mf_contact_rows_end: wp.array[int]
    mf_meta: wp.array2d[int]
    mf_impulses: wp.array2d[float]
    mf_J_a: wp.array3d[float]
    mf_J_b: wp.array3d[float]
    mf_MiJt_a: wp.array3d[float]
    mf_MiJt_b: wp.array3d[float]
    mf_row_mu: wp.array2d[float]
    row_w: wp.array2d[float]
    mf_row_w: wp.array2d[float]
    torsion_group: wp.array2d[int]
    torsion_row_radius: wp.array2d[float]
    dense_gamma: wp.array2d[float]
    mf_gamma: wp.array2d[float]
    momentum_scratch: wp.array2d[float]
    split_previous_velocity: wp.array[float]


@wp.struct
class ColoredArgs:
    constraint_count: wp.array[int]
    body_a: wp.array2d[int]
    body_b: wp.array2d[int]
    MiJt_a: wp.array3d[float]
    MiJt_b: wp.array3d[float]
    J_a: wp.array3d[float]
    J_b: wp.array3d[float]
    eff_mass_inv: wp.array2d[float]
    rhs: wp.array2d[float]
    row_w: wp.array2d[float]
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    row_mu: wp.array2d[float]
    torsion_first: wp.array2d[int]
    torsion_next: wp.array2d[int]
    torsion_row_radius: wp.array2d[float]
    compliance_gamma: wp.array2d[float]
    row_order: wp.array[int]
    color_offsets: wp.array[int]
    body_list: wp.array2d[int]
    body_count: wp.array[int]
    body_local_slot: wp.array[int]
    unit_meta: wp.array[int]
    unit_friction_setup: wp.array[float]
    unit_payload: wp.array[float]
    ring_overflow: wp.array[int]
    contact_units: wp.array[float]
    contact_responses: wp.array[float]
    body_response_split: wp.array3d[float]


@wp.struct
class CacheArgs:
    world_flag: wp.array[int]
    body_count: wp.array[int]
    art_eligible: wp.array[int]
    R: wp.array4d[float]
    B: wp.array4d[float]
    qd_base: wp.array3d[float]


@wp.struct
class TreeArgs:
    """Arrays every tree group's refresh and propagation read."""

    art_to_world: wp.array[int]
    dense_contact_world_flag: wp.array[int]
    # Tree joint range of each articulation, interleaved [first, end) pairs.
    articulation_joint_range: wp.array[int]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_parent_slot: wp.array[int]
    # Depth-level joint order, read by the level-walk refresh variants.
    level_joint: wp.array[int]
    level_offsets: wp.array[int]
    level_art_start: wp.array[int]
    S_flat: wp.array2d[float]
    com_rel: wp.array2d[float]
    U: wp.array2d[float]
    D_inv: wp.array3d[float]


@wp.struct
class TreeSlot:
    """One tree group: its articulations, its propagation guard and its articulations per world (CSR)."""

    group_to_art: wp.array[int]
    guard_flag: wp.array[int]
    world_offsets: wp.array[int]
    world_groups: wp.array[int]


@wp.struct
class MomentumRows:
    """Impulses and row properties for one family in the world sweep."""

    count: wp.array[int]
    impulses: wp.array2d[float]
    kind: wp.array2d[int]
    parent: wp.array2d[int]


@wp.struct
class SplitSweep:
    """Sweep history and the native row and velocity arrays it extrapolates."""

    dense: MomentumRows
    free: MomentumRows
    prop: MomentumRows
    prop_order: wp.array2d[int]
    dofs: wp.array2d[int]
    velocity: wp.array[float]
    bodies: wp.array[int]
    body_start: wp.array[int]
    twist: wp.array2d[float]
    prev_twist: wp.array2d[float]
    prev_velocity: wp.array[float]
    history: wp.array2d[wp.vec2]
    age: wp.array[int]
    anchor_sq: wp.array[float]
    # Rod trees (``rods``): their links' twists follow the tree propagation, which installs the impulses pending on
    # each body, so the step adds each propagation row's move to its rod bodies' pending impulses (and its twist
    # estimate) instead of extrapolating their twists. ``rod_body`` flags the rod trees' bodies.
    rod_body: wp.array[int]
    prop_body_a: wp.array2d[int]
    prop_body_b: wp.array2d[int]
    prop_J_a: wp.array3d[float]
    prop_J_b: wp.array3d[float]
    prop_MiJt_a: wp.array3d[float]
    prop_MiJt_b: wp.array3d[float]
    row_contact: wp.array2d[int]
    contact_units: wp.array[float]
    contact_unit_response: wp.array[int]
    contact_responses: wp.array[float]
    response_split: wp.array3d[float]
    body_impulses: wp.array2d[float]


@wp.struct
class ExtraArgs:
    """The optional stages' arrays in one struct, so they cost the launch one parameter: the rod chains (their
    articulations by world and what ``rod_tree.chain_*_native`` read) and the Nesterov momentum."""

    split: wp.array[SplitSweep]  # SolverSteel(nesterov_momentum=True)'s sweep history
    body_to_articulation: wp.array[int]
    rod_group_to_art: wp.array[int]
    rod_world_offsets: wp.array[int]
    rod_world_groups: wp.array[int]
    joint_meta: wp.array[int]
    chain_flag: wp.array[int]
    cache_world_flag: wp.array[int]
    BR: wp.array2d[float]
    FR: wp.array2d[float]
    CR: wp.array[float]
    chain_off: wp.array[int]
    chain_start: wp.array[int]
    body_extra: wp.array2d[float]
    body_d: wp.array2d[float]
    body_o: wp.array2d[float]
    u: wp.array[float]
    n_active: wp.array[int]
    active_body: wp.array[int]
    active_P: wp.array[float]
    response: wp.array3d[float]
    active_qd0: wp.array[float]
    momentum_start: wp.array2d[float]
    momentum_prev: wp.array2d[float]
    momentum_t: wp.array[float]
    momentum_beta: wp.array[float]
    row_contact: wp.array2d[int]
    unit_response: wp.array[int]
    # Free-rigid bodies per world (CSR) and their first DOFs.
    free_world_offsets: wp.array[int]
    free_bodies: wp.array[int]
    free_dof_start: wp.array[int]
    # propagation_interval of the rod trees (1 without rods).
    every: int


@wp.func
def _free_refresh(world: int, lane: int, free: ExtraArgs, v_out: wp.array[float], body_qd: wp.array2d[float]):
    """``refresh_propagation_free_body_qd_from_vout`` for one world's free-rigid bodies."""
    for i in range(free.free_world_offsets[world] + lane, free.free_world_offsets[world + 1], 32):
        body = free.free_bodies[i]
        dof = free.free_dof_start[i]
        for r in range(6):
            body_qd[body, r] = v_out[dof + r]


@wp.func
def _free_flush(
    world: int,
    lane: int,
    free: ExtraArgs,
    body_qd: wp.array2d[float],
    body_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """``flush_propagation_free_body_qd_to_vout`` for one world's free-rigid bodies."""
    for i in range(free.free_world_offsets[world] + lane, free.free_world_offsets[world + 1], 32):
        body = free.free_bodies[i]
        dof = free.free_dof_start[i]
        for r in range(6):
            v_out[dof + r] = body_qd[body, r]
            body_impulses[body, r] = 0.0


@wp.func
def _cache_snapshot(
    world: int,
    lane: int,
    cache: CacheArgs,
    cache_max_bodies: int,
    body_list: wp.array2d[int],
    body_qd: wp.array2d[float],
):
    """``snapshot_propagation_cache_qd_base`` for one world."""
    if cache.world_flag[world] != 0:
        n = wp.min(cache.body_count[world], cache_max_bodies)
        for slot in range(lane, n, 32):
            body = body_list[world, slot]
            if body >= 0:
                for r in range(6):
                    cache.qd_base[world, slot, r] = body_qd[body, r]


def warp_count(tree_slots: int, rods: bool = False) -> int:
    """Warps per block: the row warp, one per tree group, the rod-chain warp (with rods) and the free-body warp."""
    return 2 + tree_slots + int(rods)


# Dot product, squared correction and lookahead, primal motion, squared angular correction and displacement.
_SweepProgress = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def _update_rows(
    world: int,
    lane: int,
    stride: int,
    rows: MomentumRows,
    offset: int,
    beta: float,
    has_velocity_limits: bool,
    h: SplitSweep,
):
    for i in range(lane, wp.min(rows.count[world], rows.impulses.shape[1]), stride):
        x = rows.impulses[world, i]
        e = float(0.0)
        if beta != 0.0 and (not has_velocity_limits or rows.kind[world, i] != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT):
            y = x + beta * (x - h.history[world, offset + i][0])
            rows.impulses[world, i] = y
            e = y - x
        h.history[world, offset + i] = wp.vec2(x, e)


@wp.func
def _rod_inject_add(h: SplitSweep, body: int, impulse: wp.spatial_vector, twist: wp.spatial_vector):
    if body >= 0:
        if h.rod_body[body] != 0:
            for k in range(6):
                wp.atomic_add(h.twist, body, k, twist[k])
                wp.atomic_add(h.body_impulses, body, k, impulse[k])


@wp.func
def _rod_inject_row(world: int, i: int, offset: int, h: SplitSweep):
    """Add row ``i``'s move (its history's second entry) to the pending impulses and twists of its rod bodies.

    The step moved the row's impulse by ``e``; a rod body's twist is ``v0 + R lambda`` installed by the tree
    propagation from its pending impulses, so adding ``J^T e`` there makes the propagation install exactly the
    extrapolated twist ``v + beta (v - v_prev)``. A contact unit's three rows move together on its first row.
    """
    rc = propagation_row_unit(h.row_contact, world, i)
    ba = h.prop_body_a[world, i]
    bb = h.prop_body_b[world, i]
    if rc >= 0:
        if (rc & 3) != 0:
            return
        p = rc >> 2
        slot = h.contact_unit_response[p]
        pa = wp.spatial_vector()
        pb = wp.spatial_vector()
        va = wp.spatial_vector()
        vb = wp.spatial_vector()
        for r in range(3):
            e = h.history[world, offset + i + r][1]
            pa += contact_unit_row(h.contact_units, p, r, 0) * e
            pb += contact_unit_row(h.contact_units, p, r, 1) * e
            if slot >= 0:
                va += contact_response(h.contact_responses, slot, r, 0) * e
                vb += contact_response(h.contact_responses, slot, r, 1) * e
        if slot < 0:
            if ba >= 0:
                va = _response_apply(h.response_split, ba, pa)
            if bb >= 0:
                vb = _response_apply(h.response_split, bb, pb)
        _rod_inject_add(h, ba, pa, va)
        _rod_inject_add(h, bb, pb, vb)
        return
    e = h.history[world, offset + i][1]
    if e == 0.0:
        return
    pa = wp.spatial_vector()
    pb = wp.spatial_vector()
    va = wp.spatial_vector()
    vb = wp.spatial_vector()
    for k in range(6):
        pa[k] = h.prop_J_a[world, i, k] * e
        va[k] = h.prop_MiJt_a[world, i, k] * e
        pb[k] = h.prop_J_b[world, i, k] * e
        vb[k] = h.prop_MiJt_b[world, i, k] * e
    _rod_inject_add(h, ba, pa, va)
    _rod_inject_add(h, bb, pb, vb)


@cache
def get_split_momentum(
    warps: int, propagation: bool, torsion: bool, ordered: bool, patch_anchors: bool, rods: bool = False
):
    """Use the owner block to advance the iterate before its first row phase."""
    stride = 32 * warps

    @wp.func(module="unique")
    def progress(
        world: int, lane: int, rows: MomentumRows, offset: int, propagation_rows: bool, initial: bool, h: SplitSweep
    ):
        sums = _SweepProgress()
        for row in range(lane, wp.min(rows.count[world], rows.impulses.shape[1]), 32):
            i = row
            if wp.static(ordered):
                if propagation_rows:
                    i = h.prop_order[world, row]
            if wp.static(patch_anchors):
                if initial:
                    if rows.kind[world, i] == PGS_CONSTRAINT_TYPE_FRICTION:
                        normal = rows.parent[world, i]
                        if normal >= 0 and rows.parent[world, normal] >= 0:
                            value = rows.impulses[world, i]
                            sums[0] += value * value
                    continue
            if not propagation_rows:
                if rows.kind[world, i] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                    continue
            previous = h.history[world, offset + i]
            move = rows.impulses[world, i] - previous[0]
            correction = move - previous[1]
            sums[0] += correction * previous[1]
            sums[1] += correction * correction
            sums[2] += previous[1] * previous[1]
            if wp.static(torsion):
                if rows.kind[world, i] == PGS_CONSTRAINT_TYPE_TORSION:
                    sums[4] += correction * correction
                    sums[5] += move * move
        return sums

    # Sum in the same row order in the fused and per-pass launches so their
    # coefficients match bit for bit. All warps update the impulses and velocities.
    metric_count = 6 if torsion else 4
    reduction_source = f"""
#if defined(__CUDA_ARCH__)
    if (threadIdx.x < 32) {{
        #pragma unroll
        for (int i = 0; i < {metric_count}; ++i)
            for (int d = 16; d; d >>= 1) x[i] += __shfl_xor_sync(0xffffffff, x[i], d);
    }}
    #if {warps} > 1
        __shared__ float result[{metric_count}];
        if (threadIdx.x < {metric_count}) result[threadIdx.x] = x[threadIdx.x];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < {metric_count}; ++i) x[i] = result[i];
        __syncthreads();
    #endif
#endif
    return x;
"""

    @wp.func_native(reduction_source)
    def sum_progress(x: _SweepProgress) -> _SweepProgress: ...

    @wp.func(module="unique")
    def advance(world: int, lane: int, iteration: int, previous_age: int, friction_transition: bool, h: SplitSweep):
        """Advance the complete iterate once, before its first row phase, including cached tree twists."""
        free_offset = h.dense.impulses.shape[1]
        prop_offset = free_offset + h.free.impulses.shape[1]
        age = int(0)
        beta = float(0.0)
        if iteration > 0 or wp.static(patch_anchors):
            sums = _SweepProgress()
            if lane < 32:
                initial = iteration == 0
                if initial or previous_age > 1:
                    sums = progress(world, lane, h.dense, 0, False, initial, h)
                    sums += progress(world, lane, h.free, free_offset, False, initial, h)
                    if wp.static(propagation):
                        sums += progress(world, lane, h.prop, prop_offset, True, initial, h)
                if iteration > 0:
                    for i in range(lane, h.dofs.shape[1], 32):
                        dof = h.dofs[world, i]
                        if dof >= 0 and h.velocity[dof] != h.prev_velocity[dof]:
                            sums[3] = 1.0
                            break
            sums = sum_progress(sums)
            if wp.static(patch_anchors):
                if iteration == 0 and lane == 0:
                    h.anchor_sq[world] = sums[0]
            if iteration > 0:
                anchor_sq = float(0.0)
                if wp.static(patch_anchors):
                    anchor_sq = h.anchor_sq[world]
                age = next_momentum_age(previous_age, sums[0], friction_transition or sums[3] == 0.0)
                beta = momentum_beta(age, sums[0], sums[1], sums[2], sums[4], sums[5], anchor_sq)
        _update_rows(world, lane, stride, h.dense, 0, beta, True, h)
        _update_rows(world, lane, stride, h.free, free_offset, beta, True, h)
        if wp.static(propagation):
            _update_rows(world, lane, stride, h.prop, prop_offset, beta, False, h)
            if wp.static(rods):
                # The injection reads other threads' history rows (a contact unit's three rows; every row in the
                # ordered walk), so every row's move must be recorded first. beta is uniform over the block.
                _block_sync()
                if beta != 0.0:
                    if wp.static(ordered):
                        # One thread in row order keeps run-to-run runs bitwise identical.
                        if lane == 0:
                            for i in range(wp.min(h.prop.count[world], h.prop.impulses.shape[1])):
                                _rod_inject_row(world, i, prop_offset, h)
                    else:
                        for i in range(lane, wp.min(h.prop.count[world], h.prop.impulses.shape[1]), stride):
                            _rod_inject_row(world, i, prop_offset, h)
        for i in range(lane, h.dofs.shape[1], stride):
            dof = h.dofs[world, i]
            if dof >= 0:
                v = h.velocity[dof]
                if beta != 0.0:
                    h.velocity[dof] = v + beta * (v - h.prev_velocity[dof])
                h.prev_velocity[dof] = v
        if wp.static(propagation):
            start = h.body_start[world]
            for i in range(lane, 6 * (h.body_start[world + 1] - start), stride):
                body = h.bodies[start + i // 6]
                skip = False
                if wp.static(rods):
                    # A rod body's move arrives through its pending impulses (above).
                    skip = h.rod_body[body] != 0
                if not skip:
                    axis = i % 6
                    v = h.twist[body, axis]
                    if beta != 0.0:
                        h.twist[body, axis] = v + beta * (v - h.prev_twist[body, axis])
                    h.prev_twist[body, axis] = v
        if wp.static(warps > 1):
            _block_sync()
        return age

    return advance


@cache
def get_advance_split_sweep(
    propagation: bool, torsion: bool, ordered: bool, patch_anchors: bool, rods: bool = False
) -> wp.Kernel:
    advance = get_split_momentum(1, propagation, torsion, ordered, patch_anchors, rods)

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


def get_fused_iteration_kernel(
    colored_native: Any,
    row_native: Any,
    gemv_native: Any | None,
    tree_natives: tuple[tuple[Any, Any, bool], ...],
    has_phase3: bool,
    has_phase5: bool,
    rod_natives: tuple[Any, Any] | None = None,
    rod_refresh_every: bool = True,
    momentum: bool = False,
    warps: int = 0,
    split_momentum: bool = False,
    split_propagation: bool = True,
    split_rods: bool = False,
    torsion: bool = False,
    ordered: bool = False,
    patch_anchors: bool = False,
) -> wp.Kernel:
    """Build the fused loop kernel for one solver configuration.

    ``colored_native`` is the block-wide sweep built for the launch's block size, ``tree_natives`` holds
    ``(propagate, refresh, refresh_takes_levels)`` per tree group in the order of the ``slots`` argument (a
    level-walk refresh also reads ``TreeArgs``' level order), and ``gemv_native`` is
    ``None`` without cached responses. Launch with ``dim=[world_count]`` and ``32 * warp_count(...)`` threads.
    """
    n_slots = len(tree_natives)
    # Without generalized or free-body rows (every articulation a rod tree) the row phases are compiled out.
    has_rows = row_native is not None
    has_rods = rod_natives is not None
    # The rod chains' warp (one warp walks its world's chains, one after another) and the free bodies' after it.
    rod_warp = 1 + n_slots
    free_warp = 1 + n_slots + int(has_rods)
    rod_propagate = rod_natives[0] if has_rods else None
    rod_refresh = rod_natives[1] if has_rods else None
    block_threads = 32 * (2 + n_slots + int(has_rods))
    momentum_step = momentum_step_native(block_threads) if momentum else None
    # SolverSteel(nesterov_momentum=True): the split sweep's own Nesterov, for solves without rod-tree momentum.
    advance_momentum = (
        get_split_momentum(warps, split_propagation, torsion, ordered, patch_anchors, split_rods)
        if split_momentum
        else None
    )

    @wp.func(module="unique")
    def row_phase_impl(
        world: int,
        phase: int,
        rows: RowArgs,
        omega: float,
        regularize: int,
        freeze_drive_rows: int,
        friction_start_iteration: int,
        iteration: int,
        v_out: wp.array[float],
    ):
        wp.static(row_native)(
            world, rows.constraint_count, rows.world_dof_indices, rows.rhs, rows.diag, rows.impulses,
            rows.J_world, rows.Y_world, rows.row_type, rows.row_parent, rows.row_mu, rows.drive_target_vel_bias,
            rows.drive_vel_multiplier, rows.drive_impulse_multiplier, rows.drive_max_impulse, rows.drive_vel_limit,
            rows.velocity_limit_slot, rows.velocity_limit_sign, rows.mf_constraint_count, rows.mf_contact_rows_end, rows.mf_meta, rows.mf_impulses, rows.mf_J_a,
            rows.mf_J_b, rows.mf_MiJt_a, rows.mf_MiJt_b, rows.mf_row_mu, rows.row_w, rows.mf_row_w,
            rows.torsion_group, rows.torsion_row_radius, rows.dense_gamma, rows.mf_gamma, rows.momentum_scratch, rows.split_previous_velocity, 1, omega, regularize,
            freeze_drive_rows, phase, friction_start_iteration, iteration, v_out,
        )  # fmt: skip

    @wp.func(module="unique")
    def row_phase(
        world: int,
        phase: int,
        rows: RowArgs,
        omega: float,
        regularize: int,
        freeze_drive_rows: int,
        friction_start_iteration: int,
        iteration: int,
        v_out: wp.array[float],
    ):
        if wp.static(has_rows):
            row_phase_impl(
                world, phase, rows, omega, regularize, freeze_drive_rows, friction_start_iteration, iteration, v_out
            )

    def steel_fused_propagation_loop(
        world_count: int,
        iterations: int,
        iteration_offset: int,
        friction_start_iteration: int,
        omega: float,
        regularize: int,
        freeze_drive_rows: int,
        force_refresh: int,
        cache_max_bodies: int,
        colored: ColoredArgs,
        rows: RowArgs,
        cache: wp.array[CacheArgs],
        tree: wp.array[TreeArgs],
        slots: wp.array[TreeSlot],
        extra: wp.array[ExtraArgs],
        articulation_active: wp.array[int],
        propagation_impulses: wp.array2d[float],
        body_qd: wp.array2d[float],
        body_impulses: wp.array2d[float],
        v_out: wp.array[float],
    ):
        world, t = wp.tid()
        warp_id = t // 32
        lane = t - warp_id * 32
        rows_n = wp.min(colored.constraint_count[world], propagation_impulses.shape[1])
        every = extra[0].every
        if wp.static(momentum):
            # The solve's start: the previous point is the warm start, and the sequence restarts.
            for r in range(t, rows_n, wp.static(block_threads)):
                extra[0].momentum_prev[world, r] = propagation_impulses[world, r]
            if t == 0:
                extra[0].momentum_t[world] = 1.0
            _block_sync()
        momentum_age = int(0)
        for i in range(iterations):
            g = iteration_offset + i
            if wp.static(split_momentum):
                momentum_age = wp.static(advance_momentum)(
                    world, t, i, momentum_age, g == friction_start_iteration, extra[0].split[0]
                )
            # propagation_interval: the rod trees propagate every ``every``-th sweep and after the last, and the
            # momentum steps with them (the sweeps' body estimates carry the impulses in between).
            last = i == iterations - 1
            propagates = every == 1 or last or i % every == every - 1
            # row phases 3 (internal rows) and 4 (dense and free-body contacts)
            if warp_id == 0:
                if wp.static(has_phase3):
                    row_phase(world, 3, rows, omega, regularize, freeze_drive_rows, friction_start_iteration, g, v_out)
                row_phase(world, 4, rows, omega, regularize, freeze_drive_rows, friction_start_iteration, g, v_out)
            _block_sync()
            # live twists from the generalized velocities
            for s in range(wp.static(n_slots)):
                if warp_id == 1 + s:
                    slot = slots[s]
                    for k in range(slot.world_offsets[world], slot.world_offsets[world + 1]):
                        if wp.static(tree_natives[s][2]):
                            wp.static(tree_natives[s][1])(
                                slot.world_groups[k], slot.group_to_art, articulation_active, tree[0].art_to_world,
                                tree[0].dense_contact_world_flag, force_refresh, tree[0].articulation_joint_range,
                                tree[0].joint_parent, tree[0].joint_child, tree[0].joint_qd_start,
                                tree[0].joint_parent_slot, tree[0].level_joint, tree[0].level_offsets,
                                tree[0].level_art_start, tree[0].S_flat, tree[0].com_rel, v_out, body_qd,
                            )  # fmt: skip
                        else:
                            wp.static(tree_natives[s][1])(
                                slot.world_groups[k], slot.group_to_art, articulation_active, tree[0].art_to_world,
                                tree[0].dense_contact_world_flag, force_refresh, tree[0].articulation_joint_range,
                                tree[0].joint_parent, tree[0].joint_child, tree[0].joint_qd_start,
                                tree[0].joint_parent_slot, tree[0].S_flat, tree[0].com_rel, v_out, body_qd,
                            )  # fmt: skip
            if wp.static(has_rods):
                # Rod trees stepped by their factorization keep their twists between sweeps: refresh once.
                if warp_id == wp.static(rod_warp) and (wp.static(rod_refresh_every) or i == 0):
                    for k in range(extra[0].rod_world_offsets[world], extra[0].rod_world_offsets[world + 1]):
                        wp.static(rod_refresh)(
                            extra[0].rod_world_groups[k], extra[0].joint_meta, extra[0].rod_group_to_art, extra[0].chain_flag,
                            articulation_active, tree[0].articulation_joint_range, extra[0].FR, extra[0].CR,
                            extra[0].chain_off, extra[0].chain_start, extra[0].body_o, v_out, body_qd,
                        )  # fmt: skip
            if warp_id == wp.static(free_warp):
                _free_refresh(world, lane, extra[0], v_out, body_qd)
            if wp.static(momentum):
                # The impulses a block of sweeps starts from: the momentum's restart test reads them.
                if every == 1 or i % every == 0:
                    for r in range(t, rows_n, wp.static(block_threads)):
                        extra[0].momentum_start[world, r] = propagation_impulses[world, r]
            _block_sync()
            # The snapshot overlaps the sweep's start: the sweep only stages the twists in shared memory before
            # a block barrier that warp 0 reaches after the snapshot.
            if wp.static(gemv_native is not None):
                if warp_id == 0:
                    _cache_snapshot(world, lane, cache[0], cache_max_bodies, colored.body_list, body_qd)
            wp.static(colored_native)(
                world, world_count, colored.constraint_count, colored.body_a, colored.body_b, colored.MiJt_a,
                colored.MiJt_b, colored.J_a, colored.J_b, colored.eff_mass_inv, colored.rhs, colored.row_w,
                colored.row_type, colored.row_parent, colored.row_mu, colored.torsion_first, colored.torsion_next,
                colored.torsion_row_radius, colored.compliance_gamma, colored.row_order, colored.color_offsets,
                colored.body_list, colored.body_count, colored.body_local_slot, colored.unit_meta,
                colored.unit_friction_setup, colored.unit_payload, colored.ring_overflow, colored.contact_units, colored.contact_responses,
                colored.body_response_split, omega, regularize,
                friction_start_iteration, g, propagation_impulses, body_qd, body_impulses,
            )  # fmt: skip
            _block_sync()
            if wp.static(momentum):
                if propagates and not last:
                    # The momentum's move joins the sweep's pending impulses, so the propagation installs both.
                    wp.static(momentum_step)(
                        world_count, colored.constraint_count, extra[0].momentum_start, propagation_impulses, extra[0].momentum_prev,
                        extra[0].momentum_t, extra[0].momentum_beta,
                    )  # fmt: skip
                    _block_sync()
                    beta = extra[0].momentum_beta[world]
                    for r in range(t, rows_n, wp.static(block_threads)):
                        propagation_momentum_apply_row(
                            world, r, beta, rows_n, colored.body_a, colored.body_b, colored.J_a, colored.J_b,
                            colored.MiJt_a, colored.MiJt_b, colored.row_type, colored.row_parent, colored.row_mu,
                            extra[0].row_contact, colored.contact_units,
                            extra[0].unit_response, colored.contact_responses, colored.body_response_split,
                            propagation_impulses, extra[0].momentum_prev, body_qd, body_impulses,
                        )  # fmt: skip
                    _block_sync()
            # The GEMV, the trees and the free bodies own disjoint DOFs, bodies and impulses.
            if wp.static(gemv_native is not None):
                if warp_id == 0:
                    wp.static(gemv_native)(
                        world, world_count, cache[0].world_flag, cache[0].body_count, colored.body_list,
                        cache[0].art_eligible, extra[0].body_to_articulation, tree[0].articulation_joint_range,
                        tree[0].joint_qd_start, cache[0].R, cache[0].B, cache[0].qd_base, body_impulses, body_qd,
                        v_out,
                    )  # fmt: skip
            for s in range(wp.static(n_slots)):
                if warp_id == 1 + s:
                    slot = slots[s]
                    for k in range(slot.world_offsets[world], slot.world_offsets[world + 1]):
                        wp.static(tree_natives[s][0])(
                            slot.world_groups[k], slot.group_to_art, articulation_active, tree[0].art_to_world,
                            slot.guard_flag, tree[0].articulation_joint_range, tree[0].joint_parent, tree[0].joint_child,
                            tree[0].joint_qd_start, tree[0].joint_parent_slot, tree[0].S_flat, tree[0].com_rel,
                            tree[0].U, tree[0].D_inv, body_impulses, body_qd, v_out,
                        )  # fmt: skip
            if wp.static(has_rods):
                if warp_id == wp.static(rod_warp) and propagates:
                    for k in range(extra[0].rod_world_offsets[world], extra[0].rod_world_offsets[world + 1]):
                        wp.static(rod_propagate)(
                            extra[0].rod_world_groups[k], extra[0].joint_meta, extra[0].rod_group_to_art, extra[0].chain_flag,
                            articulation_active, tree[0].art_to_world, extra[0].cache_world_flag,
                            tree[0].articulation_joint_range, extra[0].BR, extra[0].FR, extra[0].CR, extra[0].chain_off,
                            extra[0].chain_start, extra[0].body_extra, extra[0].body_d, extra[0].body_o, extra[0].u,
                            extra[0].n_active, 0, extra[0].active_body, extra[0].active_P, extra[0].response,
                            extra[0].active_qd0, body_impulses, body_qd, v_out,
                        )  # fmt: skip
            if warp_id == wp.static(free_warp):
                _free_flush(world, lane, extra[0], body_qd, body_impulses, v_out)
            _block_sync()
            # row phase 5 (velocity limits); warp 0 runs it before the next iteration's phases 3 and 4
            if wp.static(has_phase5):
                if warp_id == 0:
                    row_phase(world, 5, rows, omega, regularize, freeze_drive_rows, friction_start_iteration, g, v_out)
                # The next lookahead reads phase 5 across the whole block.
                if wp.static(split_momentum):
                    _block_sync()

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(
        steel_fused_propagation_loop
    )
