# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-parallel solve of ``L L^T qdd = tau`` for one large articulation per thread block.

:func:`~newton._src.solvers.steel.kernels.trisolve_loop` substitutes with one thread per articulation, which for a few
hundred DOFs (a cable mechanism of rod and revolute joints) is a chain of ``n^2`` dependent loads. Here a block keeps
the right-hand side in shared memory and, after each unknown, every thread updates the rows below it (forward) or above
it (backward), so the chain is ``2 n`` block steps. The forward pass subtracts in the order of ``trisolve_loop``; the
backward pass subtracts the later unknowns first.
"""

import functools

import warp as wp

from .friction_patches import WARP_DETERMINISM_OFF

wp.set_module_options(WARP_DETERMINISM_OFF)

MAX_DOFS = 4096
"""Largest articulation the block solve takes (its right-hand side in 16 KB of shared memory)."""

BLOCK_DIM = 256

_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    __shared__ float s[TS_MAXN];
    const int art = group_to_art.data[group];
    if (articulation_active.data[art] == 0) return;
    const int start = articulation_dof_start.data[art];
    const int n = n_dofs;
    const int t = threadIdx.x, nt = blockDim.x;
    const float* L = &L_group.data[(size_t)group * n * n];
    for (int i = t; i < n; i += nt) s[i] = joint_tau.data[start + i];
    __syncthreads();
    // Forward: L z = tau, one column at a time.
    for (int i = 0; i < n; ++i) {
        const float lii = L[(size_t)i * n + i];
        const float zi = lii != 0.0f ? s[i] / lii : 0.0f;
        __syncthreads();
        if (t == 0) s[i] = zi;
        for (int j = i + 1 + t; j < n; j += nt) s[j] -= L[(size_t)j * n + i] * zi;
        __syncthreads();
    }
    // Backward: L^T qdd = z, one row of L at a time.
    for (int i = n - 1; i >= 0; --i) {
        const float lii = L[(size_t)i * n + i];
        const float xi = lii != 0.0f ? s[i] / lii : 0.0f;
        __syncthreads();
        if (t == 0) s[i] = xi;
        for (int k = t; k < i; k += nt) s[k] -= L[(size_t)i * n + k] * xi;
        __syncthreads();
    }
    for (int i = t; i < n; i += nt) joint_qdd.data[start + i] = s[i];
#endif
"""


@functools.cache
def trisolve_block_kernel() -> wp.Kernel:
    """Kernel solving every articulation of one size group; launch tiled over the group, ``BLOCK_DIM`` threads."""
    source = _SOURCE.replace(
        "#if defined(__CUDA_ARCH__)\n", f"#if defined(__CUDA_ARCH__)\n#define TS_MAXN {MAX_DOFS}\n", 1
    )

    @wp.func_native(source)
    def trisolve_block_native(
        group: int,
        L_group: wp.array3d[float],
        group_to_art: wp.array[int],
        articulation_dof_start: wp.array[int],
        n_dofs: int,
        joint_tau: wp.array[float],
        articulation_active: wp.array[int],
        joint_qdd: wp.array[float],
    ): ...

    def trisolve_block(
        L_group: wp.array3d[float],
        group_to_art: wp.array[int],
        articulation_dof_start: wp.array[int],
        n_dofs: int,
        joint_tau: wp.array[float],
        articulation_active: wp.array[int],
        joint_qdd: wp.array[float],
    ):
        group, _thread = wp.tid()
        trisolve_block_native(
            group, L_group, group_to_art, articulation_dof_start, n_dofs, joint_tau, articulation_active, joint_qdd
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(trisolve_block)


_ROWS_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    __shared__ float s[TS_MAXN];
    const int idx = block / max_constraints;
    const int c = block - idx * max_constraints;
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    if (c >= world_constraint_count.data[world]) return;
    const int n = n_dofs;
    const int t = threadIdx.x, nt = blockDim.x;
    const float* L = &L_group.data[(size_t)idx * n * n];
    const size_t row = ((size_t)idx * max_constraints + c) * n;
    for (int i = t; i < n; i += nt) s[i] = J_group.data[row + i];
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        const float lii = L[(size_t)i * n + i];
        const float zi = lii != 0.0f ? s[i] / lii : 0.0f;
        __syncthreads();
        if (t == 0) s[i] = zi;
        for (int j = i + 1 + t; j < n; j += nt) s[j] -= L[(size_t)j * n + i] * zi;
        __syncthreads();
    }
    for (int i = n - 1; i >= 0; --i) {
        const float lii = L[(size_t)i * n + i];
        const float xi = lii != 0.0f ? s[i] / lii : 0.0f;
        __syncthreads();
        if (t == 0) s[i] = xi;
        for (int k = t; k < i; k += nt) s[k] -= L[(size_t)i * n + k] * xi;
        __syncthreads();
    }
    for (int i = t; i < n; i += nt) Y_group.data[row + i] = s[i];
    if (write_world != 0) {
        const int off = articulation_world_dof_offset.data[art];
        const int wd = J_world.shape[2];
        const size_t wrow = ((size_t)world * J_world.shape[1] + c) * wd + off;
        for (int i = t; i < n; i += nt) {
            J_world.data[wrow + i] = J_group.data[row + i];
            Y_world.data[wrow + i] = s[i];
        }
    }
#endif
"""


@functools.cache
def hinv_jt_block_kernel() -> wp.Kernel:
    """``Y = H^-1 J^T`` of each (articulation, row) of one size group in a block of its own; launch tiled over
    ``n_arts * max_constraints`` with ``BLOCK_DIM`` threads. Same arguments as ``hinv_jt_par_row``."""
    source = _ROWS_SOURCE.replace(
        "#if defined(__CUDA_ARCH__)\n", f"#if defined(__CUDA_ARCH__)\n#define TS_MAXN {MAX_DOFS}\n", 1
    )

    @wp.func_native(source)
    def hinv_jt_block_native(
        block: int,
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        n_dofs: int,
        max_constraints: int,
        n_arts: int,
        write_world: int,
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
    ): ...

    def hinv_jt_block(
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        n_dofs: int,
        max_constraints: int,
        n_arts: int,
        write_world: int,
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
    ):
        block, _thread = wp.tid()
        hinv_jt_block_native(
            block,
            L_group,
            J_group,
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            world_constraint_count,
            n_dofs,
            max_constraints,
            n_arts,
            write_world,
            Y_group,
            J_world,
            Y_world,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(hinv_jt_block)
