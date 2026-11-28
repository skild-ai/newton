# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test and tuning seam that pins the kernel and launch choices of SolverSteel."""

import contextlib
import contextvars
from collections.abc import Iterator, Mapping

_KERNEL_OVERRIDES: contextvars.ContextVar[Mapping[str, object] | None] = contextvars.ContextVar(
    "steel_kernel_overrides", default=None
)


@contextlib.contextmanager
def kernel_overrides(overrides: Mapping[str, object]) -> Iterator[None]:
    """Pin the kernel and launch choices of every solver constructed in this context.

    A solver reads the overrides once at construction and keeps them. Tasks and threads that
    inherit this context see the overrides; others do not. Keys: ``cholesky_kernel``, ``trisolve_kernel``,
    ``hinv_jt_kernel``, ``tile_threads``, ``serial_kernel_block_dim``,
    ``sparse_mass_matrix``, ``diagonal_mass``, ``propagation_tree_kernel``, ``mass_solve`` and the
    ``colored_*`` sweep variants. ``mass_solve="articulated_body"`` replaces the dense Cholesky
    factor, solve and ``H^-1 J^T`` of the dense articulation groups with articulated-body passes.

    ``closure_direct_factor="sparse"`` replaces the serial sweep over loop-closure rows by the
    experimental sparse direct solve (see ``closure_direct_sparse.py``), tuned by
    ``closure_direct_eps`` (row regularization, default ``1e-3``), ``closure_direct_bias``
    (``"baumgarte"`` or ``"implicit"`` with ``closure_direct_timeconst`` [s] and
    ``closure_direct_dampratio``), ``closure_direct_pivot_floor`` and ``closure_direct_pivot_shift``
    (defaults ``1e-4`` and ``1e-3``: a world whose factor meets a pivot at or below the floor times its
    diagonal entry of ``K``, which float32 cannot resolve, is factored again with the shift times
    ``diag(K)`` added, also in the refinements' proximal term; ``_closure_direct_sparse.status[0]``
    counts those refactors), ``closure_ds_threads`` (threads per world block),
    ``closure_ds_bc`` (update columns per pass) and ``closure_ds_timing`` (per-world phase times).
    The sparse direct solve is correct on either mass solve, but on large looped mechanisms the dense Cholesky
    mass solve dominates the step (the Hoberman flower at resolution 2, one world: 199 ms per frame against
    2.37 ms with ``mass_solve="articulated_body"``), so pair it with the articulated-body solve there. Past the
    device's shared memory for its tiled triangular solve (157 DOFs at 99 KiB per block), the Cholesky
    mass solve raises :class:`ValueError` at construction and suggests the articulated-body solve.
    """
    token = _KERNEL_OVERRIDES.set(dict(overrides))
    try:
        yield
    finally:
        _KERNEL_OVERRIDES.reset(token)


def current_kernel_overrides() -> dict[str, object]:
    """Return a copy of the overrides active in the current context."""
    return dict(_KERNEL_OVERRIDES.get() or {})
