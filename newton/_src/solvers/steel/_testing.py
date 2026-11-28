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
    ``sparse_mass_matrix``, ``diagonal_mass``, ``propagation_tree_kernel``, ``contact_unit_sweep`` (``False``
    keeps rod worlds on the fused row sweep), ``rod_cholesky`` (``"panels"`` factors rod articulations with the
    double-precision panels), ``split_contact_jacobian`` (``False`` builds rod contact Jacobians one thread per
    contact), ``propagation_refine_paths`` (``False`` refines same-articulation propagation rows one
    articulation per thread), ``propagation_split_same_articulation_rows`` (``False`` steps those rows with
    their exact, unsplit response), ``tree_dynamics`` (``False`` keeps rod trees on the dense unconstrained
    velocity), ``rod_tree_kernel`` and ``rod_tree_factor``/``rod_tree_response``/``rod_tree_propagate``
    (``False`` keeps rod trees on the generic multi-DOF tree kernels), ``rod_serial_immediate`` (``False``
    colors the immediate response of rod worlds in ``solve_mode="colored"``), ``propagation_serial_units``
    (``False`` keeps the serial propagation sweep on its row-at-a-time kernel) and the ``colored_*`` sweep
    variants.
    """
    token = _KERNEL_OVERRIDES.set(dict(overrides))
    try:
        yield
    finally:
        _KERNEL_OVERRIDES.reset(token)


def current_kernel_overrides() -> dict[str, object]:
    """Return a copy of the overrides active in the current context."""
    return dict(_KERNEL_OVERRIDES.get() or {})
