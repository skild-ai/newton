# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental rolling and spin friction rows for free-body contacts.

A shape opts in through the ``steel:condim`` custom attribute, borrowing MuJoCo's 3/4/6
convention: at 4 it contributes :attr:`~newton.Model.shape_material_mu_torsional` [m]
to a spin row about the contact normal, at 6 also :attr:`~newton.Model.shape_material_mu_rolling`
[m] to a rolling pair about the contact tangents. A contact takes the larger coefficient of its
two shapes. The rows follow the contact's sliding pair, ``[normal, t1, t2, roll1, roll2, spin]``,
and are bounded by their coefficient times the pooled normal load of the contact's patch: the
rolling pair on a disk, the spin row on a box. Only contacts on the free-body route (free bodies
against free bodies or static and kinematic geometry) receive them.
"""

import numpy as np
import warp as wp

from ...sim.collide import _shape_collide_mask
from .friction_patches import WARP_DETERMINISM_OFF
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
    PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
    accumulate_free_rigid_angular_wrenches_from_impulses,
    build_mf_angular_friction_rows,
    build_propagation_free_rigid_angular_rows,
    gate_free_rigid_angular_coefficients,
)

wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})


def configure_free_rigid_angular_friction(solver, model, torsional: float | None, rolling: float | None) -> None:
    """Decide from ``steel:condim`` and the solver-wide overrides whether free-body contacts get angular rows.

    Raises :class:`ValueError` for a condim other than 3, 4 or 6, an invalid gated coefficient or override, and
    :class:`NotImplementedError` for an option the angular rows do not support.
    """
    for name, value in (("free_rigid_torsional_friction", torsional), ("free_rigid_rolling_friction", rolling)):
        if value is not None and (not np.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"{name} must be finite and non-negative [m], or None")
    solver._free_rigid_torsional_friction = None if torsional is None else float(torsional)
    solver._free_rigid_rolling_friction = None if rolling is None else float(rolling)
    condim = solver._namespaced_attribute("condim")
    values = condim.numpy() if condim is not None and model.shape_count else None
    if values is not None:
        _check_condim_values(values)
    overridden = torsional is not None or rolling is not None
    # Kernel argument for a model without steel:condim; the gate kernel ignores it.
    solver._free_rigid_condim_placeholder = wp.zeros(1, dtype=wp.int32, device=model.device)
    routed = False
    solver._free_rigid_unrouted_condim = None
    if values is not None and solver._has_free_rigid_bodies:
        reachable, on_route_body, solve_world = _free_route_contact_shapes(solver)
        routed = bool(np.any(values[reachable] >= 4))
        unrouted = (values >= 4) & on_route_body & ~reachable
        if not (overridden or routed) and np.any(unrouted):
            # Pairs given only to a CollisionPipeline can still bring these shapes into contact; warn if one does.
            solver._free_rigid_unrouted_condim = wp.array(
                unrouted.astype(np.int32), dtype=wp.int32, device=model.device
            )
            route_body = _free_route_body_mask(solver)
            route_body[solver._model_plan.response_free_rigid_body_indices] = 2
            solver._free_rigid_unrouted_route_body = wp.array(route_body, dtype=wp.int32, device=model.device)
            solver._free_rigid_unrouted_solve_world = wp.array(solve_world, dtype=wp.int32, device=model.device)
            solver._free_rigid_unrouted_warned = wp.zeros(1, dtype=wp.int32, device=model.device)
    solver._free_rigid_angular_rows = bool(overridden or routed)
    if not solver._free_rigid_angular_rows:
        return
    if values is not None:
        _check_gated_coefficients(model, values)
    unsupported = [
        name
        for name, requested in (
            (f"row_schedule={solver._row_schedule!r}", solver._row_schedule != "interleaved"),
            (f"friction_mode={solver._friction_mode!r}", solver._friction_mode != "fixed_normal"),
            ("contact_compliance=True", solver._contact_compliance),
        )
        if requested
    ]
    if unsupported:
        source = "free_rigid_*_friction" if overridden else "steel:condim above 3"
        raise NotImplementedError(f"{source} does not support {', '.join(unsupported)}")


def refresh_free_rigid_angular_friction(solver) -> None:
    """Write each shape's gated torsional and rolling coefficients [m] on the device; capturable."""
    if not solver._free_rigid_angular_rows:
        return
    model = solver.model
    condim = solver._namespaced_attribute("condim")
    torsional, rolling = solver._free_rigid_torsional_friction, solver._free_rigid_rolling_friction
    wp.launch(
        gate_free_rigid_angular_coefficients,
        dim=model.shape_count,
        inputs=[
            condim if condim is not None else solver._free_rigid_condim_placeholder,
            int(condim is not None),
            model.shape_material_mu_torsional,
            model.shape_material_mu_rolling,
            -1.0 if torsional is None else torsional,
            -1.0 if rolling is None else rolling,
        ],
        outputs=[solver._free_rigid_mu_torsional, solver._free_rigid_mu_rolling],
        device=model.device,
    )


def validate_free_rigid_angular_friction(solver) -> None:
    """Read ``steel:condim`` and the gated coefficients on the host and raise :class:`ValueError` for invalid values."""
    condim = solver._namespaced_attribute("condim")
    if solver._free_rigid_angular_rows and condim is not None:
        values = condim.numpy()
        _check_condim_values(values)
        _check_gated_coefficients(solver.model, values)


def free_rigid_route_bodies(solver) -> wp.array | None:
    """Mark the bodies whose contacts can take the free-body route; ``None`` without angular rows.

    A contact takes that route when each side is a free rigid body or has no response DOFs, as the allocator decides.
    """
    if not solver._free_rigid_angular_rows:
        return None
    return wp.array(_free_route_body_mask(solver), dtype=wp.int32, device=solver.model.device)


def warn_unrouted_condim_contacts(solver, contacts) -> None:
    """Warn once from the device if a free-body contact touches a condim 4 or 6 shape the construction gate left out."""
    if solver._free_rigid_unrouted_condim is None or contacts is None or contacts.rigid_contact_max == 0:
        return
    wp.launch(
        _warn_unrouted_condim_contact,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            solver.model.shape_body,
            solver._free_rigid_unrouted_condim,
            solver._free_rigid_unrouted_route_body,
            solver._free_rigid_unrouted_solve_world,
        ],
        outputs=[solver._free_rigid_unrouted_warned],
        device=solver.model.device,
    )


def launch_free_rigid_angular_friction_rows(solver, state_aug, contacts, threads: int) -> None:
    """Write the rolling and spin rows that follow each free-body contact's sliding pair."""
    wp.launch(
        build_mf_angular_friction_rows,
        dim=threads,
        inputs=[
            contacts.rigid_contact_count,
            threads,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            solver._contact_world,
            solver._contact_slot,
            solver._contact_path,
            solver._contact_slots_needed,
            solver._contact_art_a,
            solver._contact_art_b,
            solver._articulation_response_dof_count,
            solver.model.shape_body,
            state_aug._body_v_s,
            solver._prescribed_articulation,
            int(solver._has_prescribed_response),
            solver._free_rigid_mu_torsional,
            solver._free_rigid_mu_rolling,
        ],
        outputs=[
            solver._mf_body_a,
            solver._mf_body_b,
            solver._mf_J_a,
            solver._mf_J_b,
            solver._mf_row_type,
            solver._mf_row_parent,
            solver._mf_row_mu,
            solver._mf_phi,
            solver._mf_target_velocity,
            solver._mf_row_rebound_velocity,
        ],
        device=solver.model.device,
    )


def launch_propagation_free_rigid_angular_friction_rows(solver, state_aug, contacts, threads: int) -> None:
    """Write the rolling and spin rows that follow each free-body propagation contact's sliding pair."""
    wp.launch(
        build_propagation_free_rigid_angular_rows,
        dim=threads,
        inputs=[
            contacts.rigid_contact_count,
            threads,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            solver._contact_world,
            solver._contact_slot,
            solver._contact_path,
            solver._contact_slots_needed,
            solver._contact_art_a,
            solver._contact_art_b,
            solver._articulation_response_dof_count,
            solver._is_free_rigid,
            solver.model.shape_body,
            state_aug._body_v_s,
            solver._prescribed_articulation,
            solver._free_rigid_mu_torsional,
            solver._free_rigid_mu_rolling,
        ],
        outputs=[
            solver._propagation_body_a,
            solver._propagation_body_b,
            solver._propagation_J_a,
            solver._propagation_J_b,
            solver._propagation_row_type,
            solver._propagation_row_parent,
            solver._propagation_row_mu,
            solver._propagation_phi,
            solver._propagation_target_velocity,
            solver._propagation_row_rebound_velocity,
        ],
        device=solver.model.device,
    )


def accumulate_free_rigid_angular_wrenches(solver, contacts, rigid_contact_max: int, inv_dt: float, wrench) -> None:
    """Add the solved rolling and spin torques of free-body contacts to the ``BODY_PARENT_F`` contact wrenches."""
    wp.launch(
        accumulate_free_rigid_angular_wrenches_from_impulses,
        dim=rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            rigid_contact_max,
            solver._contact_world,
            solver._contact_slot,
            solver._contact_path,
            solver._contact_slots_needed,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            solver.model.shape_body,
            solver._mf_constraint_count,
            solver._mf_row_type,
            solver._mf_impulses,
            solver._propagation_constraint_count,
            # Contacts take path 2 only under an active propagation response, which allocates its row types.
            solver._propagation_row_type if solver._propagation_active else solver._mf_row_type,
            solver._propagation_impulses,
            inv_dt,
        ],
        outputs=[wrench],
        device=solver.model.device,
    )


def free_rigid_angular_friction_gs_sources() -> tuple[str, str, str]:
    """Return the fused GS sweep's angular-row skip, bound and projection sources.

    The rolling pair is solved as one disk block at its first row, as the sliding pair is; the spin row is a box.
    Both take their bound from the coefficient times the pooled normal load at each visit.
    """
    rolling = int(PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
    spin = int(PGS_CONSTRAINT_TYPE_CONTACT_SPIN)
    angular = f"(mf_rt == {rolling} || mf_rt == {spin})"
    skip = f"""
            if ({angular} && global_iter < friction_start_iteration) {{
                s_lam_mf[i] = 0.0f;
                __syncwarp();
                continue;
            }}
            // The rolling pair is solved at its first row, three after the normal.
            if (mf_rt == {rolling} && i != mf_par + 3) continue;"""
    radius = f"""
            float angular_radius = 0.0f;
            if ({angular}) {{
                // The contact's sliding pair, visited just before in this sweep, pooled the same normals.
                float lambda_n = angular_load;
                if (mf_par != angular_load_parent) {{
                    lambda_n = s_lam_mf[mf_par];
                    for (int patch_row = (mf_meta.data[off_meta + mf_par * 4 + 3] >> 16);
                         patch_row >= 0 && patch_row != mf_par;
                         patch_row = (mf_meta.data[off_meta + patch_row * 4 + 3] >> 16))
                        lambda_n += s_lam_mf[patch_row];
                }}
                angular_radius = fmaxf(mf_row_mu.data[off_mf + i] * lambda_n, 0.0f);
                // A zero bound with no carried impulse cannot change the velocity.
                if (angular_radius == 0.0f && s_lam_mf[i] == 0.0f && (mf_rt != {rolling} || s_lam_mf[i + 1] == 0.0f))
                    continue;
            }}"""
    update = f""" else if (mf_rt == {spin}) {{
                new_impulse = fminf(fmaxf(new_impulse, -angular_radius), angular_radius);
            }} else if (mf_rt == {rolling}) {{
                int sib = i + 1;
                int sib_mf6 = mf6_base + sib * 6;
                float2 pair = make_float2(0.0f, 0.0f);
                if (angular_radius > 0.0f) {{
                    float sibling_residual = 0.0f;
                    float cross = 0.0f;
                    if (lane < 6 && dof_a >= 0) {{
                        sibling_residual = mf_J_a.data[sib_mf6 + lane] * s_v[dof_a + lane];
                        cross = cur_Ja * mf_MiJt_a.data[sib_mf6 + lane];
                    }}
                    if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                        sibling_residual = mf_J_b.data[sib_mf6 + lane - 6] * s_v[dof_b + lane - 6];
                        cross = cur_Jb * mf_MiJt_b.data[sib_mf6 + lane - 6];
                    }}
                    sibling_residual = warp_sum(sibling_residual) + __int_as_float(mf_meta.data[off_meta + sib * 4 + 2]);
                    cross = warp_sum(cross);
                    float inv_sib = __int_as_float(mf_meta.data[off_meta + sib * 4 + 1]);
                    pair = friction_pair_candidate(mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f, cross,
                        inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f, residual, sibling_residual, old_impulse,
                        s_lam_mf[sib], angular_radius, omega);
                }}
                float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                float scale = mag > angular_radius ? angular_radius / mag : 1.0f;
                new_impulse = pair.x * scale;
                float sib_delta = pair.y * scale - s_lam_mf[sib];
                __syncwarp();
                s_lam_mf[sib] = pair.y * scale;
                if (sib_delta != 0.0f) {{
                    iteration_changed = 1;
                    if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                    if (lane >= 6 && lane < 12 && dof_b >= 0)
                        s_v[dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                }}
            }}"""
    return skip, radius, update


def _check_condim_values(values: np.ndarray) -> None:
    """Reject a ``steel:condim`` other than the implemented 3, 4 and 6."""
    invalid = np.setdiff1d(values, (3, 4, 6))
    if invalid.size:
        raise ValueError(f"steel:condim must be 3, 4 or 6, got {int(invalid[0])}")


def _check_gated_coefficients(model, condim: np.ndarray) -> None:
    """Reject a negative or non-finite coefficient on a shape whose condim enables its row."""
    for name, array, gate in (
        ("shape_material_mu_torsional", model.shape_material_mu_torsional, 4),
        ("shape_material_mu_rolling", model.shape_material_mu_rolling, 6),
    ):
        values = array.numpy()[condim >= gate]
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must be finite and non-negative on shapes with steel:condim {gate} or above")


def _free_route_body_mask(solver) -> np.ndarray:
    """Return 1 for each body whose contacts can take the free-body route, padded to at least one entry."""
    model = solver.model
    route = np.zeros(max(model.body_count, 1), dtype=np.int32)
    if model.body_count and solver._has_free_rigid_bodies:
        body_art = solver._body_to_articulation.numpy()[: model.body_count]
        response = solver._articulation_response_dof_count.numpy()
        free = solver._is_free_rigid.numpy()
        safe = np.maximum(body_art, 0)
        route[: model.body_count] = (body_art < 0) | (response[safe] == 0) | (free[safe] != 0)
    return route


def _free_route_contact_shapes(solver) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mark the shapes that can meet a free-body-route partner across a responding free body.

    Returns that mask, the mask of shapes whose body takes the free-body route and each shape's solve world (-1
    for a body without response DOFs). Partners must share a world for
    collision and, like the solve, a solve world for the bodies with response DOFs; a global body solves in world 0.
    A user-assigned :attr:`~newton.Model.shape_contact_pairs` counts its pairs whatever their collision flags, groups
    and collision worlds. Collision groups other than 0 and filter pairs are ignored, so this may include shapes that
    never touch.
    """
    model = solver.model
    shape_body, shape_world = model.shape_body.numpy(), model.shape_world.numpy()
    body = np.maximum(shape_body, 0)
    on_route_body = (shape_body < 0) | (_free_route_body_mask(solver)[body] != 0)
    colliding = _shape_collide_mask(model) & (model.shape_collision_group.numpy() != 0)
    on_route = colliding & on_route_body
    responding = np.zeros(max(model.body_count, 1), dtype=bool)
    responding[solver._model_plan.response_free_rigid_body_indices] = True
    on_free_body = on_route_body & (shape_body >= 0) & responding[body]
    on_free = on_route & on_free_body
    # The solve world of a shape whose body has response DOFs, as the contact allocator assigns it; -1 otherwise.
    solve_world = np.full(len(shape_body), -1, dtype=np.int64)
    if model.body_count:
        body_art = solver._body_to_articulation.numpy()[body]
        art = np.maximum(body_art, 0)
        has_dofs = (shape_body >= 0) & (body_art >= 0) & (solver._articulation_response_dof_count.numpy()[art] > 0)
        solve_world = np.where(has_dofs, solver._art_to_world.numpy()[art], -1).astype(np.int64)
    # Collision slots: the global (-1) world in slot 0; a global shape meets every slot, others slot 0 and their own.
    slot = shape_world.astype(np.int64) + 1
    slots = model.world_count + 1
    is_global = shape_world < 0

    def partner_count(mask, world):
        """Count ``mask`` shapes in a compatible collision slot, in solve world ``world`` per shape, or any if None."""
        if world is None:
            counts = np.bincount(slot[mask], minlength=slots)
            return np.where(is_global, counts.sum(), counts[slot] + counts[0])
        keys = slot[mask] * (slots + 1) + solve_world[mask] + 1
        unique, counts = np.unique(keys, return_counts=True)

        def lookup(query):
            if not len(unique):
                return np.zeros(len(query), dtype=np.int64)
            index = np.minimum(np.searchsorted(unique, query), len(unique) - 1)
            return np.where(unique[index] == query, counts[index], 0)

        totals = np.bincount(solve_world[mask] + 1, minlength=slots + 1)
        local = lookup(slot * (slots + 1) + world + 1) + lookup(world + 1)
        return np.where(is_global, totals[np.clip(world + 1, 0, slots)], local)

    wildcard = on_route & (solve_world < 0)
    held = on_route & ~on_free & (solve_world >= 0)
    same_body = np.bincount(body[on_free], minlength=len(responding))[body]
    free_partners = (
        partner_count(wildcard, None)
        + partner_count(on_free, solve_world)
        + partner_count(held, solve_world)
        - same_body
    )
    other_partners = np.where(solve_world >= 0, partner_count(on_free, solve_world), partner_count(on_free, None))
    partners = np.where(on_free, free_partners, other_partners)
    reachable = on_route & (partners > 0)
    if model._shape_contact_pair_data is None and model._shape_contact_pairs is not None:
        pairs = model._shape_contact_pairs.numpy().reshape(-1, 2)
        pairs = pairs[(pairs >= 0).all(axis=1) & (pairs < model.shape_count).all(axis=1)]
        a, b = pairs[:, 0], pairs[:, 1]
        world_a, world_b = solve_world[a], solve_world[b]
        same_solve_world = (world_a < 0) | (world_b < 0) | (world_a == world_b)
        routed = on_route_body[a] & on_route_body[b] & (on_free_body[a] | on_free_body[b]) & same_solve_world
        reachable[a[routed]] = True
        reachable[b[routed]] = True
    return reachable, on_route_body, solve_world


@wp.kernel
def _warn_unrouted_condim_contact(
    contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    unrouted: wp.array[wp.int32],
    route_body: wp.array[wp.int32],
    solve_world: wp.array[wp.int32],
    warned: wp.array[wp.int32],
):
    """Print one warning for a free-body contact on a shape whose condim was gated off; ``route_body`` is 2 on free bodies."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    a = shape0[tid]
    b = shape1[tid]
    if a < 0 or b < 0 or (unrouted[a] == 0 and unrouted[b] == 0):
        return
    route_a = 1
    route_b = 1
    if shape_body[a] >= 0:
        route_a = route_body[shape_body[a]]
    if shape_body[b] >= 0:
        route_b = route_body[shape_body[b]]
    if route_a == 0 or route_b == 0 or (route_a != 2 and route_b != 2):
        return
    # The solve drops a contact between two solve worlds, so the gate is right to leave it out.
    if solve_world[a] >= 0 and solve_world[b] >= 0 and solve_world[a] != solve_world[b]:
        return
    shape = b
    if unrouted[a] != 0:
        shape = a
    if wp.atomic_exch(warned, 0, 1) == 0:
        wp.printf(
            "Warning: SolverSteel: a free-body contact touches shape %d, whose steel:condim above 3 adds no rows "
            "because no default contact pair reaches it. Assign explicit pairs to model.shape_contact_pairs "
            "before constructing the solver.\n",
            shape,
        )
