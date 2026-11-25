# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Theta-weighted treatment of Steel's velocity-product bias (Coriolis, centrifugal and gyroscopic forces).

Steel evaluates the bias ``c(q, v)`` of the inverse dynamics at the start-of-step velocity. MuJoCo's ``implicit``
integrator instead linearizes it at the end of the step: ``(M + h D) dv = M dv_c`` with ``D = dc/dv`` (the RNE
velocity derivative, not symmetric) and ``dv_c`` the constrained velocity change. This module applies the
first-order (Neumann) form of that factor, weighted by ``theta``,

``delta = -theta h H^-1 D(q, v) (v_out - v)``,

measured on each step's solved velocity ``v_out`` and added to the next step's unconstrained prediction, so that
step's constraint solve projects it onto every loop closure and contact; the one-step delay changes the step by
``O(h^3)``. ``H`` stays symmetric, so every mass solve and constraint path is unchanged. ``D x`` is formed without
``D``: the bias is quadratic in the velocity, so ``D(v) x = (c(v + s x) - c(v - s x)) / (2 s)`` for any ``s``;
the kernel picks ``s = max|v| / max|x|`` per articulation so neither evaluation cancels in float32.
"""

import warp as wp

from .kernels import _active_free_root_dof_start, compute_link_kinematics, write_body_dynamics

wp.set_module_options({"enable_backward": False})


@wp.kernel
def eval_velocity_derivative_product(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    joint_X_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    articulation_origin: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    articulation_active: wp.array[int],
    velocity_start: wp.array[float],
    velocity_end: wp.array[float],
    # scratch
    joint_qd_probe: wp.array[float],
    joint_S_probe: wp.array[wp.spatial_vector],
    body_v_probe: wp.array[wp.spatial_vector],
    body_a_probe: wp.array[wp.spatial_vector],
    body_f_plus: wp.array[wp.spatial_vector],
    body_f_minus: wp.array[wp.spatial_vector],
    body_I_unused: wp.array[wp.spatial_matrix],
    body_terms_unused: wp.array2d[float],
    # outputs
    tau_derivative: wp.array[float],
):
    """Write ``(d tau / d qd) (velocity_end - velocity_start)`` at ``qd = velocity_start``, ``tau = -c(q, qd)``."""
    art = wp.tid()
    start = articulation_start[art]
    end = articulation_joint_end[art]
    if start >= end:
        return
    d0 = joint_qd_start[start]
    d1 = joint_qd_start[end]
    if articulation_active[art] == 0:
        return
    if is_free_rigid[art] != 0:
        # Single free bodies keep Steel's own stabilized gyroscopic update.
        for d in range(d0, d1):
            tau_derivative[d] = 0.0
        return
    v_max = float(0.0)
    x_max = float(0.0)
    for d in range(d0, d1):
        v_max = wp.max(v_max, wp.abs(velocity_start[d]))
        x_max = wp.max(x_max, wp.abs(velocity_end[d] - velocity_start[d]))
    if v_max == 0.0 or x_max == 0.0:
        # D is linear in the velocity: no velocity, or no change, means no correction.
        for d in range(d0, d1):
            tau_derivative[d] = 0.0
        return
    s = v_max / x_max
    origin = articulation_origin[art]
    for sweep in range(2):
        sign = 1.0
        if sweep == 1:
            sign = -1.0
        for d in range(d0, d1):
            joint_qd_probe[d] = velocity_start[d] + sign * s * (velocity_end[d] - velocity_start[d])
        for i in range(start, end):
            parent = joint_parent[i]
            child = joint_child[i]
            parent_v = wp.spatial_vector()
            parent_a = wp.spatial_vector()
            # A parent outside the articulation (a static anchor) does not move with the probed velocity.
            if parent >= 0 and body_to_articulation[parent] == art:
                parent_v = body_v_probe[parent]
                parent_a = body_a_probe[parent]
            v_s, a_s = compute_link_kinematics(
                i,
                parent,
                child,
                parent_v,
                parent_a,
                origin,
                joint_type,
                joint_q_start,
                joint_qd_start,
                joint_q,
                joint_qd_probe,
                joint_axis,
                joint_dof_dim,
                body_q,
                joint_X_p,
                joint_S_probe,
                body_v_probe,
                body_a_probe,
            )
            f_out = body_f_plus
            if sweep == 1:
                f_out = body_f_minus
            write_body_dynamics(
                child,
                origin,
                wp.vec3(),
                body_q_com[child],
                body_mass[child],
                body_inertia[child],
                v_s,
                a_s,
                0,
                0,
                body_I_unused,
                body_terms_unused,
                f_out,
            )
    # Backward pass on the wrench difference; children come after their parent in joint order.
    for i in range(start, end):
        child = joint_child[i]
        body_f_plus[child] = body_f_plus[child] - body_f_minus[child]
    inv = 0.5 / s
    for k in range(end - start):
        i = end - 1 - k
        child = joint_child[i]
        f = body_f_plus[child]
        q_start = joint_qd_start[i]
        q_end = joint_qd_start[i + 1]
        for d in range(q_start, q_end):
            tau_derivative[d] = -wp.dot(joint_S_probe[d], f) * inv
        parent = joint_parent[i]
        if parent >= 0 and body_to_articulation[parent] == art:
            body_f_plus[parent] = body_f_plus[parent] + f


@wp.kernel
def add_free_root_transport_derivative(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_child: wp.array[int],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    joint_active: wp.array[int],
    velocity_start: wp.array[float],
    velocity_end: wp.array[float],
    # outputs
    qdd_derivative: wp.array[float],
):
    """Add the derivative of the free-root transport ``w x v`` (added to the predictor outside the bias) along
    ``x = velocity_end - velocity_start`` to the root's linear entries: ``x_w x v + w x x_v``.

    The bias of a free root, taken at its COM, holds ``m w x v``; the transport cancels it in the COM's linear
    acceleration. Differentiating only the bias would leave ``m (x_w x v + w x x_v)`` on the root's linear motion.
    """
    root_index = wp.tid()
    joint = free_root_joint_indices[root_index]
    if joint_active[joint] == 0:
        return
    if is_free_rigid[body_to_articulation[joint_child[joint]]] != 0:
        return
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    v = wp.vec3(velocity_start[d + 0], velocity_start[d + 1], velocity_start[d + 2])
    w = wp.vec3(velocity_start[d + 3], velocity_start[d + 4], velocity_start[d + 5])
    xv = wp.vec3(velocity_end[d + 0] - v[0], velocity_end[d + 1] - v[1], velocity_end[d + 2] - v[2])
    xw = wp.vec3(velocity_end[d + 3] - w[0], velocity_end[d + 4] - w[1], velocity_end[d + 5] - w[2])
    c = wp.cross(xw, v) + wp.cross(w, xv)
    for i in range(3):
        qdd_derivative[d + i] = qdd_derivative[d + i] + c[i]


@wp.kernel
def store_velocity_derivative_correction(
    correction: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    dof_active: wp.array[int],
    scale: float,
    # outputs
    lagged: wp.array[float],
):
    d = wp.tid()
    if dof_active[d] == 0 or kinematic_dof_mask[d] != 0:
        lagged[d] = 0.0
        return
    lagged[d] = scale * correction[d]


@wp.kernel
def flag_discontinuous_articulations(
    dof_articulation: wp.array[int],
    qd_in: wp.array[float],
    qd_published: wp.array[float],
    # outputs
    articulation_flag: wp.array[int],
):
    """Flag the articulations whose input velocity is not, bitwise, the one the previous step published."""
    d = wp.tid()
    art = dof_articulation[d]
    if art >= 0 and qd_in[d] != qd_published[d]:
        articulation_flag[art] = 1


@wp.kernel
def add_lagged_velocity_terms(
    dof_articulation: wp.array[int],
    articulation_flag: wp.array[int],
    lagged: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    dof_active: wp.array[int],
    scale: float,
    # outputs
    target: wp.array[float],
):
    """``target += scale * lagged`` on the DOFs of articulations that continue from the previous step."""
    d = wp.tid()
    art = dof_articulation[d]
    if art < 0 or articulation_flag[art] != 0:
        return
    if dof_active[d] == 0 or kinematic_dof_mask[d] != 0:
        return
    target[d] = target[d] + scale * lagged[d]
