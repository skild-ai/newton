# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit discrete-rod material in Steel's generalized impulse response."""

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType
from .friction_patches import WARP_DETERMINISM_OFF
from .kernels import PGS_CONSTRAINT_TYPE_CONNECT

wp.set_module_options(WARP_DETERMINISM_OFF)


@wp.func
def _angular_measure(q: wp.quat):
    """Return material curvature, transported twist and their left-rotation Jacobian."""
    z = wp.vec3(0.0, 0.0, 1.0)
    t = wp.quat_rotate(q, z)
    # 1 + t.z = 2 (w² + z²). This form retains precision near a fold.
    denom = 2.0 * (q[3] * q[3] + q[2] * q[2])
    if denom <= 1.0e-12:
        # At antiparallel tangents parallel transport is undefined. Choose the
        # parent material X as the half-turn axis, as in the authored rod model.
        normal = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
        angle = wp.atan2(-normal[1], normal[0])
        return wp.vec3(20.0, 0.0, angle), wp.matrix_from_rows(wp.vec3(), wp.vec3(), t)

    cross = wp.cross(z, t)
    bend = 2.0 * cross / denom
    d_bend = (2.0 / denom) * (t[2] * wp.identity(3, float) - wp.outer(t, z))
    d_bend += wp.outer(bend, cross) / denom
    magnitude = wp.length(bend)
    if magnitude > 20.0:
        unit = bend / magnitude
        d_bend = (20.0 / magnitude) * (wp.identity(3, float) - wp.outer(unit, unit)) * d_bend
        bend = 20.0 * unit

    # The twist left after minimal tangent transport is the Z part of the
    # swing-twist decomposition. Both atan2 operands are invariant to q -> -q.
    twist = wp.atan2(2.0 * q[2] * q[3], q[3] * q[3] - q[2] * q[2])
    twist_row = wp.vec3(t[0] / denom, t[1] / denom, 1.0)
    return wp.vec3(bend[0], bend[1], twist), wp.matrix_from_rows(d_bend[0], d_bend[1], twist_row)


@wp.struct
class DahlHistory:
    limit: wp.array[float]
    decay: wp.array[float]
    strain: wp.array[wp.vec3]
    stress: wp.array[wp.vec3]
    initialized: wp.array[bool]


@wp.func
def _angle_delta(a: wp.vec3, b: wp.vec3):
    delta = a - b
    delta[2] = wp.atan2(wp.sin(delta[2]), wp.cos(delta[2]))
    return delta


@wp.func
def _strain(q: wp.array[float], rest: wp.array[float], start: int):
    rotation = wp.quat(q[start + 3], q[start + 4], q[start + 5], q[start + 6])
    reference = wp.quat(rest[start + 3], rest[start + 4], rest[start + 5], rest[start + 6])
    angular, jacobian = _angular_measure(rotation)
    intrinsic, _ = _angular_measure(reference)
    return rotation, _angle_delta(angular, intrinsic), jacobian


@wp.func
def _advance_stress(previous: float, increment: float, envelope: float, decay: float):
    # Exact integral of d sigma/d kappa along one monotonic loading segment.
    direction = wp.sign(increment)
    x = wp.abs(increment) / decay
    fraction = 1.0 - wp.exp(-x)
    if x < 1.0e-3:
        fraction = x * (1.0 - 0.5 * x)
    return previous + (direction * envelope - previous) * fraction


@wp.kernel
def _commit_history(
    rods: wp.array[int],
    q_start: wp.array[int],
    v_start: wp.array[int],
    dof_dim: wp.array2d[int],
    q: wp.array[float],
    rest: wp.array[float],
    ke: wp.array[float],
    enabled: wp.array[bool],
    history: DahlHistory,
):
    j = rods[wp.tid()]
    # The rotation and angular DOFs follow the linear ones; an inextensible rod joint has none.
    lin = dof_dim[j, 0]
    _, strain, _jac = _strain(q, rest, q_start[j] + lin - 3)
    delta = _angle_delta(strain, history.strain[j])
    stress = wp.vec3()
    decay = history.decay[j]
    if enabled[j] and history.initialized[j] and decay > 0.0 and history.limit[j] > 0.0:
        for i in range(3):
            bound = history.limit[j] * ke[v_start[j] + lin + i]
            stress[i] = wp.clamp(_advance_stress(history.stress[j][i], delta[i], bound, decay), -bound, bound)
    history.strain[j] = strain
    history.stress[j] = stress
    history.initialized[j] = enabled[j]


@wp.kernel
def _reset_history(rods: wp.array[int], worlds: wp.array[int], mask: wp.array[bool], count: int, history: DahlHistory):
    j = rods[wp.tid()]
    world = worlds[j]
    if not mask or mask[wp.where(world >= 0, world, count)]:
        history.initialized[j] = False
        history.stress[j] = wp.vec3()
        history.strain[j] = wp.vec3()


@wp.kernel
def _linearize(
    joints: wp.array[int],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    dof_dim: wp.array2d[int],
    child: wp.array[int],
    body_com: wp.array[wp.vec3],
    X_c: wp.array[wp.transform],
    q: wp.array[float],
    qd: wp.array[float],
    rest: wp.array[float],
    ke: wp.array[float],
    kd: wp.array[float],
    enabled: wp.array[bool],
    owners: wp.array[int],
    h: float,
    history: DahlHistory,
    write_blocks: int,
    tau: wp.array[float],
    blocks: wp.array[wp.spatial_matrix],
    joint_blocks: wp.array[wp.spatial_matrix],
):
    r = wp.tid()
    joint = joints[r]
    block = wp.spatial_matrix()
    lin = dof_dim[joint, 0]
    if owners[joint] < 0:
        for i in range(lin + 3):
            tau[qd_start[joint] + i] = 0.0
    if not enabled[joint]:
        if write_blocks != 0:
            blocks[r] = block
        if owners[joint] >= 0:
            joint_blocks[joint] = block
        return
    iq = q_start[joint]
    iv = qd_start[joint]
    if lin == 0:
        # Inextensible: bend and twist only. The block's top-left 3x3 is the joint's own three DOFs.
        _rotation0, a0, Jw0 = _strain(q, rest, iq - 3)
        if history.initialized and not history.initialized[joint]:
            history.strain[joint] = a0
            history.initialized[joint] = True
        w = wp.vec3(qd[iv], qd[iv + 1], qd[iv + 2])
        rate0 = Jw0 * w
        force0 = wp.vec3()
        for i in range(3):
            k = ke[iv + i]
            d = kd[iv + i] + h * k
            force0[i] = -(k * a0[i] + d * rate0[i])
            if history.initialized:
                bound = history.limit[joint] * k
                decay = history.decay[joint]
                if bound > 0.0 and decay > 0.0:
                    delta = _angle_delta(a0, history.strain[joint])
                    stress = _advance_stress(history.stress[joint][i], delta[i], bound, decay)
                    direction = wp.sign(rate0[i])
                    tangent = wp.max(0.0, (bound - direction * stress) / decay)
                    force0[i] -= stress + h * tangent * rate0[i]
                    d += h * tangent
            for j in range(3):
                for l in range(3):
                    block[j, l] += Jw0[i, j] * (h * d) * Jw0[i, l]
        load0 = wp.transpose(Jw0) * force0
        for i in range(3):
            tau[iv + i] += load0[i]
        if write_blocks != 0:
            blocks[r] = block
        if owners[joint] >= 0:
            joint_blocks[joint] = block
        return
    rotation, a, Jw = _strain(q, rest, iq)
    if history.initialized and not history.initialized[joint]:
        history.strain[joint] = a
        history.initialized[joint] = True
    error = wp.spatial_vector(wp.vec3(q[iq], q[iq + 1], q[iq + 2]), a)
    # Newton stores relative linear velocity at the child COM; material
    # strain is measured at the child anchor in the rotating parent frame.
    lever = wp.quat_rotate(rotation, wp.transform_point(wp.transform_inverse(X_c[joint]), body_com[child[joint]]))
    J = wp.spatial_matrix()
    arm = wp.skew(lever)
    for i in range(3):
        J[i, i] = 1.0
        for j in range(3):
            J[i, j + 3] = arm[i, j]
            J[i + 3, j + 3] = Jw[i, j]
    speed = wp.spatial_vector()
    for i in range(6):
        speed[i] = qd[iv + i]
    rate = J * speed
    force = wp.spatial_vector()
    for i in range(6):
        k = ke[iv + i]
        d = kd[iv + i] + h * k
        force[i] = -(k * error[i] + d * rate[i])
        if i >= 3 and history.initialized:
            bound = history.limit[joint] * k
            decay = history.decay[joint]
            if bound > 0.0 and decay > 0.0:
                axis = i - 3
                delta = _angle_delta(a, history.strain[joint])
                stress = _advance_stress(history.stress[joint][axis], delta[axis], bound, decay)
                direction = wp.sign(rate[i])
                tangent = wp.max(0.0, (bound - direction * stress) / decay)
                force[i] -= stress + h * tangent * rate[i]
                d += h * tangent
        for j in range(6):
            for l in range(6):
                block[j, l] += J[i, j] * (h * d) * J[i, l]
    load = wp.transpose(J) * force
    for i in range(6):
        tau[iv + i] += load[i]
    if write_blocks != 0:
        blocks[r] = block
    # The factor reads tree rods' blocks by joint index (scatter_joint_blocks).
    if owners[joint] >= 0:
        joint_blocks[joint] = block


@wp.kernel
def _augment(
    layout: wp.array2d[int],
    blocks: wp.array[wp.spatial_matrix],
    H: wp.array3d[float],
):
    r, i, j = wp.tid()
    rod = layout[r, 0]
    group = layout[r, 1]
    start = layout[r, 2]
    if i < layout[r, 3] and j < layout[r, 3]:  # the joint's own DOFs: 3 for an inextensible rod joint
        H[group, start + i, start + j] += blocks[rod][i, j]


@wp.kernel
def _write_loop_state(
    loops: wp.array[int],
    parents: wp.array[int],
    children: wp.array[int],
    Xp: wp.array[wp.transform],
    Xc: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    positions: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    qs: wp.array[int],
    vs: wp.array[int],
    q: wp.array[float],
    qd: wp.array[float],
):
    j = loops[wp.tid()]
    p = parents[j]
    c = children[j]
    frame = Xp[j]
    x = wp.transform_point(positions[c], com[c])
    parent_v = wp.vec3()
    parent_w = wp.vec3()
    if p >= 0:
        frame = positions[p] * frame
        parent_w = wp.spatial_bottom(velocities[p])
        parent_v = wp.spatial_top(velocities[p]) + wp.cross(parent_w, x - wp.transform_point(positions[p], com[p]))
    frame = wp.transform(wp.transform_get_translation(frame), wp.normalize(wp.transform_get_rotation(frame)))
    inv = wp.transform_inverse(frame)
    relative = inv * positions[c] * Xc[j]
    relative = wp.transform(wp.transform_get_translation(relative), wp.normalize(wp.transform_get_rotation(relative)))
    linear = wp.transform_vector(inv, wp.spatial_top(velocities[c]) - parent_v)
    angular = wp.transform_vector(inv, wp.spatial_bottom(velocities[c]) - parent_w)
    for i in range(7):
        q[qs[j] + i] = relative[i]
    for i in range(3):
        qd[vs[j] + i] = linear[i]
        qd[vs[j] + 3 + i] = angular[i]


@wp.kernel
def _loop_columns(
    info: wp.array2d[int],
    signs: wp.array2d[int],
    parents: wp.array[int],
    children: wp.array[int],
    Xp: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    origins: wp.array[wp.vec3],
    S: wp.array[wp.spatial_vector],
    vs: wp.array[int],
    tau: wp.array[float],
    cols: wp.array2d[wp.spatial_vector],
):
    l, d = wp.tid()
    j = info[l, 0]
    art = info[l, 1]
    k = info[l, 3] + d
    column = wp.spatial_vector()
    sign = float(signs[l, d])
    if sign != 0.0:
        p = parents[j]
        c = children[j]
        frame = Xp[j]
        if p >= 0:
            frame = poses[p] * frame
        inv = wp.quat_inverse(wp.transform_get_rotation(frame))
        arm = wp.transform_point(poses[c], com[c]) - origins[art]
        w = wp.spatial_bottom(S[k])
        v = wp.spatial_top(S[k]) + wp.cross(w, arm)
        column = wp.spatial_vector(sign * wp.quat_rotate(inv, v), sign * wp.quat_rotate(inv, w))
        wrench = wp.spatial_vector()
        for i in range(6):
            wrench[i] = tau[vs[j] + i]
        wp.atomic_add(tau, k, wp.dot(column, wrench))
    cols[l, d] = column


@wp.kernel
def _augment_loops(
    info: wp.array2d[int],
    cols: wp.array2d[wp.spatial_vector],
    blocks: wp.array[wp.spatial_matrix],
    H: wp.array3d[float],
):
    l, a, b = wp.tid()
    val = wp.dot(cols[l, a], blocks[info[l, 4]] * cols[l, b])
    wp.atomic_add(H, info[l, 2], a, b, val)


@wp.kernel
def _cartesian_inertial_wrench(
    integrated: wp.array[bool],
    children: wp.array[int],
    owners: wp.array[int],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    com: wp.array[wp.vec3],
    pose: wp.array[wp.transform],
    twist: wp.array[wp.spatial_vector],
    origin: wp.array[wp.vec3],
    worlds: wp.array[int],
    gravity: wp.array[wp.vec3],
    disabled: wp.array[bool],
    wrench: wp.array[wp.spatial_vector],
):
    # The generalized solve uses the current pose's motion basis throughout
    # this impulse step. In world COM coordinates the only velocity bias is
    # gyroscopic. The change of basis at the new pose is made after integration.
    j = wp.tid()
    art = owners[j]
    if art < 0 or not integrated[art]:
        return
    b = children[j]
    rot = wp.transform_get_rotation(pose[b])
    w = wp.spatial_bottom(twist[b])
    Iw = wp.quat_rotate(rot, inertia[b] * wp.quat_rotate_inv(rot, w))
    f = wp.vec3()
    if not disabled[b]:
        f = -mass[b] * gravity[worlds[b]]
    arm = wp.transform_point(pose[b], com[b]) - origin[art]
    wrench[b] = wp.spatial_vector(f, wp.cross(arm, f) + wp.cross(w, Iw))


@wp.kernel
def _world_twists(
    integrated: wp.array[int],
    skip: wp.array[int],
    active: wp.array[int],
    order: wp.array[int],
    starts: wp.array[int],
    ends: wp.array[int],
    parent: wp.array[int],
    child: wp.array[int],
    v_start: wp.array[int],
    com: wp.array[wp.vec3],
    orig: wp.array[wp.vec3],
    S: wp.array[wp.spatial_vector],
    speed: wp.array[float],
    pose: wp.array[wp.transform],
    accumulated: wp.array[wp.spatial_vector],
    next_twist: wp.array[wp.spatial_vector],
):
    art = integrated[wp.tid()]
    if active[art] == 0 or skip[art] != 0:
        return
    for slot in range(starts[art], ends[art]):
        j = slot
        if order:
            j = order[slot]
        p = parent[j]
        b = child[j]
        v = wp.spatial_vector()
        if p >= 0:
            v = accumulated[p]
        for d in range(6):
            k = v_start[j] + d
            v += S[k] * speed[k]
        accumulated[b] = v
        old = pose[b]
        cx = wp.transform_point(old, com[b])
        w = wp.spatial_bottom(v)
        linear = wp.spatial_top(v) + wp.cross(w, cx - orig[art])
        next_twist[b] = wp.spatial_vector(linear, w)


@wp.kernel
def _carry_twists(
    carried: wp.array[int],
    skip: wp.array[int],
    active: wp.array[int],
    order: wp.array[int],
    starts: wp.array[int],
    ends: wp.array[int],
    parent: wp.array[int],
    child: wp.array[int],
    v_start: wp.array[int],
    com: wp.array[wp.vec3],
    orig: wp.array[wp.vec3],
    S: wp.array[wp.spatial_vector],
    speed: wp.array[float],
    pose: wp.array[wp.transform],
    accumulated: wp.array[wp.spatial_vector],
    twist: wp.array[wp.spatial_vector],
):
    """Each body's COM twist from the step's motion basis (start-of-step pose) and its new joint velocities."""
    art = carried[wp.tid()]
    if active[art] == 0 or skip[art] != 0:
        return
    for slot in range(starts[art], ends[art]):
        j = slot
        if order:
            j = order[slot]
        p = parent[j]
        b = child[j]
        v = wp.spatial_vector()
        if p >= 0:
            v = accumulated[p]
        for k in range(v_start[j], v_start[j + 1]):
            v += S[k] * speed[k]
        accumulated[b] = v
        w = wp.spatial_bottom(v)
        cx = wp.transform_point(pose[b], com[b])
        twist[b] = wp.spatial_vector(wp.spatial_top(v) + wp.cross(w, cx - orig[art]), w)


@wp.kernel
def _validate_carried(
    articulations: wp.array[int],
    articulation_start: wp.array[int],
    joint_end: wp.array[int],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    q: wp.array[float],
    carried_q: wp.array[float],
    qd: wp.array[float],
    carried_qd: wp.array[float],
    valid: wp.array[int],
):
    """Mark the carried articulations whose every joint coordinate and velocity is the state the last step left."""
    art = articulations[wp.tid()]
    ok = int(1)
    for j in range(articulation_start[art], joint_end[art]):
        for k in range(q_start[j], q_start[j + 1]):
            if q[k] != carried_q[k]:
                ok = 0
        for k in range(qd_start[j], qd_start[j + 1]):
            if qd[k] != carried_qd[k]:
                ok = 0
    valid[art] = ok


@wp.kernel
def _invalidate_carried(
    worlds: wp.array[int], mask: wp.array[bool], count: int, q_start: wp.array[int], carried_q: wp.array[float]
):
    """Drop the carried state of the reset worlds' joints (NaN compares unequal to every coordinate)."""
    j = wp.tid()
    world = worlds[j]
    if not mask or mask[wp.where(world >= 0, world, count)]:
        for k in range(q_start[j], q_start[j + 1]):
            carried_q[k] = wp.nan


@wp.kernel
def _momentum_correction(
    carried_valid: wp.array[int],
    children: wp.array[int],
    owners: wp.array[int],
    flags: wp.array[int],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    com: wp.array[wp.vec3],
    pose: wp.array[wp.transform],
    body_v: wp.array[wp.spatial_vector],
    origin: wp.array[wp.vec3],
    carried_twist: wp.array[wp.spatial_vector],
    inv_dt: float,
    wrench: wp.array[wp.spatial_vector],
):
    """Start the step from the carried body momenta instead of the joint velocities at the new pose.

    Adds the wrench M (V_carried - V) / dt, so the joint-space momentum the solve starts from is J^T M V_carried:
    the previous step's body velocities projected onto the joints at the new pose. A projection never adds
    kinetic energy, where the joint-space velocity-product terms of a long chain do.
    """
    j = wp.tid()
    art = owners[j]
    # Only a state this solver produced carries its twists (_validate_carried): an edit to any joint's coordinates
    # or velocities, or a reset, drops the whole articulation's.
    if art < 0 or carried_valid[art] == 0:
        return
    b = children[j]
    if (flags[b] & BodyFlags.KINEMATIC) != 0:
        return
    cx = wp.transform_point(pose[b], com[b])
    arm = cx - origin[art]
    v = body_v[b]
    w = wp.spatial_bottom(v)
    now = wp.spatial_vector(wp.spatial_top(v) + wp.cross(w, arm), w)
    dv = carried_twist[b] - now
    rot = wp.transform_get_rotation(pose[b])
    f = mass[b] * wp.spatial_top(dv) * inv_dt
    tau = wp.quat_rotate(rot, inertia[b] * wp.quat_rotate_inv(rot, wp.spatial_bottom(dv))) * inv_dt
    # The wrench array holds the bias, the negative of an applied wrench, about the articulation origin.
    wrench[b] = wrench[b] - wp.spatial_vector(f, wp.cross(arm, f) + tau)


@wp.func
def _rotation_increment(w: wp.vec3, dt: float):
    half = 0.5 * dt
    magnitude = wp.length(w)
    if magnitude * dt < 1.0e-4:
        return wp.normalize(wp.quat(w * half, 1.0))
    return wp.quat_from_axis_angle(w / magnitude, magnitude * dt)


@wp.func
def _rotation_displacement(q: wp.quat, arm: wp.vec3):
    # (R - I) arm without cancellation for a nearly stationary frame.
    u = wp.vec3(q[0], q[1], q[2])
    return 2.0 * wp.cross(u, q[3] * arm + wp.cross(u, arm))


@wp.kernel
def _integrate_relative_twists(
    joints: wp.array[int],
    arts: wp.array[int],
    active: wp.array[int],
    parents: wp.array[int],
    children: wp.array[int],
    flags: wp.array[int],
    Xp: wp.array[wp.transform],
    Xc: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    twists: wp.array[wp.spatial_vector],
    qs: wp.array[int],
    vs: wp.array[int],
    q: wp.array[float],
    dt: float,
    result: wp.array[float],
    velocity: wp.array[float],
):
    j = joints[wp.tid()]
    p, c = parents[j], children[j]
    if active[arts[j]] == 0 or (flags[c] & BodyFlags.KINEMATIC) != 0:
        return
    iq, iv = qs[j], vs[j]
    frame = wp.transform_get_rotation(Xp[j])
    vp, wp_parent = wp.vec3(), wp.vec3()
    parent_arm = wp.vec3()
    dp = wp.quat_identity()
    parent_shift = wp.vec3()
    if p >= 0:
        frame = wp.normalize(wp.transform_get_rotation(poses[p]) * frame)
        vp = wp.quat_rotate_inv(frame, wp.spatial_top(twists[p]))
        wp_parent = wp.quat_rotate_inv(frame, wp.spatial_bottom(twists[p]))
        parent_arm = wp.transform_point(wp.transform_inverse(Xp[j]), com[p])
        if (flags[p] & BodyFlags.KINEMATIC) == 0:
            dp = _rotation_increment(wp_parent, dt)
            parent_shift = dt * vp - _rotation_displacement(dp, parent_arm)

    vc = wp.quat_rotate_inv(frame, wp.spatial_top(twists[c]))
    wc = wp.quat_rotate_inv(frame, wp.spatial_bottom(twists[c]))
    dc = _rotation_increment(wc, dt)
    x = wp.vec3(q[iq], q[iq + 1], q[iq + 2])
    r = wp.quat(q[iq + 3], q[iq + 4], q[iq + 5], q[iq + 6])
    child_arm = wp.transform_point(wp.transform_inverse(Xc[j]), com[c])
    child_shift = dt * vc - _rotation_displacement(dc, wp.quat_rotate(r, child_arm))
    # Difference the COM increments in the old parent frame. Subtracting
    # two new world positions would erase small elastic displacements.
    x1 = wp.quat_rotate_inv(dp, x + (child_shift - parent_shift))
    r1 = wp.normalize(wp.quat_inverse(dp) * dc * r)
    separation = x1 + wp.quat_rotate(r1, child_arm) - parent_arm
    linear = wp.quat_rotate_inv(dp, vc - vp) - wp.cross(wp.quat_rotate_inv(dp, wp_parent), separation)
    angular = wp.quat_rotate_inv(dp, wc - wp_parent)
    for i in range(3):
        result[iq + i] = x1[i]
        velocity[iv + i] = linear[i]
        velocity[iv + 3 + i] = angular[i]
    for i in range(4):
        result[iq + 3 + i] = r1[i]


@wp.kernel
def build_propagation_rod_loop_rows(
    connect_world: wp.array[int],
    connect_kind: wp.array[int],
    connect_joint: wp.array[int],
    connect_body_p: wp.array[int],
    connect_body_c: wp.array[int],
    body_responds: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_rest_q: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_enabled: wp.array[bool],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_slot: wp.array[int],
    virtual_base: int,
    connect_impulse: wp.array2d[float],
    seed_impulses: int,
    h: float,
    # outputs
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_phi: wp.array2d[float],
    propagation_target_velocity: wp.array2d[float],
    propagation_row_rebound_velocity: wp.array2d[float],
    propagation_impulses: wp.array2d[float],
):
    """Write a loop-closing rod's six soft rows: its material as compliant constraints on the strain rates.

    Strain ``e = (R_p^T (o_c - o_p), a)`` with ``a`` the rotation measure of :func:`_strain`; the rates are
    ``G_p v_p + G_c v_c`` for the bodies' COM twists. With ``d = kd + h ke`` a row is ``G v + b + gamma m = 0``,
    ``gamma = 1 / (h d)`` and ``b = ke e / d``: eliminating the impulse gives the tree rods' implicit spring.
    ``gamma`` rides in ``propagation_row_mu`` (a closure has no friction) and ``b`` in ``propagation_phi``. A side
    that does not respond (kinematic) moves its rate into the bias.
    """
    k = wp.tid()
    slot0 = contact_slot[virtual_base + k]
    if slot0 < 0 or connect_kind[k] != 1:
        return
    world = connect_world[k]
    j = connect_joint[k]
    p = connect_body_p[k]
    c = connect_body_c[k]
    Xwp = joint_X_p[j]
    if p >= 0:
        Xwp = body_q[p] * joint_X_p[j]
    Xwc = joint_X_c[j]
    if c >= 0:
        Xwc = body_q[c] * joint_X_c[j]
    Rp = wp.quat_to_matrix(wp.transform_get_rotation(Xwp))
    RpT = wp.transpose(Rp)
    o_c = wp.transform_get_translation(Xwc)
    o_p = wp.transform_get_translation(Xwp)
    rel_rot = wp.normalize(wp.quat_inverse(wp.transform_get_rotation(Xwp)) * wp.transform_get_rotation(Xwc))
    iq = joint_q_start[j]
    iv = joint_qd_start[j]
    reference = wp.quat(joint_rest_q[iq + 3], joint_rest_q[iq + 4], joint_rest_q[iq + 5], joint_rest_q[iq + 6])
    angular, Jw = _angular_measure(rel_rot)
    intrinsic, _unused = _angular_measure(reference)
    error = wp.spatial_vector(RpT * (o_c - o_p), _angle_delta(angular, intrinsic))

    lever_p = wp.vec3()
    if p >= 0:
        lever_p = o_c - wp.transform_point(body_q[p], body_com[p])
    lever_c = wp.vec3()
    if c >= 0:
        lever_c = o_c - wp.transform_point(body_q[c], body_com[c])
    skp = wp.skew(lever_p)
    skc = wp.skew(lever_c)
    JwRT = Jw * RpT
    Ga = wp.spatial_matrix()
    Gb = wp.spatial_matrix()
    for i in range(3):
        for col in range(3):
            # linear strain rate = Rp^T (v_c - [lever_c]x w_c - v_p + [lever_p]x w_p)
            Ga[i, col] = -RpT[i, col]
            Gb[i, col] = RpT[i, col]
            acc_p = float(0.0)
            acc_c = float(0.0)
            for l in range(3):
                acc_p += RpT[i, l] * skp[l, col]
                acc_c += RpT[i, l] * skc[l, col]
            Ga[i, col + 3] = acc_p
            Gb[i, col + 3] = -acc_c
            Ga[i + 3, col + 3] = -JwRT[i, col]
            Gb[i + 3, col + 3] = JwRT[i, col]
    a_on = p >= 0 and body_responds[p] != 0
    b_on = c >= 0 and body_responds[c] != 0
    prescribed = wp.spatial_vector()
    if p >= 0 and not a_on:
        prescribed += Ga * body_qd[p]
    if c >= 0 and not b_on:
        prescribed += Gb * body_qd[c]
    body_a = int(-1)
    body_b = int(-1)
    if a_on:
        body_a = p
    if b_on:
        body_b = c
    for i in range(6):
        ke = joint_target_ke[iv + i]
        d = joint_target_kd[iv + i] + h * ke
        live = joint_enabled[j] and d > 0.0 and (a_on or b_on)
        slot = slot0 + i
        propagation_body_a[world, slot] = body_a
        propagation_body_b[world, slot] = body_b
        for r in range(6):
            va = float(0.0)
            vb = float(0.0)
            if live and a_on:
                va = Ga[i, r]
            if live and b_on:
                vb = Gb[i, r]
            propagation_J_a[world, slot, r] = va
            propagation_J_b[world, slot, r] = vb
        propagation_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONNECT
        propagation_row_parent[world, slot] = slot0
        propagation_target_velocity[world, slot] = 0.0
        propagation_row_rebound_velocity[world, slot] = 0.0
        if live:
            propagation_row_mu[world, slot] = 1.0 / (h * d)
            propagation_phi[world, slot] = ke * error[i] / d + prescribed[i]
            seed = float(0.0)
            if seed_impulses != 0:
                seed = connect_impulse[k, i]
            propagation_impulses[world, slot] = seed
        else:
            propagation_row_mu[world, slot] = 1.0
            propagation_phi[world, slot] = 0.0
            propagation_impulses[world, slot] = 0.0


class RodMaterial:
    """Eliminate the six soft bilateral impulses of each tree rod into its local mass block."""

    def __init__(self, model, rods, plan, loops_as_rows=False):
        self.history = DahlHistory()
        self.history_snapshot = ()
        attrs = getattr(model, "steel", None)
        self.has_history = attrs is not None and getattr(attrs, "dahl_tau", None) is not None
        if self.has_history:
            self.history.limit = attrs.dahl_eps_max
            self.history.decay = attrs.dahl_tau
            self.history.strain = wp.zeros(model.joint_count, dtype=wp.vec3, device=model.device)
            self.history.stress = wp.zeros_like(self.history.strain)
            self.history.initialized = wp.zeros(model.joint_count, dtype=bool, device=model.device)
        self.joints = wp.array(rods, dtype=int, device=model.device)
        self.blocks = wp.empty(len(rods), dtype=wp.spatial_matrix, device=model.device)
        # Only the dense mass matrix reads the blocks by rod (augment); see release_rod_blocks.
        self.write_blocks = True
        # The same blocks indexed by joint, for the articulated-body factorization of the propagation response.
        self.joint_blocks = wp.zeros(model.joint_count, dtype=wp.spatial_matrix, device=model.device)
        joint_art = model.joint_articulation.numpy()
        dof_start = model.joint_qd_start.numpy()
        sizes = plan.response_dof_count
        groups = np.zeros(len(sizes), dtype=np.int32)
        for size in np.unique(sizes):
            groups[sizes == size] = np.arange(np.count_nonzero(sizes == size))
        rows = {}
        self.loop_groups = {}
        loops = rods[joint_art[rods] < 0]
        if np.any(model.joint_dof_dim.numpy()[loops, 0] == 0):
            raise NotImplementedError("Steel does not yet support a loop-closing inextensible rod joint.")
        if loops_as_rows:
            if (
                self.has_history
                and len(loops)
                and bool(np.any((attrs.dahl_eps_max.numpy()[loops] > 0.0) & (attrs.dahl_tau.numpy()[loops] > 0.0)))
            ):
                # The closure rows (build_propagation_rod_loop_rows) carry no Dahl history: the seam would answer
                # elastically while the tree joints hysteretically.
                raise NotImplementedError(
                    "Steel does not yet apply Dahl hysteresis to a closed rod's closing joint under the propagation "
                    "response; use articulated_contact_response='immediate' for a closed rod with Dahl friction."
                )
            # The solver enforces loop-closing rods as rows; the mass matrix keeps only the tree rods.
            loops = loops[:0]
        # In a full-spatial rod tree the six generalized velocities per link
        # also span its world COM twist. Integrate articulated free flight in
        # that basis to avoid subtracting large rotating-parent accelerations.
        kinds = model.joint_type.numpy()
        tree = joint_art >= 0
        spatial = np.zeros(model.articulation_count, dtype=bool)
        spatial[joint_art[tree & (kinds == JointType.ROD)]] = True
        spatial[joint_art[tree & ~np.isin(kinds, (JointType.FREE, JointType.ROD))]] = False
        # An inextensible rod joint has three velocities, too few to span its link's COM twist.
        inextensible = model.joint_dof_dim.numpy()[:, 0] == 0
        spatial[joint_art[tree & (kinds == JointType.ROD) & inextensible]] = False
        self.world_twists = bool(np.any(spatial))
        # Articulations with inextensible rod joints step from carried body momenta (see _momentum_correction).
        carried = np.zeros(model.articulation_count, dtype=bool)
        carried[joint_art[tree & (kinds == JointType.ROD) & inextensible]] = True
        self.carried = bool(np.any(carried))
        if self.carried:
            self.carried_mask = wp.array(carried, dtype=bool, device=model.device)
            self.carried_articulations = wp.array(np.flatnonzero(carried), dtype=int, device=model.device)
            self.carried_articulations_np = np.flatnonzero(carried)
            self.no_skip_carried = wp.zeros(max(model.articulation_count, 1), dtype=int, device=model.device)
            self.carried_twist = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
            self.carried_accumulated = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
            self.carried_q = wp.full(model.joint_coord_count, float("nan"), dtype=float, device=model.device)
            self.carried_qd = wp.full(model.joint_dof_count, float("nan"), dtype=float, device=model.device)
            self.carried_valid = wp.zeros(max(model.articulation_count, 1), dtype=int, device=model.device)
        self.all_spatial = bool(np.all(spatial))
        self.spatial_mask = spatial
        # Rod trees the propagation response can step without a dense mass matrix: full-spatial ones (their
        # velocities span each link's COM twist) and inextensible ones, which step from carried body momenta. The
        # articulated-body factorization solves either from the same joint loads the dense solve would.
        self.tree_mask = spatial | carried
        if self.world_twists:
            self.integrated = wp.array(spatial, dtype=bool, device=model.device)
            self.integrated_articulations = wp.array(np.flatnonzero(spatial), dtype=int, device=model.device)
            self._integrated_np = np.flatnonzero(spatial)
            selected = tree & spatial[np.maximum(joint_art, 0)]
            self.tree_joints = wp.array(np.flatnonzero(selected), dtype=int, device=model.device)
            self.accumulated = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
            self.no_skip = wp.zeros(max(model.articulation_count, 1), dtype=int, device=model.device)
        self.loops = None
        if len(loops):
            self.loops = wp.array(loops, dtype=int, device=model.device)
            self.loop_q, self.loop_qd = wp.empty_like(model.joint_q), wp.empty_like(model.joint_qd)
            parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
            inbound = {int(children[j]): j for j in np.flatnonzero(joint_art >= 0)}
            by_size = {}
            for j in loops:
                art = plan.loop_joint_articulation[j]
                if art < 0 or (parents[j] >= 0 and joint_art[inbound[parents[j]]] != art):
                    raise NotImplementedError("A loop-closing rod must connect links of one articulation.")
                size = int(sizes[art])
                if size:
                    start = plan.articulation_dof_start[art]
                    signs = np.zeros(size, dtype=np.int32)
                    for endpoint, sign in ((children[j], 1), (parents[j], -1)):
                        body = endpoint
                        while body in inbound:
                            edge = inbound[body]
                            signs[dof_start[edge] - start : dof_start[edge + 1] - start] += sign
                            body = parents[edge]
                    info = (j, art, groups[art], start, np.searchsorted(rods, j))
                    by_size.setdefault(size, []).append((info, signs))
            for size, group in by_size.items():
                info, signs = zip(*group, strict=True)
                self.loop_groups[size] = (
                    wp.array(np.asarray(info), dtype=int, device=model.device),
                    wp.array(np.asarray(signs), dtype=int, device=model.device),
                    wp.empty((len(group), size), dtype=wp.spatial_vector, device=model.device),
                )
        for r, j in enumerate(rods):
            art = joint_art[j]
            if art < 0:
                continue
            size = int(sizes[art])
            if size:
                rows.setdefault(size, []).append(
                    (r, groups[art], dof_start[j] - plan.articulation_dof_start[art], dof_start[j + 1] - dof_start[j])
                )
        self.layout = {size: wp.array(layout, dtype=int, device=model.device) for size, layout in rows.items()}

    def write_loop_state(self, model, state, q, qd, joints=None):
        if joints is None:
            joints = self.loops
        wp.launch(
            _write_loop_state,
            dim=len(joints),
            inputs=[
                joints,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.body_com,
                state.body_q,
                state.body_qd,
                model.joint_q_start,
                model.joint_qd_start,
            ],
            outputs=[q, qd],
            device=model.device,
        )

    def linearize(self, model, state, augmented, origins, dt):
        tau = augmented._joint_tau
        q, qd = state.joint_q, state.joint_qd
        if self.loops is not None:
            wp.copy(self.loop_q, q)
            wp.copy(self.loop_qd, qd)
            q, qd = self.loop_q, self.loop_qd
            self.write_loop_state(model, state, q, qd)
        wp.launch(
            _linearize,
            dim=len(self.joints),
            inputs=[
                self.joints,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_child,
                model.body_com,
                model.joint_X_c,
                q,
                qd,
                model.joint_rest_q,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_enabled,
                model.joint_articulation,
                dt,
                self.history,
                int(self.write_blocks),
            ],
            outputs=[tau, self.blocks, self.joint_blocks],
            device=model.device,
        )

        for info, signs, cols in self.loop_groups.values():
            wp.launch(
                _loop_columns,
                dim=cols.shape,
                inputs=[
                    info,
                    signs,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.body_com,
                    state.body_q,
                    origins,
                    augmented._joint_S_s,
                    model.joint_qd_start,
                    tau,
                ],
                outputs=[cols],
                device=model.device,
            )

    def release_rod_blocks(self, device):
        """Keep no per-rod blocks: no group assembles a dense mass matrix, so only the joint-indexed copy is read."""
        self.blocks = wp.zeros(1, dtype=wp.spatial_matrix, device=device)
        self.write_blocks = False

    def scatter_joint_blocks(self, model):
        """Each tree rod's material block at its joint index (other joints zero), as the linearization wrote it."""
        del model
        return self.joint_blocks

    def augment(self, size, H):
        if size in self.layout:
            layout = self.layout[size]
            wp.launch(_augment, dim=(layout.shape[0], 6, 6), inputs=[layout, self.blocks], outputs=[H], device=H.device)
        if size in self.loop_groups:
            info, _, cols = self.loop_groups[size]
            wp.launch(
                _augment_loops,
                dim=(len(info), size, size),
                inputs=[info, cols, self.blocks],
                outputs=[H],
                device=H.device,
            )

    def inertial_bias(self, solver, state, augmented):
        m = solver.model
        wp.launch(
            _cartesian_inertial_wrench,
            dim=m.joint_count,
            inputs=[
                self.integrated,
                m.joint_child,
                m.joint_articulation,
                m.body_mass,
                m.body_inertia,
                m.body_com,
                state.body_q,
                augmented._body_v_s,
                solver._articulation_origin,
                m.body_world,
                m.gravity,
                m.body_disable_gravity,
            ],
            outputs=[augmented._body_f_s],
            device=m.device,
        )

    def carried_bias(self, solver, state, augmented, dt):
        """The bias of the carried articulations: gyroscopic and gravity only, plus the momentum correction."""
        m = solver.model
        wp.launch(
            _cartesian_inertial_wrench,
            dim=m.joint_count,
            inputs=[
                self.carried_mask,
                m.joint_child,
                m.joint_articulation,
                m.body_mass,
                m.body_inertia,
                m.body_com,
                state.body_q,
                augmented._body_v_s,
                solver._articulation_origin,
                m.body_world,
                m.gravity,
                m.body_disable_gravity,
            ],
            outputs=[augmented._body_f_s],
            device=m.device,
        )
        wp.launch(
            _validate_carried,
            dim=len(self.carried_articulations),
            inputs=[
                self.carried_articulations,
                m.articulation_start,
                solver._articulation_joint_end,
                m.joint_q_start,
                m.joint_qd_start,
                state.joint_q,
                self.carried_q,
                state.joint_qd,
                self.carried_qd,
            ],
            outputs=[self.carried_valid],
            device=m.device,
        )
        wp.launch(
            _momentum_correction,
            dim=m.joint_count,
            inputs=[
                self.carried_valid,
                m.joint_child,
                m.joint_articulation,
                m.body_flags,
                m.body_mass,
                m.body_inertia,
                m.body_com,
                state.body_q,
                augmented._body_v_s,
                solver._articulation_origin,
                self.carried_twist,
                1.0 / dt,
            ],
            outputs=[augmented._body_f_s],
            device=m.device,
        )

    def carry(self, solver, state, augmented, result):
        """Record the step's body twists (start-of-step basis, new joint velocities) and the state they go with."""
        m = solver.model
        chain = getattr(solver, "_rod_chain_any", False)
        if chain:
            # Chain-shaped trees sum their twists with the warp scan the full-spatial rods use (any joint width).
            from . import rod_tree  # noqa: PLC0415

            wp.launch(
                rod_tree.chain_world_twists_kernel(),
                dim=rod_tree.chain_launch_dim(len(self.carried_articulations)),
                inputs=[
                    len(self.carried_articulations),
                    self.carried_articulations,
                    solver._rod_chain_flag,
                    solver._dynamics_art_active,
                    solver._propagation_joint_range,
                    solver._rod_tree_meta,
                    augmented._joint_S_s,
                    result.joint_qd,
                    state.body_q,
                    m.body_com,
                    solver._articulation_origin,
                ],
                outputs=[self.carried_twist],
                block_dim=rod_tree.BLOCK_DIM,
                device=m.device,
            )
        chain_np = getattr(solver, "_rod_chain_flag_np", None)
        all_chains = chain and chain_np is not None and bool(np.all(chain_np[self.carried_articulations_np] != 0))
        if not all_chains:
            self._carry_walk(solver, state, augmented, result, chain)
        wp.copy(self.carried_q, result.joint_q)
        wp.copy(self.carried_qd, result.joint_qd)

    def _carry_walk(self, solver, state, augmented, result, chain):
        """The serial walk of :meth:`carry` for the carried trees that are not chains."""
        m = solver.model
        wp.launch(
            _carry_twists,
            dim=len(self.carried_articulations),
            inputs=[
                self.carried_articulations,
                solver._rod_chain_flag if chain else self.no_skip_carried,
                solver._dynamics_art_active,
                m._fk_joint_order,
                m.articulation_start,
                solver._articulation_joint_end,
                m.joint_parent,
                m.joint_child,
                m.joint_qd_start,
                m.body_com,
                solver._articulation_origin,
                augmented._joint_S_s,
                result.joint_qd,
                state.body_q,
            ],
            outputs=[self.carried_accumulated, self.carried_twist],
            device=m.device,
        )

    def integrate(self, solver, state, augmented, result, velocity, dt):
        m = solver.model
        chain = getattr(solver, "_rod_chain_any", False)
        if chain:
            # Chain-shaped rod trees sum their twists with a warp scan.
            from . import rod_tree  # noqa: PLC0415

            wp.launch(
                rod_tree.chain_world_twists_kernel(),
                dim=rod_tree.chain_launch_dim(len(self.integrated_articulations)),
                inputs=[
                    len(self.integrated_articulations),
                    self.integrated_articulations,
                    solver._rod_chain_flag,
                    solver._dynamics_art_active,
                    solver._propagation_joint_range,
                    solver._rod_tree_meta,
                    augmented._joint_S_s,
                    velocity,
                    state.body_q,
                    m.body_com,
                    solver._articulation_origin,
                ],
                outputs=[result.body_qd],
                block_dim=rod_tree.BLOCK_DIM,
                device=m.device,
            )
        # The walk skips chains; with only chains it is not launched (the chain flags are host-known at setup).
        chain_np = getattr(solver, "_rod_chain_flag_np", None)
        all_chains = chain and chain_np is not None and bool(np.all(chain_np[self._integrated_np] != 0))
        if not all_chains:
            wp.launch(
                _world_twists,
                dim=len(self.integrated_articulations),
                inputs=[
                    self.integrated_articulations,
                    solver._rod_chain_flag if chain else self.no_skip,
                    solver._dynamics_art_active,
                    m._fk_joint_order,
                    m.articulation_start,
                    solver._articulation_joint_end,
                    m.joint_parent,
                    m.joint_child,
                    m.joint_qd_start,
                    m.body_com,
                    solver._articulation_origin,
                    augmented._joint_S_s,
                    velocity,
                    state.body_q,
                ],
                outputs=[self.accumulated, result.body_qd],
                block_dim=solver._serial_block_dim(len(self.integrated_articulations)),
                device=m.device,
            )
        if self.all_spatial:
            wp.copy(result.joint_q, state.joint_q)
            wp.copy(result.joint_qd, state.joint_qd)
        if len(self.tree_joints):
            wp.launch(
                _integrate_relative_twists,
                dim=len(self.tree_joints),
                inputs=[
                    self.tree_joints,
                    m.joint_articulation,
                    solver._dynamics_art_active,
                    m.joint_parent,
                    m.joint_child,
                    m.body_flags,
                    m.joint_X_p,
                    m.joint_X_c,
                    m.body_com,
                    state.body_q,
                    result.body_qd,
                    m.joint_q_start,
                    m.joint_qd_start,
                    state.joint_q,
                    dt,
                ],
                outputs=[result.joint_q, result.joint_qd],
                device=m.device,
            )

    def commit(self, model, state):
        if self.loops is not None:
            self.write_loop_state(model, state, state.joint_q, state.joint_qd)
        if self.has_history:
            wp.launch(
                _commit_history,
                dim=len(self.joints),
                inputs=[
                    self.joints,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    state.joint_q,
                    model.joint_rest_q,
                    model.joint_target_ke,
                    model.joint_enabled,
                    self.history,
                ],
                device=model.device,
            )

    def reserve_history_snapshot(self):
        if (self.has_history or getattr(self, "carried", False)) and not self.history_snapshot:
            live = ()
            if self.has_history:
                live += (self.history.strain, self.history.stress, self.history.initialized)
            if getattr(self, "carried", False):
                # A coupled solver's restarted iteration starts from the carried momenta of the step's start.
                live += (self.carried_q, self.carried_qd, self.carried_twist)
            self.history_snapshot = tuple((field, wp.clone(field)) for field in live)

    def transfer_history(self, *, restore):
        for live, saved in self.history_snapshot:
            if restore:
                wp.copy(live, saved)
            else:
                wp.copy(saved, live)

    def reset(self, model, world_mask):
        if getattr(self, "carried", False):
            # A reset state carries no momenta from the previous episode.
            wp.launch(
                _invalidate_carried,
                dim=model.joint_count,
                inputs=[model.joint_world, world_mask, model.world_count, model.joint_q_start, self.carried_q],
                device=model.device,
            )
        if self.has_history:
            wp.launch(
                _reset_history,
                dim=len(self.joints),
                inputs=[
                    self.joints,
                    model.joint_world,
                    world_mask,
                    model.world_count,
                    self.history,
                ],
                device=model.device,
            )
