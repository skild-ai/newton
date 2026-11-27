# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Franka Bimanual Cable (coupling benchmark)

Two fixed-base Franka FR3 arms, side by side and both facing a working area, manipulate one cable
with Newton's GPU IK solver. Each task loads the coupling between articulated arms, the cable and
rigid contacts in a different way:

- ``twist``: both grippers hold the cable's ends with their wrists along the cable and roll them in
  opposite directions while closing in, so the twisted span buckles out of line.
- ``wrap``: one gripper pins the cable's start on a horizontal pole while the other carries the far
  end around the pole and along it, winding the cable into a helix.

The arms are simulated by MuJoCo or Steel with PD position targets from the IK result and the cable is
a VBD rod, coupled through gripper proxies (SolverCoupledProxy); with ``--solver steel`` one Steel solver
owns the arms and the rod. The same scene and trajectory run in every configuration, so they compare
the cost of the coupling with the native solve.

Success criteria:
- Two Franka arms hold a cable between their grippers above a table.
- ``twist``: the wrists roll in opposite directions, the grasps hold, and the twisted cable buckles.
- ``wrap``: the moving gripper orbits the pole and the cable winds around it more than once and a half.

Runs:
- `python -m newton.examples franka_bimanual_cable --task twist`
- `python -m newton.examples franka_bimanual_cable --task wrap --rigid-solver steel`
- `python -m newton.examples franka_bimanual_cable --task wrap --solver steel --solve-mode colored`
"""

from __future__ import annotations

import itertools
import math
import os

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.examples.cable import solver as cable_solver
from newton.solvers import SolverMuJoCo, SolverSteel, SolverVBD

# Seed configuration for the IK solves (7 arm + 2 finger).
FRANKA_Q = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.785, 0.04, 0.04]
FRANKA_COORDS = len(FRANKA_Q)

SURFACE_Z = 0.25  # table height; both arms are mounted on it [m]
BASES = ((0.0, 0.32), (0.0, -0.32))  # arm A (left), arm B (right) base positions on the table [m]
# The knot spreads a long cable wide: its arms stand farther apart, so their hands pass each other.
KNOT_BASES = ((0.0, 0.42), (0.0, -0.42))
TCP_OFFSET = 0.107  # hand-frame offset of the point between the fingers [m]

CABLE_RADIUS = 0.006
CABLE_CONTACT_KE = 1.0e3
CABLE_CONTACT_KD = 1.0e-1
# Contact gap of the cable and the jaws; segments are longer than two radii plus this gap, so only touching
# neighbours (filtered by their joint) would otherwise meet.
CABLE_GAP = 0.005
CABLE_SEGMENT = 0.022

GRIP_STIFFNESS = 1000.0
GRIP_FORCE = 1500.0
GRIP_OPEN = 0.03
KNOT_OPEN = 0.04  # the knot's regrasps open wide: the cable lies within a few centimetres of where it was put
# Durations of the knot's scripted tie and of its pull, planned from the cable's state [s].
_KNOT_TIE_TIME = 8.5
# From the start of the pull to the hands closing on the ends (1.2 + 0.5 + 0.3 + 0.5) [s].
_KNOT_GRAB_CLOSE = 2.5
_KNOT_PULL_TIME = 8.1  # 1.2 + 0.5 + 0.3 + 0.5 + 0.3 + 0.6 + 4.0 + 0.3 + 0.4
_KNOT_GRIP_IN = 8  # links from each end where the hands take the cable
_KNOT_GRIP_CLEAR = 0.07  # [m] from the gripped link to any other strand of cable
# [m] of cable the cinched overhand knot keeps (its loop alone takes ~11 cm of this 12 mm cable). Pulling farther
# apart only loads the grips and the knot's contacts.
_KNOT_TAKE_UP = 0.14
# From the grasp on (friction combines by average): the cable slides on itself (0.3) so the pull can cinch the
# knot, while the fingers (1.7) still hold it at 1.0. At 1.0 the knot's own friction outlasted the grips; the tie
# keeps 1.0, which it needs to form.
_KNOT_CABLE_MU = 0.3
_KNOT_FINGER_MU = 1.7

# Knot cable: Dahl bending friction (saturated plastic curvature [rad], memory length [rad]).
DAHL_EPS_MAX = 2.0
DAHL_TAU = 0.1

POLE_X = 0.45
POLE_Z = SURFACE_Z + 0.42
POLE_RADIUS = 0.008
# The pole ends just short of arm B's hand, so arm B's hand and forearm pass beyond its end as they orbit it.
POLE_Y = (-0.07, 0.30)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _axis_angle(axis, angle):
    s = math.sin(0.5 * angle)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle))


# Hand orientations (x, y, z, w): the hand's z axis is the approach direction of the fingers.
DOWN = (1.0, 0.0, 0.0, 0.0)  # hand z -> -z
ALONG_MINUS_Y = _axis_angle((1.0, 0.0, 0.0), 0.5 * math.pi)  # hand z -> -y
ALONG_PLUS_Y = _axis_angle((1.0, 0.0, 0.0), -0.5 * math.pi)  # hand z -> +y


@wp.kernel
def _set_targets(
    positions: wp.array[wp.vec3],
    rotations: wp.array[wp.vec4],
    fingers: wp.array[float],
    pos_a: wp.vec3,
    rot_a: wp.vec4,
    grip_a: float,
    pos_b: wp.vec3,
    rot_b: wp.vec4,
    grip_b: float,
):
    p = wp.tid()
    if p % 2 == 0:
        positions[p] = pos_a
        rotations[p] = rot_a
        fingers[p] = grip_a
    else:
        positions[p] = pos_b
        rotations[p] = rot_b
        fingers[p] = grip_b


@wp.kernel
def _scatter_targets(
    ik_q: wp.array2d[float],
    fingers: wp.array[float],
    coords: int,
    inv_dt: float,
    previous_q: wp.array2d[float],
    joint_target_q: wp.array2d[float],
    joint_target_qd: wp.array2d[float],
):
    # Each arm's joints follow the IK solution with its frame-to-frame rate as the velocity target, so the PD
    # drives track a moving target without the lag of a pure position servo. The arms' coordinates and degrees
    # of freedom coincide and lead each world's layout.
    p = wp.tid()
    world = p // 2
    base = (p % 2) * coords
    for c in range(coords - 2):
        q = ik_q[p, c]
        joint_target_q[world, base + c] = q
        joint_target_qd[world, base + c] = (q - previous_q[p, c]) * inv_dt
        previous_q[p, c] = q
    joint_target_q[world, base + coords - 2] = fingers[p]
    joint_target_q[world, base + coords - 1] = fingers[p]


@wp.kernel
def _apply_grips(
    substep: int,
    grip_joints: wp.array[int],
    targets: wp.array2d[wp.vec3],
    enabled: wp.array2d[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_X_p: wp.array[wp.transform],
):
    # One virtual-gripper closure per thread: engaged at this substep's target, or released.
    i = wp.tid()
    j = grip_joints[i]
    on = enabled[substep, i] != 0
    joint_enabled[j] = on
    if on:
        joint_X_p[j] = wp.transform(targets[substep, i], wp.transform_get_rotation(joint_X_p[j]))


def _find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


def _nlerp(a, b, t):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.dot(a, b) < 0.0:
        b = -b
    q = (1.0 - t) * a + t * b
    return q / np.linalg.norm(q)


def _linking_turns(points, quats):
    """Twist plus writhe of a segmented cable, in turns.

    Twist sums each joint's relative rotation about the segment axis (body ``+Z``); writhe is the discrete Gauss
    integral of the centerline.
    """
    twist = 0.0
    for q0, q1 in itertools.pairwise(quats):
        rel = wp.mul(wp.quat_inverse(wp.quat(*q0)), wp.quat(*q1))
        twist += 2.0 * math.atan2(float(rel[2]), float(rel[3]))
    mids = 0.5 * (points[1:] + points[:-1])
    steps = np.diff(points, axis=0)
    writhe = 0.0
    for i in range(len(steps)):
        for j in range(len(steps)):
            if abs(i - j) < 2:
                continue
            d = mids[i] - mids[j]
            writhe += float(np.dot(np.cross(steps[i], steps[j]), d)) / float(np.linalg.norm(d)) ** 3
    return twist / (2.0 * math.pi) + writhe / (4.0 * math.pi)


def _knot_determinant(points, seed=7):
    """Determinant of the knot formed by a cable closed far away (1: no knot, 3: an overhand / trefoil knot).

    Each end is extended radially away from the cable's centroid and the two extensions are joined far outside it;
    the closed curve is projected along a generic direction and the determinant of its coloring matrix is taken.
    """
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0)
    far = 50.0 * (np.linalg.norm(pts - c, axis=1).max() + 1.0)

    def out(p):
        d = p - c
        return c + far * d / max(np.linalg.norm(d), 1e-9)

    ring = [out(pts[-1]), c + far * np.cross(out(pts[-1]) - c, out(pts[0]) - c), out(pts[0])]
    if np.linalg.norm(ring[1] - c) < 1e-9:
        ring[1] = c + far * np.array([0.0, 0.0, 1.0])
    else:
        ring[1] = c + far * (ring[1] - c) / np.linalg.norm(ring[1] - c)
    loop = np.vstack([pts, ring])
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    xyz = loop @ q
    n = len(xyz)
    crossings = []  # (position along the under strand, position along the over strand)
    for i in range(n):
        a0, a1 = xyz[i], xyz[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b0, b1 = xyz[j], xyz[(j + 1) % n]
            da, db = a1[:2] - a0[:2], b1[:2] - b0[:2]
            den = da[0] * db[1] - da[1] * db[0]
            if abs(den) < 1e-14:
                continue
            w = b0[:2] - a0[:2]
            s = (w[0] * db[1] - w[1] * db[0]) / den
            t = (w[0] * da[1] - w[1] * da[0]) / den
            if 0.0 <= s < 1.0 and 0.0 <= t < 1.0:
                za, zb = a0[2] + s * (a1[2] - a0[2]), b0[2] + t * (b1[2] - b0[2])
                crossings.append((j + t, i + s) if za > zb else (i + s, j + t))
    if not crossings:
        return 1
    unders = sorted(u for u, _ in crossings)

    def arc(position):
        # Arcs run from one under-crossing to the next; arc k starts after the k-th under-crossing.
        return (int(np.searchsorted(unders, position, side="right")) - 1) % len(unders)

    k = len(crossings)
    m = np.zeros((k, k))
    for row, (under, over) in enumerate(crossings):
        before = arc(under - 1e-9)
        after = arc(under + 1e-9)
        m[row, arc(over)] += 2.0
        m[row, before] -= 1.0
        m[row, after] -= 1.0
    return round(abs(np.linalg.det(m[1:, 1:]))) if k > 1 else 1


class Track:
    """A hand's task-space keyframes: times [s], TCP positions [m], hand quaternions and finger targets [m]."""

    def __init__(self):
        self.times, self.positions, self.rotations, self.grips = [0.0], [], [], []

    def start(self, pos, rot, grip):
        self.positions.append(np.asarray(pos, dtype=np.float64))
        self.rotations.append(np.asarray(rot, dtype=np.float64))
        self.grips.append(float(grip))

    def to(self, duration, pos=None, rot=None, grip=None):
        self.times.append(self.times[-1] + float(duration))
        self.positions.append(self.positions[-1] if pos is None else np.asarray(pos, dtype=np.float64))
        self.rotations.append(self.rotations[-1] if rot is None else np.asarray(rot, dtype=np.float64))
        self.grips.append(self.grips[-1] if grip is None else float(grip))

    def sample(self, t):
        times = self.times
        t = min(max(t, 0.0), times[-1])
        i = int(np.searchsorted(times, t, side="right")) - 1
        i = min(max(i, 0), len(times) - 2) if len(times) > 1 else 0
        if len(times) == 1:
            return self.positions[0], self.rotations[0], self.grips[0]
        a = (t - times[i]) / max(times[i + 1] - times[i], 1e-9)
        pos = (1.0 - a) * self.positions[i] + a * self.positions[i + 1]
        rot = _nlerp(self.rotations[i], self.rotations[i + 1], a)
        grip = (1.0 - a) * self.grips[i] + a * self.grips[i + 1]
        return pos, rot, grip


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.task = args.task
        self.twist_roll = float(args.twist_roll)
        self.knot_handoff = not args.knot_control
        self.clamps = cable_solver.CableClamps(args)
        self.native = getattr(args, "solver", "coupled") == "steel"
        # Levenberg-Marquardt iterations per frame, warm-started from the previous frame's solution.
        self.ik_iterations = max(1, int(args.ik_iterations))
        self.rigid_solver = args.rigid_solver
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.world_count = max(1, int(args.world_count))
        # Under the immediate response eight substeps of six iterations hold both grasps; the propagation response's
        # split sweep needs 24 iterations and runs faster from cable_solver.ARM_PROPAGATION_WORLDS worlds. The knot keeps
        # the immediate response at any world count: it holds every grip (100 of 100 runs, about 21 ms per frame at one
        # world), while under the propagation response (8 substeps of 20 iterations, the tree propagating every sweep,
        # about 17 ms) a jaw loses the cable in about 2 runs of 100.
        if args.contact_response is None and args.task == "knot" and self.native:
            args.contact_response = "immediate"
        if self.native and args.task == "knot" and args.contact_response == "propagation":
            native_substeps, native_iterations = 8, 20
        elif self.native and cable_solver.arm_contact_response(args, self.world_count) == "propagation":
            native_substeps, native_iterations = 8, 24
        else:
            native_substeps, native_iterations = 8, 6
        self.sim_substeps = (
            int(args.substeps) if args.substeps is not None else (native_substeps if self.native else 10)
        )
        if args.steel_iterations is None:
            args.steel_iterations = native_iterations if self.native else 24
        self.sim_dt = self.frame_dt / self.sim_substeps
        # The knot's tie holds the cable with the virtual grippers' closures, which need fewer substeps than the
        # jaws' friction grasp of the pull: six tie the knot in every run (five miss about one in ten).
        tie_substeps = getattr(args, "tie_substeps", None)
        knot_tie = 6 if self.native and self.task == "knot" and args.contact_response == "propagation" else None
        self.tie_substeps = int(tie_substeps or knot_tie or self.sim_substeps)
        self.tie_dt = self.frame_dt / self.tie_substeps
        self.use_graph = bool(args.graph_capture)

        self._plan_task()
        self._build_ik_model()
        self.initial_q = self._solve_initial_pose()
        self._build_scene()
        self.use_graph = self.use_graph and self.device.is_cuda

        self.control = self.model.control()
        # Coupled: the scene pipeline carries the static contacts (table, pole) and the cable's own, and the proxies
        # the jaws'.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=None if self.native else self._static_shape_pairs(),
            # Native Steel warm starts the contacts from their matches (the jaws' squeeze carries over).
            **cable_solver.pipeline_options(args),
        )
        self.contacts = self.collision_pipeline.contacts()
        if self.native or self.rigid_solver == "steel":
            self.model.rigid_contact_max = self.contacts.rigid_contact_max
        self._build_solvers(args)
        self._build_ik()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        if not self.native:
            self.solver.prepare_contacts(self.contacts)

        newton.examples.configure_coupled_view(self, args)
        if self.world_count > 1:
            self.viewer.set_world_offsets((1.2, 1.4, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(1.5, -0.9, 0.95), pitch=-20.0, yaw=150.0)
            self.viewer.camera.look_at(wp.vec3(0.45, 0.0, 0.45))

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.capture()

    # ------------------------------------------------------------------
    # Task plans
    # ------------------------------------------------------------------
    def _plan_task(self):
        """Keyframes of both hands and the cable's initial centerline for the selected task."""
        r = CABLE_RADIUS
        grip_hold = r  # each finger at the cable radius: the jaws start on the cable, then squeeze it
        self.track_a, self.track_b = Track(), Track()
        self.bases = KNOT_BASES if self.task == "knot" else BASES
        self.pole = False
        self.anchor = False
        self.held = (0, -1)  # cable bodies the arms hold at the end of the task (None: free)
        self.ik_seeds = [list(FRANKA_Q), list(FRANKA_Q)]
        if self.task == "twist":
            z = SURFACE_Z + 0.25
            half = 0.20
            pa, pb = np.array([0.55, half, z]), np.array([0.55, -half, z])
            # The fingers pinch the end segments: the cable runs past each TCP by half a segment.
            self.cable_segments = round(2.0 * half / CABLE_SEGMENT) + 1
            seg = 2.0 * half / (self.cable_segments - 1)
            self.cable_points = np.linspace(
                pa + np.array([0.0, 0.5 * seg, 0.0]), pb - np.array([0.0, 0.5 * seg, 0.0]), self.cable_segments + 1
            )
            # Each wrist rolls about its own approach axis; the two axes point at each other, so equal rolls
            # twist the cable in opposite senses at its ends. Joint 7 follows the roll one to one, so each roll
            # starts where it leaves both wrists inside joint 7's range for the whole stroke.
            start_roll = (1.0, 2.4)
            roll_total = self.twist_roll
            rest = (ALONG_MINUS_Y, ALONG_PLUS_Y)

            def hand(arm, a):
                return _quat_mul(rest[arm], _axis_angle((0.0, 0.0, 1.0), start_roll[arm] + roll_total * a))

            self.track_a.start(pa, hand(0, 0.0), grip_hold)
            self.track_b.start(pb, hand(1, 0.0), grip_hold)
            # The jaws squeeze the cable before the wrists roll.
            self.track_a.to(0.6, grip=0.0)
            self.track_b.to(0.6, grip=0.0)
            closing = 0.09
            steps = 30
            for i in range(1, steps + 1):
                a = i / steps
                self.track_a.to(3.5 / steps, pos=pa - np.array([0.0, closing * a, 0.0]), rot=hand(0, a))
                self.track_b.to(3.5 / steps, pos=pb + np.array([0.0, closing * a, 0.0]), rot=hand(1, a))
            self.track_a.to(1.0)
            self.track_b.to(1.0)
            self.commanded_roll = 2.0 * abs(roll_total)
        elif self.task == "wrap":
            self.pole = True
            # Both hands point along the pole, away from their own arm, with the jaws closing along x: the hands
            # stay beside the pole's axis wherever they go, and the orbiting wrist never winds up.
            hand_a = _quat_mul(ALONG_MINUS_Y, _axis_angle((0.0, 0.0, 1.0), 0.5 * math.pi))
            hand_b = _quat_mul(ALONG_PLUS_Y, _axis_angle((0.0, 0.0, 1.0), 0.5 * math.pi))
            wound = POLE_RADIUS + r  # radius of the cable's centerline on the pole
            turns = 1.6
            phi0 = 0.55 * math.pi  # arm B starts just past the top of the pole
            r0 = 0.26
            # Arm B starts just past the pole's end and lays each turn toward arm A, away from that end.
            y_a, y_b0, pitch = 0.10, POLE_Y[0] + 0.035, -0.03
            # Arm A holds the cable's start above the pole; the cable hangs onto the pole and up to arm B.
            pa = np.array([POLE_X, y_a, POLE_Z + 0.12])

            def orbit(phi):
                # Arm B spirals in by the cable it winds onto the pole, so the free span stays taut.
                rad = r0 - wound * (phi - phi0)
                y = y_b0 - pitch * (phi - phi0) / (2.0 * math.pi)
                return np.array([POLE_X + rad * math.cos(phi), y, POLE_Z + rad * math.sin(phi)])

            pb = orbit(phi0)
            seg = CABLE_SEGMENT
            # The cable hangs straight down from arm A's jaws, along the pads rather than out past the fingertips.
            top = np.array([POLE_X, y_a, POLE_Z + wound + 0.002])
            out = (pb - top) / np.linalg.norm(pb - top)
            path = [pa + np.array([0.0, 0.0, 0.5 * seg]), top, pb + 0.5 * seg * out]
            self.cable_points, self.cable_segments = self._resample(path, seg)
            self.track_a.start(pa, hand_a, grip_hold)
            self.track_b.start(pb, hand_b, grip_hold)
            self.track_a.to(0.6, grip=0.0)
            self.track_b.to(0.6, grip=0.0)
            steps = int(48 * turns)
            for i in range(1, steps + 1):
                phi = phi0 + 2.0 * math.pi * turns * i / steps
                self.track_b.to(5.0 * turns / steps, pos=orbit(phi))
                self.track_a.to(5.0 * turns / steps)
            self.track_a.to(0.75)
            self.track_b.to(0.75)
            self.wrap_turns = turns
            # Arm B starts on the elbow branch that reaches the pole's far side without flipping.
            self.ik_seeds[1] = [1.1, 0.0, -0.9, -2.3, 1.9, 1.5, 1.5, 0.04, 0.04]
        elif self.task == "knot":
            self._plan_knot(r, grip_hold)
        else:
            raise ValueError(f"unknown task {self.task!r}")
        self.duration = max(self.track_a.times[-1], self.track_b.times[-1]) + getattr(self, "planned_tail", 0.0)

    def _plan_knot(self, r, grip_hold):
        """Overhand knot: virtual grippers tie it, then the hands pull it tight.

        The cable starts straight on the table. Virtual grippers (a pinned link moved along waypoints; the rest of
        the cable is physics) tie it as a hand would: one holds the left end; the other drags the right end around
        into a U and lifts it over the other leg and down past it, a crossed U. A second lifts the U's bottom (the
        ear) and the end is driven into the loop and out under it. Then everything lets go, and the hands take the
        two ends where they lie and pull them wide apart: the knot collapses tight at the center.
        """
        rest = SURFACE_Z + r
        length, x0 = 1.2, 0.62
        self.cable_points, self.cable_segments = self._resample(
            [np.array([x0, 0.5 * length, rest]), np.array([x0, -0.5 * length, rest])], CABLE_SEGMENT
        )
        self.held = (_KNOT_GRIP_IN, -1 - _KNOT_GRIP_IN)  # the hands take the ends this many links in
        z = SURFACE_Z + r + 0.002

        def down(yaw):
            return _quat_mul(DOWN, _axis_angle((0.0, 0.0, 1.0), yaw))

        up = np.array([0.0, 0.0, 0.10])
        # The hands wait high and out of the way while the cable is tied.
        self.track_a.start(np.array([0.30, 0.34, z + 0.25]), down(0.0), KNOT_OPEN)
        self.track_b.start(np.array([0.30, -0.34, z + 0.25]), down(0.0), KNOT_OPEN)
        self.tie_start, self.tie_end = 0.5, 0.5 + _KNOT_TIE_TIME
        settle = 1.0
        self.track_a.to(self.tie_end + settle)
        self.track_b.to(self.tie_end + settle)
        self._knot = {"z": z, "up": up, "down": down, "rest": rest}
        self.drives = []  # (link, Track) of the virtual grippers
        # The virtual grippers hand the ends over: they hold them until the hands have closed on them.
        self.drive_until = self.tie_end + settle + _KNOT_GRAB_CLOSE
        self.replans = [
            (self.tie_start, self._plan_tie_cross),
            (self.tie_start + 4.0, self._plan_tie_thread),
            (self.tie_end + settle, self._plan_knot_pull),
        ]
        self.planned_tail = _KNOT_PULL_TIME

    def _drive(self, link, times, points):
        """A virtual gripper on ``link`` through ``points`` [m] at ``times`` [s], eased between them."""
        track = Track()
        track.times = [float(times[0])]
        track.start(points[0], (0.0, 0.0, 0.0, 1.0), 0.0)
        for t0, t1, p in zip(times[:-1], times[1:], points[1:], strict=True):
            track.to(t1 - t0, pos=p)
        self.drives.append((link, track))

    def _plan_tie_cross(self):
        """Pin the left end; drag the right end into a U toward the arms, then slide it under the left leg (lifted
        into a small tent) and out past it. The end's crossings then alternate along it, under here, then over and
        under at the loop: an overhand knot."""
        rest = self._knot["rest"]
        pts = self._cable_centerline()
        n = len(pts)
        t0 = self.tie_start
        left, right = pts[0], pts[n - 1]
        self._drive(0, [t0, t0 + 4.0], [left, left])
        y_cross = 0.30
        u_end = np.array([0.36, 0.05, rest + 0.01])  # the right end beside the left leg: a U, bottom out at -y
        before = np.array([0.40, y_cross, rest + 0.01])
        past = np.array([0.80, y_cross, rest + 0.01])
        self._drive(n - 1, [t0, t0 + 1.4, t0 + 2.2, t0 + 3.6, t0 + 4.0], [right, u_end, before, past, past])
        # The left leg's link where the end passes under it lifts into a tent meanwhile.
        k = int(np.argmin(np.abs(pts[: n // 2, 1] - y_cross)))
        leg = pts[k]
        tent = leg + np.array([0.0, 0.0, 0.10])
        self._drive(k, [t0 + 1.4, t0 + 2.0, t0 + 3.6, t0 + 4.0], [leg, tent, tent, leg + np.array([0.0, 0.0, 0.01])])

    def _plan_tie_thread(self):
        """Lift the U's bottom (the ear) and drive the end into the loop and out under the ear, then let go."""
        rest = self._knot["rest"]
        pts = self._cable_centerline()
        n = len(pts)
        t0 = self.tie_start + 4.0
        self.drives = []
        left, end = pts[0], pts[n - 1]
        # The loop: the cable between the two spans that cross (seen from above).
        q = pts[:, :2]
        cross, lo, hi = None, 0, n - 1
        for i in range(n - 1):
            for j in range(i + 3, n - 1):
                da, db = q[i + 1] - q[i], q[j + 1] - q[j]
                den = da[0] * db[1] - da[1] * db[0]
                if abs(den) < 1e-12:
                    continue
                w = q[j] - q[i]
                sa, sb = (w[0] * db[1] - w[1] * db[0]) / den, (w[0] * da[1] - w[1] * da[0]) / den
                if 0.0 <= sa <= 1.0 and 0.0 <= sb <= 1.0 and (cross is None or j - i < hi - lo):
                    cross, lo, hi = q[i] + sa * da, i + 1, j
        if cross is None:
            cross, lo, hi = np.array([0.55, 0.15]), n // 4, n // 2
        loop = pts[lo : hi + 1]
        ear_k = lo + int(np.argmax(np.linalg.norm(loop[:, :2] - cross, axis=1)))
        ear = pts[ear_k]
        out_dir = (ear[:2] - cross) / max(np.linalg.norm(ear[:2] - cross), 1e-6)
        inside = cross + 0.45 * (ear[:2] - cross)
        lifted = ear + np.array([0.0, 0.0, 0.12])
        hold = self.drive_until
        self._drive(0, [t0, hold], [left, left])
        self._drive(ear_k, [t0, t0 + 0.8, t0 + 3.6, t0 + 4.2], [ear, lifted, lifted, ear + np.array([0, 0, 0.01])])
        hop = np.array([*inside, rest + 0.10])
        dip = np.array([*inside, rest + 0.01])
        # Well past the ear: the hand pulls a tail through the knot, not the end back out of it.
        under = np.array([*(ear[:2] + 0.22 * out_dir), rest + 0.01])
        self._drive(
            n - 1,
            [t0, t0 + 0.8, t0 + 1.6, t0 + 2.2, t0 + 3.6, hold],
            [end, end + np.array([0.0, 0.0, 0.10]), hop, dip, under, under],
        )

    def _grip_frame(self):
        """Every substep's virtual-gripper targets [m] and engagement for this frame, uploaded at once."""
        n = self.cable_bodies_per_world
        targets = np.zeros((self.tie_substeps, len(self.grip_joints), 3), dtype=np.float32)
        enabled = np.zeros((self.tie_substeps, len(self.grip_joints)), dtype=np.int32)
        for substep in range(self.tie_substeps):
            time = self.sim_time + substep * self.tie_dt
            active = {
                k: np.asarray(tr.sample(time)[0], dtype=np.float64)
                for k, tr in self.drives
                if tr.times[0] <= time <= tr.times[-1]
            }
            for k, p0 in self._reachable(active).items():
                for w in range(self.world_count):
                    targets[substep, w * n + k] = p0
                    enabled[substep, w * n + k] = 1
        if not hasattr(self, "_grip_targets"):
            self._grip_joint_ids = wp.array(self.grip_joints, dtype=int, device=self.device)
            self._grip_targets = wp.array(targets, dtype=wp.vec3, device=self.device)
            self._grip_enabled = wp.array(enabled, dtype=wp.int32, device=self.device)
        else:
            self._grip_targets.assign(targets)
            self._grip_enabled.assign(enabled)

    def _simulate_tie(self):
        """The tie's frame on the device: the hands' IK, then the virtual grippers before every substep."""
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iters)
        wp.launch(
            _scatter_targets,
            dim=2 * self.world_count,
            inputs=[self.ik_q, self.ik_fingers, FRANKA_COORDS, 1.0 / self.frame_dt, self.ik_previous_q],
            outputs=[self.control_joint_target_q, self.control_joint_target_qd],
            device=self.device,
        )
        copy_last = self.use_graph and self.tie_substeps % 2 == 1
        for substep in range(self.tie_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _apply_grips,
                dim=len(self.grip_joints),
                inputs=[substep, self._grip_joint_ids, self._grip_targets, self._grip_enabled],
                outputs=[self.model.joint_enabled, self.model.joint_X_p],
                device=self.device,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.tie_dt)
            if copy_last and substep == self.tie_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def _step_tie(self):
        """One frame of the tie: the hands' targets as usual, the virtual grippers applied before every substep."""
        if self.grip_joints and self.native:
            # The closures are re-anchored by a kernel, so the whole frame replays as one graph.
            self.update_ik_targets()
            self._grip_frame()
            if self.use_graph:
                if getattr(self, "_tie_graph", None) is None:
                    with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                        self._simulate_tie()
                    self._tie_graph = capture.graph
                with wp.ScopedDevice(self.device):
                    wp.capture_launch(self._tie_graph)
            else:
                self._simulate_tie()
            self._grips_engaged = True
            self.sim_time += self.frame_dt
            return
        self.update_ik_targets()
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iters)
        wp.launch(
            _scatter_targets,
            dim=2 * self.world_count,
            inputs=[self.ik_q, self.ik_fingers, FRANKA_COORDS, 1.0 / self.frame_dt, self.ik_previous_q],
            outputs=[self.control_joint_target_q, self.control_joint_target_qd],
            device=self.device,
        )
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._apply_drives(self.sim_time + substep * self.sim_dt, self.sim_dt)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if not self.native:
                newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            if self.use_graph and self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def _apply_drives(self, time, dt):
        """Move each driven link to its virtual gripper's position, with the gripper's velocity.

        On a stretchable cable the link is placed outright. On an inextensible one no link can be: its joints
        have no coordinates to take up the offset. Its gripper's ball closure is engaged at the gripper instead,
        and the solver brings the link there with the rest of the cable.
        """
        active = [(k, tr) for k, tr in self.drives if tr.times[0] <= time <= tr.times[-1]]
        n = self.cable_bodies_per_world
        if not active:
            if self.grip_joints and getattr(self, "_grips_engaged", False):
                self._engage_grips(np.zeros(len(self.grip_joints), dtype=bool), None)
            return
        if not hasattr(self, "_cable_arts"):
            arts = np.unique(self.model.joint_articulation.numpy()[self.cable_joints])
            self._cable_arts = wp.array(arts[arts >= 0], dtype=wp.int32, device=self.device)
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if self.grip_joints:
            targets = self._reachable({k: np.asarray(tr.sample(time)[0], dtype=np.float64) for k, tr in active})
            enabled = np.zeros(len(self.grip_joints), dtype=bool)
            frames = self.model.joint_X_p.numpy()
            for k, p0 in targets.items():
                for w in range(self.world_count):
                    enabled[w * n + k] = True
                    frames[self.grip_joints[w * n + k], :3] = p0
            self._engage_grips(enabled, frames)
        placed = [] if self.grip_joints else active
        if not placed:
            return
        for k, tr in placed:
            p0 = tr.sample(time)[0]
            p1 = tr.sample(min(time + dt, tr.times[-1]))[0]
            for w in range(self.world_count):
                b = self.cable_bodies[w * n + k]
                body_q[b, :3] = p0
                body_qd[b, :3] = (p1 - p0) / dt
                body_qd[b, 3:6] = 0.0
        self.state_0.body_q.assign(body_q)
        self.state_0.body_qd.assign(body_qd)
        if self.native and self._cable_arts.shape[0] > 0:
            newton.eval_ik(
                self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd, indices=self._cable_arts
            )

    def _reachable(self, targets):
        """Pull the grippers' targets [m] together until each pair lies within the cable between their links.

        An inextensible cable cannot reach two points farther apart than the length between the links held: the
        closures would fight, and the solve with them. Each excess is split between the two targets.
        """
        if not hasattr(self, "_link_spacing"):
            pts = self._cable_centerline()
            self._link_spacing = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        links = sorted(targets)
        for _ in range(8):
            moved = False
            for a_i, a in enumerate(links):
                for b in links[a_i + 1 :]:
                    gap = targets[b] - targets[a]
                    dist = float(np.linalg.norm(gap))
                    reach = 0.98 * self._link_spacing * (b - a)
                    if dist > reach:
                        shift = 0.5 * (dist - reach) * gap / dist
                        targets[a] = targets[a] + shift
                        targets[b] = targets[b] - shift
                        moved = True
            if not moved:
                break
        return targets

    def _engage_grips(self, enabled, frames):
        """Engage the virtual-gripper closures in ``enabled`` (per cable link), at ``frames`` [m] when given."""
        flags = self.model.joint_enabled.numpy()
        flags[self.grip_joints] = enabled
        self.model.joint_enabled.assign(flags)
        if frames is not None:
            self.model.joint_X_p.assign(frames)
        self._grips_engaged = bool(enabled.any())

    def _cable_centerline(self):
        """World 0's cable link positions, in order."""
        n = self.cable_bodies_per_world
        return self.state_0.body_q.numpy()[self.cable_bodies[:n], :3].astype(np.float64)

    @staticmethod
    def _yaw_across(points, k, reach=1):
        """The jaw yaw that closes across the cable at link k (yaw 0: the cable runs along x)."""
        a, b = points[max(k - reach, 0)], points[min(k + reach, len(points) - 1)]
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        while yaw > 0.5 * math.pi:
            yaw -= math.pi
        while yaw < -0.5 * math.pi:
            yaw += math.pi
        return yaw

    def _both(self, duration, a=None, b=None):
        self.track_a.to(duration, **(a or {}))
        self.track_b.to(duration, **(b or {}))

    def _slip_knot(self):
        """The knot's cable slides on itself from the grasp on; the fingers hold it as before (see _KNOT_CABLE_MU)."""
        if not self.inextensible:
            return
        body = self.model.shape_body.numpy()
        mu = self.model.shape_material_mu.numpy()
        mu[np.isin(body, self.cable_bodies)] = _KNOT_CABLE_MU
        mu[np.isin(body, self.gripper_bodies)] = _KNOT_FINGER_MU
        self.model.shape_material_mu.assign(mu)

    def _damp_taut_cable(self):
        """A hundred times the cable's bend and twist damping, from the grasp on.

        An inextensible cable pulled taut between the grips carries fast transverse waves that nothing else damps
        (a stretchable one loses them in its implicit stretch spring); they shake it beyond the looser grip. The
        light damping of the tie is kept until then: damped this heavily, the cable does not form the knot.
        """
        if not self.inextensible:
            return
        qd_start = self.model.joint_qd_start.numpy()
        rods = [j for j in self.cable_joints if self.model.joint_type.numpy()[j] == int(newton.JointType.ROD)]
        dofs = np.concatenate([np.arange(qd_start[j], qd_start[j + 1]) for j in rods])
        kd = self.model.joint_target_kd.numpy()
        kd[dofs] *= 100.0
        self.model.joint_target_kd.assign(kd)

    def _plan_knot_pull(self):
        """Take each end where it lies and pull the two apart: the knot closes.

        The hands grip at the table, near their fingertips, so they pull nearly level, along the cable leaving each
        grip: a pull that rises drags the cable down and out between the fingertips.
        """
        k = self._knot
        down, up, z = k["down"], k["up"], k["z"]
        self._damp_taut_cable()
        self._slip_knot()
        pts = self._cable_centerline()
        grips = []
        # Eight links (18 cm) in from each end: a grip near the very end loses it to the first centimetres of slip.
        n = len(pts)

        def clearance(i):
            # Distance in the table plane from link i to the nearest stretch of cable not next to it along the cable:
            # the jaws must close on one strand, not on the leg lying beside it.
            far = np.abs(np.arange(n) - i) > 4
            return float(np.min(np.linalg.norm(pts[far, :2] - pts[i, :2], axis=1)))

        def bend(i):
            # The cable's turn across the jaws [rad]: they close square to a straight stretch, and on a curl they
            # close askew, twist the cable as they close and let it slide.
            a, b = pts[i] - pts[max(i - 2, 0)], pts[min(i + 2, n - 1)] - pts[i]
            a[2] = b[2] = 0.0
            cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            return math.acos(min(1.0, max(-1.0, cos)))

        def pick(cands):
            # The first candidate (nearest the preferred spot) with room for the jaws on a straight stretch; else
            # the first with room; else the roomiest.
            for good in (
                lambda i: clearance(i) > _KNOT_GRIP_CLEAR and bend(i) < math.radians(20.0),
                lambda i: clearance(i) > _KNOT_GRIP_CLEAR,
            ):
                for i in cands:
                    if good(i):
                        return i
            return max(cands, key=clearance)

        # From the preferred eight links in, step inward first (more cable behind the jaws holds against axial
        # slip), and toward the end only if no spot inward has room.
        steps = list(range(_KNOT_GRIP_IN, _KNOT_GRIP_IN + 7)) + list(range(_KNOT_GRIP_IN - 1, 4, -1))
        ia = pick(steps)
        ib = pick([n - 1 - j for j in steps])
        self.held = (ia, ib - n)  # the success check follows the links actually taken
        for idx, inward in ((ia, 1), (ib, -1)):
            e = pts[idx].copy()
            e[2] = z
            # Outward along the cable at the grip: away from the knot.
            out = pts[idx] - pts[idx + 4 * inward]
            out[2] = 0.0
            out /= max(np.linalg.norm(out), 1e-9)
            # The jaws close the same with the hand turned half a turn; keep the yaw within [-135, 45] degrees so
            # the last wrist joint stays clear of its limit (near +80 degrees it ran out and the hand closed askew,
            # above the cable).
            yaw = self._yaw_across(pts, idx, reach=2)  # over four links: the stretch the jaws close on
            if yaw > 0.25 * math.pi:
                yaw -= math.pi
            grips.append((e, yaw, out))
        (ea, ya, oa), (eb, yb, ob) = grips
        both = self._both
        both(1.2, {"pos": ea + up, "rot": down(ya)}, {"pos": eb + up, "rot": down(yb)})
        both(0.5, {"pos": ea}, {"pos": eb})
        both(0.3)
        both(0.5, {"grip": 0.0}, {"grip": 0.0})
        both(0.3)
        pull = 0.22  # until the knot cinches: pulling on past it only raises the tension until a grip lets go
        if self.inextensible:
            # The cable cannot stretch: the grips may end no farther apart than the cable between them, less
            # what the cinched knot takes up. Past that the arms would only pull against each other.
            arc = float(np.linalg.norm(np.diff(pts[ia : ib + 1], axis=0), axis=1).sum())
            reach = arc - _KNOT_TAKE_UP
            lo, hi = 0.0, pull
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm((eb + mid * ob) - (ea + mid * oa)) <= reach:
                    lo = mid
                else:
                    hi = mid
            pull = max(lo, 0.02)
        lift = np.array([0.0, 0.0, 0.03])
        # A gentle start: the first centimetre slowly, so the tension does not jump from nothing.
        both(0.6, {"pos": ea + 0.01 * oa + lift}, {"pos": eb + 0.01 * ob + lift})
        both(4.0, {"pos": ea + pull * oa + lift}, {"pos": eb + pull * ob + lift})  # slow: the knot cinches
        # Ease off a centimetre: the knot stays cinched and the grips stop holding the peak tension.
        ease = pull - 0.01
        both(0.3, {"pos": ea + ease * oa + lift}, {"pos": eb + ease * ob + lift})
        both(0.4)

    @staticmethod
    def _resample(path, spacing):
        pts = [np.asarray(p, dtype=np.float64) for p in path]
        lengths = [np.linalg.norm(b - a) for a, b in itertools.pairwise(pts)]
        total = float(sum(lengths))
        n = max(2, round(total / spacing))
        out = []
        for k in range(n + 1):
            s = total * k / n
            for a, b, length in zip(pts[:-1], pts[1:], lengths, strict=True):
                if s <= length or b is pts[-1]:
                    out.append(a + (b - a) * min(s / max(length, 1e-12), 1.0))
                    break
                s -= length
        return np.asarray(out), n

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------
    @staticmethod
    def _add_franka(builder, base_xy):
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(base_xy[0], base_xy[1], SURFACE_Z), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
            joint_damping_mode="passive",
        )

    def _build_ik_model(self):
        # One Franka at the origin solves every arm's problem: targets are shifted into each arm's base frame.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, SURFACE_Z), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
            joint_damping_mode="passive",
        )
        builder.joint_q[:FRANKA_COORDS] = FRANKA_Q
        self.ik_model = builder.finalize()
        self.hand_body = _find_label_index(self.ik_model.body_label, "fr3_hand")

    def _make_ik_solver(self, problems, positions, rotations):
        pos_obj = ik.IKObjectivePosition(
            link_index=self.hand_body, link_offset=wp.vec3(0.0, 0.0, TCP_OFFSET), target_positions=positions
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=self.hand_body, link_offset_rotation=wp.quat_identity(), target_rotations=rotations
        )
        # One limit per DoF of the IK arm, shared by every problem.
        limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=10.0,
        )
        return ik.IKSolver(
            model=self.ik_model,
            n_problems=problems,
            objectives=[pos_obj, rot_obj, limits],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _local_targets(self, t):
        out = []
        for track, base in ((self.track_a, self.bases[0]), (self.track_b, self.bases[1])):
            pos, rot, grip = track.sample(t)
            out.append((pos - np.array([base[0], base[1], 0.0]), rot, grip))
        return out

    def _solve_initial_pose(self):
        device = self.ik_model.device
        targets = self._local_targets(0.0)
        positions = wp.array([wp.vec3(*t[0]) for t in targets], dtype=wp.vec3, device=device)
        rotations = wp.array([wp.vec4(*t[1]) for t in targets], dtype=wp.vec4, device=device)
        solver = self._make_ik_solver(2, positions, rotations)
        q = wp.array(np.asarray(self.ik_seeds, dtype=np.float32), dtype=float, device=device)
        for _ in range(8):
            solver.step(q, q, iterations=50)
        q_np = q.numpy()
        for arm in range(2):
            q_np[arm, FRANKA_COORDS - 2 :] = targets[arm][2]
        return q_np

    def _build_ik(self):
        device = self.device
        problems = 2 * self.world_count
        targets = self._local_targets(0.0)
        self.ik_positions = wp.array(
            [wp.vec3(*targets[p % 2][0]) for p in range(problems)], dtype=wp.vec3, device=device
        )
        self.ik_rotations = wp.array(
            [wp.vec4(*targets[p % 2][1]) for p in range(problems)], dtype=wp.vec4, device=device
        )
        self.ik_fingers = wp.array([targets[p % 2][2] for p in range(problems)], dtype=float, device=device)
        self.ik_q = wp.array(np.tile(self.initial_q, (self.world_count, 1)), dtype=float, device=device)
        self.ik_solver = self._make_ik_solver(problems, self.ik_positions, self.ik_rotations)
        self.ik_iters = self.ik_iterations
        self.control_joint_target_q = self.control.joint_target_q.reshape((self.world_count, -1))
        self.control_joint_target_qd = self.control.joint_target_qd.reshape((self.world_count, -1))
        self.ik_previous_q = wp.clone(self.ik_q)

    def update_ik_targets(self):
        (pa, ra, ga), (pb, rb, gb) = self._local_targets(self.sim_time)
        wp.launch(
            _set_targets,
            dim=2 * self.world_count,
            inputs=[
                self.ik_positions,
                self.ik_rotations,
                self.ik_fingers,
                wp.vec3(*pa),
                wp.vec4(*ra),
                float(ga),
                wp.vec3(*pb),
                wp.vec4(*rb),
                float(gb),
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _build_scene(self):
        template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        template.rigid_gap = 0.01
        if self.native:
            SolverSteel.register_custom_attributes(template)
        else:
            SolverMuJoCo.register_custom_attributes(template)
            SolverVBD.register_custom_attributes(template)
        self._emit_template(template)
        bodies, joints = template.body_count, template.joint_count

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = template.rigid_gap
        builder.replicate(template, copy_count=self.world_count)

        def expand(ids, stride):
            return [w * stride + i for w in range(self.world_count) for i in ids]

        self.arm_bodies = expand(self.arm_bodies, bodies)
        self.arm_joints = expand(self.arm_joints, joints)
        self.cable_bodies = expand(self.cable_bodies, bodies)
        self.cable_joints = expand(self.cable_joints, joints)
        self.grip_joints = expand(self.grip_joints, joints)
        self.gripper_bodies = expand(self.gripper_bodies, bodies)
        self.hand_bodies = expand(self.hand_bodies, bodies)
        gripper = set(self.gripper_bodies)
        arm = set(self.arm_bodies)
        for shape, body in enumerate(builder.shape_body):
            if body in gripper:
                builder.shape_material_ke[shape] = CABLE_CONTACT_KE
                builder.shape_material_kd[shape] = CABLE_CONTACT_KD
                builder.shape_material_mu[shape] = 1.0

        cfg = newton.ModelBuilder.ShapeConfig(ke=CABLE_CONTACT_KE, kd=CABLE_CONTACT_KD, mu=0.8, margin=0.0, gap=0.01)
        table = builder.add_ground_plane(height=SURFACE_Z, cfg=cfg, label="table")
        self.static_shapes = [table]
        # The arms stand on the table: their links never touch it, the cable and the jaws may.
        for shape, body in enumerate(builder.shape_body):
            if body in arm and body not in gripper:
                builder.add_shape_collision_filter_pair(table, shape)
        if self.pole:
            # A static horizontal pole along y.
            pole = builder.add_shape_capsule(
                body=-1,
                xform=wp.transform(
                    wp.vec3(POLE_X, 0.5 * (POLE_Y[0] + POLE_Y[1]), POLE_Z),
                    wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi),
                ),
                radius=POLE_RADIUS,
                half_height=0.5 * (POLE_Y[1] - POLE_Y[0]) - POLE_RADIUS,
                cfg=cfg,
                label="pole",
            )
            self.static_shapes.append(pole)
        if not self.native:
            builder.color()
        self.model = builder.finalize()
        self.device = self.model.device
        if self.task == "knot" and os.environ.get("KNOT_DAHL") == "1":
            # A plastic cable: Dahl friction in its bending keeps the shape it is bent into.
            material = self.model.steel if self.native else self.model.vbd
            for name, value in (("dahl_eps_max", DAHL_EPS_MAX), ("dahl_tau", DAHL_TAU)):
                values = getattr(material, name).numpy()
                values[self.cable_joints] = value
                getattr(material, name).assign(values)

    def _static_shape_pairs(self) -> wp.array:
        static = set(self.static_shapes)
        cable = set(self.cable_bodies)
        body = self.model.shape_body.numpy()
        pairs = [
            (int(a), int(b))
            for a, b in self.model.shape_contact_pairs.numpy()
            if {int(a), int(b)} & static or (body[a] in cable and body[b] in cable)
        ]
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device)

    def _emit_template(self, builder):
        arm_body_start, arm_joint_start = builder.body_count, builder.joint_count
        coord_start = builder.joint_coord_count
        for arm in range(2):
            self._add_franka(builder, self.bases[arm])
            c0 = coord_start + arm * FRANKA_COORDS
            builder.joint_q[c0 : c0 + FRANKA_COORDS] = self.initial_q[arm].tolist()
            builder.joint_target_q[c0 : c0 + FRANKA_COORDS] = self.initial_q[arm].tolist()
            d0 = arm * FRANKA_COORDS
            builder.joint_target_ke[d0 : d0 + 7] = [400.0] * 7
            builder.joint_target_kd[d0 : d0 + 7] = [80.0] * 7
            # The knot's regrasps hold the cable through turns of the wrist: they squeeze harder.
            # The knot's regrasps hold the cable through turns of the wrist: they squeeze harder.
            grip_ke = 3.0 * GRIP_STIFFNESS if self.task == "knot" else GRIP_STIFFNESS
            builder.joint_target_ke[d0 + 7 : d0 + 9] = [grip_ke] * 2
            builder.joint_target_kd[d0 + 7 : d0 + 9] = [100.0] * 2
            builder.joint_effort_limit[d0 : d0 + 4] = [87.0] * 4
            builder.joint_effort_limit[d0 + 4 : d0 + 7] = [12.0] * 3
            builder.joint_effort_limit[d0 + 7 : d0 + 9] = [GRIP_FORCE] * 2
            builder.joint_armature[d0 : d0 + 7] = [1.0e-3] * 7
            # The arm drives track the IK rate as well as its position (MuJoCo adds a velocity actuator for it).
            builder.joint_target_mode[d0 : d0 + 7] = [int(newton.JointTargetMode.POSITION_VELOCITY)] * 7
        self.arm_bodies = list(range(arm_body_start, builder.body_count))
        self.arm_joints = list(range(arm_joint_start, builder.joint_count))
        if not self.native:
            gravcomp = builder.custom_attributes["mujoco:gravcomp"]
            if gravcomp.values is None:
                gravcomp.values = {}
            for body in self.arm_bodies:
                gravcomp.values[body] = 1.0
        if self.native or self.rigid_solver == "steel":
            for body in self.arm_bodies:
                builder.body_disable_gravity[body] = True
        self.gripper_bodies = [
            b for b in self.arm_bodies if "hand" in builder.body_label[b] or "finger" in builder.body_label[b]
        ]
        self.hand_bodies = [b for b in self.arm_bodies if builder.body_label[b].endswith("fr3_hand")]

        cable_body_start, cable_joint_start = builder.body_count, builder.joint_count
        cfg = newton.ModelBuilder.ShapeConfig(
            density=200.0,
            ke=CABLE_CONTACT_KE,
            kd=CABLE_CONTACT_KD,
            mu=1.0,
            margin=0.0,
            gap=CABLE_GAP,
        )
        rod = newton.Rod(
            self.cable_points, rest_straight=getattr(self, "cable_rest_straight", True), radius=CABLE_RADIUS
        )
        bend = {"twist": 2.0e-3, "wrap": 1.0e-3, "knot": 3.0e-3}[self.task]
        # The knot's cable does not stretch: under native Steel its rod joints have no linear DOFs at all.
        self.inextensible = self.task == "knot" and self.native
        stretch = {} if self.inextensible else {"stretch_stiffness": 2.0e2, "stretch_damping": 1.0e-1}
        bend_damping = 2.0e-3 * bend
        builder.add_rod(
            rod=rod,
            cfg=cfg,
            inextensible=self.inextensible,
            **stretch,
            bend_stiffness=bend,
            bend_damping=bend_damping,
            twist_stiffness=bend,
            twist_damping=bend_damping,
            label="cable",
        )
        self.cable_bodies = list(range(cable_body_start, builder.body_count))
        self.cable_joints = list(range(cable_joint_start, builder.joint_count))
        self.cable_bodies_per_world = len(self.cable_bodies)
        self.grip_joints = []
        if self.inextensible:
            # Virtual grippers: a released ball closure from the world to each cable link. One is engaged and
            # re-anchored at its gripper every substep while that link is driven (Steel reads joint_enabled and
            # joint_X_p of a loop closure every step), so the link follows exactly, solved with the cable.
            for body in self.cable_bodies:
                position = wp.transform_get_translation(builder.body_q[body])
                self.grip_joints.append(
                    builder.add_joint_ball(
                        -1, body, parent_xform=wp.transform(position, wp.quat_identity()), enabled=False
                    )
                )
        if self.anchor:
            # The cable's start is fixed to the table.
            self.clamps.pin(builder, cable_body_start)
            if self.clamps.bodies:
                raise RuntimeError("the cable's anchor must be its rod root")

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def _build_solvers(self, args):
        if self.native:
            self.solver = cable_solver.make_solver(
                self.model,
                args,
                args.steel_iterations,
                contact_rows=int(args.contact_rows) if args.contact_rows else 24 * self.cable_bodies_per_world + 128,
                enable_joint_limits=True,
                coupled_contact_passes=int(args.coupled_contact_passes),
                # The knot's inextensible chain is rigidly coupled: every sweep's impulses must reach the whole tree.
                propagation_interval=1 if self.task == "knot" else 3,
            )
            return
        if self.rigid_solver == "steel":
            rigid_name = "steel"

            def rigid_solver(v):
                return SolverSteel(
                    v, iterations=int(args.steel_iterations), enable_joint_limits=True, articulation_constraint_max=96
                )

        else:
            rigid_name = "mjc"

            def rigid_solver(v):
                return SolverMuJoCo(
                    model=v,
                    solver="newton",
                    integrator="implicitfast",
                    cone="elliptic",
                    iterations=int(args.mujoco_iterations),
                    ls_iterations=int(args.mujoco_ls_iterations),
                    use_mujoco_contacts=False,
                    njmax=max(512, 128 * self.world_count),
                    nconmax=max(128, 32 * self.world_count),
                )

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name=rigid_name, solver=rigid_solver, bodies=self.arm_bodies, joints=self.arm_joints
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v, iterations=int(args.vbd_iterations), rigid_contact_history=False
                    ),
                    bodies=self.cable_bodies,
                    joints=self.cable_joints,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source=rigid_name,
                        destination="vbd",
                        bodies=self.gripper_bodies,
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                            model, broad_phase="explicit"
                        ),
                        collide_interval=1,
                    ),
                ],
                iterations=1,
            ),
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if self.use_graph:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iters)
        wp.launch(
            _scatter_targets,
            dim=2 * self.world_count,
            inputs=[self.ik_q, self.ik_fingers, FRANKA_COORDS, 1.0 / self.frame_dt, self.ik_previous_q],
            outputs=[self.control_joint_target_q, self.control_joint_target_qd],
            device=self.device,
        )
        copy_last = self.use_graph and self.sim_substeps % 2 == 1
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if not self.native:
                newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            if copy_last and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Steps planned from the cable's state are planned when they start.
        while getattr(self, "replans", None) and self.sim_time + 1e-9 >= self.replans[0][0]:
            self.replans.pop(0)[1]()
        if getattr(self, "drives", None) is not None and self.sim_time < self.drive_until + self.frame_dt:
            # While the virtual grippers tie the cable, they act before every substep.
            self._step_tie()
            return
        if self.grip_joints and getattr(self, "_grips_engaged", False):
            # The tie is over: release every virtual gripper before the captured steps.
            self._engage_grips(np.zeros(len(self.grip_joints), dtype=bool), None)
        self.update_ik_targets()
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    def _tcp_positions(self, body_q):
        out = []
        for hand in self.hand_bodies:
            q = body_q[hand]
            tf = wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:]))
            out.append(np.asarray(wp.transform_point(tf, wp.vec3(0.0, 0.0, TCP_OFFSET))))
        return np.asarray(out).reshape(self.world_count, 2, 3)

    def test_final(self):
        cable_solver.check_capacity(self.solver)
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(body_q)), "Body positions contain NaN or inf values"
        if self.sim_time < self.duration - 1.0e-6:
            raise AssertionError(f"Run through the whole task ({self.duration:.1f} s; ran {self.sim_time:.1f} s)")
        tcp = self._tcp_positions(body_q)
        n = self.cable_bodies_per_world
        for world in range(self.world_count):
            ids = self.cable_bodies[world * n : (world + 1) * n]
            pts = body_q[ids, :3]
            # The grasps that should hold the cable at the end do.
            for arm, index in enumerate(self.held):
                if index is None:
                    continue
                held = float(np.linalg.norm(pts[index] - tcp[world, arm]))
                assert held < 0.04, f"World {world}: arm {'AB'[arm]} lost the cable ({held * 100:.1f} cm off)"
            if self.task == "knot":
                dets = [_knot_determinant(pts, seed) for seed in range(3)]
                det = max(set(dets), key=dets.count)
                want = 1 if not self.knot_handoff else 3
                assert det == want, f"World {world}: knot determinant {det} (expected {want}: an overhand knot is 3)"
            elif self.task == "twist":
                a, b = tcp[world, 0], tcp[world, 1]
                axis = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                rel = pts - a
                off = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
                assert float(off.max()) > 0.02, f"World {world}: the twisted cable did not buckle ({off.max():.3f} m)"
                # The wrists' roll went into the cable: its twist plus writhe (the linking number) holds a good part
                # of the commanded roll instead of the cable spinning in the jaws.
                link = _linking_turns(pts, body_q[ids, 3:])
                need = max(0.2, 0.4 * self.commanded_roll / (2.0 * math.pi))
                assert abs(link) > need, f"World {world}: the cable holds {abs(link):.2f} turns of twist (< {need:.2f})"
            else:
                # Winding: the cable's angle around the pole axis (y), summed along the cable.
                ang = np.unwrap(np.arctan2(pts[:, 2] - POLE_Z, pts[:, 0] - POLE_X))
                winding = abs(float(ang[-1] - ang[0])) / (2.0 * math.pi)
                assert winding > 1.5, f"World {world}: the cable wound {winding:.2f} turns around the pole"
                # ... and lies on it rather than circling it in the air.
                radial = np.hypot(pts[:, 0] - POLE_X, pts[:, 2] - POLE_Z)
                on_pole = int(np.count_nonzero(radial < POLE_RADIUS + CABLE_RADIUS + 0.012))
                assert on_pole >= 6, f"World {world}: only {on_pole} cable segments rest on the pole"

    @staticmethod
    def task_frames(args) -> int:
        """Frames that run the selected task through its last keyframe."""
        plan = Example.__new__(Example)
        plan.task, plan.twist_roll, plan.knot_handoff = args.task, float(args.twist_roll), not args.knot_control
        plan._plan_task()
        return math.ceil(plan.duration * 60) + 1

    @staticmethod
    def create_parser():
        parser = cable_solver.create_parser(baseline="coupled")
        parser.set_defaults(contact_response=None)
        newton.examples.add_coupled_view_args(parser)
        newton.examples.add_world_count_arg(parser)
        parser.add_argument("--task", choices=("twist", "wrap", "knot"), default="twist", help="Bimanual task.")
        parser.add_argument(
            "--knot-control", action="store_true", help="Knot: skip the handoff (a negative control; no knot forms)."
        )
        parser.add_argument(
            "--rigid-solver", choices=("mujoco", "steel"), default="mujoco", help="Solver that owns the arms."
        )
        parser.add_argument(
            "--twist-roll", type=float, default=-3.0, help="Twist: each wrist's roll about the cable [rad]."
        )
        parser.add_argument("--substeps", type=int, default=None, help="Substeps per frame.")
        parser.add_argument(
            "--tie-substeps", type=int, default=None, help="Knot: substeps per frame while the virtual grippers tie it."
        )
        parser.add_argument("--contact-rows", type=int, default=None, help="Native Steel: per-world contact rows.")
        parser.add_argument("--steel-iterations", type=int, default=None, help="Steel PGS iterations.")
        parser.add_argument("--vbd-iterations", type=int, default=20, help="VBD iterations per coupled substep.")
        parser.add_argument("--mujoco-iterations", type=int, default=100, help="MuJoCo solver iterations.")
        parser.add_argument("--mujoco-ls-iterations", type=int, default=20, help="MuJoCo line-search iterations.")
        parser.add_argument(
            "--coupled-contact-passes", type=int, default=4, help="Native Steel: extra jaw-contact passes."
        )
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable graph capture.",
        )
        parser.add_argument("--ik-iterations", type=int, default=8, help="IK iterations per frame (warm-started).")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=Example.task_frames(parser.parse_known_args()[0]))
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
