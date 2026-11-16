# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-style fixed and spatial tendons in SolverSteel, checked against closed-form mechanics."""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverSteel
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_test_devices

G = 9.81
_HEAD = (
    '<option gravity="0 0 -9.81"><flag contact="disable"/></option>'
    '<default><geom contype="0" conaffinity="0"/></default>'
)


def _slider(name, pos, axis="0 0 1", mass=1.0):
    return (
        f'<body name="{name}" pos="{pos}"><joint name="{name}" type="slide" axis="{axis}"/>'
        f'<geom type="box" size="0.05 0.05 0.05" mass="{mass}"/><site name="{name}_s" pos="0 0 0.05"/></body>'
    )


def _frame(children):
    return f'<body name="frame"><geom type="sphere" size="0.02" mass="0.1"/>{children}</body>'


def _atwood_mjcf(m1, m2, r=0.1, zp=2.5, z0=1.0):
    length = 2.0 * (zp - (z0 + 0.05)) + math.pi * r
    return f"""<mujoco>{_HEAD}<worldbody>
      <geom name="pulley" type="cylinder" size="{r} 0.02" pos="0 0 {zp}" euler="90 0 0"/>
      <site name="side" pos="0 0 {zp + 0.3}"/>
      {_frame(_slider("m1", f"{-r} 0 {z0}", mass=m1) + _slider("m2", f"{r} 0 {z0}", mass=m2))}
    </worldbody><tendon><spatial limited="true" range="0 {length:.9f}">
      <site site="m1_s"/><geom geom="pulley" sidesite="side"/><site site="m2_s"/></spatial></tendon></mujoco>"""


class _Run:
    """A one-world model from MJCF stepped by SolverSteel, with joint access by name."""

    def __init__(
        self,
        mjcf,
        device,
        *,
        solve_mode="serial",
        qd=None,
        q=None,
        ctrl=None,
        joint_f=None,
        parent_body=False,
        gravity=-G,
        contacts=False,
        mjcf_kwargs=None,
        **solver_kwargs,
    ):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, gravity))
        parent = -1
        if parent_body:
            parent = builder.add_link(mass=0.0)
            builder.add_articulation([builder.add_joint_fixed(parent=-1, child=parent)])
        builder.add_mjcf(mjcf, parent_body=parent, **(mjcf_kwargs or {}))
        builder.joint_armature = [0.0] * len(builder.joint_armature)
        self.model = builder.finalize(device=device)
        self.labels = list(self.model.joint_label)
        joint_q, joint_qd = self.model.joint_q.numpy(), self.model.joint_qd.numpy()
        for values, target, start in ((q, joint_q, "joint_q_start"), (qd, joint_qd, "joint_qd_start")):
            for name, value in (values or {}).items():
                first = getattr(self.model, start).numpy()[self.joint(name)]
                target[first : first + np.size(value)] = value
        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)
        if solve_mode == "colored":
            solver_kwargs.setdefault("articulated_contact_response", "propagation")
        with warnings.catch_warnings():
            # solve_mode="colored" on CPU warns that it runs the serial sweep; the CPU cases cover exactly that.
            warnings.filterwarnings(
                "ignore", message="SolverSteel: solve_mode='colored' runs", category=newton.exceptions.NewtonWarning
            )
            self.solver = SolverSteel(self.model, solve_mode=solve_mode, **solver_kwargs)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.pipeline = newton.CollisionPipeline(self.model) if contacts else None
        self.contacts = self.pipeline.contacts() if contacts else None
        if ctrl is not None:
            self.control.mujoco.ctrl.assign(ctrl)
        if joint_f:
            forces = self.control.joint_f.numpy()
            for name, value in joint_f.items():
                forces[self.model.joint_qd_start.numpy()[self.joint(name)]] = value
            self.control.joint_f.assign(forces)

    def joint(self, name):
        return next(i for i, label in enumerate(self.labels) if label.endswith(name))

    def step(self, dt, steps):
        for _ in range(steps):
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def q(self, name, offset=0):
        return float(self.state_0.joint_q.numpy()[self.model.joint_q_start.numpy()[self.joint(name)] + offset])

    def qd(self, name, offset=0):
        return float(self.state_0.joint_qd.numpy()[self.model.joint_qd_start.numpy()[self.joint(name)] + offset])


_MODES = ("serial", "colored")


def test_tendon_atwood_machine(test, device):
    """A rope over a cylinder wrap: a = (m1 - m2) g / (m1 + m2) and tension 2 m1 m2 g / (m1 + m2)."""
    m1, m2, dt, steps = 3.0, 1.0, 1.0e-3, 300
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(_atwood_mjcf(m1, m2), device, solve_mode=mode, iterations=32)
            run.step(dt, steps - 100)
            # one step's impulse carries the float32 position round-off over dt^2; average the force instead
            tension = 0.0
            for _ in range(100):
                run.step(dt, 1)
                tension -= float(run.solver.tendon_forces(dt).numpy()[0, 0]) / 100.0
            accel = (m1 - m2) * G / (m1 + m2)
            # semi-implicit Euler integrates a constant acceleration exactly in velocity
            test.assertAlmostEqual(run.qd("m1"), -accel * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(run.qd("m2"), accel * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(tension, 2.0 * m1 * m2 * G / (m1 + m2), delta=0.02)
            length = float(run.solver.tendon_lengths().numpy()[0])
            test.assertAlmostEqual(length, 2.0 * 1.45 + math.pi * 0.1, delta=1.0e-3)


def test_tendon_slack_rope_jerks_plastically(test, device):
    """A thrown mass slackens the rope (free fall), which then snaps taut with an inelastic impulse.

    The default speed-aware limit gate creates the rope's row before it snaps taut, so it matches the
    ungated (``inf``) run and the analytic impulse.
    """
    m1, m2, v0, dt = 1.5, 1.0, 2.0, 1.0e-3
    t_star = v0 / G
    for gap in (None, float("inf")):
        with test.subTest(tendon_limit_activation_gap=gap):
            run = _Run(_atwood_mjcf(m1, m2), device, qd={"m2": v0}, iterations=32, tendon_limit_activation_gap=gap)
            run.step(dt, int(0.15 / dt))
            test.assertAlmostEqual(run.qd("m1"), -G * 0.15, delta=2.0e-3)  # no rope force while slack
            test.assertAlmostEqual(run.qd("m2"), v0 - G * 0.15, delta=2.0e-3)
            run.step(dt, int(0.15 / dt))
            impulse = v0 / (1.0 / m1 + 1.0 / m2)
            accel = (m1 - m2) * G / (m1 + m2)
            after = 0.3 - t_star
            test.assertAlmostEqual(run.qd("m1"), -v0 + impulse / m1 - accel * after, delta=0.02)
            test.assertAlmostEqual(run.qd("m2"), impulse / m2 + accel * after, delta=0.02)
            length = float(run.solver.tendon_lengths().numpy()[0])
            test.assertLess(length - float(run.model.mujoco.tendon_range.numpy()[0][1]), 1.0e-5)


def test_tendon_pulley_divisor_tackle(test, device):
    """A 2:1 tackle, L = |A - s2| + |B - s1| / 2, moves the effort mass twice as far as the load."""
    m1, m2, dt, steps = 1.0, 3.0, 1.0e-3, 300
    length = (2.5 - 1.05) + (2.5 - 1.25) / 2.0
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      <site name="A" pos="0.3 0 2.5"/><site name="B" pos="-0.3 0 2.5"/>
      {_frame(_slider("m1", "-0.3 0 1.2", mass=m1) + _slider("m2", "0.3 0 1.0", mass=m2))}
    </worldbody><tendon><spatial limited="true" range="0 {length:.9f}">
      <site site="A"/><site site="m2_s"/><pulley divisor="2"/><site site="B"/><site site="m1_s"/>
    </spatial></tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, iterations=32)
            run.step(dt, steps)
            accel = (m2 - 2.0 * m1) * G / (2.0 * m1 + 0.5 * m2)
            test.assertAlmostEqual(run.qd("m1"), accel * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(run.qd("m2"), -0.5 * accel * steps * dt, delta=2.0e-3)


def test_tendon_fixed_spring_damper(test, device):
    """A fixed tendon L = q1 + 2 q2 with stiffness and damping rings as the damped oscillator of L."""
    m1, m2, k, b, q10, dt = 1.0, 2.0, 50.0, 1.0, 0.1, 5.0e-4
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0", mass=m1) + _slider("m2", "0 0.2 1", axis="1 0 0", mass=m2))}
    </worldbody><tendon><fixed stiffness="{k}" damping="{b}" springlength="0">
      <joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed></tendon></mujoco>"""
    inv_mass = 1.0 / m1 + 4.0 / m2
    wn = math.sqrt(k * inv_mass)
    zeta = b * inv_mass / (2.0 * wn)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    for mode, passive in ((m, p) for m in _MODES for p in ("implicit_damping", "implicit")):
        with test.subTest(solve_mode=mode, tendon_passive_mode=passive):
            run = _Run(mjcf, device, solve_mode=mode, q={"m1": q10}, iterations=32, tendon_passive_mode=passive)
            for t in (0.2, 0.4, 0.6):
                run.step(dt, int(round(0.2 / dt)))
                exact = q10 * math.exp(-zeta * wn * t) * (math.cos(wd * t) + zeta * wn / wd * math.sin(wd * t))
                length = run.q("m1") + 2.0 * run.q("m2")
                # implicit Euler adds O(wn dt) numerical damping
                test.assertAlmostEqual(length, exact, delta=3.0e-3)
            # the tendon force acts along J = (1, 2), so the momentum combination 2 p1 - p2 stays zero
            test.assertAlmostEqual(m1 * run.qd("m1") * 2.0 - m2 * run.qd("m2"), 0.0, delta=1.0e-4)


def test_tendon_friction_loss_sticks_and_slips(test, device):
    """Tendon friction loss holds an incline below its bound and slows a slip above it by f / m."""
    theta, dt, steps = math.radians(30.0), 1.0e-3, 400
    axis = f"{math.cos(theta)} 0 {-math.sin(theta)}"
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("stick", "0 -0.4 1", axis=axis) + _slider("slip", "0 0 1", axis=axis) + _slider("stop", "0 0.4 1", axis="1 0 0"))}
    </worldbody><tendon>
      <fixed frictionloss="6"><joint joint="stick" coef="1"/></fixed>
      <fixed frictionloss="3"><joint joint="slip" coef="1"/></fixed>
      <fixed frictionloss="2"><joint joint="stop" coef="1"/></fixed>
    </tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, qd={"stop": 1.0}, iterations=32)
            run.step(dt, steps)
            test.assertAlmostEqual(run.qd("stick"), 0.0, delta=1.0e-4)
            test.assertAlmostEqual(run.qd("slip"), (G * math.sin(theta) - 3.0) * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(run.qd("stop"), 1.0 - 2.0 * steps * dt, delta=2.0e-3)
            run.step(dt, 200)  # past t = m v0 / f = 0.5 s the slide has stopped and stays stopped
            test.assertAlmostEqual(run.qd("stop"), 0.0, delta=1.0e-4)


def test_tendon_across_articulations_is_ignored(test, device):
    """A tendon coupling two articulations warns and is not enforced."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_slider("a", "-0.2 0 1")}{_slider("b", "0.2 0 1")}
    </worldbody><tendon><fixed stiffness="100"><joint joint="a" coef="1"/><joint joint="b" coef="1"/></fixed></tendon></mujoco>"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = _Run(mjcf, device, iterations=8)
    test.assertTrue(
        any(
            "cross-articulation" in str(w.message) and issubclass(w.category, newton.exceptions.NewtonWarning)
            for w in caught
        )
    )
    test.assertIsNone(run.solver.tendon_lengths())
    run.step(1.0e-3, 100)
    test.assertAlmostEqual(run.qd("a"), -G * 0.1, delta=1.0e-3)


def _winch_mjcf(actuator):
    return f"""<mujoco>{_HEAD}<worldbody><site name="top" pos="0 0 2.5"/>
      {_frame(_slider("m", "0 0 1"))}
    </worldbody><tendon><spatial name="rope"><site site="top"/><site site="m_s"/></spatial></tendon>
    <actuator>{actuator}</actuator></mujoco>"""


def test_tendon_motor_lifts_a_mass(test, device):
    """A tendon motor's force acts along dL/dq: a negative (shortening) force lifts a hanging mass."""
    pull, dt, steps = 14.715, 1.0e-3, 300
    mjcf = _winch_mjcf('<motor tendon="rope" ctrlrange="-100 100"/>')
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, ctrl=[-pull], iterations=32)
            run.step(dt, steps)
            test.assertAlmostEqual(run.qd("m"), (pull - G) * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(float(run.solver.tendon_actuator_forces().numpy()[0]), -pull, delta=1.0e-3)


def test_tendon_servo_saturates_at_its_force_range(test, device):
    """A saturated tendon servo reports its force-range bound and accelerates just below the bound's rate.

    Like a joint drive's effort limit, the range clamps the explicit force while the servo's stiffness and
    damping stay implicit in the mass matrix, which slows the saturated lift slightly (2 % here).
    """
    kp, fmax, dt = 200.0, 30.0, 1.0e-3
    mjcf = _winch_mjcf(
        f'<general tendon="rope" biastype="affine" gainprm="{kp}" biasprm="0 {-kp} -20" ctrlrange="0 3" '
        f'forcerange="{-fmax} {fmax}"/>'
    )
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, ctrl=[1.15], iterations=32)  # 0.3 m short of L = 1.45
            run.step(dt, 50)
            test.assertAlmostEqual(float(run.solver.tendon_actuator_forces().numpy()[0]), -fmax, delta=1.0e-3)
            saturated = (fmax - G) * 50 * dt
            test.assertLess(run.qd("m"), saturated)
            test.assertGreater(run.qd("m"), 0.95 * saturated)


def test_tendon_armature_couples_joint_inertia(test, device):
    """Fixed-tendon armature adds a J^T J to the mass matrix, so a force on q1 alone also accelerates q2."""
    a, dt, steps = 0.5, 1.0e-3, 200
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0") + _slider("m2", "0 0.2 1", axis="1 0 0", mass=2.0))}
    </worldbody><tendon><fixed armature="{a}"><joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed>
    </tendon></mujoco>"""
    jac = np.array([1.0, 2.0])
    accel = np.linalg.solve(np.diag([1.0, 2.0]) + a * np.outer(jac, jac), [1.0, 0.0])
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, joint_f={"m1": 1.0}, iterations=8)
            run.step(dt, steps)
            test.assertAlmostEqual(run.qd("m1"), accel[0] * steps * dt, delta=1.0e-4)
            test.assertAlmostEqual(run.qd("m2"), accel[1] * steps * dt, delta=1.0e-4)


def test_tendon_implicit_spring_survives_large_steps(test, device):
    """With tendon_passive_mode="implicit" a stiff tendon spring stays bounded at omega * dt = 4.6."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0", mass=0.02) + _slider("m2", "0 0.2 1", axis="1 0 0", mass=0.04))}
    </worldbody><tendon><fixed stiffness="2000" springlength="0">
      <joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed></tendon></mujoco>"""
    run = _Run(mjcf, device, q={"m1": 0.01}, iterations=8, tendon_passive_mode="implicit")
    run.step(1.0 / 120.0, 60)  # omega = 548 rad/s, dt = 8.3 ms
    test.assertLessEqual(abs(run.q("m1") + 2.0 * run.q("m2")), 0.01)


def test_tendon_passive_mode_is_validated(test, device):
    """An unknown tendon_passive_mode raises."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1"))}</worldbody>
    <tendon><fixed stiffness="1"><joint joint="m" coef="1"/></fixed></tendon></mujoco>"""
    with test.assertRaises(ValueError):
        _Run(mjcf, device, tendon_passive_mode="explicit")


def test_tendon_lengths_include_tendons_without_rows(test, device):
    """Tendons without a limit, spring, friction loss, or actuator still report their length, as MuJoCo's ten_length."""
    dt = 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0") + _slider("m2", "0 0.2 1", axis="1 0 0"))}
    </worldbody><tendon><fixed><joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed>
      <fixed armature="0.5"><joint joint="m1" coef="-1"/></fixed>
      <spatial><site site="m1_s"/><site site="m2_s"/></spatial></tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, joint_f={"m1": 1.0}, iterations=8)
            run.step(dt, 99)
            q1, q2 = run.q("m1"), run.q("m2")  # lengths are evaluated at the input state of the step
            run.step(dt, 1)
            test.assertGreater(q1, 1.0e-3)
            lengths = run.solver.tendon_lengths().numpy()
            np.testing.assert_allclose(lengths, [q1 + 2.0 * q2, -q1, math.hypot(q2 - q1, 0.4)], atol=1.0e-5)


def test_tendon_rows_beyond_the_contact_budget_are_kept(test, device):
    """More tendon rows than articulation_constraint_max: every tendon still holds its limit."""
    count, dt, steps = 40, 1.0e-3, 200
    sliders = "".join(_slider(f"m{i}", f"0 {0.2 * i} 1", axis="1 0 0") for i in range(count))
    tendons = "".join(
        f'<fixed limited="true" range="-0.05 0.05"><joint joint="m{i}" coef="1"/></fixed>' for i in range(count)
    )
    mjcf = f"<mujoco>{_HEAD}<worldbody>{_frame(sliders)}</worldbody><tendon>{tendons}</tendon></mujoco>"
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, joint_f={f"m{i}": 10.0 for i in range(count)}, iterations=16)
            run.step(dt, steps)
            q = np.array([run.q(f"m{i}") for i in range(count)])
            # a free 1 kg slider pushed by 10 N would travel 0.2 m; the limit holds it at 0.05 m
            np.testing.assert_allclose(q, 0.05, atol=2.0e-3)


def test_tendon_length_is_reported_when_its_rows_are_dropped(test, device):
    """A tendon whose limit rows do not fit the dense row capacity still reports this step's length.

    Mimic rows are reserved before the tendon rows. Two of them leave one free row of the three
    (``articulation_constraint_max=1`` plus the tendon's two), so the tendon's pair is dropped whole.
    """
    sliders = "".join(_slider(f"s{i}", f"0 {0.3 * i} 1") for i in range(4))
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(sliders)}</worldbody>
      <tendon><fixed limited="true" range="-5 5"><joint joint="s0" coef="1"/><joint joint="s1" coef="1"/></fixed>
      </tendon><equality><joint joint1="s2" joint2="s0"/><joint joint1="s3" joint2="s1"/></equality></mujoco>"""
    q = {"s0": 0.3, "s1": 0.2, "s2": 0.3, "s3": 0.2}
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(
                mjcf, device, solve_mode=mode, q=q, gravity=0.0, articulation_constraint_max=1,
                tendon_limit_activation_gap=float("inf"),
            )  # fmt: skip
            capture = StdOutCapture()  # the step prints the row-overflow warning
            capture.begin()
            try:
                run.step(1.0e-3, 1)
                wp.synchronize()
            finally:
                capture.end()
            test.assertTrue(run.solver.constraint_overflow.numpy()[0])
            test.assertAlmostEqual(float(run.solver.tendon_lengths().numpy()[0]), 0.5, delta=1.0e-6)


def test_tendon_limit_rows_are_gated_by_distance(test, device):
    """A limited tendon far from its range owns no limit row; one pushed into its bound gets a row that holds it.

    A gated tendon whose range is wider than twice its gap reserves one limit row instead of two, so the dense
    row capacity grows by one row per tendon instead of two.
    """
    dt, steps = 1.0e-3, 200
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("far", "0 -0.2 1", axis="1 0 0") + _slider("near", "0 0.2 1", axis="1 0 0"))}
    </worldbody><tendon>
      <fixed limited="true" range="-0.5 0.5"><joint joint="far" coef="1"/></fixed>
      <fixed limited="true" range="-0.05 0.05"><joint joint="near" coef="2"/></fixed>
    </tendon></mujoco>"""
    for mode, gap in ((m, g) for m in _MODES for g in (None, 0.01, float("inf"))):
        with test.subTest(solve_mode=mode, tendon_limit_activation_gap=gap):
            run = _Run(
                mjcf,
                device,
                solve_mode=mode,
                q={"far": 0.1},
                joint_f={"far": 1.0, "near": 10.0},
                iterations=16,
                tendon_limit_activation_gap=gap,
            )
            gated = gap is None or gap < float("inf")
            test.assertEqual(run.solver.articulation_constraint_max, 32 + (2 if gated else 4))
            run.step(dt, steps)
            slots = run.solver._tendon.slot.numpy()
            # the far slider covers 0.02 m of its 0.4 m way to the bound
            test.assertEqual(slots[0] < 0, gated)
            test.assertGreaterEqual(slots[1], 0)
            test.assertAlmostEqual(run.q("near"), 0.025, delta=1.0e-3)
            q_far = run.q("far")
            test.assertAlmostEqual(q_far, 0.1 + 0.5 * (steps * dt) ** 2, delta=1.0e-3)
            run.step(dt, 1)
            np.testing.assert_allclose(run.solver.tendon_lengths().numpy(), [q_far, 0.05], atol=1.0e-4)


def test_tendon_limit_gate_keeps_the_rows_of_mimic_coupled_tendons(test, device):
    """Under the default gate, tendons moving a mimic-coupled joint keep both limit rows; the others reserve one.

    The mimic row moves its joints inside the solve, where the unconstrained velocity the gate reads cannot see it.
    """
    mjcf = f"""<mujoco>{_HEAD}<worldbody><site name="w" pos="0 0.6 3"/>
      {_frame(_slider("a", "0 -0.2 1", axis="1 0 0") + _slider("b", "0 0.2 1", axis="1 0 0") + _slider("c", "0 0.6 1", axis="1 0 0"))}
    </worldbody><equality><joint joint1="b" joint2="a" polycoef="0 1 0 0 0"/></equality><tendon>
      <fixed limited="true" range="-1 1"><joint joint="a" coef="1"/></fixed>
      <fixed limited="true" range="-1 1"><joint joint="c" coef="1"/></fixed>
      <spatial limited="true" range="0 5"><site site="w"/><site site="a_s"/></spatial>
      <spatial limited="true" range="0 5"><site site="w"/><site site="c_s"/></spatial>
    </tendon></mujoco>"""
    for gap, extra in ((None, 2 + 1 + 2 + 1), (0.02, 4), (float("inf"), 8)):
        with test.subTest(tendon_limit_activation_gap=gap):
            run = _Run(mjcf, device, tendon_limit_activation_gap=gap)
            test.assertEqual(run.solver.articulation_constraint_max, 32 + extra)


def test_tendon_friction_follows_an_upper_only_limit_row(test, device):
    """The friction row keeps the tendon Jacobian's sign when the gate gives the tendon only its upper limit row.

    The slider sits near the upper bound of a wide range, so it reserves and activates the upper row alone. A push
    away from the bound is held by friction, which must push back along +L, and the slide must stay stopped.
    """
    dt, push = 1.0e-3, 2.0
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody><tendon>
      <fixed limited="true" range="-1 0.05" frictionloss="5"><joint joint="m" coef="1"/></fixed>
    </tendon></mujoco>"""
    for mode, gap in ((m, g) for m in _MODES for g in (None, float("inf"))):
        with test.subTest(solve_mode=mode, tendon_limit_activation_gap=gap):
            run = _Run(
                mjcf,
                device,
                solve_mode=mode,
                q={"m": 0.04},
                joint_f={"m": -push},
                iterations=16,
                tendon_limit_activation_gap=gap,
            )
            run.step(dt, 20)
            test.assertAlmostEqual(run.qd("m"), 0.0, delta=1.0e-5)
            test.assertAlmostEqual(float(run.solver.tendon_forces(dt).numpy()[0, 2]), push, delta=1.0e-3)


def test_tendon_spring_is_slack_inside_its_dead_band(test, device):
    """A tendon spring inside its springlength dead band exerts no force in either passive mode.

    The slider moves at 1 m/s from L = 0.5 and stays inside the band [0, 2] for the whole run, so an undamped
    spring must leave its velocity unchanged and report zero spring force.
    """
    dt, steps = 0.01, 10
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody><tendon>
      <fixed stiffness="1000" springlength="0 2"><joint joint="m" coef="1"/></fixed>
    </tendon></mujoco>"""
    for passive in ("implicit_damping", "implicit"):
        with test.subTest(tendon_passive_mode=passive):
            run = _Run(mjcf, device, q={"m": 0.5}, qd={"m": 1.0}, iterations=8, tendon_passive_mode=passive)
            run.step(dt, steps)
            test.assertAlmostEqual(run.qd("m"), 1.0, delta=1.0e-6)
            test.assertAlmostEqual(float(run.solver.tendon_forces(dt).numpy()[0, 1]), 0.0, delta=1.0e-6)


def _guyed_slider_speed(q, q_start, qd_start, armature=1.0):
    """Speed of the unit slider of the guyed-slider model at ``q`` after starting at ``q_start`` with ``qd_start``.

    The tendon runs from the world site, 1.45 m above the slider's site, to the slider at x = 1 + q, so
    ``J = dL/dq = x / sqrt(x^2 + 1.45^2)``. No force does work along the slider, so the kinetic energy
    ``(1 + armature J^2) qd^2 / 2`` is conserved and the armature slows the slider as J grows.
    """

    def inertia(x):
        return 1.0 + armature * x * x / (x * x + 1.45**2)

    return qd_start * math.sqrt(inertia(1.0 + q_start) / inertia(1.0 + q))


def test_tendon_armature_history_restarts_on_reset(test, device):
    """A reset, or notify_state_changed, restarts the spatial-tendon armature dJ/dt history.

    After a step from rest, the slider is teleported and reset. The next step must match a fresh solver started at
    the teleported state; a stale previous Jacobian would turn the teleport into an armature impulse.
    """
    dt = 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody><site name="w" pos="0 0 2.5"/>{_frame(_slider("m", "1 0 1", axis="1 0 0"))}
    </worldbody><tendon><spatial armature="1"><site site="w"/><site site="m_s"/></spatial></tendon></mujoco>"""
    fresh = _Run(mjcf, device, q={"m": 1.0}, qd={"m": 1.0}, iterations=8)
    fresh.step(dt, 1)
    for how in ("reset", "masked reset", "notify_state_changed"):
        with test.subTest(how=how):
            run = _Run(mjcf, device, iterations=8)
            run.step(dt, 1)
            q, qd = run.state_0.joint_q.numpy(), run.state_0.joint_qd.numpy()
            q[run.model.joint_q_start.numpy()[run.joint("m")]] = 1.0
            qd[run.model.joint_qd_start.numpy()[run.joint("m")]] = 1.0
            run.state_0.joint_q.assign(q)
            run.state_0.joint_qd.assign(qd)
            newton.eval_fk(run.model, run.state_0.joint_q, run.state_0.joint_qd, run.state_0)
            mask = wp.array(np.ones(run.model.world_count + 1, dtype=bool), dtype=wp.bool, device=device)
            if how == "reset":
                run.solver.reset(run.state_0)
            elif how == "masked reset":
                run.solver.reset(run.state_0, world_mask=mask)
            else:
                run.solver.notify_state_changed(run.state_0, world_mask=mask)
            run.step(dt, 1)
            test.assertAlmostEqual(run.qd("m"), fresh.qd("m"), delta=1.0e-6)
            test.assertAlmostEqual(run.q("m"), fresh.q("m"), delta=1.0e-6)
            # From the restarted history the armature term then slows the slider as energy conservation requires:
            # 6.4e-3 m/s over these steps, which a solver without the tendon term would not lose.
            run.step(dt, 99)
            test.assertAlmostEqual(run.qd("m"), _guyed_slider_speed(run.q("m"), 1.0, 1.0), delta=2.0e-4)
            test.assertLess(run.qd("m"), 1.0 - 2.0e-3)


def test_tendon_margin_and_routed_tendons_warn(test, device):
    """Steel warns about what it ignores: a limited tendon's margin and the routed tendons of add_tendon."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody>
    <tendon><fixed name="t" limited="true" range="-1 1" margin="0.1"><joint joint="m" coef="1"/></fixed></tendon>
    </mujoco>"""
    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "margin is not supported.*'t'"):
        _Run(mjcf, device, iterations=4)
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, 0.5)), mass=1.0)
    builder.add_tendon([newton.TendonGuide(body=-1, offset=(0.0, 0.0, 1.0)), newton.TendonGuide(body=body)])
    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "1 routed tendons"):
        SolverSteel(builder.finalize(device=device))


def test_tendon_position_dampratio_is_critically_damped(test, device):
    """A tendon position actuator's dampratio resolves to MuJoCo's kv = 2 dampratio sqrt(kp m) at the rest state.

    The reflected inertia of L = q1 + 2 q2 with tendon armature a is ``m = 1 / (1/m1 + 4/m2) + a``, so dampratio 1
    makes L a critically damped oscillator: L(t) = u + (L0 - u)(1 + wn t) exp(-wn t), with wn = sqrt(kp / m).
    """
    m1, m2, kp, u, armature, dt = 1.0, 2.0, 30.0, 0.5, 0.25, 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0", mass=m1) + _slider("m2", "0 0.2 1", axis="1 0 0", mass=m2))}
    </worldbody><tendon><fixed name="t" armature="{armature}"><joint joint="m1" coef="1"/><joint joint="m2" coef="2"/>
    </fixed></tendon><actuator><position tendon="t" kp="{kp}" dampratio="1"/></actuator></mujoco>"""
    wn = math.sqrt(kp / (1.0 / (1.0 / m1 + 4.0 / m2) + armature))
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, ctrl=[u], iterations=16)
            elapsed = 0.0
            for t in (0.25, 0.5, 1.0, 2.0):
                run.step(dt, int(round((t - elapsed) / dt)))
                elapsed = t
                exact = u - u * (1.0 + wn * t) * math.exp(-wn * t)
                # implicit Euler adds O(wn dt) numerical damping
                test.assertAlmostEqual(run.q("m1") + 2.0 * run.q("m2"), exact, delta=5.0e-3)
            test.assertTrue(np.isfinite(run.state_0.joint_qd.numpy()).all())


def test_tendon_dampratio_inertia_is_taken_at_the_reference_pose(test, device):
    """A dampratio's reflected inertia is evaluated at MuJoCo's qpos0, not at the model's initial joint_q.

    The tendon's moment arm on the hinge changes with the angle, so its reflected inertia does too. Two solvers
    built from models that differ only in their initial hinge angle must resolve the same damping gain, and so
    trace the same motion from a common start state.
    """
    dt, steps = 1.0e-3, 200
    mjcf = f"""<mujoco>{_HEAD}<worldbody><site name="w" pos="-0.2 0 1.3"/><body name="arm" pos="0 0 1">
      <joint name="h" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"/>
      <site name="a" pos="0.15 0 0.05"/></body></worldbody><tendon><spatial name="t"><site site="w"/><site site="a"/>
    </spatial></tendon><actuator><position tendon="t" kp="30" dampratio="1"/></actuator></mujoco>"""
    angles = []
    for start in (0.0, 0.9):
        run = _Run(mjcf, device, q={"h": start}, ctrl=[0.3], iterations=8)
        run.state_0.joint_q.zero_()
        newton.eval_fk(run.model, run.state_0.joint_q, run.state_0.joint_qd, run.state_0)
        run.solver.reset(run.state_0)
        run.step(dt, steps)
        angles.append(run.q("h"))
    test.assertGreater(abs(angles[0]), 1.0e-2)
    test.assertAlmostEqual(angles[1], angles[0], delta=1.0e-6)


def test_tendon_dampratio_leaves_the_model_unchanged(test, device):
    """Resolving dampratio writes the solver's copy of biasprm, so a later kp change re-resolves the same ratio."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody>
    <tendon><fixed name="t"><joint joint="m" coef="1"/></fixed></tendon>
    <actuator><position tendon="t" kp="30" dampratio="1"/></actuator></mujoco>"""
    run = _Run(mjcf, device)
    mujoco = run.model.mujoco
    test.assertEqual(float(mujoco.actuator_biasprm.numpy()[0, 2]), 1.0)
    # the slider's reflected inertia is 1 kg
    test.assertAlmostEqual(float(run.solver._tendon.biasprm.numpy()[0, 2]), -2.0 * math.sqrt(30.0), places=4)
    gain, bias = mujoco.actuator_gainprm.numpy(), mujoco.actuator_biasprm.numpy()
    gain[0, 0], bias[0, 1] = 120.0, -120.0
    mujoco.actuator_gainprm.assign(gain)
    mujoco.actuator_biasprm.assign(bias)
    run.solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES)
    test.assertEqual(float(mujoco.actuator_biasprm.numpy()[0, 2]), 1.0)
    test.assertAlmostEqual(float(run.solver._tendon.biasprm.numpy()[0, 2]), -2.0 * math.sqrt(120.0), places=4)


def test_tendon_auto_spring_rests_at_the_joint_rest_pose(test, device):
    """An automatic springlength is the tendon's length at the joints' rest pose (MuJoCo's qpos_spring).

    The slider starts 0.15 m from its ``springref``, so a fixed or a spatial tendon spring swings it about the
    rest pose as x = rest + (q0 - rest) cos(w t). A rest length taken at the initial joint_q would leave it still.
    """
    rest, q0, k, dt, t = 0.1, 0.25, 100.0, 5.0e-4, 0.1
    tendons = {
        "fixed": '<fixed stiffness="{k}"><joint joint="m" coef="1"/></fixed>',
        "spatial": '<spatial stiffness="{k}"><site site="w"/><site site="m_s"/></spatial>',
    }
    exact = rest + (q0 - rest) * math.cos(math.sqrt(k) * t)
    for kind, tendon in tendons.items():
        mjcf = f"""<mujoco>{_HEAD}<worldbody><site name="w" pos="0 0 1"/>
          <body name="frame"><geom type="sphere" size="0.02" mass="0.1"/><body name="m" pos="1 0 1">
          <joint name="m" type="slide" axis="1 0 0" springref="{rest}"/><geom type="box" size="0.05 0.05 0.05" mass="1"/>
          <site name="m_s" pos="0 0 0"/></body></body>
        </worldbody><tendon>{tendon.format(k=k)}</tendon></mujoco>"""
        for mode in _MODES:
            with test.subTest(tendon=kind, solve_mode=mode):
                run = _Run(mjcf, device, solve_mode=mode, q={"m": q0}, iterations=8)
                run.step(dt, int(round(t / dt)))
                test.assertAlmostEqual(run.q("m"), exact, delta=2.0e-3)
                # A new rest pose moves the rest length with it.
                run.model.joint_rest_q.fill_(q0)
                run.solver.notify_model_changed(newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES)
                run.state_0.joint_q.fill_(q0)
                run.state_0.joint_qd.zero_()
                newton.eval_fk(run.model, run.state_0.joint_q, run.state_0.joint_qd, run.state_0)
                run.solver.reset(run.state_0)
                run.step(dt, 20)
                test.assertAlmostEqual(run.q("m"), q0, delta=1.0e-6)


def test_tendon_fixed_length_includes_the_joint_reference(test, device):
    """A fixed tendon's length is sum(c * qpos) with qpos = joint_q + ref, as MuJoCo's ten_length."""
    ref, k, dt = 0.2, 100.0, 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody><body name="frame"><geom type="sphere" size="0.02" mass="0.1"/>
      <body name="m" pos="0 0 1"><joint name="m" type="slide" axis="1 0 0" ref="{ref}"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/></body></body>
    </worldbody><tendon><fixed stiffness="{k}" springlength="0"><joint joint="m" coef="1"/></fixed>
      <fixed limited="true" range="0.25 1"><joint joint="m" coef="1"/></fixed></tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, iterations=16)
            run.step(dt, 1)
            np.testing.assert_allclose(run.solver.tendon_lengths().numpy(), [ref, ref], atol=1.0e-6)
            # The spring pulls the length -k ref; the lower limit at 0.25 pushes back harder, so the slider moves
            # +0.05 m to it. With the ref dropped, the spring is relaxed and the limit sits 0.25 m away.
            run.step(dt, 300)
            test.assertAlmostEqual(run.q("m") + ref, 0.25, delta=2.0e-3)


def test_tendon_actuator_force_range_caps_the_summed_force(test, device):
    """A tendon's actuatorfrcrange scales its actuators' forces so their sum stays in range, as MuJoCo."""
    dt, steps = 1.0e-3, 100
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody>
    <tendon><fixed name="t" actuatorfrcrange="-1 1"><joint joint="m" coef="1"/></fixed></tendon>
    <actuator><motor tendon="t"/><motor tendon="t" gear="2"/></actuator></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, ctrl=[10.0, 10.0], iterations=8)
            run.step(dt, steps)
            # both actuator forces are 10, sum to 20 and scale to 0.5; the joint force is 1 * 0.5 + 2 * 0.5 N
            np.testing.assert_allclose(run.solver.tendon_actuator_forces().numpy(), [0.5, 0.5], atol=1.0e-6)
            test.assertAlmostEqual(run.qd("m"), 1.5 * steps * dt, delta=1.0e-5)


def test_tendon_auto_limit_follows_autolimits(test, device):
    """A tendon with limited="auto" is limited by its range only while the model's autolimits is set.

    The slider moves at 1 m/s toward a range bound 0.1 m away. With autolimits off, the automatic tendon keeps
    going, as tendon actuators' automatic control and force ranges do, and an explicit limited="true" still stops.
    """
    dt, steps = 1.0e-3, 300
    for limited, autolimits, stops in (("auto", True, True), ("auto", False, False), ("true", False, True)):
        mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody><tendon>
          <fixed limited="{limited}" range="-1 0.1"><joint joint="m" coef="1"/></fixed></tendon></mujoco>"""
        with test.subTest(limited=limited, autolimits=autolimits):
            run = _Run(mjcf, device, qd={"m": 1.0}, iterations=8)
            run.model.mujoco.autolimits.fill_(autolimits)
            run.solver = SolverSteel(run.model, iterations=8)
            run.step(dt, steps)
            test.assertAlmostEqual(run.q("m"), 0.1 if stops else steps * dt, delta=1.0e-3)


def test_tendon_force_range_scales_before_actuator_clamps(test, device):
    """The tendon's actuatorfrcrange scales the unclamped actuator forces; each forcerange clamps afterwards.

    Forces 10 and 0.5 sum to 10.5 and scale by 1 / 10.5 into the [-1, 1] cap, which leaves both inside their own
    [-2, 2] range: 0.952 and 0.048, so 0.952 + 2 * 0.048 = 1.048 N on the joint, as MuJoCo's mj_fwdActuation.
    Clamping first would give 2 and 0.5, scaled to 0.8 and 0.2 and 1.2 N.
    """
    dt, steps = 1.0e-3, 100
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody>
    <tendon><fixed name="t" actuatorfrcrange="-1 1"><joint joint="m" coef="1"/></fixed></tendon>
    <actuator><motor tendon="t" forcerange="-2 2"/><motor tendon="t" gear="2" forcerange="-2 2"/></actuator>
    </mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, ctrl=[10.0, 0.5], iterations=8)
            run.step(dt, steps)
            forces = np.array([10.0, 0.5]) / 10.5
            np.testing.assert_allclose(run.solver.tendon_actuator_forces().numpy(), forces, atol=1.0e-6)
            test.assertAlmostEqual(run.qd("m"), (forces[0] + 2.0 * forces[1]) * steps * dt, delta=1.0e-5)


def test_tendon_property_changes_reach_the_solver(test, device):
    """notify_model_changed(TENDON_PROPERTIES) re-derives the spring terms; edits that add tendon rows raise.

    The second tendon starts with zero stiffness. Giving it a stiffness, or moving its springlength, must change
    the next step's acceleration to -k (L - L_s) / m along L = q.
    """
    dt = 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("a", "0 -0.2 1", axis="1 0 0") + _slider("b", "0 0.2 1", axis="1 0 0"))}
    </worldbody><tendon><fixed stiffness="10" springlength="0"><joint joint="a" coef="1"/></fixed>
      <fixed springlength="0"><joint joint="b" coef="1"/></fixed></tendon></mujoco>"""
    flags = newton.ModelFlags.TENDON_PROPERTIES
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, q={"b": 0.1}, iterations=8)
            stiffness = run.model.mujoco.tendon_stiffness.numpy()
            stiffness[1] = 50.0
            run.model.mujoco.tendon_stiffness.assign(stiffness)
            run.solver.notify_model_changed(flags)
            run.step(dt, 1)
            # implicit damping mode: the spring is explicit, the slider started at rest
            test.assertAlmostEqual(run.qd("b"), -50.0 * 0.1 * dt, delta=1.0e-6)
            band = run.model.mujoco.tendon_springlength.numpy()
            band[1] = (0.3, 0.3)
            run.model.mujoco.tendon_springlength.assign(band)
            run.solver.notify_model_changed(flags)
            qd, q = run.qd("b"), run.q("b")
            run.step(dt, 1)
            test.assertAlmostEqual(run.qd("b") - qd, -50.0 * (q - 0.3) * dt, delta=1.0e-6)
            limited = run.model.mujoco.tendon_limited.numpy()
            limited[1] = 1
            run.model.mujoco.tendon_limited.assign(limited)
            with test.assertRaises(NotImplementedError):
                run.solver.notify_model_changed(flags)


def test_tendon_range_edits_report_the_reserved_rows(test, device):
    """A limit range edited within its reserved rows takes effect; one that needs more rows says so.

    A gated tendon whose range is wider than twice its activation gap reserves one limit row. Moving the bound
    keeps that row, so the slider stops at the new bound. Narrowing the range below twice the gap would need two
    rows, which raises with that reason rather than claiming that the set of limited tendons changed.
    """
    dt = 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1", axis="1 0 0"))}</worldbody><tendon>
      <fixed limited="true" range="-1 1"><joint joint="m" coef="1"/></fixed></tendon></mujoco>"""
    flags = newton.ModelFlags.TENDON_PROPERTIES
    run = _Run(mjcf, device, qd={"m": 1.0}, iterations=8, tendon_limit_activation_gap=0.05)
    run.model.mujoco.tendon_range.assign(np.array([[-1.0, 0.2]], dtype=np.float32))
    run.solver.notify_model_changed(flags)
    run.step(dt, 400)
    test.assertAlmostEqual(run.q("m"), 0.2, delta=1.0e-3)
    run.model.mujoco.tendon_range.assign(np.array([[0.17, 0.23]], dtype=np.float32))
    with test.assertRaisesRegex(NotImplementedError, "how many limit rows a tendon reserves"):
        run.solver.notify_model_changed(flags)


def _fresh_copy(run, mode, **solver_kwargs):
    """A solver constructed from ``run``'s current model, stepping a copy of its current state."""
    if mode == "colored":
        solver_kwargs.setdefault("articulated_contact_response", "propagation")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="SolverSteel: solve_mode='colored' runs", category=newton.exceptions.NewtonWarning
        )
        solver = SolverSteel(run.model, solve_mode=mode, **solver_kwargs)
    state_0, state_1 = run.model.state(), run.model.state()
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        wp.copy(getattr(state_0, name), getattr(run.state_0, name))
    return solver, state_0, state_1


def _step_alongside(run, fresh, state_0, state_1, dt, steps):
    """Step ``run`` and a fresh solver from the same state; return the fresh solver's final state."""
    run.step(dt, steps)
    for _ in range(steps):
        fresh.step(state_0, state_1, run.control, None, dt)
        state_0, state_1 = state_1, state_0
    return state_0


def test_tendon_stiffness_switched_on_by_notify(test, device):
    """A spring given at runtime to a tendon on an articulation without tendon mass terms rings as the closed form.

    ``L = qa + 2 qb`` on a two-slider chain (``H = [[2, 1], [1, 1]]``) whose articulation size has no tendon mass
    terms at construction (only a separate one-slider articulation has a damped tendon), so the notification moves
    that size onto the tendon mass path (and off the fused CRBA-Cholesky kernel with parallel streams). From the
    switch state ``(q_s, v_s)`` the motion is ``q_s + v_s t + H^-1 J^T (L - L_s - dL_s t) / (J H^-1 J^T)`` with
    ``L = L_s cos wt + dL_s / w sin wt`` and ``w^2 = k J H^-1 J^T``, and it matches a solver constructed with the
    spring. Switched off, the tendon keeps its zero mass terms and the motion is free; switched on again, nothing
    has to change.
    """
    k, dt, steps = 50.0, 1.0e-3, 600
    J, H = np.array([1.0, 2.0]), np.array([[2.0, 1.0], [1.0, 1.0]])
    chain = _slider("a", "0 0 1", axis="1 0 0")[: -len("</body>")] + _slider("b", "0 0 0.2", axis="1 0 0") + "</body>"
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{chain}{_slider("anchor", "0 1 1", axis="1 0 0")}</worldbody><tendon>
      <fixed damping="1"><joint joint="anchor" coef="1"/></fixed>
      <fixed springlength="0"><joint joint="a" coef="1"/><joint joint="b" coef="2"/></fixed></tendon></mujoco>"""
    flags = newton.ModelFlags.TENDON_PROPERTIES
    for mode, streams in [(m, s) for m in _MODES for s in (False, True)]:
        with test.subTest(solve_mode=mode, use_parallel_streams=streams):
            options = {"iterations": 8, "use_parallel_streams": streams}
            run = _Run(mjcf, device, solve_mode=mode, q={"a": 0.1, "b": -0.05}, qd={"a": 0.3, "b": 0.1}, **options)
            run.step(dt, 100)
            q_s, v_s = np.array([run.q("a"), run.q("b")]), np.array([run.qd("a"), run.qd("b")])
            np.testing.assert_allclose(v_s, (0.3, 0.1), atol=1.0e-6)  # free motion before the switch
            run.model.mujoco.tendon_stiffness.assign(np.array([0.0, k], dtype=np.float32))
            run.solver.notify_model_changed(flags)
            fresh = _step_alongside(run, *_fresh_copy(run, mode, **options), dt, steps)
            np.testing.assert_array_equal(run.state_0.joint_q.numpy(), fresh.joint_q.numpy())
            np.testing.assert_array_equal(run.state_0.joint_qd.numpy(), fresh.joint_qd.numpy())
            Hinv_J = np.linalg.solve(H, J)
            m_eff = float(J @ Hinv_J)
            w, t = math.sqrt(k * m_eff), steps * dt
            L_s, dL_s = float(J @ q_s), float(J @ v_s)
            L = L_s * math.cos(w * t) + dL_s / w * math.sin(w * t)
            q = q_s + v_s * t + Hinv_J * (L - L_s - dL_s * t) / m_eff
            # symplectic Euler's O((w dt)^2) phase error over 0.6 s
            np.testing.assert_allclose([run.q("a"), run.q("b")], q, atol=2.0e-4)
            run.model.mujoco.tendon_stiffness.assign(np.zeros(2, dtype=np.float32))
            run.solver.notify_model_changed(flags)
            v_off = np.array([run.qd("a"), run.qd("b")])
            run.step(dt, 50)
            np.testing.assert_allclose([run.qd("a"), run.qd("b")], v_off, atol=1.0e-6)
            run.model.mujoco.tendon_stiffness.assign(np.array([0.0, k], dtype=np.float32))
            run.solver.notify_model_changed(flags)


def test_tendon_friction_loss_switched_by_notify(test, device):
    """A friction-loss row keeps its slot at zero friction loss, so switching it off and on needs no new solver.

    On a 30 degree incline, with friction loss switched off, both sliders accelerate at g sin(theta). Switched
    back on, the one whose friction loss is below m g sin(theta) keeps accelerating at g sin(theta) - f / m; the
    other decelerates at that rate until it stops and then stays stopped, as a solver constructed with the
    friction loss does. Friction loss on a tendon that had no friction row adds a row and raises.
    """
    theta, dt, off, on = math.radians(30.0), 1.0e-3, 200, 500
    axis = f"{math.cos(theta)} 0 {-math.sin(theta)}"
    bodies = _slider("slip", "0 -0.4 1", axis=axis) + _slider("stop", "0 0 1", axis=axis) + _slider("free", "0 0.4 1")
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(bodies)}</worldbody><tendon>
      <fixed frictionloss="3"><joint joint="slip" coef="1"/></fixed>
      <fixed frictionloss="8"><joint joint="stop" coef="1"/></fixed>
      <fixed><joint joint="free" coef="1"/></fixed></tendon></mujoco>"""
    gs, f = G * math.sin(theta), np.array([3.0, 8.0, 0.0], dtype=np.float32)
    flags = newton.ModelFlags.TENDON_PROPERTIES
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, iterations=16)
            run.model.mujoco.tendon_frictionloss.assign(np.zeros(3, dtype=np.float32))
            run.solver.notify_model_changed(flags)
            run.step(dt, off)
            v_s = gs * off * dt
            test.assertAlmostEqual(run.qd("slip"), v_s, delta=1.0e-4)
            test.assertAlmostEqual(run.qd("stop"), v_s, delta=1.0e-4)
            run.model.mujoco.tendon_frictionloss.assign(f)
            run.solver.notify_model_changed(flags)
            fresh = _step_alongside(run, *_fresh_copy(run, mode, iterations=16), dt, on)
            np.testing.assert_array_equal(run.state_0.joint_qd.numpy(), fresh.joint_qd.numpy())
            np.testing.assert_array_equal(run.state_0.joint_q.numpy(), fresh.joint_q.numpy())
            test.assertAlmostEqual(run.qd("slip"), v_s + (gs - f[0]) * on * dt, delta=1.0e-4)
            test.assertAlmostEqual(run.qd("stop"), 0.0, delta=1.0e-5)  # stopped at t = 0.32 s, then held
            run.model.mujoco.tendon_frictionloss.assign(np.array([3.0, 8.0, 1.0], dtype=np.float32))
            with test.assertRaises(NotImplementedError):
                run.solver.notify_model_changed(flags)


def test_tendon_inside_sphere_wrap_matches_mujoco(test, device):
    """A span wrapping a sphere whose side site is inside it matches MuJoCo's length and varies continuously.

    The reference lengths are MuJoCo 3.17's ten_length for these points (unit sphere at the origin, side site at
    its center); the iteration on z from 1 - 1e-7 failed in float32 here and fell back to the midpoint (4.788 m).
    """
    a = (-1.7458, -0.1959, 3.3557)
    cases = {(-0.0199, -1.1801, -0.9182): 4.714231404, (-0.0199, -1.1801, -0.9192): 4.715135496}
    for b, expected in cases.items():
        mjcf = f"""<mujoco>{_HEAD}<worldbody><geom name="ball" type="sphere" size="1"/><site name="side" pos="0 0 0"/>
          <site name="b" pos="{b[0]} {b[1]} {b[2]}"/>
          <body name="m" pos="{a[0]} {a[1]} {a[2]}"><joint name="m" type="slide" axis="1 0 0"/>
          <geom type="box" size="0.05 0.05 0.05" mass="1"/><site name="a" pos="0 0 0"/></body>
        </worldbody><tendon><spatial><site site="a"/><geom geom="ball" sidesite="side"/><site site="b"/></spatial>
        </tendon></mujoco>"""
        with test.subTest(b=b):
            run = _Run(mjcf, device, iterations=4)
            run.step(1.0e-3, 1)
            test.assertAlmostEqual(float(run.solver.tendon_lengths().numpy()[0]), expected, delta=2.0e-5)


def test_tendon_armature_history_survives_coupled_iteration_restarts(test, device):
    """A coupler's restarted solve of a step starts from the same spatial-armature dJ/dt history as the first."""
    dt = 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody><site name="w" pos="0 0 2.5"/>{_frame(_slider("m", "1 0 1", axis="1 0 0"))}
    </worldbody><tendon><spatial armature="1"><site site="w"/><site site="m_s"/></spatial></tendon></mujoco>"""
    run = _Run(mjcf, device, qd={"m": 1.0}, iterations=8)
    solver = run.solver
    solver.coupling_notify_iteration_restarts(True)
    run.step(dt, 1)
    kinematic = newton.StateFlags.JOINT_Q | newton.StateFlags.JOINT_QD
    solver.coupling_notify_input_state_update(run.state_0, kinematic, dt=dt)
    solver.step(run.state_0, run.state_1, run.control, None, dt)
    first = run.state_1.joint_qd.numpy().copy()
    solver.coupling_notify_input_state_update(run.state_0, newton.StateFlags.NONE, iteration_restart=True, dt=dt)
    solver.step(run.state_0, run.state_1, run.control, None, dt)
    np.testing.assert_allclose(run.state_1.joint_qd.numpy(), first, atol=1.0e-7)
    # Both solves carry the armature's -a J dJ/dt qd term, which slows the slider by 1.7e-4 m/s in this step; a
    # restart that lost the history, or a solver without the tendon term, would leave the speed unchanged.
    dof = run.model.joint_qd_start.numpy()[run.joint("m")]
    q_new = float(run.state_1.joint_q.numpy()[run.model.joint_q_start.numpy()[run.joint("m")]])
    expected = _guyed_slider_speed(q_new, run.q("m"), run.qd("m"))
    test.assertAlmostEqual(float(first[dof]), expected, delta=5.0e-6)
    test.assertLess(float(first[dof]), run.qd("m") - 1.0e-4)


def _shared_dof_tendons_mjcf(count):
    """Three hinge arms whose DOFs are shared by ``count`` fixed tendons, two spatial tendons and position actuators."""
    rng = np.random.default_rng(7)
    arms = "".join(
        f'<body name="a{i}" pos="{0.2 * i} 0 0"><joint name="j{i}" type="hinge" axis="0 1 0"/>'
        f'<geom type="box" size="0.05 0.02 0.1" pos="0 0 -0.1" mass="0.2"/><site name="s{i}" pos="0 0 -0.2"/></body>'
        for i in range(3)
    )
    tendons, actuators = "", ""
    for t in range(count):
        coef = rng.uniform(-1.0, 1.0, 3)
        joints = "".join(f'<joint joint="j{i}" coef="{coef[i]:.4f}"/>' for i in range(3))
        tendons += (
            f'<fixed name="t{t}" stiffness="{rng.uniform(1.0, 20.0):.3f}" damping="{rng.uniform(0.01, 0.2):.3f}" '
            f'armature="{rng.uniform(0.001, 0.01):.4f}" springlength="0">{joints}</fixed>'
        )
        if t % 8 == 0:
            actuators += f'<position tendon="t{t}" kp="{rng.uniform(1.0, 5.0):.3f}" kv="0.05"/>'
    for end, rest in ((1, 0.2), (2, 0.4)):
        tendons += (
            f'<spatial stiffness="30" damping="0.1" armature="0.005" springlength="{rest}">'
            f'<site site="s0"/><site site="s{end}"/></spatial>'
        )
    return (
        f"<mujoco>{_HEAD}<worldbody>{_frame(arms)}</worldbody>"
        f"<tendon>{tendons}</tendon><actuator>{actuators}</actuator></mujoco>"
    )


def test_tendon_mass_terms_are_bitwise_run_to_run(test, device):
    """With ``deterministic=RUN_TO_RUN``, tendons that share DOFs add their mass terms and forces in a fixed order.

    34 tendons with springs, dampers, armature and actuators act on the same three DOFs in each of 16 worlds,
    so their sums into the mass matrix and the joint forces depend on the order of the additions. Three runs
    must agree bit for bit.
    """
    worlds = 16

    def run():
        world = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -G))
        SolverMuJoCo.register_custom_attributes(world)
        SolverSteel.register_custom_attributes(world)
        world.add_mjcf(_shared_dof_tendons_mjcf(32))
        world.joint_armature = [0.0] * len(world.joint_armature)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -G))
        builder.replicate(world, worlds)
        model = builder.finalize(device=device)
        speed = np.linspace(0.5, 1.5, worlds)[:, None] * np.array([1.0, -2.0, 1.5])
        model.joint_qd.assign(speed.reshape(-1).astype(np.float32))
        solver = SolverSteel(model, deterministic=wp.DeterministicMode.RUN_TO_RUN)
        state_0, state_1, control = model.state(), model.state(), model.control()
        control.mujoco.ctrl.fill_(0.1)
        for _ in range(20):
            solver.step(state_0, state_1, control, None, 0.002)
            state_0, state_1 = state_1, state_0
        return np.concatenate([state_0.joint_q.numpy(), state_0.joint_qd.numpy()])

    first = run()
    test.assertTrue(np.isfinite(first).all())
    test.assertGreater(float(np.abs(first).max()), 1.0e-3)
    for _ in range(2):
        np.testing.assert_array_equal(run().view(np.uint32), first.view(np.uint32))


_NO_GRAVITY = '<option gravity="0 0 0"/><default><geom condim="1" friction="0 0 0"/></default>'
_ARM_GEOM = '<geom type="box" size="0.25 0.02 0.02" pos="0.25 0 0" mass="0.2"/>'
# Arm inertia about its hinge: a 0.5 x 0.04 m box of 0.2 kg centered 0.25 m out.
_ARM_INERTIA = 0.2 * ((0.5**2 + 0.04**2) / 12.0 + 0.25**2)
_PROPAGATION = {"articulated_contact_response": "propagation"}


def _base(children):
    """A fixed root whose own geom takes no part in collisions."""
    return (
        f'<body name="base"><geom type="sphere" size="0.02" mass="0.1" contype="0" conaffinity="0"/>{children}</body>'
    )


def _arm(name, pos, inner=""):
    return f'<body name="{name}" pos="{pos}"><joint name="{name}" type="hinge" axis="0 0 1"/>{_ARM_GEOM}{inner}</body>'


def _ball(pos, mass=0.5):
    return (
        f'<body name="ball" pos="{pos}"><freejoint name="ball"/><geom type="sphere" size="0.05" mass="{mass}"/></body>'
    )


def test_tendon_armature_reaches_propagation_contacts(test, device):
    """A contact on a tendon-coupled arm responds through the tendon armature under the propagation response.

    Two sibling arms of hinge inertia ``I`` share a tendon ``q1 - q2`` with armature ``a``, so
    ``H = I + a [[1, -1], [-1, 1]]``. A ball of mass ``m`` and speed ``v0`` hits arm 1 squarely at radius ``r``.
    The contact is inelastic: with ``y = H^-1 e1`` the impulse is ``p = m v0 / (1 + m r^2 y1)``, the arms leave at
    ``r p y`` and the ball at ``v0 - p / m``. The articulated-body tree of the propagation response has no place
    for the off-tree armature term; without it arm 2 would stay at rest.
    """
    a, m, v0, r, dt = 20.0, 0.5, 2.0, 0.4, 2.0e-3
    mjcf = f"""<mujoco>{_NO_GRAVITY}<worldbody>{_base(_arm("h1", "0 0 0") + _arm("h2", "0 0 0.5"))}
      {_ball(f"{r} 0.075 0", m)}</worldbody>
      <tendon><fixed armature="{a}"><joint joint="h1" coef="1"/><joint joint="h2" coef="-1"/></fixed></tendon>
    </mujoco>"""
    y = np.linalg.solve(_ARM_INERTIA * np.eye(2) + a * np.array([[1.0, -1.0], [-1.0, 1.0]]), [1.0, 0.0])
    p = m * v0 / (1.0 + m * r * r * y[0])
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(
                mjcf, device, solve_mode=mode, gravity=0.0, contacts=True, qd={"ball": (0.0, -v0)}, iterations=8,
                **_PROPAGATION,
            )  # fmt: skip
            run.step(dt, 6)
            # The arms have turned 0.02 rad since the impact, which changes the contact geometry by O(1e-3).
            test.assertAlmostEqual(run.qd("h1"), -r * p * y[0], delta=2.0e-2)
            test.assertAlmostEqual(run.qd("h2"), -r * p * y[1], delta=2.0e-2)
            test.assertAlmostEqual(run.qd("ball", 1), -(v0 - p / m), delta=1.0e-2)


def test_tendon_single_joint_armature_converges_in_one_propagation_iteration(test, device):
    """A tendon armature on the struck joint itself scales the row's effective mass, not only the velocity update.

    A ball of mass ``m`` hits a slider of mass ``M`` whose own tendon has armature ``a``: both leave at
    ``m v0 / (m + M + a)``. A propagation row that kept the tree's effective mass ``M`` would under-step by
    ``M / (M + a)`` per iteration; with the dense-factor response one iteration is exact.
    """
    a, m, big_m, v0, dt = 20.0, 1.0, 1.0, 1.0, 2.0e-3
    mjcf = f"""<mujoco>{_NO_GRAVITY}<worldbody>{_base(_slider("s1", "0 0 0", axis="1 0 0", mass=big_m))}
      {_ball("-0.105 0 0", m)}</worldbody>
      <tendon><fixed armature="{a}"><joint joint="s1" coef="1"/></fixed></tendon></mujoco>"""
    speed = m * v0 / (m + big_m + a)
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(
                mjcf, device, solve_mode=mode, gravity=0.0, contacts=True, qd={"ball": (v0,)}, iterations=1,
                **_PROPAGATION,
            )  # fmt: skip
            run.step(dt, 6)
            test.assertAlmostEqual(run.qd("s1"), speed, delta=1.0e-4)
            test.assertAlmostEqual(run.qd("ball"), speed, delta=1.0e-4)


def test_tendon_actuated_gripper_holds_a_box_under_propagation(test, device):
    """A tendon position actuator's grip force reaches the contacts of the propagation response in full.

    Two sibling finger sliders share the tendon ``(q_l + q_r) / 2`` driven by a position actuator, which squeezes
    a box with ``N = kp c / 2`` per finger. With friction 1.3 times the critical ``m g / (2 N)`` the box must not
    slip. The actuator's implicit gains ``dt kv + dt^2 kp`` are tendon mass terms; a contact response without
    them scales the normal force by about ``H0 / H`` and drops the box.
    """
    kp, kv, c, m, dt = 1000.0, 100.0, 0.02, 0.1, 2.0e-3
    mu = 1.3 * m * G / (kp * c)
    fingers = "".join(
        f'<body name="{name}" pos="0 {y} 0"><joint name="{name}" type="slide" axis="0 {axis} 0"/>'
        '<geom type="box" size="0.04 0.02 0.04" mass="0.1"/></body>'
        for name, y, axis in (("fl", -0.05, 1), ("fr", 0.05, -1))
    )
    mjcf = f"""<mujoco><compiler autolimits="true"/><option gravity="0 0 {-G}"/>
      <default><geom friction="{mu} 0 0"/></default><worldbody>{_base(fingers)}
      <body name="box"><freejoint name="box"/><geom type="box" size="0.03 0.03 0.03" mass="{m}"/></body></worldbody>
      <tendon><fixed name="t"><joint joint="fl" coef="0.5"/><joint joint="fr" coef="0.5"/></fixed></tendon>
      <actuator><position tendon="t" kp="{kp}" kv="{kv}"/></actuator></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, contacts=True, ctrl=[c], iterations=16, **_PROPAGATION)
            run.step(dt, 150)
            test.assertAlmostEqual(run.q("box", 2), 0.0, delta=1.0e-3)
            test.assertAlmostEqual(run.q("fl"), 0.0, delta=1.0e-3)
            test.assertAlmostEqual(run.q("fr"), 0.0, delta=1.0e-3)


def test_tendon_mass_terms_reach_same_articulation_propagation_rows(test, device):
    """Contacts between two links of one articulation see the tendon mass terms as propagation rows too.

    Arm 1 swings into arm 2; a tendon ``q1 - q3`` with armature ``a`` couples it to a third arm that nothing
    touches. No generalized force acts on joint 3, so its momentum ``(I + a) w3 - a w1`` keeps its initial value
    ``-a w0`` through the impact, whether the contact is a dense row (the default) or a propagation row
    (``propagation_same_articulation_rows``).
    """
    a, w0, dt = 20.0, 3.0, 2.0e-3
    mjcf = f"""<mujoco>{_NO_GRAVITY}<worldbody>
      {_base(_arm("h1", "0 0 0") + _arm("h2", "0 0.15 0") + _arm("h3", "0 0 0.5"))}</worldbody>
      <tendon><fixed armature="{a}"><joint joint="h1" coef="1"/><joint joint="h3" coef="-1"/></fixed></tendon>
    </mujoco>"""
    for mode, same_rows in [(m, s) for m in _MODES for s in (False, True)]:
        with test.subTest(solve_mode=mode, propagation_same_articulation_rows=same_rows):
            run = _Run(
                mjcf, device, solve_mode=mode, gravity=0.0, contacts=True, qd={"h1": w0}, iterations=16,
                mjcf_kwargs={"enable_self_collisions": True}, propagation_same_articulation_rows=same_rows,
                **_PROPAGATION,
            )  # fmt: skip
            run.step(dt, 100)
            test.assertLess(run.qd("h1"), 0.9 * w0)  # arm 1 has hit arm 2
            momentum = (_ARM_INERTIA + a) * run.qd("h3") - a * run.qd("h1")
            test.assertAlmostEqual(momentum, -a * w0, delta=2.0e-3 * a * w0)


def test_tendon_propagation_response_follows_a_stale_mass_matrix(test, device):
    """With ``update_mass_matrix_interval`` above one, the propagation response reuses the factor like the dense rows.

    A ball hits the distal link of a two-link finger whose joints share a tendon with armature. The finger's mass
    matrix changes with its pose, so both responses work from a factor up to three steps old; they agree because
    the tendon sizes' propagation response is refreshed with that factor.
    """
    mjcf = f"""<mujoco>{_NO_GRAVITY}<worldbody>{_base(_arm("h1", "0 0 0", inner=_arm("h2", "0.5 0 0")))}
      {_ball("0.8 0.075 0")}</worldbody>
      <tendon><fixed armature="0.05"><joint joint="h1" coef="1"/><joint joint="h2" coef="-1"/></fixed></tendon>
    </mujoco>"""
    options = {"gravity": 0.0, "contacts": True, "qd": {"ball": (0.0, -2.0)}, "iterations": 32}
    reference = _Run(mjcf, device, update_mass_matrix_interval=4, **options)
    reference.step(2.0e-3, 100)
    test.assertLess(reference.q("h2"), -0.05)  # the ball has pushed the distal link
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, update_mass_matrix_interval=4, **options, **_PROPAGATION)
            run.step(2.0e-3, 100)
            for name in ("h1", "h2"):
                test.assertAlmostEqual(run.q(name), reference.q(name), delta=2.0e-3)
                test.assertAlmostEqual(run.qd(name), reference.qd(name), delta=2.0e-2)


def test_tendon_armature_switched_on_by_notify_under_propagation(test, device):
    """Tendon mass terms that appear at runtime move their articulation size onto the dense propagation response.

    The arms' tendon starts without mass terms (only a separate slider has a damped tendon), so the arms take the
    tree pass and, on CUDA, the cached responses. After the armature is switched on and notified, the solver
    matches one constructed with it.
    """
    mjcf = f"""<mujoco>{_NO_GRAVITY}<worldbody>{_base(_arm("h1", "0 0 0") + _arm("h2", "0 0 0.5"))}
      {_ball("0.4 0.1 0")}{_slider("anchor", "5 0 0", axis="1 0 0")}</worldbody><tendon>
      <fixed><joint joint="h1" coef="1"/><joint joint="h2" coef="-1"/></fixed>
      <fixed damping="1"><joint joint="anchor" coef="1"/></fixed></tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            options = {"iterations": 8, **_PROPAGATION}
            run = _Run(mjcf, device, solve_mode=mode, gravity=0.0, contacts=True, qd={"ball": (0.0, -2.0)}, **options)
            run.step(2.0e-3, 5)
            run.model.mujoco.tendon_armature.assign(np.array([20.0, 0.0], dtype=np.float32))
            run.solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
            fresh, state_0, state_1 = _fresh_copy(run, mode, **options)
            contacts = run.pipeline.contacts()
            for _ in range(40):
                run.pipeline.collide(state_0, contacts)
                fresh.step(state_0, state_1, run.control, contacts, 2.0e-3)
                state_0, state_1 = state_1, state_0
            run.step(2.0e-3, 40)
            test.assertLess(run.qd("h2"), -1.0)  # the armature drags arm 2 along
            np.testing.assert_allclose(run.state_0.joint_qd.numpy(), state_0.joint_qd.numpy(), atol=1.0e-5)


class TestSteelTendon(unittest.TestCase):
    pass


for _name in (
    "test_tendon_atwood_machine",
    "test_tendon_slack_rope_jerks_plastically",
    "test_tendon_pulley_divisor_tackle",
    "test_tendon_fixed_spring_damper",
    "test_tendon_friction_loss_sticks_and_slips",
    "test_tendon_across_articulations_is_ignored",
    "test_tendon_motor_lifts_a_mass",
    "test_tendon_servo_saturates_at_its_force_range",
    "test_tendon_armature_couples_joint_inertia",
    "test_tendon_implicit_spring_survives_large_steps",
    "test_tendon_passive_mode_is_validated",
    "test_tendon_lengths_include_tendons_without_rows",
    "test_tendon_rows_beyond_the_contact_budget_are_kept",
    "test_tendon_limit_rows_are_gated_by_distance",
    "test_tendon_limit_gate_keeps_the_rows_of_mimic_coupled_tendons",
    "test_tendon_friction_follows_an_upper_only_limit_row",
    "test_tendon_spring_is_slack_inside_its_dead_band",
    "test_tendon_armature_history_restarts_on_reset",
    "test_tendon_margin_and_routed_tendons_warn",
    "test_tendon_position_dampratio_is_critically_damped",
    "test_tendon_dampratio_inertia_is_taken_at_the_reference_pose",
    "test_tendon_dampratio_leaves_the_model_unchanged",
    "test_tendon_auto_spring_rests_at_the_joint_rest_pose",
    "test_tendon_fixed_length_includes_the_joint_reference",
    "test_tendon_actuator_force_range_caps_the_summed_force",
    "test_tendon_auto_limit_follows_autolimits",
    "test_tendon_force_range_scales_before_actuator_clamps",
    "test_tendon_property_changes_reach_the_solver",
    "test_tendon_range_edits_report_the_reserved_rows",
    "test_tendon_stiffness_switched_on_by_notify",
    "test_tendon_friction_loss_switched_by_notify",
    "test_tendon_inside_sphere_wrap_matches_mujoco",
    "test_tendon_armature_history_survives_coupled_iteration_restarts",
    "test_tendon_mass_terms_are_bitwise_run_to_run",
    "test_tendon_armature_reaches_propagation_contacts",
    "test_tendon_single_joint_armature_converges_in_one_propagation_iteration",
    "test_tendon_actuated_gripper_holds_a_box_under_propagation",
    "test_tendon_mass_terms_reach_same_articulation_propagation_rows",
    "test_tendon_propagation_response_follows_a_stale_mass_matrix",
    "test_tendon_armature_switched_on_by_notify_under_propagation",
):
    add_function_test(TestSteelTendon, _name, globals()[_name], devices=get_test_devices())
# The dropped rows print the device-side overflow warning, which the test captures itself.
add_function_test(
    TestSteelTendon,
    "test_tendon_length_is_reported_when_its_rows_are_dropped",
    test_tendon_length_is_reported_when_its_rows_are_dropped,
    devices=get_test_devices(),
    check_output=False,
)


if __name__ == "__main__":
    unittest.main()
