# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.rod import _angular_measure
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_axial_impulse(test, device, mode):
    """Match one implicit spring step for a displaced rod joint under a constant load."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_rod(
        -1,
        body,
        parent_xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)),
        stretch_stiffness=2000.0,
        stretch_damping=5.0,
        shear_stiffness=0.0,
        bend_stiffness=0.0,
    )
    builder.add_articulation([joint])
    builder.joint_q[2] = 0.05
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverSteel(model, solve_mode=mode)
    state, result = model.state(), model.state()
    control = model.control()
    control.joint_f.assign([2.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    h = 0.01
    solver.step(state, result, control, None, h)
    speed = h * (2.5 - 2000.0 * 0.05) / (2.0 + h * 5.0 + h * h * 2000.0)
    test.assertAlmostEqual(float(result.joint_qd.numpy()[2]), speed, places=5)
    test.assertAlmostEqual(float(result.body_q.numpy()[0, 0]), 0.05 + h * speed, places=6)


class TestSteelRod(unittest.TestCase):
    pass


for mode in ("serial", "colored"):
    add_function_test(
        TestSteelRod, f"test_axial_impulse_{mode}", test_axial_impulse, devices=get_cuda_test_devices(), mode=mode
    )


def test_colored_ground_contacts(test, device):
    """Resolve concurrent resting contacts on two separate open cables."""
    builder = newton.ModelBuilder()
    for y in (-0.5, 0.5):
        rod = newton.Rod.create_straight(
            start=(-0.2, y, 0.019), direction=(1.0, 0.0, 0.0), length=0.4, segment_count=4, radius=0.02
        )
        builder.add_rod(rod=rod, bend_stiffness=2, twist_stiffness=1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    ends = []
    for mode in ("serial", "colored"):
        solver = newton.solvers.SolverSteel(model, solve_mode=mode, articulation_constraint_max=128, iterations=16)
        a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        for _ in range(30):
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, 1.0 / 240.0)
            a, b = b, a
        positions = a.body_q.numpy()[:, :3].copy()
        test.assertTrue(np.all(np.isfinite(positions)))
        test.assertGreater(float(positions[:, 2].min()), 0.018)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.check_constraint_capacity()
        if mode == "colored":
            schedule = solver._immediate_colors
            colors = int(schedule.count.numpy()[0])
            offsets = schedule.offsets.numpy()[0, : colors + 1]
            test.assertGreaterEqual(int(np.diff(offsets).max()), 2)
        ends.append(positions)
    np.testing.assert_allclose(ends[0], ends[1], atol=2.0e-4)


add_function_test(
    TestSteelRod, "test_colored_ground_contacts", test_colored_ground_contacts, devices=get_cuda_test_devices()
)


def test_packed_contact_batch(test, device, mode):
    """Five open rods should settle identically to an isolated dense reference."""

    def settle(cables):
        builder = newton.ModelBuilder()
        for i in range(cables):
            rod = newton.Rod.create_straight(
                start=(0.0, i * 0.25, 0.018), direction=(1.0, 0.0, 0.0), length=0.4, segment_count=4, radius=0.02
            )
            builder.add_rod(rod=rod, bend_stiffness=2)
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model)
        solver = newton.solvers.SolverSteel(model, solve_mode=mode, articulation_constraint_max=192)
        test.assertEqual(solver._packed_rod_size is not None, cables == 5)
        a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        for _ in range(12):
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, 1.0 / 240.0)
            a, b = b, a
        solver.check_constraint_capacity()
        return a.body_q.numpy()[:, :3]

    reference = settle(1)
    batch = settle(5).reshape(5, 4, 3)
    for i in range(5):
        np.testing.assert_allclose(batch[i], reference + np.array([0.0, i * 0.25, 0.0]), atol=2.0e-4)


for mode in ("serial", "colored"):
    add_function_test(
        TestSteelRod,
        f"test_packed_contact_batch_{mode}",
        test_packed_contact_batch,
        devices=get_cuda_test_devices(),
        mode=mode,
    )


@wp.kernel
def _fill_factor_fixture(H: wp.array3d[float], R: wp.array2d[float]):
    g, i, j = wp.tid()
    H[g, i, j] = wp.exp(-0.05 * wp.abs(float(i - j))) + float(i == j)
    if i == j:
        R[g, i] = float(g + 1) * 0.25


def test_large_factor(test, device):
    from newton._src.solvers.steel import blocked_cholesky  # noqa: PLC0415

    size = 197
    H = wp.empty((2, size, size), dtype=float, device=device)
    R = wp.empty((2, size), dtype=float, device=device)
    L = wp.zeros_like(H)
    wp.launch(_fill_factor_fixture, dim=H.shape, inputs=[H, R], device=device)
    arts = wp.array([0, 1], dtype=int, device=device)
    active = wp.array([1, 0], dtype=int, device=device)
    blocked_cholesky.factor(H, R, arts, active, L)
    factors = L.numpy()
    np.testing.assert_allclose(factors[0] @ factors[0].T, H.numpy()[0] + np.diag(R.numpy()[0]), atol=2e-6)
    test.assertTrue(np.all(factors[1] == 0.0))


add_function_test(TestSteelRod, "test_large_factor", test_large_factor, devices=get_cuda_test_devices())


def test_closed_ring_rigid_translation(test, device):
    """A translating material ring and a prescribed ring retain their rest shapes."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    ring_size = 12
    angles = np.linspace(0.0, 2.0 * np.pi, ring_size + 1)
    for ring in range(2):
        points = [wp.vec3(ring + 0.2 * np.cos(t), 0.2 * np.sin(t), 0.3) for t in angles]
        points[-1] = points[0]
        rod = newton.Rod(points, radius=0.01, closed=True)
        bodies, _ = builder.add_rod(rod=rod, stretch_stiffness=1.0e5, bend_stiffness=2.0)
        if ring == 0:
            for body in bodies:
                builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverSteel(model)
    a, b = model.state(), model.state()
    initial = a.body_q.numpy()
    root = int(model.articulation_start.numpy()[1])
    v = a.joint_qd.numpy()
    v[model.joint_qd_start.numpy()[root] : model.joint_qd_start.numpy()[root] + 3] = [0.2, 0.0, 0.0]
    a.joint_qd.assign(v)
    newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    dt, steps = 1.0 / 480.0, 96
    control = model.control()
    for _ in range(steps):
        solver.step(a, b, control, None, dt)
        a, b = b, a
    expected = initial[:, :3].copy()
    expected[ring_size:, 0] += 0.2 * dt * steps
    np.testing.assert_allclose(a.body_q.numpy()[:, :3], expected, atol=3e-5)
    np.testing.assert_allclose(a.body_q.numpy()[:, 3:], initial[:, 3:], atol=4e-5)
    test.assertLess(float(np.abs(a.body_qd.numpy()[:ring_size]).max()), 1e-7)


add_function_test(
    TestSteelRod,
    "test_closed_ring_rigid_translation",
    test_closed_ring_rigid_translation,
    devices=get_cuda_test_devices(),
)


@wp.kernel
def _differentiate_material_error(errors: wp.array2d[float]):
    pose, axis_id = wp.tid()
    axis = wp.vec3()
    axis[axis_id] = 1.0
    bend_axis = wp.normalize(wp.vec3(0.7, 0.5, 0.0))
    q = wp.quat_from_axis_angle(bend_axis, 0.3 + 0.4 * float(pose)) * wp.quat_from_axis_angle(
        wp.vec3(0.0, 0.0, 1.0), 2.7 - 0.5 * float(pose)
    )
    _, jac = _angular_measure(q)
    eps = 0.001
    before, _left = _angular_measure(wp.quat_from_axis_angle(axis, -eps) * q)
    after, _right = _angular_measure(wp.quat_from_axis_angle(axis, eps) * q)
    difference = after - before
    difference[2] = wp.atan2(wp.sin(difference[2]), wp.cos(difference[2]))
    predicted = jac * axis
    errors[pose, axis_id] = wp.length(difference / (2.0 * eps) - predicted) / wp.max(1.0, wp.length(predicted))


def test_material_jacobian(test, device):
    """Compare finite rotations to the curvature/twist Jacobian used in impulse work."""
    errors = wp.empty((7, 3), dtype=float, device=device)
    wp.launch(_differentiate_material_error, dim=errors.shape, outputs=[errors], device=device)
    test.assertLess(float(np.max(errors.numpy())), 5.0e-4)


add_function_test(TestSteelRod, "test_material_jacobian", test_material_jacobian, devices=get_cuda_test_devices())


def test_single_block_factor(test, device):
    """The single-launch compensated factor matches the double-precision panels on an ill-conditioned rod matrix."""
    from newton._src.solvers.steel import blocked_cholesky  # noqa: PLC0415

    rng = np.random.default_rng(3)
    for size in (65, 114, 128):
        # A chain-like matrix whose diagonal spans the mass and inertia scales of a thin rod.
        scale = np.logspace(-4.0, -8.5, size)
        q, _ = np.linalg.qr(rng.standard_normal((size, size)))
        core = q @ np.diag(np.logspace(0.0, -5.0, size)) @ q.T + np.eye(size) * 1.0e-6
        matrix = (np.sqrt(scale)[:, None] * core * np.sqrt(scale)[None, :]).astype(np.float32)
        H = wp.array(np.stack([matrix, matrix]), dtype=float, device=device)
        R = wp.array(np.stack([scale * 1.0e-3, scale * 1.0e-3]).astype(np.float32), dtype=float, device=device)
        arts = wp.array([0, 1], dtype=int, device=device)
        active = wp.array([1, 0], dtype=int, device=device)
        exact = matrix.astype(np.float64) + np.diag(R.numpy()[0].astype(np.float64))
        d = np.sqrt(np.diag(exact))
        residual = {}
        for single in (True, False):
            L = wp.zeros_like(H)
            blocked_cholesky.factor(H, R, arts, active, L, single_block=single)
            factors = L.numpy().astype(np.float64)
            test.assertTrue(np.all(factors[1] == 0.0))
            lower = np.tril(factors[0])
            residual[single] = np.abs((lower @ lower.T - exact) / np.outer(d, d)).max()
        test.assertLess(residual[True], 2.0 * residual[False] + 1.0e-7, f"size {size}: {residual}")


add_function_test(TestSteelRod, "test_single_block_factor", test_single_block_factor, devices=get_cuda_test_devices())


def _crossing_rods(device):
    builder = newton.ModelBuilder()
    lower = newton.Rod.create_straight(
        start=(-0.2, 0.0, 0.019), direction=(1.0, 0.0, 0.0), length=0.4, segment_count=8, radius=0.02
    )
    upper = newton.Rod.create_straight(
        start=(0.0, -0.2, 0.06), direction=(0.0, 1.0, 0.0), length=0.4, segment_count=8, radius=0.02
    )
    builder.add_rod(rod=lower, bend_stiffness=2, twist_stiffness=1)
    builder.add_rod(rod=upper, bend_stiffness=2, twist_stiffness=1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _settle(model, steps, **options):
    pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, **options)
    a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
    for _ in range(steps):
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, 1.0 / 240.0)
        a, b = b, a
    solver.check_constraint_capacity()
    return solver, a, contacts


def test_contact_unit_sweep(test, device):
    """Visiting each contact as one update reproduces the fused row sweep of a serial rod world.

    Both sweeps take one step from the same state. The two differ by rounding, and the sweep's own
    fast-math rounding moves this unconverged step's angular velocities by about 5e-3 rad/s, so the
    velocity tolerances follow that spread. Trajectories are not compared: rounding can flip a contact
    between sticking and sliding.
    """
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    model = _crossing_rods(device)
    _, rest, contacts = _settle(model, 60, iterations=64)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    # Fixed contact and row order, so both sweeps solve the same rows in the same sequence.
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    results = {}
    for enabled in (True, False):
        with kernel_overrides({"contact_unit_sweep": enabled}):
            solver = newton.solvers.SolverSteel(
                model, articulation_constraint_max=640, iterations=64, deterministic=wp.DeterministicMode.RUN_TO_RUN
            )
        test.assertEqual(solver._contact_unit_sweep is not None, enabled)
        a, b, control, step_contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        a.assign(rest)
        pipeline.collide(a, step_contacts)
        solver.step(a, b, control, step_contacts, 1.0 / 240.0)
        results[enabled] = (b.body_q.numpy(), b.body_qd.numpy())
    np.testing.assert_allclose(results[True][0][:, :3], results[False][0][:, :3], atol=1.0e-6)
    np.testing.assert_allclose(results[True][1][:, :3], results[False][1][:, :3], atol=5.0e-4)
    np.testing.assert_allclose(results[True][1][:, 3:], results[False][1][:, 3:], atol=2.0e-2)


add_function_test(TestSteelRod, "test_contact_unit_sweep", test_contact_unit_sweep, devices=get_cuda_test_devices())


def test_coupled_contact_passes(test, device):
    """Revisiting rod-on-rod contacts changes the sweep order, not the converged contact solution."""
    model = _crossing_rods(device)
    with test.assertRaises(ValueError):
        newton.solvers.SolverSteel(model, coupled_contact_passes=-1)
    positions = {}
    for passes in (0, 4):
        solver, state, _ = _settle(model, 20, iterations=200, coupled_contact_passes=passes)
        test.assertEqual(solver._contact_unit_sweep.coupled_passes, passes)
        positions[passes] = state.body_q.numpy()[:, :3]
        test.assertTrue(np.all(np.isfinite(positions[passes])))
    np.testing.assert_allclose(positions[4], positions[0], atol=2.0e-4)


add_function_test(
    TestSteelRod, "test_coupled_contact_passes", test_coupled_contact_passes, devices=get_cuda_test_devices()
)


def test_split_contact_jacobian(test, device):
    """Splitting each rod contact's Jacobian over its rows and DOFs writes the same rows as one thread per contact."""
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    model = _crossing_rods(device)
    _, state, _ = _settle(model, 30, iterations=32)
    pipeline = newton.CollisionPipeline(model)
    rows = {}
    for split in (True, False):
        with kernel_overrides({"split_contact_jacobian": split}):
            solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, iterations=1)
        a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        a.assign(state)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, 1.0 / 240.0)
        count = int(solver._constraint_count.numpy()[0])
        test.assertGreater(count, 0)
        rows[split] = solver._J_world.numpy()[0, :count].astype(np.float64)
    # Contacts reserve their rows through atomic counters, so the row order differs between solvers:
    # every row of one build must have a matching row in the other.
    test.assertEqual(rows[True].shape, rows[False].shape)
    distance = np.abs(rows[True][:, None, :] - rows[False][None, :, :]).max(axis=2)
    test.assertLess(float(distance.min(axis=1).max()), 1.0e-6)
    test.assertLess(float(distance.min(axis=0).max()), 1.0e-6)


add_function_test(
    TestSteelRod, "test_split_contact_jacobian", test_split_contact_jacobian, devices=get_cuda_test_devices()
)


_PROPAGATION = {"articulated_contact_response": "propagation", "propagation_same_articulation_rows": True}


def _step_from(model, state, solver, with_contacts=True):
    # A fixed contact order, so solvers compared from one state see the same rows in the same order.
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
    a.assign(state)
    if with_contacts:
        pipeline.collide(a, contacts)
    solver.step(a, b, control, contacts if with_contacts else None, 1.0 / 240.0)
    return b


def test_propagation_matches_immediate(test, device):
    """The propagation response, with the rod material in each joint's articulated block, solves the same step.

    Frictionless contacts make the converged step unique, so both responses must approach it as the
    iterations grow; without contacts both predictors agree to rounding.
    """
    model = _crossing_rods(device)
    model.shape_material_mu.fill_(0.0)
    _, rest, contacts = _settle(model, 40, iterations=64)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    gaps = []
    for iterations in (256, 2048):
        qd = {}
        for name, options in (("immediate", {}), ("propagation", _PROPAGATION)):
            solver = newton.solvers.SolverSteel(
                model,
                articulation_constraint_max=640,
                iterations=iterations,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
                **options,
            )
            qd[name] = _step_from(model, rest, solver).joint_qd.numpy()
        gaps.append(float(np.abs(qd["immediate"] - qd["propagation"]).max()))
    test.assertLess(gaps[1], 0.5 * gaps[0])
    test.assertLess(gaps[1], 2.0e-3)
    free = {}
    for name, options in (("immediate", {}), ("propagation", _PROPAGATION)):
        solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, iterations=8, **options)
        free[name] = _step_from(model, rest, solver, with_contacts=False).joint_qd.numpy()
    np.testing.assert_allclose(free["propagation"], free["immediate"], atol=1.0e-4)


add_function_test(
    TestSteelRod,
    "test_propagation_matches_immediate",
    test_propagation_matches_immediate,
    devices=get_cuda_test_devices(),
)


def _clamped_rod(device):
    builder = newton.ModelBuilder()
    rod = newton.Rod.create_straight(
        start=(0.0, 0.0, 0.5), direction=(1.0, 0.0, 0.0), length=0.4, segment_count=8, radius=0.01
    )
    bodies, _joints = builder.add_rod(rod=rod, bend_stiffness=0.5, twist_stiffness=0.5, wrap_in_articulation=True)
    builder.body_flags[int(bodies[0])] = int(newton.BodyFlags.KINEMATIC)
    tip = int(bodies[-1])
    # The clamp closes a loop after the articulation's joints, inside its joint range.
    builder.add_joint_fixed(-1, tip, parent_xform=builder.body_q[tip])
    return builder.finalize(device=device), tip


def test_propagation_loop_closure(test, device):
    """A rod clamped to the world at both ends sags under gravity as with the immediate response.

    The closure is solved as body rows of the propagation sweep, serial and colored, and the tree passes
    walk only the articulation's tree joints.
    """
    positions = {}
    for name, options in (
        ("immediate", {}),
        ("serial", _PROPAGATION),
        ("colored", {**_PROPAGATION, "solve_mode": "colored"}),
    ):
        model, tip = _clamped_rod(device)
        solver = newton.solvers.SolverSteel(model, iterations=16, articulation_constraint_max=64, **options)
        a, b, control = model.state(), model.state(), model.control()
        start = a.body_q.numpy()[tip, :3].copy()
        for _ in range(120):
            solver.step(a, b, control, None, 1.0 / 600.0)
            a, b = b, a
        q = a.body_q.numpy()
        test.assertTrue(np.all(np.isfinite(q)))
        test.assertLess(float(np.abs(q[tip, :3] - start).max()), 1.0e-5)
        positions[name] = q[:, :3]
    for name in ("serial", "colored"):
        np.testing.assert_allclose(positions[name], positions["immediate"], atol=1.0e-4)


add_function_test(
    TestSteelRod, "test_propagation_loop_closure", test_propagation_loop_closure, devices=get_cuda_test_devices()
)


def test_propagation_contracted_chain(test, device):
    """A chain with few contact-active links iterates on those links alone and ends where the full pass ends.

    The clamp of a long rod is its only row, so its chain contracts to one active link; the contracted
    iterations and the propagation of their summed impulses must match propagating every iteration.
    """
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    for mode in ("serial", "colored"):
        positions = {}
        for contracted in (True, False):
            builder = newton.ModelBuilder()
            rod = newton.Rod.create_straight(
                start=(0.0, 0.0, 0.5), direction=(1.0, 0.0, 0.0), length=0.8, segment_count=32, radius=0.01
            )
            bodies, _joints = builder.add_rod(
                rod=rod, bend_stiffness=0.5, twist_stiffness=0.5, wrap_in_articulation=True
            )
            builder.body_flags[int(bodies[0])] = int(newton.BodyFlags.KINEMATIC)
            tip = int(bodies[-1])
            builder.add_joint_fixed(-1, tip, parent_xform=builder.body_q[tip])
            model = builder.finalize(device=device)
            with kernel_overrides({"rod_chain_contracted": contracted}):
                solver = newton.solvers.SolverSteel(
                    model, iterations=16, articulation_constraint_max=64, solve_mode=mode, **_PROPAGATION
                )
            test.assertEqual(solver._rod_contracted, contracted)
            a, b, control = model.state(), model.state(), model.control()
            for _ in range(60):
                solver.step(a, b, control, None, 1.0 / 600.0)
                a, b = b, a
            if contracted:
                counts = solver._rod_ctr_count.numpy()
                test.assertTrue(np.any((counts > 0) & (4 * counts <= 32)), f"{mode}: no chain contracted")
            positions[contracted] = a.body_q.numpy()[:, :3]
        test.assertTrue(np.all(np.isfinite(positions[True])))
        np.testing.assert_allclose(positions[True], positions[False], atol=1.0e-5, err_msg=mode)


add_function_test(
    TestSteelRod,
    "test_propagation_contracted_chain",
    test_propagation_contracted_chain,
    devices=get_cuda_test_devices(),
)


def test_propagation_interval(test, device):
    """Propagating every third iteration still settles a clamped rod where the exact response does."""
    with test.assertRaisesRegex(ValueError, "propagation_interval must be at least 1"):
        newton.solvers.SolverSteel(_clamped_rod(device)[0], propagation_interval=0, **_PROPAGATION)
    positions = {}
    for interval in (1, 3):
        model, _tip = _clamped_rod(device)
        solver = newton.solvers.SolverSteel(
            model, iterations=16, articulation_constraint_max=64, propagation_interval=interval, **_PROPAGATION
        )
        test.assertEqual(solver.propagation_interval, interval)
        a, b, control = model.state(), model.state(), model.control()
        for _ in range(120):
            solver.step(a, b, control, None, 1.0 / 600.0)
            a, b = b, a
        positions[interval] = a.body_q.numpy()[:, :3]
    test.assertTrue(np.all(np.isfinite(positions[3])))
    np.testing.assert_allclose(positions[3], positions[1], atol=2.0e-3)


add_function_test(TestSteelRod, "test_propagation_interval", test_propagation_interval, devices=get_cuda_test_devices())


def test_propagation_tree_dynamics(test, device):
    """Rod trees take their unconstrained velocity from the propagation factorization, as the dense solve does."""
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    model = _crossing_rods(device)
    _, rest, _ = _settle(model, 20, iterations=32)
    qd = {}
    for tree in (True, False):
        with kernel_overrides({"tree_dynamics": tree}):
            solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, iterations=8, **_PROPAGATION)
        test.assertEqual(bool(solver._tree_dynamics_sizes), tree)
        qd[tree] = _step_from(model, rest, solver, with_contacts=False).joint_qd.numpy()
    np.testing.assert_allclose(qd[True], qd[False], atol=1.0e-4, rtol=1.0e-5)


add_function_test(
    TestSteelRod, "test_propagation_tree_dynamics", test_propagation_tree_dynamics, devices=get_cuda_test_devices()
)
