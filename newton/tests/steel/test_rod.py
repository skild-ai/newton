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
    # Contact warm starting reads the pipeline's matching.
    matching = (
        {"contact_matching": "latest", "contact_matching_sort": "keys"} if options.get("contact_warmstart") else {}
    )
    pipeline = newton.CollisionPipeline(model, **matching)
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


def test_propagation_contact_units(test, device):
    """Contact units reproduce the rows of the propagation response's colored sweep.

    With contact units every contact's rows read their Jacobians and responses from its unit (one record per
    contact, PROPAGATION_CONTACT_UNIT_BLOCKS) instead of per-row arrays. Both take one step from the same settled
    state; they differ by rounding (the sweep rebuilds each row from the unit's directions and lever arms), so the
    tolerances follow the unconverged step's spread, as in :func:`test_contact_unit_sweep`.
    """
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    model = _crossing_rods(device)
    options = {
        "iterations": 64,
        "solve_mode": "colored",
        "articulated_contact_response": "propagation",
        "propagation_same_articulation_rows": True,
        "friction_anchor_beta": 0.0,
        "contact_warmstart": True,
    }
    with kernel_overrides({"colored_packed": False}):
        _, rest, contacts = _settle(model, 60, **options)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="latest")
    results = {}
    for enabled in (True, False):
        with kernel_overrides({"colored_packed": False, "contact_units": enabled}):
            solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, **options)
        test.assertEqual(solver._contact_units_on, enabled)
        a, b, control, step_contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        a.assign(rest)
        pipeline.collide(a, step_contacts)
        solver.step(a, b, control, step_contacts, 1.0 / 240.0)
        if enabled:
            # Every contact row belongs to a contact unit: per row, or implied when every row is a unit's.
            count = int(solver._propagation_constraint_count.numpy()[0])
            test.assertGreater(count, 0)
            links = solver._propagation_row_contact.numpy()
            if links.shape == (1, 1):
                test.assertEqual(int(links[0, 0]), solver._propagation_unit_max)
                test.assertEqual(count % 3, 0)
            else:
                test.assertTrue(np.all(links[0, :count] >= 0))
        results[enabled] = (b.body_q.numpy(), b.body_qd.numpy())
    np.testing.assert_allclose(results[True][0][:, :3], results[False][0][:, :3], atol=1.0e-6)
    np.testing.assert_allclose(results[True][1][:, :3], results[False][1][:, :3], atol=5.0e-4)
    np.testing.assert_allclose(results[True][1][:, 3:], results[False][1][:, 3:], atol=2.0e-2)


add_function_test(
    TestSteelRod, "test_propagation_contact_units", test_propagation_contact_units, devices=get_cuda_test_devices()
)


def _hairpin_rod(device):
    """A rod folded back on itself on the ground, its legs touching, with a kinematic first segment."""
    builder = newton.ModelBuilder()
    radius, gap, z = 0.02, 0.0405, 0.0205
    points = [wp.vec3(x, 0.0, z) for x in np.linspace(-0.25, 0.15, 9)]
    turn = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 7)[1:-1]
    points += [wp.vec3(0.15 + 0.5 * gap * np.cos(t), 0.5 * gap + 0.5 * gap * np.sin(t), z) for t in turn]
    points += [wp.vec3(x, gap, z) for x in np.linspace(0.15, -0.25, 9)]
    rod = newton.Rod(points, radius=radius)
    bodies, _ = builder.add_rod(rod=rod, bend_stiffness=2, twist_stiffness=1)
    builder.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_propagation_contact_units_self_contact(test, device):
    """Contact units of a rod touching itself store its exact responses and reproduce the rows.

    Contacts between the hairpin's legs join two links of one articulation: their units take response slots and the
    refinement stores the exact responses there; ground contacts on the kinematic first segment fold their
    prescribed side out. A response capacity below the contacts is reported as an overflow.
    """
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    model = _hairpin_rod(device)
    options = {
        "iterations": 64,
        "solve_mode": "colored",
        "articulated_contact_response": "propagation",
        "propagation_same_articulation_rows": True,
        "friction_anchor_beta": 0.0,
        "contact_warmstart": True,
    }
    with kernel_overrides({"colored_packed": False}):
        _, rest, contacts = _settle(model, 30, **options)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="latest")
    results = {}
    for enabled in (True, False):
        with kernel_overrides({"colored_packed": False, "contact_units": enabled}):
            solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, **options)
        a, b, control, step_contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        a.assign(rest)
        pipeline.collide(a, step_contacts)
        solver.step(a, b, control, step_contacts, 1.0 / 240.0)
        if enabled:
            test.assertGreater(int(solver._contact_response_count.numpy()[0]), 0)
        results[enabled] = (b.body_q.numpy(), b.body_qd.numpy())
    np.testing.assert_allclose(results[True][0][:, :3], results[False][0][:, :3], atol=1.0e-6)
    np.testing.assert_allclose(results[True][1][:, :3], results[False][1][:, :3], atol=5.0e-4)
    np.testing.assert_allclose(results[True][1][:, 3:], results[False][1][:, 3:], atol=2.0e-2)

    with kernel_overrides({"colored_packed": False, "contact_units": True}):
        solver = newton.solvers.SolverSteel(
            model, articulation_constraint_max=640, propagation_same_articulation_contact_max=1, **options
        )
    a, b, control, step_contacts = model.state(), model.state(), model.control(), pipeline.contacts()
    a.assign(rest)
    pipeline.collide(a, step_contacts)
    solver.step(a, b, control, step_contacts, 1.0 / 240.0)
    test.assertTrue(np.all(np.isfinite(b.body_q.numpy())))
    with test.assertRaises(RuntimeError):
        solver.check_constraint_capacity()


add_function_test(
    TestSteelRod,
    "test_propagation_contact_units_self_contact",
    test_propagation_contact_units_self_contact,
    devices=get_cuda_test_devices(),
)


def test_propagation_contact_units_need_the_warp_sweep(test, device):
    """A world whose bodies the warp sweep cannot stage takes the block sweep, which keeps its contacts as rows."""
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    builder = newton.ModelBuilder()
    for k in range(2):
        rod = newton.Rod.create_straight(
            start=(-1.5, 0.05 * k, 0.019), direction=(1.0, 0.0, 0.0), length=5.0, segment_count=500, radius=0.02
        )
        builder.add_rod(rod=rod, bend_stiffness=2, twist_stiffness=1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    with kernel_overrides({"colored_packed": False, "contact_units": True}):
        solver = newton.solvers.SolverSteel(
            model, solve_mode="colored", articulated_contact_response="propagation", friction_anchor_beta=0.0
        )
    test.assertFalse(solver._contact_units_on)
    test.assertIsNotNone(solver._pgs_solve_propagation_colored_block_kernel)


add_function_test(
    TestSteelRod,
    "test_propagation_contact_units_need_the_warp_sweep",
    test_propagation_contact_units_need_the_warp_sweep,
    devices=get_cuda_test_devices(),
)


def test_propagation_rows_fit_32_bit_indices(test, device):
    """A row budget whose Jacobian offsets over every world pass 32 bits is refused at construction."""
    builder = newton.ModelBuilder()
    rod = newton.Rod.create_straight(start=(0.0, 0.0, 0.1), direction=(1.0, 0.0, 0.0), length=0.5, segment_count=4)
    builder.add_rod(rod=rod, bend_stiffness=2, twist_stiffness=1)
    world = newton.ModelBuilder()
    world.replicate(builder, copy_count=4)
    model = world.finalize(device=device)
    # The solver's index-range check (_check_index_range) bounds the 6-wide propagation rows over all worlds.
    with test.assertRaisesRegex(ValueError, "propagation row Jacobians.*propagation_constraint_max"):
        newton.solvers.SolverSteel(
            model, articulated_contact_response="propagation", propagation_constraint_max=(2**31) // 24 + 1
        )


add_function_test(
    TestSteelRod,
    "test_propagation_rows_fit_32_bit_indices",
    test_propagation_rows_fit_32_bit_indices,
    devices=get_cuda_test_devices(),
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
    # The split sweep alone approaches the immediate step as the iterations grow; momentum reaches it at least as
    # fast (it may already sit at the rounding floor at 256 iterations).
    gaps = {False: [], True: []}
    for iterations in (256, 2048):
        qd = {}
        variants = (
            ("immediate", {}),
            ("split", {**_PROPAGATION, "propagation_momentum": False}),
            ("momentum", {**_PROPAGATION, "propagation_momentum": True}),
        )
        for name, options in variants:
            solver = newton.solvers.SolverSteel(
                model,
                articulation_constraint_max=640,
                iterations=iterations,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
                **options,
            )
            qd[name] = _step_from(model, rest, solver).joint_qd.numpy()
        gaps[False].append(float(np.abs(qd["immediate"] - qd["split"]).max()))
        gaps[True].append(float(np.abs(qd["immediate"] - qd["momentum"]).max()))
    test.assertLess(gaps[False][1], 0.5 * gaps[False][0])
    test.assertLess(gaps[False][1], 2.0e-3)
    test.assertLess(gaps[True][1], 2.0e-3)
    test.assertLessEqual(gaps[True][0], gaps[False][0] + 1.0e-5)
    free = {}
    for name, options in (("immediate", {}), ("propagation", _PROPAGATION)):
        solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, iterations=8, **options)
        free[name] = _step_from(model, rest, solver, with_contacts=False).joint_qd.numpy()
    # These solvers are not run-to-run deterministic; a joint rate can differ by about 1e-4 between runs.
    np.testing.assert_allclose(free["propagation"], free["immediate"], atol=2.0e-4)


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
    iterations and the propagation of their summed impulses must match propagating every iteration. Contracted
    iterations run only without momentum, which the rod material otherwise turns on.
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
                    model,
                    iterations=16,
                    articulation_constraint_max=64,
                    solve_mode=mode,
                    propagation_momentum=False,
                    # Nesterov's moves on rod rows arrive through the chains' pending impulses (no contraction).
                    nesterov_momentum=False,
                    **_PROPAGATION,
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


def _pushed_cable_penetration(device, momentum, iterations=8, steps=60, dt=1.0 / 120.0, speed=0.6):
    """Deepest penetration of a kinematic capsule driven sideways through a 40-link cable lying on the ground."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.rigid_gap = 0.05
    rod = newton.Rod.create_straight(
        start=(-1.5, 0.0, 0.021), direction=(1.0, 0.0, 0.0), length=4.0, segment_count=40, radius=0.02
    )
    builder.add_rod(rod=rod, bend_stiffness=1.0e2, bend_damping=5.0)
    pusher = builder.add_body(xform=wp.transform(wp.vec3(0.0, -0.08, 0.3), wp.quat_identity()))
    builder.add_shape_capsule(body=pusher, radius=0.05, half_height=0.25, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    builder.body_flags[pusher] = int(newton.BodyFlags.KINEMATIC)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    model.rigid_contact_max = contacts.rigid_contact_max
    solver = newton.solvers.SolverSteel(
        model,
        iterations=iterations,
        articulation_constraint_max=64,
        propagation_constraint_max=512,
        propagation_momentum=momentum,
        **_PROPAGATION,
    )
    a, b, control = model.state(), model.state(), model.control()
    joint = int(np.flatnonzero(model.joint_child.numpy() == pusher)[0])
    q0, v0 = int(model.joint_q_start.numpy()[joint]), int(model.joint_qd_start.numpy()[joint])
    links = [i for i in range(model.body_count) if i != pusher]
    deepest = 0.0
    for step in range(steps):
        for state, ahead in ((a, 0), (b, 1)):
            q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
            q[q0 + 1] = -0.08 + speed * dt * (step + ahead)
            qd[v0 : v0 + 6] = 0.0
            qd[v0 + 1] = speed
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, dt)
        a, b = b, a
        body_q = a.body_q.numpy()
        gap = np.hypot(body_q[links, 0] - body_q[pusher, 0], body_q[links, 1] - body_q[pusher, 1])
        deepest = max(deepest, float((0.07 - gap).max()))
    test_finite = np.all(np.isfinite(a.body_q.numpy()))
    return deepest, bool(test_finite), solver


def test_propagation_momentum(test, device):
    """Momentum lets a cable lying on the ground follow a pusher that the split sweep alone lets through.

    Every link carries ground rows, so the split sweep gives each row a sliver of the cable's mobility and the
    sideways push converges slowly; momentum recovers that convergence at the same iterations.
    """
    plain, finite, solver = _pushed_cable_penetration(device, False)
    test.assertFalse(solver.propagation_momentum)
    test.assertTrue(finite)
    accelerated, finite, solver = _pushed_cable_penetration(device, True)
    test.assertTrue(solver.propagation_momentum)
    test.assertTrue(finite)
    test.assertGreater(plain, 6.0e-3)
    test.assertLess(accelerated, 0.6 * plain)


add_function_test(TestSteelRod, "test_propagation_momentum", test_propagation_momentum, devices=get_cuda_test_devices())


def test_propagation_weighted_split_weights(test, device):
    """Bodies whose rows carry the impulse take most of each joint's mobility; every row-bearing body keeps a share.

    A three-link chain with rows on links 1 and 2: link 2's rows carry ten times link 1's impulse. Each joint's weight
    sums its row-bearing bodies' weights, so a body's share of a joint (its weight over the joint's) sums to one.
    """
    from newton._src.solvers.steel.kernels import (  # noqa: PLC0415
        accumulate_propagation_body_activity,
        compute_propagation_body_split_weight,
        count_propagation_joint_split,
        max_propagation_articulation_activity,
    )

    rows = 4
    count = wp.array([rows], dtype=wp.int32, device=device)
    body_a = wp.array([[1, 1, 2, 2]], dtype=wp.int32, device=device)
    body_b = wp.full((1, rows), -1, dtype=wp.int32, device=device)
    row_type = wp.zeros((1, rows), dtype=wp.int32, device=device)
    impulses = wp.array([[0.1, 0.0, 1.0, 0.0]], dtype=wp.float32, device=device)
    body_to_art = wp.zeros(3, dtype=wp.int32, device=device)
    body_to_joint = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    joint_parent = wp.array([-1, 0, 1], dtype=wp.int32, device=device)
    activity = wp.zeros(3, dtype=wp.float32, device=device)
    art_activity = wp.zeros(1, dtype=wp.float32, device=device)
    weight = wp.ones(3, dtype=wp.int32, device=device)
    wp.launch(
        accumulate_propagation_body_activity,
        dim=rows,
        inputs=[count, body_a, body_b, row_type, impulses, rows],
        outputs=[activity],
        device=device,
    )
    wp.launch(
        max_propagation_articulation_activity,
        dim=3,
        inputs=[activity, body_to_art],
        outputs=[art_activity],
        device=device,
    )
    rod = wp.ones(1, dtype=wp.int32, device=device)
    wp.launch(
        compute_propagation_body_split_weight,
        dim=3,
        inputs=[activity, body_to_art, art_activity, rod, 15],
        outputs=[weight],
        device=device,
    )
    np.testing.assert_array_equal(weight.numpy(), [1, 3, 16])
    seen = wp.zeros(3, dtype=wp.int32, device=device)
    split = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        count_propagation_joint_split,
        dim=rows,
        inputs=[count, body_a, body_b, rows, body_to_joint, joint_parent, 3, weight],
        outputs=[seen, split],
        device=device,
    )
    np.testing.assert_array_equal(split.numpy(), [19, 19, 16])
    w = weight.numpy()
    test.assertAlmostEqual((w[1] + w[2]) / split.numpy()[1], 1.0)
    # An articulation that is not a rod (a free body, an arm) keeps weight one.
    free = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        compute_propagation_body_split_weight,
        dim=3,
        inputs=[activity, body_to_art, art_activity, free, 15],
        outputs=[weight],
        device=device,
    )
    np.testing.assert_array_equal(weight.numpy(), [1, 1, 1])


add_function_test(
    TestSteelRod,
    "test_propagation_weighted_split_weights",
    test_propagation_weighted_split_weights,
    devices=get_cuda_test_devices(),
)


def test_propagation_momentum_restarts_on_a_turn(test, device):
    """Momentum restarts when the sweeps reverse its last step, even if they agree with the latest move.

    The impulses stood at 1, the momentum carried them on to 1.5, and the sweeps brought them back to 0.2: the sweeps
    moved the same way as the block's move (the gradient test passes), but against the momentum's step, which on an
    oscillating mode would otherwise grow each block.
    """
    from newton._src.solvers.steel.kernels import (  # noqa: PLC0415
        propagation_momentum_dot,
        propagation_momentum_step,
        propagation_momentum_world,
    )

    rows = 3
    count = wp.array([rows], dtype=wp.int32, device=device)
    prev = wp.full((1, rows), 1.0, dtype=wp.float32, device=device)
    start = wp.full((1, rows), 1.5, dtype=wp.float32, device=device)
    impulses = wp.full((1, rows), 0.2, dtype=wp.float32, device=device)
    for fused in (True, False):
        t = wp.full(1, 5.0, dtype=wp.float32, device=device)
        beta = wp.full(1, 0.7, dtype=wp.float32, device=device)
        if fused:
            wp.launch(
                propagation_momentum_step,
                dim=256,
                inputs=[1, count, start, impulses, prev],
                outputs=[t, beta],
                block_dim=256,
                device=device,
            )
        else:
            dot = wp.zeros((1, 2), dtype=wp.float32, device=device)
            wp.launch(
                propagation_momentum_dot,
                dim=rows,
                inputs=[count, rows, impulses, prev, start],
                outputs=[dot],
                device=device,
            )
            test.assertGreater(float(dot.numpy()[0, 0]), 0.0)
            wp.launch(propagation_momentum_world, dim=1, inputs=[dot], outputs=[t, beta], device=device)
        test.assertEqual(float(beta.numpy()[0]), 0.0)
        test.assertEqual(float(t.numpy()[0]), 1.0)


add_function_test(
    TestSteelRod,
    "test_propagation_momentum_restarts_on_a_turn",
    test_propagation_momentum_restarts_on_a_turn,
    devices=get_cuda_test_devices(),
)


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


def test_propagation_tree_dynamics_frees_dense_buffers(test, device):
    """Rod trees no dense row reaches keep no dense mass matrix, factor or row blocks, and step as before.

    With joint limits on, the limit rows reach the rod trees, so they keep their dense buffers.
    """
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    model = _crossing_rods(device)
    _, rest, contacts = _settle(model, 20, iterations=32)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    solver = newton.solvers.SolverSteel(model, articulation_constraint_max=640, iterations=8, **_PROPAGATION)
    test.assertTrue(solver._tree_dynamics_sizes)
    test.assertEqual(solver._row_free_sizes, frozenset(solver._tree_dynamics_sizes))
    for size in solver._row_free_sizes:
        test.assertNotIn(size, solver._dense_row_sizes)
        test.assertEqual(solver._H_by_size[size].size, 1)
        test.assertEqual(solver._L_by_size[size].size, 1)
    # A step with contacts runs on the freed buffers; without contacts it matches the dense predictor (the two
    # contact solves follow different paths, so only the unconstrained step is compared, as above).
    test.assertTrue(np.all(np.isfinite(_step_from(model, rest, solver).joint_qd.numpy())))
    freed = _step_from(model, rest, solver, with_contacts=False).joint_qd.numpy()
    with kernel_overrides({"tree_dynamics": False}):
        dense = newton.solvers.SolverSteel(model, articulation_constraint_max=640, iterations=8, **_PROPAGATION)
    test.assertFalse(dense._row_free_sizes)
    reference = _step_from(model, rest, dense, with_contacts=False).joint_qd.numpy()
    np.testing.assert_allclose(freed, reference, atol=1.0e-4, rtol=1.0e-5)
    limited = newton.solvers.SolverSteel(
        model, articulation_constraint_max=640, iterations=8, enable_joint_limits=True, **_PROPAGATION
    )
    if limited._joint_limit_sizes & limited._tree_dynamics_sizes:
        test.assertFalse(limited._row_free_sizes & limited._joint_limit_sizes)


add_function_test(
    TestSteelRod,
    "test_propagation_tree_dynamics_frees_dense_buffers",
    test_propagation_tree_dynamics_frees_dense_buffers,
    devices=get_cuda_test_devices(),
)


def _hanging_rod(device, inextensible, ground=True):
    """An 8-link rod held at its root link, falling as a pendulum onto the ground."""
    builder = newton.ModelBuilder()
    rod = newton.Rod.create_straight(
        start=(0.0, 0.0, 0.3), direction=(1.0, 0.0, 0.0), length=0.4, segment_count=8, radius=0.01
    )
    stretch = {} if inextensible else {"stretch_stiffness": 1.0e9, "shear_stiffness": 1.0e9}
    bodies, _joints = builder.add_rod(
        rod=rod, bend_stiffness=0.05, twist_stiffness=0.05, inextensible=inextensible, **stretch
    )
    builder.body_flags[int(bodies[0])] = int(newton.BodyFlags.KINEMATIC)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device), bodies


def _junction_gaps(model, state):
    """Distance between each rod joint's parent and child anchors [m]."""
    q = state.body_q.numpy()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    X_p, X_c = model.joint_X_p.numpy(), model.joint_X_c.numpy()

    def point(pose, local):
        return np.asarray(wp.transform_point(wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])), wp.vec3(*local[:3])))

    rods = np.flatnonzero((model.joint_type.numpy() == int(newton.JointType.ROD)) & (parent >= 0))
    return np.array([np.linalg.norm(point(q[parent[j]], X_p[j]) - point(q[child[j]], X_c[j])) for j in rods])


def test_inextensible_layout(test, device):
    """An inextensible rod joint has three angular DOFs and a quaternion, and keeps only bend and twist gains."""
    model, _ = _hanging_rod(device, inextensible=True, ground=False)
    rods = model.joint_type.numpy() == int(newton.JointType.ROD)
    test.assertTrue(np.all(model.joint_dof_dim.numpy()[rods] == (0, 3)))
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()  # each with an end sentinel
    test.assertTrue(np.all(np.diff(q_start)[rods] == 4))
    test.assertTrue(np.all(np.diff(qd_start)[rods] == 3))
    ke = model.joint_target_ke.numpy()
    for j in np.flatnonzero(rods):
        np.testing.assert_allclose(ke[qd_start[j] : qd_start[j] + 3], 0.05)
    with test.assertRaises(ValueError):
        newton.ModelBuilder().add_rod(
            rod=newton.Rod.create_straight(start=(0, 0, 0), direction=(1, 0, 0), length=0.2, segment_count=2),
            stretch_stiffness=1.0,
            inextensible=True,
        )
    with test.assertRaises(ValueError):
        newton.solvers.SolverVBD(model)


def test_inextensible_pendulum(test, device, mode, response):
    """A falling inextensible rod keeps its links joined, reaches the ground, and in free swing tracks a very
    stiff six-DOF rod."""
    model, bodies = _hanging_rod(device, inextensible=True)
    pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverSteel(
        model, solve_mode=mode, articulated_contact_response=response, articulation_constraint_max=256, iterations=16
    )
    a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
    for _ in range(180):
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, 1.0 / 240.0)
        a, b = b, a
    positions = a.body_q.numpy()[:, :3]
    test.assertTrue(np.all(np.isfinite(positions)))
    # Reduced coordinates: the links cannot part, whatever the load.
    test.assertLess(float(_junction_gaps(model, a).max()), 1.0e-5)
    test.assertLess(float(positions[int(bodies[-1]), 2]), 0.05)  # it swung down onto the ground

    # In free swing it converges to a very stiff six-DOF rod (stepped with the immediate response, which resolves
    # that stiffness exactly). Stepping from the carried body momenta dissipates a little per step, so the gap is
    # first order in the step: it halves with the step.
    gaps = []
    for dt in (1.0 / 240.0, 1.0 / 480.0):
        tips = {}
        for inextensible, rod_response in ((True, response), (False, "immediate")):
            model, bodies = _hanging_rod(device, inextensible, ground=False)
            solver = newton.solvers.SolverSteel(
                model,
                solve_mode=mode,
                articulated_contact_response=rod_response,
                articulation_constraint_max=256,
                iterations=16,
            )
            a, b, control = model.state(), model.state(), model.control()
            for _ in range(round(0.25 / dt)):  # mid swing
                solver.step(a, b, control, None, dt)
                a, b = b, a
            tips[inextensible] = a.body_q.numpy()[int(bodies[-1]), :3]
        gaps.append(float(np.linalg.norm(tips[True] - tips[False])))
    test.assertLess(gaps[0], 0.004)
    test.assertLess(gaps[1], 0.65 * gaps[0])


add_function_test(TestSteelRod, "test_inextensible_layout", test_inextensible_layout, devices=get_cuda_test_devices())
for mode in ("serial", "colored"):
    for response in ("immediate", "propagation"):
        add_function_test(
            TestSteelRod,
            f"test_inextensible_pendulum_{mode}_{response}",
            test_inextensible_pendulum,
            devices=get_cuda_test_devices(),
            mode=mode,
            response=response,
        )


def test_inextensible_tree_dynamics(test, device):
    """Under the propagation response an inextensible rod tree steps from the articulated-body factorization, with
    no dense mass matrix, and matches the dense solve of the same joint loads."""
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415

    results = {}
    for tree in (True, False):
        model, _bodies = _hanging_rod(device, inextensible=True)
        pipeline = newton.CollisionPipeline(model, deterministic=True)
        with kernel_overrides({"tree_dynamics": tree}):
            solver = newton.solvers.SolverSteel(
                model,
                articulated_contact_response="propagation",
                articulation_constraint_max=256,
                iterations=16,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
                # The dense path's generalized rows would take Nesterov and the tree's rows propagation_momentum;
                # the two dynamics paths are compared without either.
                nesterov_momentum=False,
                propagation_momentum=False,
            )
        test.assertEqual(bool(solver._tree_dynamics_sizes), tree)
        a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        for _ in range(120):
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, 1.0 / 240.0)
            a, b = b, a
        test.assertLess(float(_junction_gaps(model, a).max()), 1.0e-5)
        results[tree] = a.body_q.numpy()
    test.assertTrue(np.all(np.isfinite(results[True])))
    np.testing.assert_allclose(results[True][:, :3], results[False][:, :3], atol=2.0e-4)


add_function_test(
    TestSteelRod, "test_inextensible_tree_dynamics", test_inextensible_tree_dynamics, devices=get_cuda_test_devices()
)


def test_inextensible_fused_loop(test, device):
    """With one world the latency-fused loop carries the rod chains, closure rows and momentum in one launch per
    solve, its parameters fit the 4 KB a CUDA graph can capture, and it steps like the per-pass launches."""
    from newton._src.solvers.steel._testing import kernel_overrides  # noqa: PLC0415
    from newton.tests.steel.test_propagation import _kernel_param_bytes  # noqa: PLC0415

    # Run to run identical without momentum (ordered rows step the momentum in row order, which the loop does not).
    results = {}
    for fused, momentum in ((True, False), (False, False), (True, True)):
        model, bodies = _hanging_rod(device, inextensible=True)
        pipeline = newton.CollisionPipeline(model, deterministic=True)
        with kernel_overrides({"propagation_latency_fused": fused}):
            solver = newton.solvers.SolverSteel(
                model,
                solve_mode="colored",
                articulated_contact_response="propagation",
                articulation_constraint_max=256,
                iterations=16,
                propagation_interval=1,
                propagation_momentum=momentum,
                deterministic=None if momentum else wp.DeterministicMode.RUN_TO_RUN,
            )
        test.assertEqual(solver._latency_fused, fused)
        if fused:
            test.assertLessEqual(_kernel_param_bytes(solver._latency_fused_kernel), 4096)
        a, b, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
        for _ in range(120):
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, 1.0 / 240.0)
            a, b = b, a
        results[fused, momentum] = a.body_q.numpy()
    for key in ((True, False), (True, True)):
        test.assertTrue(np.all(np.isfinite(results[key])))
        test.assertLess(float(results[key][int(bodies[-1]), 2]), 0.05)  # it swung down onto the ground
    np.testing.assert_allclose(results[True, False][:, :3], results[False, False][:, :3], atol=1.0e-4)


add_function_test(
    TestSteelRod, "test_inextensible_fused_loop", test_inextensible_fused_loop, devices=get_cuda_test_devices()
)


def test_inextensible_long_chain_whip(test, device):
    """A long, light inextensible rod whipping under gravity stays bounded, as a stiff six-DOF rod does.

    The joint-space velocity-product terms of such a chain feed energy into a semi-implicit step and it diverges;
    stepping from the carried body momenta (a projection) does not add energy.
    """
    peaks = {}
    for inextensible in (True, False):
        builder = newton.ModelBuilder()
        rod = newton.Rod.create_straight(
            start=(0.0, 0.0, 1.5), direction=(1.0, 0.0, 0.0), length=1.2, segment_count=54, radius=0.004
        )
        stretch = {} if inextensible else {"stretch_stiffness": 1.0e7, "stretch_damping": 0.1}
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=newton.ModelBuilder.ShapeConfig(density=200.0),
            bend_stiffness=3.0e-3,
            twist_stiffness=3.0e-3,
            inextensible=inextensible,
            **stretch,
        )
        builder.body_flags[int(bodies[0])] = int(newton.BodyFlags.KINEMATIC)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverSteel(model, iterations=8, articulation_constraint_max=64)
        a, b, control = model.state(), model.state(), model.control()
        peak = 0.0
        for step in range(960):  # 2 s; the uncorrected step diverged at 1.5 s
            solver.step(a, b, control, None, 1.0 / 480.0)
            a, b = b, a
            if step % 48 == 0:
                qd = a.body_qd.numpy()
                test.assertTrue(np.all(np.isfinite(qd)))
                peak = max(peak, float(np.abs(qd[:, 3:]).max()))
        peaks[inextensible] = peak
    test.assertLess(peaks[True], 1.5 * peaks[False])


add_function_test(
    TestSteelRod,
    "test_inextensible_long_chain_whip",
    test_inextensible_long_chain_whip,
    devices=get_cuda_test_devices(),
)


def test_carried_rod_state_follows_resets_and_velocity_edits(test, device):
    """An inextensible rod's carried momenta survive only an unedited state of the last step.

    A reset replays a fresh solver's trajectory exactly, and a velocity edit at an unchanged pose drops the carried
    momenta of its articulation for that step instead of being pulled back toward them.
    """

    def run(solver, model, steps):
        pipeline = newton.CollisionPipeline(model, deterministic=True)
        contacts = pipeline.contacts()
        a, b, control = model.state(), model.state(), model.control()
        for _ in range(steps):
            pipeline.collide(a, contacts)
            solver.step(a, b, control, contacts, 1.0 / 240.0)
            a, b = b, a
        return a

    def make(model):
        return newton.solvers.SolverSteel(
            model,
            articulated_contact_response="propagation",
            articulation_constraint_max=256,
            iterations=16,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )

    model, _bodies = _hanging_rod(device, inextensible=True)
    reused = make(model)
    test.assertTrue(reused._rod_material.carried)
    run(reused, model, 12)
    reused.reset(model.state())
    replay = run(reused, model, 12).body_q.numpy()
    fresh = run(make(model), model, 12).body_q.numpy()
    test.assertTrue(np.all(np.isfinite(replay)))
    np.testing.assert_array_equal(replay, fresh)

    # The rod's articulation keeps its carried momenta while the state is the step's own ...
    solver = make(model)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    a, b, control = model.state(), model.state(), model.control()
    art = int(model.joint_articulation.numpy()[int(model.articulation_start.numpy()[0])])
    for _ in range(6):
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, 1.0 / 240.0)
        a, b = b, a
    test.assertEqual(int(solver._rod_material.carried_valid.numpy()[art]), 1)
    # ... and drops them for a step whose joint velocities were edited at the same pose.
    a.joint_qd.zero_()
    pipeline.collide(a, contacts)
    solver.step(a, b, control, contacts, 1.0 / 240.0)
    test.assertEqual(int(solver._rod_material.carried_valid.numpy()[art]), 0)


def test_closed_rod_dahl_under_propagation_raises(test, device):
    """A closed rod with Dahl friction under the propagation response is refused: its closing joint has no history."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverSteel.register_custom_attributes(builder)
    angles = np.linspace(0.0, 2.0 * np.pi, 13)
    points = [wp.vec3(0.2 * np.cos(t), 0.2 * np.sin(t), 0.3) for t in angles]
    points[-1] = points[0]
    builder.add_rod(rod=newton.Rod(points, radius=0.01, closed=True), stretch_stiffness=1.0e5, bend_stiffness=2.0)
    model = builder.finalize(device=device)
    # Registered but inactive Dahl attributes (both zero) do not refuse the ring.
    newton.solvers.SolverSteel(model, **_PROPAGATION)
    model.steel.dahl_eps_max.fill_(0.05)
    model.steel.dahl_tau.fill_(0.1)
    with test.assertRaisesRegex(NotImplementedError, "Dahl hysteresis"):
        newton.solvers.SolverSteel(model, **_PROPAGATION)
    newton.solvers.SolverSteel(model)  # the immediate response carries the closing joint's history


_CABLE_RESPONSES = {
    "immediate": {"articulated_contact_response": "immediate", "articulation_constraint_max": 512},
    "propagation": {
        "articulated_contact_response": "propagation",
        "solve_mode": "colored",
        "propagation_same_articulation_rows": True,
        "propagation_constraint_max": 512,
    },
}


def _ground_cable(device, *, mu=0.6, links=12, length=0.6, radius=0.02, upright=False, torsion_radius=0.0, rolling=0.0):
    """A cable lying on (or, ``upright``, standing on its end on) the ground plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    newton.solvers.SolverSteel.register_custom_attributes(builder)
    builder.rigid_gap = 0.01
    if upright:
        start, direction = (0.0, 0.0, radius + 0.001), (0.0, 0.0, 1.0)
    else:
        start, direction = (-0.5 * length, 0.0, radius), (1.0, 0.0, 0.0)
    rod = newton.Rod.create_straight(
        start=start, direction=direction, length=length, segment_count=links, radius=radius
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=1200.0, mu=mu)
    builder.add_rod(rod=rod, cfg=cfg, bend_stiffness=5.0e1, bend_damping=1.0, stretch_stiffness=1.0e5)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    model = builder.finalize(device=device)
    if torsion_radius:
        model.steel.min_torsional_patch_radius.fill_(torsion_radius)
    if rolling:
        model.steel.condim.fill_(6)
        model.shape_material_mu_rolling.fill_(rolling)
    model.rigid_contact_max = max(model.rigid_contact_max or 0, 1024)
    return model


def _run_ground_cable(
    model, options, steps, *, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), settle=30, dt=1.0 / 240.0, on_step=None
):
    """Settle, set the root joint's velocity, and step; return the solver, final state and per-step body twists."""
    solver = newton.solvers.SolverSteel(model, iterations=24, **options)
    pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="latest")
    contacts = pipeline.contacts()
    a, b, control = model.state(), model.state(), model.control()
    for _ in range(settle):
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, dt)
        a, b = b, a
    if any(linear) or any(angular):
        root = int(np.flatnonzero(model.joint_parent.numpy() == -1)[0])
        dof = int(model.joint_qd_start.numpy()[root])
        qd = np.zeros_like(a.joint_qd.numpy())
        qd[dof : dof + 3] = linear
        qd[dof + 3 : dof + 6] = angular
        a.joint_qd.assign(qd)
        newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    twists = []
    for step in range(steps):
        if on_step is not None:
            on_step(step, a, solver)
        pipeline.collide(a, contacts)
        solver.step(a, b, control, contacts, dt)
        a, b = b, a
        twists.append(a.body_qd.numpy().copy())
    solver.check_constraint_capacity()
    return solver, a, np.array(twists)


def test_cable_friction_modes_stop_a_slide(test, device):
    """Every friction model stops a cable sliding along its axis at the Coulomb rate, under both responses."""
    dt, speed, mu = 1.0 / 240.0, 1.0, 0.5
    expected = speed / (mu * 9.81)
    for mode in ("fixed_normal", "bisection", "bisection_desaxce", "coulomb_newton"):
        for response, options in _CABLE_RESPONSES.items():
            with test.subTest(mode=mode, response=response):
                model = _ground_cable(device, mu=mu)
                opts = {**options, "friction_mode": mode, "friction_anchor_beta": 0.0}
                _, state, twists = _run_ground_cable(model, opts, 120, linear=(speed, 0.0, 0.0), dt=dt)
                test.assertTrue(np.all(np.isfinite(state.body_q.numpy())))
                stopped = np.flatnonzero(np.abs(twists[:, :, 0]).mean(axis=1) < 0.02)
                test.assertGreater(stopped.size, 0)
                test.assertAlmostEqual(stopped[0] * dt, expected, delta=0.03)
                test.assertLess(np.abs(twists[-1, :, 0]).max(), 0.01)


def test_cable_torsion_stops_a_spin_with_warm_start(test, device):
    """Contact torsion stops a cable spinning on its end, with the cable presets' warm start, under both responses."""
    for response, options in _CABLE_RESPONSES.items():
        for radius in (0.0, 0.01):
            with test.subTest(response=response, radius=radius):
                model = _ground_cable(device, links=3, length=0.12, radius=0.03, upright=True, torsion_radius=radius)
                opts = {
                    **options,
                    "contact_torsion_device": True,
                    "contact_warmstart": True,
                    "friction_anchor_beta": 0.0,
                }
                _, _, twists = _run_ground_cable(model, opts, 120, angular=(0.0, 0.0, 4.0))
                spin = np.abs(twists[-1, :, 5]).mean()
                if radius:
                    test.assertLess(spin, 1.0e-3)
                else:
                    test.assertGreater(spin, 3.9)


def test_cable_rolling_friction(test, device):
    """Rolling friction (steel:condim 6) stops a cable rolling sideways under both responses, at about the rate of
    a rolling cylinder, alone and with contact torsion; the free-body override does not reach cable links."""
    dt, speed, radius, rolling = 1.0 / 240.0, 0.5, 0.02, 0.002
    roll = {"linear": (0.0, speed, 0.0), "angular": (-speed / radius, 0.0, 0.0)}
    # A rolling capsule decelerates at mu_r g / (r (1 + I / (m r^2))), with I / (m r^2) between 0.4 and 0.5.
    slowest = speed / (rolling * 9.81 / (radius * 1.5))
    fastest = speed / (rolling * 9.81 / (radius * 1.4))
    for response, options in _CABLE_RESPONSES.items():
        for torsion in (0.0, 0.01):
            with test.subTest(response=response, torsion=torsion):
                model = _ground_cable(device, rolling=rolling, torsion_radius=torsion)
                opts = {**options, "friction_anchor_beta": 0.0}
                if torsion:
                    opts.update(contact_torsion_device=True, contact_warmstart=True)
                solver, state, twists = _run_ground_cable(model, opts, 240, dt=dt, **roll)
                test.assertTrue(solver._articulated_rolling)
                test.assertTrue(np.all(np.isfinite(state.body_q.numpy())))
                stopped = np.flatnonzero(np.abs(twists[:, :, 1]).mean(axis=1) < 0.01)
                test.assertGreater(stopped.size, 0)
                test.assertGreater(stopped[0] * dt, 0.8 * fastest)
                test.assertLess(stopped[0] * dt, 1.2 * slowest)
        with test.subTest(response=response, override_only=True):
            # Without condim the cable keeps rolling, even with the free-body override set.
            model = _ground_cable(device)
            opts = {**options, "free_rigid_rolling_friction": rolling}
            solver, _, twists = _run_ground_cable(model, opts, 240, dt=dt, **roll)
            test.assertFalse(solver._articulated_rolling)
            test.assertGreater(twists[-1, :, 1].mean(), 0.9 * speed)
        with test.subTest(response=response, upright=True):
            # Contact torsion still stops an upright spinning cable whose contact also has a rolling pair.
            model = _ground_cable(
                device, links=3, length=0.12, radius=0.03, upright=True, torsion_radius=0.01, rolling=rolling
            )
            opts = {**options, "contact_torsion_device": True, "contact_warmstart": True, "friction_anchor_beta": 0.0}
            _, _, twists = _run_ground_cable(model, opts, 120, angular=(0.0, 0.0, 4.0))
            test.assertLess(np.abs(twists[-1, :, 5]).mean(), 1.0e-3)


def test_cable_sleeps_and_wakes_with_warm_start(test, device):
    """A resting cable falls asleep (its rod gains are material, not a drive) and a force wakes it, warm-started or not."""
    for response, options in _CABLE_RESPONSES.items():
        for warm in (False, True):
            with test.subTest(response=response, warm=warm):
                model = _ground_cable(device)
                opts = {
                    **options,
                    "enable_sleeping": True,
                    "sleep_quiet_time": 0.25,
                    "contact_warmstart": warm,
                    "friction_anchor_beta": 0.0,
                }
                seen = {}

                def on_step(step, state, solver, seen=seen):
                    asleep = bool(np.any(solver._sleeping.asleep_steps.numpy() > 0))
                    if asleep and "slept" not in seen:
                        seen["slept"] = step
                        seen["pose"] = state.body_q.numpy().copy()
                    if step == 200:
                        force = state.body_f.numpy()
                        force[0, 2] = 5.0
                        state.body_f.assign(force)
                    if step == 202:
                        seen["woken"] = not asleep
                        state.body_f.zero_()

                solver, state, _ = _run_ground_cable(model, opts, 400, settle=0, on_step=on_step)
                test.assertIn("slept", seen)
                test.assertLess(seen["slept"], 150)
                test.assertTrue(seen["woken"])
                test.assertTrue(bool(np.any(solver._sleeping.asleep_steps.numpy() > 0)))
                test.assertTrue(np.all(np.isfinite(state.body_q.numpy())))
                # It settles back where it slept: the poke lifts one end briefly.
                test.assertLess(np.abs(state.body_q.numpy()[:, :3] - seen["pose"][:, :3]).max(), 0.02)


for _name, _fn in (
    (
        "test_carried_rod_state_follows_resets_and_velocity_edits",
        test_carried_rod_state_follows_resets_and_velocity_edits,
    ),
    ("test_closed_rod_dahl_under_propagation_raises", test_closed_rod_dahl_under_propagation_raises),
    ("test_cable_friction_modes_stop_a_slide", test_cable_friction_modes_stop_a_slide),
    ("test_cable_torsion_stops_a_spin_with_warm_start", test_cable_torsion_stops_a_spin_with_warm_start),
    ("test_cable_sleeps_and_wakes_with_warm_start", test_cable_sleeps_and_wakes_with_warm_start),
    ("test_cable_rolling_friction", test_cable_rolling_friction),
):
    add_function_test(TestSteelRod, _name, _fn, devices=get_cuda_test_devices())
