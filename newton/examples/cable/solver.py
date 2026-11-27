# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver selection for examples that use Newton's rod material."""

import copy
import math

import warp as wp

import newton
import newton.examples


def create_parser(baseline="vbd"):
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        choices=(baseline, "steel"),
        default=baseline,
        help=f"Run the {baseline} scene or solve its rods natively with Steel on CUDA.",
    )
    parser.add_argument(
        "--solve-mode",
        choices=("serial", "colored"),
        default="serial",
        help="Contact sweep order for Steel's immediate articulation response.",
    )
    parser.add_argument(
        "--rod-substeps",
        type=int,
        default=None,
        help="Steel: substeps per frame, replacing the scene's preset.",
    )
    parser.add_argument(
        "--rod-iterations",
        type=int,
        default=None,
        help="Steel: solver iterations per substep, replacing the scene's preset.",
    )
    parser.add_argument(
        "--contact-response",
        choices=("immediate", "propagation"),
        default="propagation",
        help="Steel: contacts on rods as dense generalized rows, or as body rows propagated through the tree.",
    )
    return parser


def budget(args, substeps, iterations, propagation=None):
    """Return a scene's substeps and solver iterations, with Steel's command-line overrides applied.

    ``propagation`` is the scene's budget under Steel's propagation contact response, whose rod trees
    respond exactly through their articulated-body factorization and need fewer substeps and sweeps.
    """
    if getattr(args, "solver", "vbd") == "steel":
        if propagation is not None and getattr(args, "contact_response", "propagation") == "propagation":
            substeps, iterations = propagation
        substeps = getattr(args, "rod_substeps", None) or substeps
        iterations = getattr(args, "rod_iterations", None) or iterations
    return int(substeps), int(iterations)


def pipeline_options(args):
    """Collision-pipeline options a cable example needs for its solver.

    Native Steel warm-starts its contacts from its own friction-patch history, which matches each contact to its
    predecessor, so the pipeline needs no contact matching (and keeps none of its sorted history); VBD needs none
    either.
    """
    del args
    return {}


def add_world_count_arg(parser):
    """Add ``--world-count``: the cable examples replicate their scene into that many independent worlds."""
    return newton.examples.add_world_count_arg(parser)


def replicate_worlds(template, args, clamps, *, ground=True, ground_cfg=None):
    """Replicate a one-world ``template`` into ``--world-count`` worlds, over one shared ground plane if ``ground``.

    Pin the template's fixtures with ``clamps`` (or pass ``None``) first; they are repeated for every world. Returns the builder to
    finalize, the world count and ``expand``, which maps template body indices to every world's, world by world.
    """
    world_count = max(1, int(getattr(args, "world_count", 1) or 1))
    builder = newton.ModelBuilder(up_axis=template.up_axis, gravity=template.gravity)
    builder.rigid_gap = template.rigid_gap
    builder.default_shape_cfg = copy.deepcopy(template.default_shape_cfg)
    if ground:
        builder.add_ground_plane(cfg=ground_cfg)
    builder.replicate(template, copy_count=world_count)
    stride = template.body_count
    if clamps is not None:
        clamps.replicate(world_count, stride, template.joint_count, template.articulation_count)

    def expand(ids):
        return [w * stride + int(i) for w in range(world_count) for i in ids]

    return builder, world_count, expand


def contact_budget(model, per_body, *, minimum=64):
    """Rigid contacts one world holds: ``per_body`` contacts for each body of a world, at least ``minimum``.

    Each example's ``per_body`` is its scene's peak contact count over the whole task, under Steel and VBD alike,
    with a quarter on top. Without it the collision pipeline sizes its buffer for every shape pair's worst case,
    many times what a cable scene makes, and Steel keeps friction-patch and warm-start state for every slot.
    Both solvers flag a pass that makes more contacts than this (Steel's ``check_constraint_capacity``).
    Pass ``rigid_contact_max=contact_budget(...) * model.world_count`` to the pipeline.
    """
    bodies = max(model.body_count // max(model.world_count, 1), 1)
    return 32 * math.ceil(max(minimum, per_body * bodies) / 32)


def make_solver(
    model,
    args,
    iterations,
    *,
    contact_rows=32,
    contacts=None,
    same_articulation_contacts=None,
    parallel_tree=False,
    enable_joint_limits=False,
    coupled_contact_passes=0,
    propagation_interval=3,
    deterministic=False,
    **vbd_options,
):
    if getattr(args, "solver", "vbd") == "steel":
        if not model.device.is_cuda:
            raise ValueError("Steel cable examples require a CUDA device.")
        if model.rigid_contact_max is None:
            # No collision pipeline published a capacity: the scene makes no rigid contacts, so Steel keeps no
            # per-contact state (its fallback estimate sizes it for every shape pair touching at once).
            model.rigid_contact_max = 0
        propagation = getattr(args, "contact_response", "propagation") == "propagation"
        # ``contact_rows`` is the rows one world holds: three per contact of the world's budget (a normal and two
        # tangents) and six per clamped body. The propagation response solves contacts and rod clamps as body
        # rows; the generalized rows then carry only joint limits and drives.
        dense_rows = min(contact_rows, 64) if propagation else contact_rows
        return newton.solvers.SolverSteel(
            model,
            iterations=iterations,
            solve_mode=getattr(args, "solve_mode", "serial"),
            articulated_contact_response="propagation" if propagation else "immediate",
            propagation_same_articulation_rows=propagation,
            propagation_constraint_max=contact_rows if propagation else None,
            # The colored sweep keeps one unit per contact (and per clamp closure) instead of one per row.
            propagation_contact_max=contacts if propagation else None,
            # Contacts between two links of one rod that one world holds: from 256 worlds each keeps its exact row
            # responses in a slot of this many, while every other contact rebuilds them from its bodies' responses.
            propagation_same_articulation_contact_max=same_articulation_contacts if propagation else None,
            articulation_constraint_max=dense_rows,
            row_watermark=True,
            # Each step publishes its body state through the solver's own kinematics and the next step starts from
            # them instead of re-deriving them from the joint coordinates.
            reuse_kinematics=True,
            # Each step starts from the previous step's contact impulses, so a held grasp or a cable resting on a
            # surface keeps its load instead of rebuilding it every substep (Steel matches the contacts itself, see
            # pipeline_options).
            contact_warmstart=True,
            # Point friction: two tangent rows per contact, warm-started, without friction-patch anchors. Every
            # cable task holds its grasps and passes its checks this way, and the patches' per-contact history
            # would otherwise be Steel's largest buffer in contact-dense scenes.
            friction_anchor_beta=0.0,
            # Rod trees propagate every third iteration, where the sweeps' momentum steps too; the sweeps' body
            # estimates carry the impulses between. Scenes that push long cables sideways propagate every one.
            propagation_interval=propagation_interval if propagation else 1,
            parallel_tree=parallel_tree,
            enable_joint_limits=enable_joint_limits,
            coupled_contact_passes=coupled_contact_passes,
            # Rows reserved in a fixed order: with a deterministic collision pipeline every world of a replicated
            # scene then steps bitwise alike, run to run (cheap with few contacts per world, as in a grasp).
            deterministic=wp.DeterministicMode.RUN_TO_RUN if deterministic else None,
        )
    return newton.solvers.SolverVBD(model, iterations=iterations, **vbd_options)


def check_capacity(solver):
    if isinstance(solver, newton.solvers.SolverSteel):
        solver.check_constraint_capacity()


@wp.kernel
def _set_clamp_frames(
    bodies: wp.array[int],
    joints: wp.array[int],
    targets: wp.array[wp.transform],
    frames: wp.array[wp.transform],
):
    i = wp.tid()
    frames[joints[i]] = targets[bodies[i]]


@wp.kernel
def _publish_root_targets(
    bodies: wp.array[int],
    targets: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    dt: float,
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
):
    body = bodies[wp.tid()]
    before = poses[body]
    after = targets[body]
    displacement = wp.transform_point(after, com[body]) - wp.transform_point(before, com[body])
    dq = wp.transform_get_rotation(after) * wp.quat_inverse(wp.transform_get_rotation(before))
    if dq[3] < 0.0:
        dq = -dq
    axis, angle = wp.quat_to_axis_angle(dq)
    velocities[body] = wp.spatial_vector(displacement / dt, axis * (angle / dt))
    poses[body] = after


class CableClamps:
    """Prescribed world poses for rod fixtures in the cable examples."""

    def __init__(self, args):
        self.steel = getattr(args, "solver", "vbd") == "steel"
        self.bodies = []
        self.joints = []
        self.moving_roots = []
        self.root_articulations = []

    def pin(self, builder, body, *, moving=False):
        if self.steel:
            for j, (parent, child, kind) in enumerate(
                zip(builder.joint_parent, builder.joint_child, builder.joint_type, strict=True)
            ):
                if parent < 0 and child == body and kind == newton.JointType.FREE:
                    builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
                    if moving:
                        self.moving_roots.append(body)
                        self.root_articulations.append(builder.joint_articulation[j])
                    return
            self.bodies.append(body)
        else:
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

    def finish(self, builder):
        # Append closures only after the tree joint ranges are complete.
        for body in self.bodies:
            self.joints.append(builder.add_joint_fixed(-1, body, parent_xform=builder.body_q[body]))

    def replicate(self, world_count, bodies, joints, articulations):
        """Repeat the pins made on a one-world template for each of ``world_count`` replicated worlds.

        ``bodies``, ``joints`` and ``articulations`` are the template's counts (each world's index stride).
        """
        self.bodies = [w * bodies + b for w in range(world_count) for b in self.bodies]
        self.joints = [w * joints + j for w in range(world_count) for j in self.joints]
        self.moving_roots = [w * bodies + b for w in range(world_count) for b in self.moving_roots]
        self.root_articulations = [w * articulations + a for w in range(world_count) for a in self.root_articulations]

    def bind(self, model):
        if self.steel:
            self.body_ids = wp.array(self.bodies, dtype=int, device=model.device)
            self.joint_ids = wp.array(self.joints, dtype=int, device=model.device)
            self.targets = wp.clone(model.body_q)
            self.frames = model.joint_X_p
            self.model = model
            if self.moving_roots:
                self.roots = wp.array(self.moving_roots, dtype=int, device=model.device)
                self.articulations = wp.array(self.root_articulations, dtype=int, device=model.device)

    def pose_buffers(self, state_in, state_out):
        if self.steel:
            return [self.targets, self.targets]
        return [state_in.body_q, state_out.body_q]

    def update(self, state, dt):
        if self.steel and self.moving_roots:
            wp.launch(
                _publish_root_targets,
                dim=len(self.moving_roots),
                inputs=[self.roots, self.targets, self.model.body_com, dt],
                outputs=[state.body_q, state.body_qd],
                device=self.model.device,
            )
            newton.eval_ik(self.model, state, state.joint_q, state.joint_qd, indices=self.articulations)
        if self.steel and self.bodies:
            wp.launch(
                _set_clamp_frames,
                dim=len(self.bodies),
                inputs=[self.body_ids, self.joint_ids, self.targets],
                outputs=[self.frames],
                device=self.frames.device,
            )
