# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver selection for examples that use Newton's rod material."""

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
    """Collision-pipeline options a native Steel run needs: contact matching, which its contact warm starting reads,
    in the bodies' frames so contacts that moving bodies carry together (a grasp) keep their match."""
    if getattr(args, "solver", "vbd") == "steel":
        return {"contact_matching": "latest", "contact_matching_body_frame": True}
    return {}


def make_solver(
    model,
    args,
    iterations,
    *,
    contact_rows=32,
    parallel_tree=False,
    enable_joint_limits=False,
    coupled_contact_passes=0,
    propagation_interval=3,
    **vbd_options,
):
    if getattr(args, "solver", "vbd") == "steel":
        if not model.device.is_cuda:
            raise ValueError("Steel cable examples require a CUDA device.")
        propagation = getattr(args, "contact_response", "propagation") == "propagation"
        # The propagation response solves contacts and rod clamps as body rows; the generalized rows then
        # carry only joint limits and drives. Its capacity is per world: the contact buffer is shared by all
        # worlds, so each world takes its share with headroom for uneven worlds, and at most 64 rows per body
        # of a world (the cable bundle peaks at 52), which the pipeline's bound for many replicated worlds
        # would otherwise exceed many times over.
        dense_rows = min(contact_rows, 64) if propagation else contact_rows
        if propagation and model.world_count > 1:
            share = max(64, 2 * (contact_rows + model.world_count - 1) // model.world_count)
            per_body = 64 * max(model.body_count // model.world_count, 1)
            contact_rows = min(contact_rows, share, max(64, per_body))
        return newton.solvers.SolverSteel(
            model,
            iterations=iterations,
            solve_mode=getattr(args, "solve_mode", "serial"),
            articulated_contact_response="propagation" if propagation else "immediate",
            propagation_same_articulation_rows=propagation,
            propagation_constraint_max=contact_rows if propagation else None,
            articulation_constraint_max=dense_rows,
            row_watermark=True,
            # Each step publishes its body state through the solver's own kinematics and the next step starts from
            # them instead of re-deriving them from the joint coordinates.
            reuse_kinematics=True,
            # Each step starts from the previous step's contact impulses, so a held grasp or a cable resting on a
            # surface keeps its load instead of rebuilding it every substep (the pipeline matches contacts, see
            # pipeline_options).
            contact_warmstart=True,
            # Rod trees propagate every third iteration, where the sweeps' momentum steps too; the sweeps' body
            # estimates carry the impulses between. Scenes that push long cables sideways propagate every one.
            propagation_interval=propagation_interval if propagation else 1,
            parallel_tree=parallel_tree,
            enable_joint_limits=enable_joint_limits,
            coupled_contact_passes=coupled_contact_passes,
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
