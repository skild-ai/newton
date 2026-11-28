# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Forward kinematics of chain articulations without a serial walk over their joints.

:func:`~.kernels.eval_rigid_fk_kinematics` walks an articulation's joints on one thread, and on a long chain (a
cable) every step of that walk waits on the previous one's memory and transforms. Here only the two compositions
that really chain run serially, on one lane over values staged in shared memory: the joints' relative transforms
into body poses, and the joint twists into body twists and accelerations. Everything else runs one joint per
thread with the same functions as the walk (:func:`~.kernels.link_joint_motion`).

A chain here lists its joints parent first, each joint's parent being the previous joint's child, and holds only
joints whose relative transform comes from their coordinates alone (ROD, FREE, DISTANCE, FIXED).
"""

import functools

import numpy as np
import warp as wp

from ...sim import JointType
from .friction_patches import WARP_DETERMINISM_OFF
from .kernels import _compute_body_net_wrench, jcalc_tau, jcalc_transform, link_joint_motion

# Longest chain the serial compositions stage in shared memory; longer ones keep the walk.
CHAIN_FK_MAX = 256
CHAIN_FK_BLOCK = 32


def chain_fk_plan(model, articulation_joint_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the chain flag per articulation and the joints of the flagged ones (in order)."""
    count = max(model.articulation_count, 1)
    flags = np.zeros(count, dtype=np.int32)
    if not model.articulation_count or not model.joint_count:
        return flags, np.zeros(0, dtype=np.int32)
    starts = model.articulation_start.numpy()
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()
    jtype = model.joint_type.numpy()
    supported = {int(JointType.ROD), int(JointType.FREE), int(JointType.DISTANCE), int(JointType.FIXED)}
    joints = []
    for art in range(model.articulation_count):
        start, end = int(starts[art]), int(articulation_joint_end[art])
        n = end - start
        if n < 2 or n > CHAIN_FK_MAX or parent[start] >= 0:
            continue
        if not all(int(jtype[i]) in supported for i in range(start, end)):
            continue
        if not any(int(jtype[i]) == int(JointType.ROD) for i in range(start, end)):
            continue
        if not all(parent[i] == child[i - 1] for i in range(start + 1, end)):
            continue
        # FREE and DISTANCE below the root reference their twist at the articulation origin: keep the walk.
        if any(int(jtype[i]) in (int(JointType.FREE), int(JointType.DISTANCE)) for i in range(start + 1, end)):
            continue
        flags[art] = 1
        joints.extend(range(start, end))
    return flags, np.asarray(joints, dtype=np.int32)


@wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
def chain_relative_transforms(
    joints: wp.array[int],
    joint_articulation: wp.array[int],
    articulation_active: wp.array[int],
    skip: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    # outputs
    relative: wp.array[wp.transform],
):
    """Each chain joint's child pose in its parent's frame: ``X_p * X_j(q) * X_c^-1``."""
    i = joints[wp.tid()]
    art = joint_articulation[i]
    if articulation_active[art] == 0 or skip[art] != 0:
        return
    X_j = jcalc_transform(
        joint_type[i],
        joint_axis,
        joint_qd_start[i],
        joint_dof_dim[i, 0],
        joint_dof_dim[i, 1],
        joint_q,
        joint_q_start[i],
    )
    relative[i] = (joint_X_p[i] * X_j) * wp.transform_inverse(joint_X_c[i])


@wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
def chain_joint_motion(
    joints: wp.array[int],
    joint_articulation: wp.array[int],
    articulation_active: wp.array[int],
    skip: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    joint_v: wp.array[wp.spatial_vector],
    joint_c: wp.array[wp.spatial_vector],
):
    """Each chain joint's motion subspace, joint twist and apparent acceleration bias (the walk's per-joint terms)."""
    i = joints[wp.tid()]
    art = joint_articulation[i]
    if articulation_active[art] == 0 or skip[art] != 0:
        return
    v_j, c_app = link_joint_motion(
        i,
        joint_parent[i],
        joint_child[i],
        articulation_origin[art],
        joint_type,
        joint_q_start,
        joint_qd_start,
        joint_q,
        joint_qd,
        joint_axis,
        joint_dof_dim,
        body_q,
        body_q_com,
        joint_X_p,
        joint_S_s,
    )
    joint_v[i] = v_j
    joint_c[i] = c_app


_PRELUDE = (
    r"""
#if defined(__CUDA_ARCH__)
    #define FK_MAX """
    + str(CHAIN_FK_MAX)
    + r"""
    const int lane = threadIdx.x & 31;
    const int slot = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (slot >= nchains) return;
    const int art = chain_arts.data[slot];
    if (articulation_active.data[art] == 0 || skip.data[art] != 0) return;
    const int start = articulation_start.data[art];
    const int n = articulation_joint_end.data[art] - start;
    const unsigned FULL = 0xffffffffu;
"""
)

_COMPOSE = (
    _PRELUDE
    + r"""
    // Body poses: X_k = X_{k-1} * R_k from the root, on lane 0 over the staged relative transforms.
    __shared__ float s_xf[FK_MAX * 7];
    wp::transform_t<float>* s_x = reinterpret_cast<wp::transform_t<float>*>(s_xf);
    for (int k = lane; k < n; k += 32) s_x[k] = relative.data[start + k];
    __syncwarp(FULL);
    if (lane == 0) {
        wp::transform_t<float> x = s_x[0];
        for (int k = 1; k < n; ++k) {
            x = wp::transform_multiply(x, s_x[k]);
            s_x[k] = x;
        }
    }
    __syncwarp(FULL);
    for (int k = lane; k < n; k += 32) {
        const int child = joint_child.data[start + k];
        body_q.data[child] = s_x[k];
        body_q_com.data[child] = wp::transform_multiply(s_x[k], body_X_com.data[child]);
    }
    if (lane == 0) {
        const int root = joint_child.data[start];
        articulation_origin.data[art] = wp::transform_point(s_x[0], body_com.data[root]);
    }
#endif
"""
)

_ACCUMULATE = (
    _PRELUDE
    + r"""
    // Body twists and accelerations: v_k = v_{k-1} + v_j, a_k = a_{k-1} + v_k x v_j + c_app, on lane 0.
    __shared__ float s_vf[FK_MAX * 6];
    __shared__ float s_af[FK_MAX * 6];
    wp::spatial_vector_t<float>* s_v = reinterpret_cast<wp::spatial_vector_t<float>*>(s_vf);
    wp::spatial_vector_t<float>* s_a = reinterpret_cast<wp::spatial_vector_t<float>*>(s_af);
    for (int k = lane; k < n; k += 32) {
        s_v[k] = joint_v.data[start + k];
        s_a[k] = joint_c.data[start + k];
    }
    __syncwarp(FULL);
    if (lane == 0) {
        wp::spatial_vector_t<float> v(0.0f), a(0.0f);
        for (int k = 0; k < n; ++k) {
            const wp::spatial_vector_t<float> vj = s_v[k];
            v = wp::add(v, vj);
            // Steel's spatial_cross: angular parts are the bottom three.
            const wp::vec_t<3, float> va(v[0], v[1], v[2]), wa(v[3], v[4], v[5]);
            const wp::vec_t<3, float> vb(vj[0], vj[1], vj[2]), wb(vj[3], vj[4], vj[5]);
            const wp::vec_t<3, float> w = wp::cross(wa, wb);
            const wp::vec_t<3, float> l = wp::add(wp::cross(wa, vb), wp::cross(va, wb));
            wp::spatial_vector_t<float> cr(0.0f);
            cr[0] = l[0]; cr[1] = l[1]; cr[2] = l[2]; cr[3] = w[0]; cr[4] = w[1]; cr[5] = w[2];
            a = wp::add(wp::add(a, cr), s_a[k]);
            s_v[k] = v;
            s_a[k] = a;
        }
    }
    __syncwarp(FULL);
    for (int k = lane; k < n; k += 32) {
        const int child = joint_child.data[start + k];
        body_v_s.data[child] = s_v[k];
        body_a_s.data[child] = s_a[k];
    }
    if (lane == 0) valid_out.data[art] = 1;
#endif
"""
)


@functools.cache
def chain_compose_kernel():
    """Kernel composing each chain's body poses from its relative transforms (one warp per chain)."""

    @wp.func_native(_COMPOSE)
    def chain_compose_native(
        nchains: int,
        chain_arts: wp.array[int],
        articulation_active: wp.array[int],
        skip: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_child: wp.array[int],
        relative: wp.array[wp.transform],
        body_X_com: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
    ): ...

    def chain_compose(  # pragma: no cover - compiled by wp.kernel
        nchains: int,
        chain_arts: wp.array[int],
        articulation_active: wp.array[int],
        skip: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_child: wp.array[int],
        relative: wp.array[wp.transform],
        body_X_com: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
    ):
        _tid = wp.tid()
        chain_compose_native(
            nchains,
            chain_arts,
            articulation_active,
            skip,
            articulation_start,
            articulation_joint_end,
            joint_child,
            relative,
            body_X_com,
            body_com,
            body_q,
            body_q_com,
            articulation_origin,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(chain_compose)


@functools.cache
def chain_accumulate_kernel():
    """Kernel accumulating each chain's body twists and accelerations from its joint terms (one warp per chain)."""

    @wp.func_native(_ACCUMULATE)
    def chain_accumulate_native(
        nchains: int,
        chain_arts: wp.array[int],
        articulation_active: wp.array[int],
        skip: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_child: wp.array[int],
        joint_v: wp.array[wp.spatial_vector],
        joint_c: wp.array[wp.spatial_vector],
        body_v_s: wp.array[wp.spatial_vector],
        body_a_s: wp.array[wp.spatial_vector],
        valid_out: wp.array[int],
    ): ...

    def chain_accumulate(  # pragma: no cover - compiled by wp.kernel
        nchains: int,
        chain_arts: wp.array[int],
        articulation_active: wp.array[int],
        skip: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_child: wp.array[int],
        joint_v: wp.array[wp.spatial_vector],
        joint_c: wp.array[wp.spatial_vector],
        body_v_s: wp.array[wp.spatial_vector],
        body_a_s: wp.array[wp.spatial_vector],
        valid_out: wp.array[int],
    ):
        _tid = wp.tid()
        chain_accumulate_native(
            nchains,
            chain_arts,
            articulation_active,
            skip,
            articulation_start,
            articulation_joint_end,
            joint_child,
            joint_v,
            joint_c,
            body_v_s,
            body_a_s,
            valid_out,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(chain_accumulate)


@wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
def chain_body_wrenches(
    joints: wp.array[int],
    joint_articulation: wp.array[int],
    articulation_active: wp.array[int],
    joint_child: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    wrench: wp.array[wp.spatial_vector],
):
    """Each chain joint's child body wrench before its descendants' are added (the walk's net wrench minus them)."""
    i = joints[wp.tid()]
    art = joint_articulation[i]
    if articulation_active[art] == 0:
        return
    wrench[i] = _compute_body_net_wrench(
        joint_child[i],
        wp.spatial_vector(),
        articulation_origin[art],
        body_fb_s,
        body_f_ext,
        body_flags,
        body_q,
        body_com,
    )


@wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
def chain_joint_tau(
    joints: wp.array[int],
    joint_articulation: wp.array[int],
    articulation_active: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_stiffness: wp.array[float],
    joint_rest_q: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
    # outputs
    tau: wp.array[float],
):
    """Each chain joint's generalized force from its subtree wrench (the walk's ``jcalc_tau``)."""
    i = joints[wp.tid()]
    art = joint_articulation[i]
    if articulation_active[art] == 0:
        return
    jcalc_tau(
        joint_type[i],
        joint_S_s,
        joint_f,
        joint_q,
        joint_qd,
        joint_stiffness,
        joint_rest_q,
        joint_damping,
        joint_q_start[i],
        joint_qd_start[i],
        joint_dof_dim[i, 0],
        joint_dof_dim[i, 1],
        wrench[i],
        0,
        tau,
    )


@wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
def and_not_flags(a: wp.array[int], b: wp.array[int], out: wp.array[int]):
    i = wp.tid()
    out[i] = 1 if (a[i] != 0 and b[i] == 0) else 0


@wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)
def or_flags(a: wp.array[int], b: wp.array[int], out: wp.array[int]):
    i = wp.tid()
    out[i] = 1 if (a[i] != 0 or b[i] != 0) else 0


_SUFFIX = (
    _PRELUDE.replace("skip.data[art] != 0", "false")
    + r"""
    // Subtree wrenches: f_k = w_k + f_{k+1} from the tip, on lane 0; a body's accumulator gets its child's.
    __shared__ float s_ff[FK_MAX * 6];
    wp::spatial_vector_t<float>* s_f = reinterpret_cast<wp::spatial_vector_t<float>*>(s_ff);
    for (int k = lane; k < n; k += 32) s_f[k] = wrench.data[start + k];
    __syncwarp(FULL);
    if (lane == 0) {
        wp::spatial_vector_t<float> f = s_f[n - 1];
        for (int k = n - 2; k >= 0; --k) {
            f = wp::add(s_f[k], f);
            s_f[k] = f;
        }
        // Rod chains carry no drive rows.
        row_counts.data[art] = 0;
    }
    __syncwarp(FULL);
    for (int k = lane; k < n; k += 32) {
        wrench.data[start + k] = s_f[k];
        if (k + 1 < n) body_ft_s.data[joint_child.data[start + k]] = s_f[k + 1];
    }
#endif
"""
)


@functools.cache
def chain_suffix_kernel():
    """Kernel summing each chain's subtree wrenches from the tip (one warp per chain)."""

    @wp.func_native(_SUFFIX)
    def chain_suffix_native(
        nchains: int,
        chain_arts: wp.array[int],
        articulation_active: wp.array[int],
        skip: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_child: wp.array[int],
        wrench: wp.array[wp.spatial_vector],
        body_ft_s: wp.array[wp.spatial_vector],
        row_counts: wp.array[int],
    ): ...

    def chain_suffix(  # pragma: no cover - compiled by wp.kernel
        nchains: int,
        chain_arts: wp.array[int],
        articulation_active: wp.array[int],
        skip: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        joint_child: wp.array[int],
        wrench: wp.array[wp.spatial_vector],
        body_ft_s: wp.array[wp.spatial_vector],
        row_counts: wp.array[int],
    ):
        _tid = wp.tid()
        chain_suffix_native(
            nchains,
            chain_arts,
            articulation_active,
            skip,
            articulation_start,
            articulation_joint_end,
            joint_child,
            wrench,
            body_ft_s,
            row_counts,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(chain_suffix)


class ChainFK:
    """The chain articulations of a model and their scratch, with a launcher matching the walk's outputs."""

    def __init__(self, model, articulation_joint_end: np.ndarray):
        flags, joints = chain_fk_plan(model, articulation_joint_end)
        device = model.device
        self.count = int(flags.sum())
        count = len(flags)
        self.flags = wp.array(flags, dtype=wp.int32, device=device)
        self.arts = wp.array(np.flatnonzero(flags).astype(np.int32), dtype=wp.int32, device=device)
        self.joints = wp.array(joints, dtype=wp.int32, device=device)
        joint_count = max(model.joint_count, 1)
        self.relative = wp.zeros(joint_count, dtype=wp.transform, device=device)
        self.joint_v = wp.zeros(joint_count, dtype=wp.spatial_vector, device=device)
        self.joint_c = wp.zeros(joint_count, dtype=wp.spatial_vector, device=device)
        self.skip = wp.zeros(count, dtype=wp.int32, device=device)
        self.walk_active = wp.zeros(count, dtype=wp.int32, device=device)
        self.wrench = wp.zeros(joint_count, dtype=wp.spatial_vector, device=device)

    def tau_walk_active(self, model, active):
        """The articulations the tau walk handles: the active ones that are not chains."""
        wp.launch(
            and_not_flags,
            dim=len(self.flags),
            inputs=[active, self.flags],
            outputs=[self.walk_active],
            device=model.device,
        )
        return self.walk_active

    def launch_tau(
        self,
        model,
        articulation_joint_end,
        active,
        tau_inputs,
        body_ft_s,
        row_counts,
        tau,
    ):
        """The tau walk's outputs for the chains: subtree wrenches, body accumulators, tau and no drive rows.

        ``tau_inputs`` is the walk's list (joint arrays, forces, springs, ``joint_S_s``, body wrenches and poses).
        """
        if self.count == 0:
            return
        device = model.device
        (
            joint_type,
            _joint_parent,
            joint_child,
            joint_articulation,
            joint_qd_start,
            joint_q_start,
            joint_dof_dim,
            joint_f,
            joint_q,
            joint_qd,
            stiffness,
            rest_q,
            damping,
            joint_S_s,
            body_fb_s,
            body_f_ext,
            body_flags,
            body_q,
            body_com,
            articulation_origin,
        ) = tau_inputs
        nj = len(self.joints)
        wp.launch(
            chain_body_wrenches,
            dim=nj,
            inputs=[
                self.joints,
                joint_articulation,
                active,
                joint_child,
                articulation_origin,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
            ],
            outputs=[self.wrench],
            device=device,
        )
        wp.launch(
            chain_suffix_kernel(),
            dim=self.count * 32,
            inputs=[
                self.count,
                self.arts,
                active,
                active,
                model.articulation_start,
                articulation_joint_end,
                joint_child,
            ],
            outputs=[self.wrench, body_ft_s, row_counts],
            block_dim=CHAIN_FK_BLOCK,
            device=device,
        )
        wp.launch(
            chain_joint_tau,
            dim=nj,
            inputs=[
                self.joints,
                joint_articulation,
                active,
                joint_type,
                joint_qd_start,
                joint_q_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                stiffness,
                rest_q,
                damping,
                joint_S_s,
                self.wrench,
            ],
            outputs=[tau],
            device=device,
        )

    def walk_skip(self, model, reuse, none):
        """The skip flags of the walk: ``reuse`` plus the chains, which this class handles."""
        if reuse is none:
            return self.flags
        wp.launch(or_flags, dim=len(self.flags), inputs=[reuse, self.flags], outputs=[self.skip], device=model.device)
        return self.skip

    def launch(
        self,
        model,
        articulation_joint_end,
        joint_q,
        joint_qd,
        body_X_com,
        articulation_active,
        skip,
        body_q,
        body_q_com,
        articulation_origin,
        joint_S_s,
        body_v_s,
        body_a_s,
        valid_out,
    ):
        """Write the walk's outputs for the chain articulations that are active and not skipped."""
        if self.count == 0:
            return
        device = model.device
        nj = len(self.joints)
        wp.launch(
            chain_relative_transforms,
            dim=nj,
            inputs=[
                self.joints,
                model.joint_articulation,
                articulation_active,
                skip,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
            ],
            outputs=[self.relative],
            device=device,
        )
        common = [self.count, self.arts, articulation_active, skip, model.articulation_start, articulation_joint_end]
        wp.launch(
            chain_compose_kernel(),
            dim=self.count * 32,
            inputs=[*common, model.joint_child, self.relative, body_X_com, model.body_com],
            outputs=[body_q, body_q_com, articulation_origin],
            block_dim=CHAIN_FK_BLOCK,
            device=device,
        )
        wp.launch(
            chain_joint_motion,
            dim=nj,
            inputs=[
                self.joints,
                model.joint_articulation,
                articulation_active,
                skip,
                articulation_origin,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q,
                joint_qd,
                model.joint_axis,
                model.joint_dof_dim,
                body_q,
                body_q_com,
                model.joint_X_p,
            ],
            outputs=[joint_S_s, self.joint_v, self.joint_c],
            device=device,
        )
        wp.launch(
            chain_accumulate_kernel(),
            dim=self.count * 32,
            inputs=[*common, model.joint_child, self.joint_v, self.joint_c],
            outputs=[body_v_s, body_a_s, valid_out],
            block_dim=CHAIN_FK_BLOCK,
            device=device,
        )
