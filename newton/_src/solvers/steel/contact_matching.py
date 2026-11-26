# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact matching across steps for SolverSteel's warm start, without the collision pipeline's matching.

Each step the matcher sorts the contacts by shape pair and matches every contact to the nearest contact of the
same shape pair that the solver solved last, with the rules of :class:`~newton.CollisionPipeline`'s body-frame
matching at its default thresholds: the contacts' world midpoints lie within :data:`MATCH_DISTANCE` with normals
within :data:`MATCH_NORMAL_DOT`, or each witness point stayed within :data:`MATCH_DISTANCE` in its own body's frame
with the normal within :data:`MATCH_NORMAL_DOT` in both frames. The nearest claimant of a previous contact wins it,
so the match is one to one.

The history is the sorted keys and slots of the solved contacts, their witness points and normal (in the pair's
order), and the body poses they were solved at: about 88 bytes per contact slot and 32 per body, against the
pipeline's sorted contact copies and matcher history.
"""

from __future__ import annotations

import warp as wp

from .friction_patches import WARP_DETERMINISM_OFF

# Steel's kernels order their own reductions (the claims below are deterministic by construction).
wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})

# CollisionPipeline's defaults, contact_matching_pos_threshold and contact_matching_normal_dot_threshold.
MATCH_DISTANCE = wp.constant(0.0005)
MATCH_NORMAL_DOT = wp.constant(0.995)
_NO_KEY = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))
_NO_CLAIM = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))


@wp.func
def _pair(sa: int, sb: int):
    """Sort key of a shape pair and whether the contact's sides are swapped into the pair's order."""
    ka = sa + 1
    kb = sb + 1
    flip = int(0)
    if ka > kb:
        t = ka
        ka = kb
        kb = t
        flip = 1
    return (wp.int64(ka) << wp.int64(32)) | wp.int64(kb), flip


@wp.func
def _local_point(q: wp.array[wp.transform], body: int, point: wp.vec3):
    result = point
    if body >= 0:
        result = wp.transform_point(wp.transform_inverse(q[body]), point)
    return result


@wp.func
def _local_vector(q: wp.array[wp.transform], body: int, v: wp.vec3):
    result = v
    if body >= 0:
        result = wp.quat_rotate_inv(wp.transform_get_rotation(q[body]), v)
    return result


@wp.func
def _witness(
    c: int,
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    q: wp.array[wp.transform],
):
    """Bodies, world witness points and normal of contact ``c``, swapped into its shape pair's order."""
    a = int(-1)
    b = int(-1)
    if shape0[c] >= 0:
        a = shape_body[shape0[c]]
    if shape1[c] >= 0:
        b = shape_body[shape1[c]]
    n = -normal[c]
    pa = point0[c]
    if a >= 0:
        pa = wp.transform_point(q[a], pa)
    pa = pa - margin0[c] * n
    pb = point1[c]
    if b >= 0:
        pb = wp.transform_point(q[b], pb)
    pb = pb + margin1[c] * n
    _key, flip = _pair(shape0[c], shape1[c])
    if flip != 0:
        return b, a, pb, pa, -n
    return a, b, pa, pb, n


@wp.kernel(enable_backward=False)
def _prepare_keys(
    count: wp.array[int],
    capacity: int,
    shape0: wp.array[int],
    shape1: wp.array[int],
    keys: wp.array[wp.int64],
    slots: wp.array[int],
):
    c = wp.tid()
    slots[c] = c
    keys[c] = _NO_KEY
    if c < count[0] and c < capacity:
        key, _flip = _pair(shape0[c], shape1[c])
        keys[c] = key


@wp.func
def _sorted_range(keys: wp.array[wp.int64], n: int, key: wp.int64):
    lo = int(0)
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    start = lo
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return start, lo


@wp.kernel(enable_backward=False)
def _match(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    q: wp.array[wp.transform],
    prev_q: wp.array[wp.transform],
    prev_keys: wp.array[wp.int64],
    prev_slots: wp.array[int],
    prev_pa: wp.array[wp.vec3],
    prev_pb: wp.array[wp.vec3],
    prev_normal: wp.array[wp.vec3],
    # out
    best_prev: wp.array[int],
    claims: wp.array[wp.int64],
):
    c = wp.tid()
    best_prev[c] = -1
    if c >= count[0]:
        return
    a, b, pa, pb, n = _witness(c, shape0, shape1, point0, point1, normal, margin0, margin1, shape_body, q)
    key, _flip = _pair(shape0[c], shape1[c])
    mid = 0.5 * (pa + pb)
    la = _local_point(q, a, pa)
    lb = _local_point(q, b, pb)
    na = _local_vector(q, a, n)
    nb = _local_vector(q, b, n)
    start, stop = _sorted_range(prev_keys, prev_keys.shape[0], key)
    limit = MATCH_DISTANCE * MATCH_DISTANCE
    best = int(-1)
    best_d = float(limit)
    for j in range(start, stop):
        p = prev_slots[j]
        ppa = prev_pa[p]
        ppb = prev_pb[p]
        pn = prev_normal[p]
        d = float(1.0e30)
        if wp.dot(n, pn) >= MATCH_NORMAL_DOT:
            d = wp.length_sq(mid - 0.5 * (ppa + ppb))
        if (
            wp.dot(na, _local_vector(prev_q, a, pn)) >= MATCH_NORMAL_DOT
            and wp.dot(nb, _local_vector(prev_q, b, pn)) >= MATCH_NORMAL_DOT
        ):
            da = wp.length_sq(la - _local_point(prev_q, a, ppa))
            db = wp.length_sq(lb - _local_point(prev_q, b, ppb))
            d = wp.min(d, wp.max(da, db))
        if d <= best_d and (best < 0 or d < best_d):
            best = p
            best_d = d
    if best < 0:
        return
    best_prev[c] = best
    # Nearer claims win; equal distances go to the lower contact index.
    rank = wp.int64(wp.min(best_d / limit, 1.0) * 1073741823.0)
    wp.atomic_min(claims, best, (rank << wp.int64(32)) | wp.int64(c))


@wp.kernel(enable_backward=False)
def _finalize(
    count: wp.array[int],
    best_prev: wp.array[int],
    claims: wp.array[wp.int64],
    history_generation: wp.array[wp.int32],
    # out
    match_index: wp.array[int],
    match_generation: wp.array[wp.int32],
):
    c = wp.tid()
    if c == 0:
        match_generation[0] = history_generation[0]
    result = int(-1)
    if c < count[0]:
        p = best_prev[c]
        if p >= 0 and (claims[p] & wp.int64(0xFFFFFFFF)) == wp.int64(c):
            result = p
    match_index[c] = result


@wp.kernel(enable_backward=False)
def _store(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    q: wp.array[wp.transform],
    keys: wp.array[wp.int64],
    slots: wp.array[int],
    # out
    prev_q: wp.array[wp.transform],
    prev_keys: wp.array[wp.int64],
    prev_slots: wp.array[int],
    prev_pa: wp.array[wp.vec3],
    prev_pb: wp.array[wp.vec3],
    prev_normal: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < prev_q.shape[0]:
        prev_q[i] = q[i]
    if i >= prev_keys.shape[0]:
        return
    prev_keys[i] = keys[i]
    prev_slots[i] = slots[i]
    if i < count[0]:
        _a, _b, pa, pb, n = _witness(i, shape0, shape1, point0, point1, normal, margin0, margin1, shape_body, q)
        prev_pa[i] = pa
        prev_pb[i] = pb
        prev_normal[i] = n


class ContactMatcher:
    """Matches each step's contacts to the contacts the solver last solved (see the module docstring)."""

    def __init__(self, capacity: int, body_count: int, device):
        n = max(int(capacity), 1)
        self.capacity = n
        # Radix-sort buffers of this step's keys and slots (twice the capacity).
        self._keys = wp.full(2 * n, _NO_KEY, dtype=wp.int64, device=device)
        self._slots = wp.zeros(2 * n, dtype=wp.int32, device=device)
        self._prev_keys = wp.full(n, _NO_KEY, dtype=wp.int64, device=device)
        self._prev_slots = wp.zeros(n, dtype=wp.int32, device=device)
        self._prev_pa = wp.zeros(n, dtype=wp.vec3, device=device)
        self._prev_pb = wp.zeros(n, dtype=wp.vec3, device=device)
        self._prev_normal = wp.zeros(n, dtype=wp.vec3, device=device)
        self._prev_q = wp.zeros(max(int(body_count), 1), dtype=wp.transform, device=device)
        self._best_prev = wp.zeros(n, dtype=wp.int32, device=device)
        self._claims = wp.zeros(n, dtype=wp.int64, device=device)
        #: Each contact's index in the contact set the solver last solved, or -1.
        self.match_index = wp.full(n, -1, dtype=wp.int32, device=device)
        #: Generation of the solved contact set :attr:`match_index` refers to.
        self.match_generation = wp.zeros(1, dtype=wp.int32, device=device)

    def _contact_inputs(self, contacts, shape_body):
        return [
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            shape_body,
        ]

    def match(self, model, state, contacts, history_generation):
        """Sort this step's contacts and match them against the stored history."""
        device = model.device
        n = self.capacity
        count = contacts.rigid_contact_count
        wp.launch(
            _prepare_keys,
            dim=n,
            inputs=[count, contacts.rigid_contact_max, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1],
            outputs=[self._keys, self._slots],
            device=device,
        )
        wp.utils.radix_sort_pairs(self._keys, self._slots, n)
        self._claims.fill_(_NO_CLAIM)
        wp.launch(
            _match,
            dim=min(n, contacts.rigid_contact_max),
            inputs=[
                count,
                *self._contact_inputs(contacts, model.shape_body),
                state.body_q,
                self._prev_q,
                self._prev_keys,
                self._prev_slots,
                self._prev_pa,
                self._prev_pb,
                self._prev_normal,
            ],
            outputs=[self._best_prev, self._claims],
            device=device,
        )
        wp.launch(
            _finalize,
            dim=n,
            inputs=[count, self._best_prev, self._claims, history_generation],
            outputs=[self.match_index, self.match_generation],
            device=device,
        )

    def store(self, model, state, contacts):
        """Keep this step's sorted contacts, witness points and body poses for the next match."""
        wp.launch(
            _store,
            dim=max(self.capacity, self._prev_q.shape[0]),
            inputs=[
                contacts.rigid_contact_count,
                *self._contact_inputs(contacts, model.shape_body),
                state.body_q,
                self._keys,
                self._slots,
            ],
            outputs=[
                self._prev_q,
                self._prev_keys,
                self._prev_slots,
                self._prev_pa,
                self._prev_pb,
                self._prev_normal,
            ],
            device=model.device,
        )
