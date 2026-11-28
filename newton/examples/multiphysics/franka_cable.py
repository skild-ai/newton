# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measurements for the Franka rod pick-and-place scenes."""

import numpy as np
import warp as wp


@wp.kernel
def _midpoint_motion(
    ids: wp.array[wp.vec2i],
    half_length: float,
    poses: wp.array[wp.transform],
    twists: wp.array[wp.spatial_vector],
    result: wp.array[wp.spatial_vector],
):
    w = wp.tid()
    position = wp.vec3()
    velocity = wp.vec3()
    for side in range(2):
        body = ids[w][side]
        arm = wp.transform_vector(poses[body], wp.vec3(0.0, 0.0, half_length * float(1 - 2 * side)))
        position += wp.transform_get_translation(poses[body]) + arm
        velocity += wp.spatial_top(twists[body]) + wp.cross(wp.spatial_bottom(twists[body]), arm)
    result[w] = wp.spatial_vector(0.5 * position, 0.5 * velocity)


def _sample_midpoints(example, length):
    # Odd meshes have a central capsule. For even meshes, average the two
    # endpoint positions and velocities at the central material joint.
    n = example.payload_body_count_per_world
    left = example.payload_bodies[(n - 1) // 2 :: n]
    right = example.payload_bodies[n // 2 :: n]
    ids = wp.array(list(zip(left, right, strict=True)), dtype=wp.vec2i, device=example.device)
    samples = wp.empty(example.world_count, dtype=wp.spatial_vector, device=example.device)
    wp.launch(
        _midpoint_motion,
        dim=example.world_count,
        inputs=[ids, 0.0 if n % 2 else 0.5 * length / n, example.state_0.body_q, example.state_0.body_qd],
        outputs=[samples],
        device=example.device,
    )
    return samples.numpy()


def check_carry(example, length, lift_key, placement_key):
    frame = round(example.sim_time * example.fps)
    lift = round(example.key_times[lift_key] * example.fps)
    placed = round(example.key_times[placement_key] * example.fps)
    if frame in (lift, placed):
        xyz = _sample_midpoints(example, length)[:, :3]
        if frame == lift:
            rest_z = example.surface_z + example.payload_radius
            assert np.all(xyz[:, 2] > 0.5 * (rest_z + example.targets[lift_key, 2])), "Cable slipped during lift"
        else:
            error = np.linalg.norm(xyz[:, :2] - example.place_target_xy, axis=1)
            assert np.all(error < 0.01), f"Cable missed target before release. Errors [m] {error}"


def check_release(example, length, open_width):
    assert example.sim_time + 0.5 * example.frame_dt >= example.key_times[-1], "Run through jaw opening and retraction"
    joint_q = example.state_0.joint_q.numpy().reshape(example.world_count, -1)
    assert np.all(joint_q[:, [example.finger_idx0, example.finger_idx1]] > 0.9 * open_width), "Jaws still closed"
    hands = [i for i in example.franka_bodies if example.model.body_label[i].endswith("fr3_hand")]
    assert len(hands) == example.world_count
    hand_z = example.state_0.body_q.numpy()[hands, 2]
    assert np.all(hand_z > example.surface_z + 0.14), "Gripper did not retract"
    sample = _sample_midpoints(example, length)
    height_error = np.abs(sample[:, 2] - (example.surface_z + example.payload_radius))
    assert np.all(height_error < 0.5 * example.payload_radius), (
        f"Cable still above the table. Height errors [m] {height_error}"
    )
    speed = np.linalg.norm(sample[:, 3:], axis=1)
    assert np.all(speed < 0.02), f"Released cable still moving. Speeds [m/s] {speed}"
    distance = np.linalg.norm(sample[:, :2] - example.place_target_xy, axis=1)
    assert np.all(distance < 0.01), f"Released midpoint missed the target. Errors [m] {distance}"
