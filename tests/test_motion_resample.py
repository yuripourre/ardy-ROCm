# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ardy.motion_resample import append_cycle_blend_frames, interpolate_local_pose, resample_local_motion


def _make_motion(num_frames: int, num_joints: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    root_trans = torch.stack(
        [
            torch.linspace(0.0, float(num_frames - 1), num_frames),
            torch.zeros(num_frames),
            torch.linspace(0.0, float(2 * (num_frames - 1)), num_frames),
        ],
        dim=1,
    )
    local_rot_mats = torch.zeros(num_frames, num_joints, 3, 3)
    for frame_idx in range(num_frames):
        angle = frame_idx * (np.pi / max(num_frames - 1, 1))
        rot = torch.from_numpy(Rotation.from_euler("y", angle).as_matrix()).float()
        for joint_idx in range(num_joints):
            local_rot_mats[frame_idx, joint_idx] = rot
    return local_rot_mats, root_trans


def test_resample_preserves_endpoints():
    local_rot_mats, root_trans = _make_motion(120)
    resampled_rots, resampled_root = resample_local_motion(local_rot_mats, root_trans, 15)

    assert resampled_rots.shape == (15, 3, 3, 3)
    assert resampled_root.shape == (15, 3)
    torch.testing.assert_close(resampled_root[0], root_trans[0])
    torch.testing.assert_close(resampled_root[-1], root_trans[-1])
    torch.testing.assert_close(resampled_rots[0], local_rot_mats[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(resampled_rots[-1], local_rot_mats[-1], atol=1e-5, rtol=1e-5)


def test_resample_downsample_and_upsample():
    local_rot_mats, root_trans = _make_motion(120)
    downsampled_rots, downsampled_root = resample_local_motion(local_rot_mats, root_trans, 15)
    assert downsampled_rots.shape[0] == 15
    assert downsampled_root.shape[0] == 15

    short_rots, short_root = _make_motion(10)
    upsampled_rots, upsampled_root = resample_local_motion(short_rots, short_root, 15)
    assert upsampled_rots.shape[0] == 15
    assert upsampled_root.shape[0] == 15
    torch.testing.assert_close(upsampled_root[0], short_root[0])
    torch.testing.assert_close(upsampled_root[-1], short_root[-1])


def test_resample_noop_when_lengths_match():
    local_rot_mats, root_trans = _make_motion(15)
    out_rots, out_root = resample_local_motion(local_rot_mats, root_trans, 15)
    assert out_rots is local_rot_mats
    assert out_root is root_trans


def test_interpolate_local_pose_endpoints():
    local_rot_mats, root_trans = _make_motion(10)
    rots_a, root_a = local_rot_mats[0], root_trans[0]
    rots_b, root_b = local_rot_mats[-1], root_trans[-1]

    at_zero_rots, at_zero_root = interpolate_local_pose(rots_a, root_a, rots_b, root_b, 0.0)
    torch.testing.assert_close(at_zero_rots, rots_a)
    torch.testing.assert_close(at_zero_root, root_a)

    at_one_rots, at_one_root = interpolate_local_pose(rots_a, root_a, rots_b, root_b, 1.0)
    torch.testing.assert_close(at_one_rots, rots_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(at_one_root, root_b)


def test_append_cycle_blend_frames_length():
    local_rot_mats, root_trans = _make_motion(10)
    blend_frames = 4
    out_rots, out_root = append_cycle_blend_frames(local_rot_mats, root_trans, blend_frames)

    expected_frames = 10 + blend_frames - 1
    assert out_rots.shape[0] == expected_frames
    assert out_root.shape[0] == expected_frames
    torch.testing.assert_close(out_rots[0], local_rot_mats[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_root[0], root_trans[0])
    last_blend_rots, _ = interpolate_local_pose(
        local_rot_mats[-1], root_trans[-1], local_rot_mats[0], root_trans[0], 0.75
    )
    torch.testing.assert_close(out_rots[-1], last_blend_rots, atol=1e-4, rtol=1e-4)


def test_append_cycle_blend_frames_k_one():
    local_rot_mats, root_trans = _make_motion(10)
    out_rots, out_root = append_cycle_blend_frames(local_rot_mats, root_trans, 1)
    assert out_rots.shape[0] == 10
    assert out_root.shape[0] == 10
