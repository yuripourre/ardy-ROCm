# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal resampling utilities for skeletal motion."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def resample_local_motion(
    local_rot_mats: torch.Tensor,
    root_trans: torch.Tensor,
    target_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly resample local rotations and root translation to ``target_frames``.

    Maps source timeline ``[0, T-1]`` onto ``[0, target_frames-1]`` via linear
    interpolation for root translation and SLERP for joint rotations.

    Args:
        local_rot_mats: Local rotation matrices, shape ``[T, J, 3, 3]``.
        root_trans: Root translations, shape ``[T, 3]``.
        target_frames: Desired output frame count (must be >= 1).

    Returns:
        Resampled ``(local_rot_mats, root_trans)`` with shape ``[target_frames, ...]``.
    """
    if target_frames < 1:
        raise ValueError(f"target_frames must be >= 1, got {target_frames}")

    source_frames = local_rot_mats.shape[0]
    if source_frames == target_frames:
        return local_rot_mats, root_trans

    device = local_rot_mats.device
    dtype = local_rot_mats.dtype

    if source_frames == 1:
        return (
            local_rot_mats.expand(target_frames, *local_rot_mats.shape[1:]).clone(),
            root_trans.expand(target_frames, *root_trans.shape[1:]).clone(),
        )

    src_times = np.linspace(0, source_frames - 1, source_frames)
    dst_times = np.linspace(0, source_frames - 1, target_frames)

    root_np = root_trans.detach().cpu().numpy()
    resampled_root = np.stack(
        [np.interp(dst_times, src_times, root_np[:, axis]) for axis in range(root_np.shape[1])],
        axis=1,
    )

    num_joints = local_rot_mats.shape[1]
    rots_np = local_rot_mats.detach().cpu().numpy()
    resampled_rots = np.zeros((target_frames, num_joints, 3, 3), dtype=rots_np.dtype)
    for joint_idx in range(num_joints):
        joint_rots = Rotation.from_matrix(rots_np[:, joint_idx])
        slerp = Slerp(src_times, joint_rots)
        resampled_rots[:, joint_idx] = slerp(dst_times).as_matrix()

    return (
        torch.from_numpy(resampled_rots).to(device=device, dtype=dtype),
        torch.from_numpy(resampled_root).to(device=device, dtype=root_trans.dtype),
    )
