# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root path interpolation utilities (Viser-free)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import interp1d

from ardy.motion_rep.smooth_root import get_smooth_root_pos


def interpolate_root_path(
    frame_indices: np.ndarray,
    root_positions: np.ndarray,
    t: np.ndarray,
    smooth: bool = False,
) -> np.ndarray:
    """Interpolate root positions along X/Z for the given frame indices.

    Args:
        frame_indices: Sorted keyframe frame numbers, shape ``[N]``.
        root_positions: Root positions at keyframes, shape ``[N, 3]``.
        t: Frame indices to sample, shape ``[T]``.
        smooth: When True and at least three keyframes exist, apply root smoothing.

    Returns:
        Interpolated root positions, shape ``[T, 3]``.
    """
    if len(frame_indices) == 0:
        raise ValueError("frame_indices must not be empty")
    if len(frame_indices) == 1:
        return np.repeat(root_positions[:1], len(t), axis=0)

    x = root_positions[:, 0]
    z = root_positions[:, 2]

    interp_x = interp1d(frame_indices, x, kind="linear")
    interp_z = interp1d(frame_indices, z, kind="linear")

    x_new = interp_x(t)
    z_new = interp_z(t)
    path3d = np.stack([x_new, np.zeros_like(x_new), z_new], axis=1)

    if smooth and len(frame_indices) >= 3:
        path3d = get_smooth_root_pos(torch.from_numpy(path3d[None]))[0].numpy()
    return path3d
