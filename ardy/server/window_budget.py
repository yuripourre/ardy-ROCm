# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-window sizing for the API server."""

import math
from typing import Optional


def compute_window_num_frames(
    history_length: int,
    gen_horizon_len: int,
    num_frames_per_token: int,
    max_window_len: int,
    history_start_idx: int = 0,
    max_constraint_idx: Optional[int] = None,
    future_crop_length: int = 0,
) -> int:
    """Number of frames of the sequence visible to the model in one step."""
    num_frames = history_length + gen_horizon_len

    if max_constraint_idx is not None:
        num_frames = max(num_frames, max_constraint_idx - history_start_idx + 1)
        num_frames = min(num_frames, future_crop_length + history_length + gen_horizon_len)
        num_frames = math.ceil(num_frames / num_frames_per_token) * num_frames_per_token

    num_frames = max(min(num_frames, max_window_len), history_length + gen_horizon_len)
    return num_frames
