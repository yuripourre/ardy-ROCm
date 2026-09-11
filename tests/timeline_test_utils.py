# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for timeline resample and animation-limit scenario tests."""

import torch

from ardy.motion_resample import (
    resample_local_motion,
    resolve_prompt_source_span,
    resolve_resized_clip_length,
    resolve_restart_from_now_keep_end,
)


def make_linear_root_motion(num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Motion with linear root travel along X for speed assertions."""
    root_trans = torch.stack(
        [
            torch.linspace(0.0, float(num_frames - 1), num_frames),
            torch.zeros(num_frames),
            torch.zeros(num_frames),
        ],
        dim=1,
    )
    local_rot_mats = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(num_frames, 1, 3, 3).clone()
    return local_rot_mats, root_trans


def mean_root_delta(root_trans: torch.Tensor) -> float:
    if root_trans.shape[0] < 2:
        return 0.0
    deltas = root_trans[1:] - root_trans[:-1]
    return float(deltas.norm(dim=1).mean())


def resample_segment(
    local_rot_mats: torch.Tensor,
    root_trans: torch.Tensor,
    source_start: int,
    source_end: int,
    new_start: int,
    new_end: int,
    has_later_prompt: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Mirror concat/crop semantics used by ``_resample_session_motion_span``."""
    source_frames = local_rot_mats.shape[0]
    new_len = new_end - new_start
    source_len = source_end - source_start

    if source_len < 1:
        segment_rots = local_rot_mats[:new_len]
        segment_root = root_trans[:new_len]
    elif source_len == new_len:
        segment_rots = local_rot_mats[source_start:source_end]
        segment_root = root_trans[source_start:source_end]
    else:
        segment_rots, segment_root = resample_local_motion(
            local_rot_mats[source_start:source_end],
            root_trans[source_start:source_end],
            new_len,
        )

    prefix_len = min(new_start, source_start, source_frames)
    prefix_rots = local_rot_mats[:prefix_len]
    prefix_roots = root_trans[:prefix_len]
    if new_start > prefix_len:
        pad_len = new_start - prefix_len
        hold_frame = max(0, prefix_len - 1)
        pad_rots = local_rot_mats[hold_frame].unsqueeze(0).expand(pad_len, -1, -1, -1)
        pad_roots = root_trans[hold_frame].unsqueeze(0).expand(pad_len, -1)
        prefix_rots = torch.cat([prefix_rots, pad_rots], dim=0)
        prefix_roots = torch.cat([prefix_roots, pad_roots], dim=0)

    if has_later_prompt:
        suffix_rots = local_rot_mats[source_end:]
        suffix_roots = root_trans[source_end:]
        out_rots = torch.cat([prefix_rots, segment_rots, suffix_rots], dim=0)
        out_root = torch.cat([prefix_roots, segment_root, suffix_roots], dim=0)
        target_frames = out_rots.shape[0]
    else:
        out_rots = torch.cat([prefix_rots, segment_rots], dim=0)
        out_root = torch.cat([prefix_roots, segment_root], dim=0)
        target_frames = resolve_resized_clip_length(
            source_frames,
            source_start,
            source_end,
            new_start,
            new_end,
            has_later_prompt=False,
        )
        out_rots = out_rots[:target_frames]
        out_root = out_root[:target_frames]

    return out_rots, out_root, target_frames


def apply_initial_animation_limit(
    local_rot_mats: torch.Tensor,
    root_trans: torch.Tensor,
    target_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``_resample_session_motion_to_length`` for a single motion sample."""
    return resample_local_motion(local_rot_mats, root_trans, target_frames)


def trim_motion_before_rfn(
    local_rot_mats: torch.Tensor,
    root_trans: torch.Tensor,
    current_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``restart_from_now`` prefix trim (keep playhead frame inclusive)."""
    keep_end = resolve_restart_from_now_keep_end(current_frame)
    return local_rot_mats[:keep_end], root_trans[:keep_end]


def apply_rfn_tail_warp(
    local_rot_mats: torch.Tensor,
    root_trans: torch.Tensor,
    current_frame: int,
    gui_frames: int,
    *,
    trim_prefix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Mirror ``restart_from_now`` post-generation span resample."""
    if trim_prefix:
        local_rot_mats, root_trans = trim_motion_before_rfn(
            local_rot_mats, root_trans, current_frame
        )
    source_frames = local_rot_mats.shape[0]
    source_start, source_end = resolve_prompt_source_span(current_frame, source_frames, None)
    new_start = current_frame
    new_end = current_frame + gui_frames
    out_rots, out_root, out_len = resample_segment(
        local_rot_mats,
        root_trans,
        source_start,
        source_end,
        new_start,
        new_end,
        has_later_prompt=False,
    )
    return out_rots, out_root, out_len
