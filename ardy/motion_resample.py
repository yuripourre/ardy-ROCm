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


def interpolate_local_pose(
    rots_a: torch.Tensor,
    root_a: torch.Tensor,
    rots_b: torch.Tensor,
    root_b: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate between two poses (SLERP joints, linear root).

    Args:
        rots_a: Local rotation matrices for pose A, shape ``[J, 3, 3]``.
        root_a: Root translation for pose A, shape ``[3]``.
        rots_b: Local rotation matrices for pose B, shape ``[J, 3, 3]``.
        root_b: Root translation for pose B, shape ``[3]``.
        alpha: Blend weight in ``[0, 1]`` (0 = A, 1 = B).

    Returns:
        Interpolated ``(local_rot_mats, root_trans)`` with shapes ``[J, 3, 3]`` and ``[3]``.
    """
    alpha = float(alpha)
    if alpha <= 0.0:
        return rots_a.clone(), root_a.clone()
    if alpha >= 1.0:
        return rots_b.clone(), root_b.clone()

    device = rots_a.device
    dtype = rots_a.dtype
    root_interp = root_a * (1.0 - alpha) + root_b * alpha

    rots_np_a = rots_a.detach().cpu().numpy()
    rots_np_b = rots_b.detach().cpu().numpy()
    num_joints = rots_a.shape[0]
    interp_rots = np.zeros((num_joints, 3, 3), dtype=rots_np_a.dtype)
    key_times = np.array([0.0, 1.0])
    for joint_idx in range(num_joints):
        joint_rots = Rotation.from_matrix(np.stack([rots_np_a[joint_idx], rots_np_b[joint_idx]]))
        slerp = Slerp(key_times, joint_rots)
        interp_rots[joint_idx] = slerp(alpha).as_matrix()

    return (
        torch.from_numpy(interp_rots).to(device=device, dtype=dtype),
        root_interp.to(device=device, dtype=root_b.dtype),
    )


def append_cycle_blend_frames(
    local_rot_mats: torch.Tensor,
    root_trans: torch.Tensor,
    blend_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append blend frames from last pose toward first, then drop the duplicate last.

    Output length is ``T + blend_frames`` where ``T`` is the input frame count.
    ``blend_frames`` is the number of closing frames to keep (``K``); interpolation
    uses ``K + 1`` steps so the final duplicate of frame 0 can be dropped.

    Args:
        local_rot_mats: Local rotation matrices, shape ``[T, J, 3, 3]``.
        root_trans: Root translations, shape ``[T, 3]``.
        blend_frames: Number of closing frames to add ``K`` (must be >= 1).

    Returns:
        Extended ``(local_rot_mats, root_trans)`` with shape ``[T + K, ...]``.
    """
    if blend_frames < 1:
        raise ValueError(f"blend_frames must be >= 1, got {blend_frames}")

    source_frames = local_rot_mats.shape[0]
    if source_frames < 1:
        raise ValueError("local_rot_mats must have at least one frame")

    rots_last = local_rot_mats[-1]
    root_last = root_trans[-1]
    rots_first = local_rot_mats[0]
    root_first = root_trans[0]

    blend_rots = []
    blend_roots = []
    num_steps = blend_frames + 1
    for step_idx in range(1, num_steps + 1):
        alpha = step_idx / num_steps
        interp_rots, interp_root = interpolate_local_pose(
            rots_last, root_last, rots_first, root_first, alpha
        )
        blend_rots.append(interp_rots)
        blend_roots.append(interp_root)

    extended_rots = torch.cat([local_rot_mats, torch.stack(blend_rots, dim=0)], dim=0)
    extended_root = torch.cat([root_trans, torch.stack(blend_roots, dim=0)], dim=0)

    return extended_rots[:-1], extended_root[:-1]


def resolve_prompt_source_span(
    old_start: int,
    source_frames: int,
    next_prompt_start: int | None,
) -> tuple[int, int]:
    """Resolve motion span owned by a prompt, ignoring padded visual bar ends.

    The visual timeline bar is often longer than generated motion (open-ended
    padding). Source duration is the real motion owned by this prompt up to the
    next prompt or clip end, never the padded ``old_end``.

    Args:
        old_start: Prompt start frame.
        source_frames: Total frames in the motion clip.
        next_prompt_start: Start frame of the next prompt, or ``None`` if last.

    Returns:
        ``(source_start, source_end)`` with exclusive ``source_end`` for slicing.
    """
    if source_frames < 1:
        return 0, 0

    source_start = max(0, min(old_start, source_frames))
    if next_prompt_start is None:
        source_end = source_frames
    else:
        source_end = max(source_start, min(next_prompt_start, source_frames))
    return source_start, source_end


def resolve_resized_clip_length(
    source_frames: int,
    source_start: int,
    source_end: int,
    new_start: int,
    new_end: int,
    has_later_prompt: bool,
) -> int:
    """Compute output clip length after resampling a prompt span.

    Args:
        source_frames: Original clip frame count.
        source_start: Inclusive start of the prompt's source motion.
        source_end: Exclusive end of the prompt's source motion.
        new_start: New prompt bar start frame.
        new_end: New prompt bar end frame.
        has_later_prompt: Whether a later prompt exists after this one.

    Returns:
        Target clip frame count after concat (prefix + warped segment + suffix).
    """
    new_len = new_end - new_start
    if not has_later_prompt:
        return new_start + new_len

    suffix_len = max(0, source_frames - source_end)
    return new_start + new_len + suffix_len


def resolve_animation_end_frame(
    animation_limit_base_frame: int,
    gui_frames: int,
    target_animation_end_frame: int | None,
) -> int | None:
    """Return the last allowed frame index when Animation Frames is set, else stored cap."""
    if gui_frames > 0:
        return animation_limit_base_frame + gui_frames - 1
    return target_animation_end_frame


def resolve_effective_playback_end(
    max_frame_idx: int,
    animation_end: int | None,
) -> int:
    """Last playable frame index, respecting an optional animation cap."""
    if max_frame_idx < 0:
        return 0
    end_frame = max_frame_idx
    if animation_end is not None:
        end_frame = min(end_frame, animation_end)
    return end_frame


def clamp_playhead_frame(
    frame_idx: int,
    max_frame_idx: int,
    effective_end: int | None,
) -> int:
    """Clamp playhead to generated frames inside the active clip/bar."""
    frame_idx = int(frame_idx)
    if max_frame_idx < 0:
        return max(0, frame_idx)
    end_frame = resolve_effective_playback_end(max_frame_idx, effective_end)
    return max(0, min(frame_idx, end_frame))


def should_skip_generation_at_limit(
    max_frame_idx: int,
    animation_end: int | None,
    skip_animation_frame_limit: bool,
) -> bool:
    """Return True when generation should stop because the animation cap is reached."""
    if skip_animation_frame_limit:
        return False
    if animation_end is None:
        return False
    return max_frame_idx >= animation_end
