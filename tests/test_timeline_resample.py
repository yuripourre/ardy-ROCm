# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for timeline span/clip-length helpers and resize semantics."""

import torch

from ardy.motion_resample import resolve_prompt_source_span, resolve_resized_clip_length
from tests.timeline_test_utils import (
    make_linear_root_motion,
    mean_root_delta,
    resample_segment,
)


def test_padded_walking_bar_shrink_source_and_clip_length():
    """Padded bar (0, 200) with 80-frame clip shrinks to 50 without honoring padding."""
    clip_frames = 80
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, None)
    assert (source_start, source_end) == (0, 80)

    target_frames = resolve_resized_clip_length(
        clip_frames, source_start, source_end, 0, 50, has_later_prompt=False
    )
    assert target_frames == 50

    rots, roots = make_linear_root_motion(clip_frames)
    out_rots, out_roots, out_len = resample_segment(
        rots, roots, source_start, source_end, 0, 50, False
    )
    assert out_len == 50
    assert out_rots.shape[0] == 50
    assert out_roots.shape[0] == 50


def test_padded_walking_bar_stretch_clip_length():
    """Stretching padded bar to 150 keeps full 80-frame source and extends clip to 150."""
    clip_frames = 80
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, None)
    assert (source_start, source_end) == (0, 80)

    target_frames = resolve_resized_clip_length(
        clip_frames, source_start, source_end, 0, 150, has_later_prompt=False
    )
    assert target_frames == 150

    rots, roots = make_linear_root_motion(clip_frames)
    _, _, out_len = resample_segment(rots, roots, source_start, source_end, 0, 150, False)
    assert out_len == 150


def test_stale_short_span_ignores_old_end():
    """Stale span (0, 16) with 80-frame clip still sources full generated motion."""
    clip_frames = 80
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, None)
    assert (source_start, source_end) == (0, 80)

    target_frames = resolve_resized_clip_length(
        clip_frames, source_start, source_end, 0, 40, has_later_prompt=False
    )
    assert target_frames == 40


def test_packed_prompts_resize_first_segment():
    """First prompt (0, 10) with next at 10: shrink to 5 keeps suffix frames."""
    clip_frames = 80
    next_prompt_start = 10
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, next_prompt_start)
    assert (source_start, source_end) == (0, 10)

    target_frames = resolve_resized_clip_length(
        clip_frames, source_start, source_end, 0, 5, has_later_prompt=True
    )
    assert target_frames == 5 + (clip_frames - 10)

    rots, roots = make_linear_root_motion(clip_frames)
    _, _, out_len = resample_segment(rots, roots, source_start, source_end, 0, 5, True)
    assert out_len == 75


def test_restart_from_now_shaped_last_segment():
    """RFN-style tail segment [20:90) warped to 40 frames yields clip length 60."""
    clip_frames = 90
    old_start = 20
    source_start, source_end = resolve_prompt_source_span(old_start, clip_frames, None)
    assert (source_start, source_end) == (20, 90)

    new_start = 20
    new_end = 60
    target_frames = resolve_resized_clip_length(
        clip_frames, source_start, source_end, new_start, new_end, has_later_prompt=False
    )
    assert target_frames == 60

    rots, roots = make_linear_root_motion(clip_frames)
    out_rots, out_root, out_len = resample_segment(
        rots, roots, source_start, source_end, new_start, new_end, False
    )
    assert out_len == 60
    torch.testing.assert_close(out_root[:20], roots[:20])


def test_shrink_increases_per_frame_speed():
    """Shrinking 80 frames to 50 increases mean root delta (faster motion)."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, None)

    original_speed = mean_root_delta(roots)
    _, shrunk_root, _ = resample_segment(rots, roots, source_start, source_end, 0, 50, False)
    shrunk_speed = mean_root_delta(shrunk_root)

    assert shrunk_speed > original_speed


def test_stretch_decreases_per_frame_speed():
    """Stretching 80 frames to 150 decreases mean root delta (slower motion)."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, None)

    original_speed = mean_root_delta(roots)
    _, stretched_root, _ = resample_segment(rots, roots, source_start, source_end, 0, 150, False)
    stretched_speed = mean_root_delta(stretched_root)

    assert stretched_speed < original_speed


def test_resolve_prompt_source_span_clamps_start():
    """Start frame beyond clip is clamped to ``source_frames``."""
    source_start, source_end = resolve_prompt_source_span(100, 80, None)
    assert source_start == 80
    assert source_end == 80
