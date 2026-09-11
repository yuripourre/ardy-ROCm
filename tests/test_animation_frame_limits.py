# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for animation frame limits, RFN warping, and multi-segment chains."""

import torch

from ardy.motion_resample import (
    clamp_playhead_frame,
    resolve_animation_end_frame,
    resolve_effective_playback_end,
    resolve_prompt_source_span,
    should_skip_generation_at_limit,
)
from tests.timeline_test_utils import (
    apply_initial_animation_limit,
    apply_rfn_tail_warp,
    make_linear_root_motion,
    mean_root_delta,
    resample_segment,
)


def test_resolve_animation_end_frame_uses_gui_frames():
    assert resolve_animation_end_frame(0, 40, None) == 39
    assert resolve_animation_end_frame(30, 25, None) == 54


def test_resolve_animation_end_frame_falls_back_to_target():
    assert resolve_animation_end_frame(0, 0, 79) == 79
    assert resolve_animation_end_frame(10, 0, None) is None


def test_resolve_effective_playback_end():
    assert resolve_effective_playback_end(99, 54) == 54
    assert resolve_effective_playback_end(40, None) == 40
    assert resolve_effective_playback_end(-1, 54) == 0


def test_clamp_playhead_frame():
    assert clamp_playhead_frame(80, 99, 54) == 54
    assert clamp_playhead_frame(30, 99, 54) == 30
    assert clamp_playhead_frame(5, -1, None) == 5


def test_should_skip_generation_at_limit():
    assert should_skip_generation_at_limit(39, 39, False) is True
    assert should_skip_generation_at_limit(38, 39, False) is False
    assert should_skip_generation_at_limit(100, 39, True) is False
    assert should_skip_generation_at_limit(100, None, False) is False


def test_initial_generation_warps_to_n_frames():
    """Full-clip animation limit warps uncapped generation down to N frames."""
    rots, roots = make_linear_root_motion(120)
    limited_rots, limited_roots = apply_initial_animation_limit(rots, roots, 40)

    assert limited_rots.shape[0] == 40
    assert limited_roots.shape[0] == 40
    torch.testing.assert_close(limited_roots[0], roots[0])
    torch.testing.assert_close(limited_roots[-1], roots[-1])

    end_frame = resolve_animation_end_frame(0, 40, None)
    assert end_frame == 39
    assert should_skip_generation_at_limit(39, end_frame, False) is True


def test_initial_limit_shrink_increases_speed():
    rots, roots = make_linear_root_motion(120)
    _, limited_roots = apply_initial_animation_limit(rots, roots, 40)
    assert mean_root_delta(limited_roots) > mean_root_delta(roots)


def test_rfn_warps_tail_preserves_prefix():
    """RFN at frame 20 warps tail to 40 frames while keeping prefix intact."""
    rots, roots = make_linear_root_motion(90)
    out_rots, out_root, out_len = apply_rfn_tail_warp(rots, roots, 20, 40)

    assert out_len == 60
    assert out_rots.shape[0] == 60
    torch.testing.assert_close(out_root[:20], roots[:20])

    end_frame = resolve_animation_end_frame(20, 40, None)
    assert end_frame == 59


def test_rfn_tail_warp_not_trimmed():
    """RFN tail warp keeps full source span, not a short stale bar."""
    rots, roots = make_linear_root_motion(90)
    source_start, source_end = resolve_prompt_source_span(20, 90, None)
    assert (source_start, source_end) == (20, 90)

    _, out_root, out_len = apply_rfn_tail_warp(rots, roots, 20, 40)
    assert out_len == 60
    assert out_root.shape[0] == 60


def test_two_segment_chain():
    """Limit 40, then RFN at 30 with 25 frames."""
    rots, roots = make_linear_root_motion(120)
    rots, roots = apply_initial_animation_limit(rots, roots, 40)
    prefix_before_rfn = roots[:30].clone()

    rots, roots, out_len = apply_rfn_tail_warp(rots, roots, 30, 25)
    assert out_len == 55
    torch.testing.assert_close(roots[:30], prefix_before_rfn)

    end_frame = resolve_animation_end_frame(30, 25, None)
    assert end_frame == 54
    source_start, source_end = resolve_prompt_source_span(30, 55, None)
    assert (source_start, source_end) == (30, 55)


def test_three_segment_chain():
    """40 frames, RFN@30->25, then RFN@50->20."""
    rots, roots = make_linear_root_motion(120)
    rots, roots = apply_initial_animation_limit(rots, roots, 40)
    rots, roots, _ = apply_rfn_tail_warp(rots, roots, 30, 25)
    prefix_before_second_rfn = roots[:50].clone()

    rots, roots, out_len = apply_rfn_tail_warp(rots, roots, 50, 20)
    assert out_len == 70
    torch.testing.assert_close(roots[:50], prefix_before_second_rfn)

    end_frame = resolve_animation_end_frame(50, 20, None)
    assert end_frame == 69


def test_resize_after_limit_matches_bar():
    """Shrinking the last bar to 50 drops leftover frames (walking-bar regression)."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, None)

    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 0, 50, False)
    assert out_len == 50
    assert out_root.shape[0] == 50

    end_frame = resolve_animation_end_frame(0, 0, 49)
    assert clamp_playhead_frame(60, 49, end_frame) == 49


def test_middle_prompt_resize_keeps_neighbors():
    """Packed prompts at 0 and 40: shrinking first segment keeps later motion."""
    clip_frames = 80
    next_prompt_start = 40
    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, next_prompt_start)

    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 0, 5, True)
    assert out_len == 5 + (clip_frames - 40)
    torch.testing.assert_close(out_root[5:], roots[40:])
