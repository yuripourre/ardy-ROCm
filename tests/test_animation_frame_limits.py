# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for animation frame limits, RFN warping, and multi-segment chains."""

import torch

from ardy.motion_resample import (
    clamp_playhead_frame,
    resolve_animation_end_frame,
    resolve_effective_playback_end,
    resolve_next_prompt_start_from_spans,
    resolve_prompt_source_span,
    resolve_restart_from_now_keep_end,
    should_pause_playback_at_clip_end,
    should_skip_generation_at_limit,
)
from tests.timeline_test_utils import (
    apply_initial_animation_limit,
    apply_rfn_tail_warp,
    make_linear_root_motion,
    mean_root_delta,
    resample_segment,
    trim_motion_before_rfn,
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


def test_resolve_restart_from_now_keep_end_includes_playhead():
    assert resolve_restart_from_now_keep_end(0) == 1
    assert resolve_restart_from_now_keep_end(39) == 40


def test_rfn_trim_preserves_playhead_frame():
    """RFN at the last capped frame must keep that frame in the prefix clip."""
    rots, roots = make_linear_root_motion(40)
    trimmed_rots, trimmed_roots = trim_motion_before_rfn(rots, roots, 39)
    assert trimmed_roots.shape[0] == 40
    torch.testing.assert_close(trimmed_roots[-1], roots[39])
    assert trimmed_roots[-1, 0] == 39.0


def test_rfn_at_last_capped_frame_with_animation_limit():
    """40-frame cap, RFN at frame 39 with limit 25: playhead frame survives trim + warp."""
    rots, roots = make_linear_root_motion(80)
    rots, roots = apply_initial_animation_limit(rots, roots, 40)
    prefix_before = roots[:40].clone()

    trimmed_rots, trimmed_roots = trim_motion_before_rfn(rots, roots, 39)
    assert trimmed_roots.shape[0] == 40
    torch.testing.assert_close(trimmed_roots, prefix_before)

    out_rots, out_root, out_len = apply_rfn_tail_warp(
        rots, roots, 39, 25, trim_prefix=True
    )
    assert out_len == 64
    torch.testing.assert_close(out_root[:39], prefix_before[:39])
    torch.testing.assert_close(out_root[39], prefix_before[39])


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


def test_two_animation_frame_regions_resize_first_uses_stored_spans():
    """Animation-Frames regions at 0 and 40: resize first bar with stored next start."""
    clip_frames = 80
    prompt_spans = {"p1": (0, 40), "p2": (40, 80)}
    next_prompt_start = resolve_next_prompt_start_from_spans(prompt_spans, "p1", 0, clip_frames)
    assert next_prompt_start == 40

    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, next_prompt_start)
    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 0, 20, True)
    assert out_len == 60
    torch.testing.assert_close(out_root[20:], roots[40:])


def test_two_animation_frame_regions_resize_second_independent():
    """Second region resize warps only its slice; first region prefix unchanged."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(40, clip_frames, None)

    prefix_before = roots[:40].clone()
    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 40, 55, False)
    assert out_len == 55
    torch.testing.assert_close(out_root[:40], prefix_before)


def test_two_regions_sequential_resize_independent():
    """Shrink p1 then p2; each region warps independently after span sync."""
    clip_frames = 80
    prompt_spans = {"p1": (0, 40), "p2": (40, 80)}
    rots, roots = make_linear_root_motion(clip_frames)

    next_start = resolve_next_prompt_start_from_spans(prompt_spans, "p1", 0, clip_frames)
    p1_src = resolve_prompt_source_span(0, clip_frames, next_start)
    packed_rots, packed_root, packed_len = resample_segment(
        rots, roots, p1_src[0], p1_src[1], 0, 20, True
    )
    assert packed_len == 60

    # After p1 resize, spans sync to packed layout; p2 owns [20, 60).
    synced_spans = {"p1": (0, 20), "p2": (20, 60)}
    assert resolve_next_prompt_start_from_spans(synced_spans, "p2", 20, packed_len) is None
    p2_src = resolve_prompt_source_span(20, packed_len, None)
    _, final_root, final_len = resample_segment(
        packed_rots, packed_root, p2_src[0], p2_src[1], 20, 35, False
    )
    assert final_len == 35
    torch.testing.assert_close(final_root[:20], packed_root[:20])


def test_three_regions_middle_resize_independent():
    """Middle of three Animation-Frames regions resizes without touching neighbors."""
    clip_frames = 90
    prompt_spans = {"p1": (0, 30), "p2": (30, 60), "p3": (60, 90)}
    rots, roots = make_linear_root_motion(clip_frames)

    next_start = resolve_next_prompt_start_from_spans(prompt_spans, "p2", 30, clip_frames)
    source_start, source_end = resolve_prompt_source_span(30, clip_frames, next_start)
    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 30, 50, True)
    assert out_len == 80
    torch.testing.assert_close(out_root[:30], roots[:30])
    torch.testing.assert_close(out_root[50:], roots[60:])


def test_unequal_animation_frame_regions_resize_independently():
    """40-frame + 25-frame regions each keep neighbor motion when resized."""
    clip_frames = 65
    prompt_spans = {"p1": (0, 40), "p2": (40, 65)}
    rots, roots = make_linear_root_motion(clip_frames)

    next_start = resolve_next_prompt_start_from_spans(prompt_spans, "p1", 0, clip_frames)
    p1_src = resolve_prompt_source_span(0, clip_frames, next_start)
    packed_rots, packed_root, packed_len = resample_segment(
        rots, roots, p1_src[0], p1_src[1], 0, 25, True
    )
    assert packed_len == 50
    torch.testing.assert_close(packed_root[25:], roots[40:])

    p2_src = resolve_prompt_source_span(25, packed_len, None)
    _, final_root, final_len = resample_segment(
        packed_rots, packed_root, p2_src[0], p2_src[1], 25, 40, False
    )
    assert final_len == 40
    torch.testing.assert_close(final_root[:25], packed_root[:25])


def test_should_pause_not_at_clip_end():
    assert should_pause_playback_at_clip_end(at_clip_end=False, animation_end=None, auto_replan=True) is False
    assert should_pause_playback_at_clip_end(at_clip_end=False, animation_end=39, auto_replan=False) is False


def test_should_pause_unbounded_auto_replan_waits():
    assert should_pause_playback_at_clip_end(at_clip_end=True, animation_end=None, auto_replan=True) is False


def test_should_pause_unbounded_no_auto_replan():
    assert should_pause_playback_at_clip_end(at_clip_end=True, animation_end=None, auto_replan=False) is True


def test_should_pause_at_animation_cap():
    assert should_pause_playback_at_clip_end(at_clip_end=True, animation_end=39, auto_replan=True) is True
    assert should_pause_playback_at_clip_end(at_clip_end=True, animation_end=39, auto_replan=False) is True
