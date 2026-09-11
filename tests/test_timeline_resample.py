# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for timeline span/clip-length helpers and resize semantics."""

import torch

from ardy.motion_resample import (
    resolve_next_prompt_start_from_spans,
    resolve_prompt_source_span,
    resolve_resized_clip_length,
)
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


def test_resolve_next_prompt_start_from_spans_packed_neighbors():
    """Stored spans keep pre-pack next start even when UI would show a packed bar."""
    prompt_spans = {"p1": (0, 40), "p2": (40, 80)}
    assert resolve_next_prompt_start_from_spans(prompt_spans, "p1", 0, 80) == 40
    # Live packed UI would place p2 at 20; stored spans must still source from 40.
    assert resolve_next_prompt_start_from_spans(prompt_spans, "p1", 0, 80) != 20


def test_resolve_next_prompt_start_from_spans_skips_self_and_loop():
    prompt_spans = {"p1": (0, 40), "p2": (40, 80), "loop": (70, 80)}
    assert resolve_next_prompt_start_from_spans(prompt_spans, "p1", 0, 80, "loop") == 40
    assert resolve_next_prompt_start_from_spans(prompt_spans, "p2", 40, 80, "loop") is None


def test_resolve_next_prompt_start_from_spans_rfn_tail():
    """RFN tail at old_start is not treated as a later prompt."""
    prompt_spans = {"p1": (20, 60)}
    assert resolve_next_prompt_start_from_spans(prompt_spans, "", 20, 60) is None


def test_live_packed_next_start_skips_warp_regression():
    """Using live packed next start (20) keeps 80 frames unwarped — the bug."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    live_next_start = 20
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, live_next_start)
    assert (source_start, source_end) == (0, 20)
    _, _, out_len = resample_segment(rots, roots, source_start, source_end, 0, 20, True)
    assert out_len == 80


def test_stored_next_start_shrinks_first_of_two_regions():
    """Two 40-frame regions: shrink first to 20 using stored next start 40."""
    clip_frames = 80
    stored_next_start = 40
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, stored_next_start)
    assert (source_start, source_end) == (0, 40)

    rots, roots = make_linear_root_motion(clip_frames)
    original_first_speed = mean_root_delta(roots[:40])
    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 0, 20, True)
    assert out_len == 60
    assert mean_root_delta(out_root[:20]) > original_first_speed
    torch.testing.assert_close(out_root[20:], roots[40:])


def test_stored_next_start_stretches_first_of_two_regions():
    """Two 40-frame regions: stretch first to 60 using stored next start 40."""
    clip_frames = 80
    stored_next_start = 40
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, stored_next_start)

    rots, roots = make_linear_root_motion(clip_frames)
    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 0, 60, True)
    assert out_len == 100
    torch.testing.assert_close(out_root[60:], roots[40:])


def test_resize_last_region_shrink_keeps_prefix():
    """Shrink second region only: prefix [0:40) unchanged, warp [40:80) to 20 frames."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(40, clip_frames, None)
    assert (source_start, source_end) == (40, 80)

    original_second_speed = mean_root_delta(roots[40:])
    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 40, 60, False)
    assert out_len == 60
    torch.testing.assert_close(out_root[:40], roots[:40])
    assert mean_root_delta(out_root[40:]) > original_second_speed


def test_resize_last_region_stretch_keeps_prefix():
    """Stretch second region only: prefix [0:40) unchanged, warp [40:80) to 60 frames."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    source_start, source_end = resolve_prompt_source_span(40, clip_frames, None)

    original_second_speed = mean_root_delta(roots[40:])
    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 40, 100, False)
    assert out_len == 100
    torch.testing.assert_close(out_root[:40], roots[:40])
    assert mean_root_delta(out_root[40:]) < original_second_speed


def _shrink_first_region(rots, roots, new_len: int):
    """Shrink first of two equal regions; returns packed clip (rots, roots, length)."""
    clip_frames = roots.shape[0]
    stored_next_start = clip_frames // 2
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, stored_next_start)
    return resample_segment(rots, roots, source_start, source_end, 0, new_len, True)


def test_resize_second_after_first_shrunk_packed_layout():
    """After p1 shrink, resize p2 with synced spans: prefix untouched, p2 warped."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)
    packed_rots, packed_root, packed_len = _shrink_first_region(rots, roots, 20)
    assert packed_len == 60
    torch.testing.assert_close(packed_root[20:], roots[40:])

    # Spans synced after p1 resize: p2 starts at 20 in packed clip.
    p2_source_start, p2_source_end = resolve_prompt_source_span(20, packed_len, None)
    assert (p2_source_start, p2_source_end) == (20, 60)

    prefix_before = packed_root[:20].clone()
    _, out_root, out_len = resample_segment(
        packed_rots, packed_root, p2_source_start, p2_source_end, 20, 40, False
    )
    assert out_len == 40
    torch.testing.assert_close(out_root[:20], prefix_before)


def test_resize_middle_of_three_regions():
    """Three 30-frame regions: shrink middle to 15; prefix and suffix unchanged."""
    clip_frames = 90
    rots, roots = make_linear_root_motion(clip_frames)
    prompt_spans = {"p1": (0, 30), "p2": (30, 60), "p3": (60, 90)}
    next_start = resolve_next_prompt_start_from_spans(prompt_spans, "p2", 30, clip_frames)
    assert next_start == 60

    source_start, source_end = resolve_prompt_source_span(30, clip_frames, next_start)
    assert (source_start, source_end) == (30, 60)

    _, out_root, out_len = resample_segment(rots, roots, source_start, source_end, 30, 45, True)
    assert out_len == 30 + 15 + 30
    torch.testing.assert_close(out_root[:30], roots[:30])
    torch.testing.assert_close(out_root[45:], roots[60:])


def test_unequal_region_sizes_resize_independently():
    """40 + 25 frame regions: each resize warps only its own slice."""
    clip_frames = 65
    rots, roots = make_linear_root_motion(clip_frames)
    prompt_spans = {"p1": (0, 40), "p2": (40, 65)}

    # Shrink first region 40 -> 20
    next_start = resolve_next_prompt_start_from_spans(prompt_spans, "p1", 0, clip_frames)
    source_start, source_end = resolve_prompt_source_span(0, clip_frames, next_start)
    after_p1_rots, after_p1, len_p1 = resample_segment(rots, roots, source_start, source_end, 0, 20, True)
    assert len_p1 == 45
    torch.testing.assert_close(after_p1[20:], roots[40:])

    # Shrink second region 25 -> 10 on packed clip (p2 at 20..45)
    p2_start, p2_end = resolve_prompt_source_span(20, len_p1, None)
    _, after_p2, len_p2 = resample_segment(
        after_p1_rots, after_p1, p2_start, p2_end, 20, 30, False
    )
    assert len_p2 == 30
    torch.testing.assert_close(after_p2[:20], after_p1[:20])


def test_sequential_resize_p1_then_p2():
    """Shrink p1 then p2; final clip length and per-region content are consistent."""
    clip_frames = 80
    rots, roots = make_linear_root_motion(clip_frames)

    after_p1_rots, after_p1, len_p1 = _shrink_first_region(rots, roots, 20)
    assert len_p1 == 60

    _, after_p2, len_p2 = resample_segment(
        after_p1_rots, after_p1, 20, 60, 20, 35, False
    )
    assert len_p2 == 35
    torch.testing.assert_close(after_p2[:20], after_p1[:20])
