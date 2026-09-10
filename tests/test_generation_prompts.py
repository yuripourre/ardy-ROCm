# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ardy.server.generation import _build_segment_prompts, _generation_start_frame, restart_from
from ardy.server.session import MotionSession, PromptSegment


def _three_segment_session() -> MotionSession:
    session = MotionSession(id="test", model_name="core", device="cpu")
    session.model = object()
    session.prompts = [
        PromptSegment(id="p1", text="walk", start_frame=0, end_frame=39),
        PromptSegment(id="p2", text="turn", start_frame=40, end_frame=79),
        PromptSegment(id="p3", text="wave", start_frame=80, end_frame=119),
    ]
    return session


def test_restart_from_preserves_future_prompts():
    session = _three_segment_session()
    restart_from(session, 39)

    assert len(session.prompts) == 3
    assert session.prompts[0].text == "walk"
    assert session.prompts[0].end_frame == 39
    assert session.prompts[1].text == "turn"
    assert session.prompts[1].start_frame == 40
    assert session.prompts[1].end_frame == 79
    assert session.prompts[2].text == "wave"
    assert session.prompts[2].start_frame == 80
    assert session.text_embedding is None


def test_active_prompt_at_generation_frame():
    session = _three_segment_session()
    restart_from(session, 39)

    assert session.active_prompt_text(39) == "walk"
    assert session.active_prompt_text(40) == "turn"
    assert session.active_prompt_text(79) == "turn"
    assert session.active_prompt_text(80) == "wave"


def test_get_history_motion_uses_frame_idx_for_playhead():
    session = _three_segment_session()
    session.frame_idx = 24
    session.max_frame_idx = 64

    import torch

    from ardy.server.generation import _get_history_motion

    session.motion_tensor = torch.zeros(1, 65, 8)
    _, _, history_end_idx, _ = _get_history_motion(session)
    assert history_end_idx == 24


def test_build_segment_prompts():
    prompts = _build_segment_prompts([("walk", 40), ("turn", 40), ("wave", 40)])
    assert len(prompts) == 3
    assert prompts[0].text == "walk"
    assert prompts[0].start_frame == 0
    assert prompts[0].end_frame == 39
    assert prompts[1].text == "turn"
    assert prompts[1].start_frame == 40
    assert prompts[1].end_frame == 79
    assert prompts[2].text == "wave"
    assert prompts[2].start_frame == 80
    assert prompts[2].end_frame == 119


def test_generation_start_frame_uses_history_end():
    session = _three_segment_session()
    session.frame_idx = 39
    session.max_frame_idx = 39

    import torch

    session.motion_tensor = torch.zeros(1, 40, 8)

    from ardy.server.generation import _get_history_motion

    _, _, history_end_idx, _ = _get_history_motion(session)
    assert _generation_start_frame(session, history_end_idx) == 40
