# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-session motion state for the HTTP API."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ardy.server.constraints import ConstraintStore

INFINITE_FRAME_IDX = 99999


@dataclass
class PromptSegment:
    id: str
    text: str
    start_frame: int
    end_frame: Optional[int] = None

    def covers_frame(self, frame: int) -> bool:
        end = self.end_frame if self.end_frame is not None else INFINITE_FRAME_IDX
        return self.start_frame <= frame <= end


@dataclass
class MotionSession:
    """Viser-free session holding model state and generated motion."""

    id: str
    model_name: str
    device: str

    model: Any = None
    motion_rep: Any = None
    model_fps: int = 20
    num_frames_per_token: int = 4
    gen_horizon_len: int = 40
    max_window_len: int = 200

    motion_tensor: Optional[torch.Tensor] = None
    joints_pos: Optional[torch.Tensor] = None
    joints_rot: Optional[torch.Tensor] = None
    foot_contacts: Optional[torch.Tensor] = None
    root_velocities: Optional[torch.Tensor] = None

    frame_idx: int = 0
    max_frame_idx: int = -1

    prompts: list[PromptSegment] = field(default_factory=list)
    constraints: ConstraintStore = field(default_factory=ConstraintStore)
    text_embedding: Optional[torch.Tensor] = None

    diffusion_steps: int = 20
    cfg_text_weight: float = 2.0
    cfg_constraint_weight: float = 2.0
    history_crop_length: int = 4
    future_crop_length: int = 160
    replan_buffer_size: int = 0
    enable_postprocess: bool = True
    postprocess_contact_threshold: float = 0.5
    postprocess_root_margin: float = 0.05

    init_global_translation: Optional[torch.Tensor] = None
    init_first_heading_angle: float = 0.0

    motion_tensor_lock: threading.Lock = field(default_factory=threading.Lock)
    generation_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def frame_count(self) -> int:
        return self.max_frame_idx + 1 if self.max_frame_idx >= 0 else 0

    @property
    def skeleton_name(self) -> Optional[str]:
        if self.motion_rep is None:
            return None
        skeleton = self.motion_rep.skeleton
        return getattr(skeleton, "name", type(skeleton).__name__)

    def active_prompt_text(self, frame: int) -> Optional[str]:
        for prompt in reversed(self.prompts):
            if prompt.covers_frame(frame):
                return prompt.text
        return None

    def configure_from_model(self) -> None:
        if self.model is None:
            return
        patch = self.model.denoiser.num_frames_per_token
        self.num_frames_per_token = patch
        self.gen_horizon_len = self.model.gen_horizon_len
        self.model_fps = self.motion_rep.fps
        self.diffusion_steps = int(self.model.diffusion.num_base_steps)
        self.max_window_len = (10 * self.model_fps // patch) * patch
        crop_max = ((self.max_window_len - self.gen_horizon_len) // patch) * patch
        self.history_crop_length = patch
        self.future_crop_length = crop_max


def new_session_id() -> str:
    return uuid.uuid4().hex
