# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class SessionResponse(BaseModel):
    id: str
    model: str
    fps: int
    horizon: int
    skeleton: Optional[str] = None
    frame_count: int = 0


class SessionInfoResponse(BaseModel):
    id: str
    model: str
    fps: int
    horizon: int
    skeleton: Optional[str] = None
    frame_count: int


class PromptInput(BaseModel):
    id: Optional[str] = None
    text: str
    start_frame: int = 0
    end_frame: Optional[int] = None


class PromptResponse(BaseModel):
    id: str
    text: str
    start_frame: int
    end_frame: Optional[int] = None


class Root2dKeyframeRequest(BaseModel):
    x: float
    z: float
    heading: Optional[float] = None


class FullbodyKeyframeRequest(BaseModel):
    joints_pos: list[list[float]]
    joints_rot: list[list[list[float]]]


class EndEffectorKeyframeRequest(BaseModel):
    joints_pos: list[list[float]]
    joints_rot: list[list[list[float]]]
    joint_names: list[str]
    end_effector_type: list[str]


class InterpolateRoot2dRequest(BaseModel):
    start_frame: int
    end_frame: int
    smooth: bool = False


class GenerateStepRequest(BaseModel):
    diffusion_steps: Optional[int] = None
    cfg_weight: Optional[list[float]] = None


class GenerateStepResponse(BaseModel):
    start_frame: int
    end_frame: int


MAX_BVH_FRAMES = 2000


class GenerateBvhSegment(BaseModel):
    prompt: str = Field(min_length=1)
    frames: int = Field(ge=1, le=MAX_BVH_FRAMES)


class GenerateBvhRequest(BaseModel):
    segments: Optional[list[GenerateBvhSegment]] = Field(default=None, min_length=1)
    prompt: Optional[str] = Field(default=None, min_length=1)
    frames: Optional[int] = Field(default=None, ge=1, le=MAX_BVH_FRAMES)
    diffusion_steps: Optional[int] = None
    cfg_weight: Optional[list[float]] = None

    @model_validator(mode="after")
    def normalize_segments(self) -> "GenerateBvhRequest":
        if self.segments is not None:
            if self.prompt is not None or self.frames is not None:
                raise ValueError("Specify either segments or prompt/frames, not both")
            total_frames = sum(segment.frames for segment in self.segments)
            if total_frames > MAX_BVH_FRAMES:
                raise ValueError(f"Total frames must be <= {MAX_BVH_FRAMES}, got {total_frames}")
            return self
        if self.prompt is not None and self.frames is not None:
            object.__setattr__(
                self,
                "segments",
                [GenerateBvhSegment(prompt=self.prompt, frames=self.frames)],
            )
            return self
        raise ValueError("Provide segments or both prompt and frames")

    @property
    def resolved_segments(self) -> list[GenerateBvhSegment]:
        if self.segments is None:
            raise ValueError("segments not resolved")
        return self.segments


class RestartFromRequest(BaseModel):
    frame: int


class FrameRangeRequest(BaseModel):
    start: int
    end: int


class HealthResponse(BaseModel):
    status: str
    device: str
    model: Optional[str] = None
    ready: bool = False


class ModelsResponse(BaseModel):
    models: list[str]


class ConstraintsResponse(BaseModel):
    root2d: list[dict[str, Any]] = Field(default_factory=list)
    fullbody: list[dict[str, Any]] = Field(default_factory=list)
    end_effector: list[dict[str, Any]] = Field(default_factory=list)
