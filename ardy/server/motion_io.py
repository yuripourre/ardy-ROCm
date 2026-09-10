# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion serialization for the HTTP API."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch

from ardy.exports.bvh import BvhExporter
from ardy.server.session import MotionSession


def _motion_tensors(session: MotionSession, start: int, end: int):
    if session.motion_tensor is None or session.motion_rep is None:
        raise RuntimeError("No motion available in session")

    total = session.motion_tensor.shape[1]
    if end is None:
        end = total
    end = min(end, total)
    if start < 0 or start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")

    tensor_unnorm = session.motion_rep.unnormalize(session.motion_tensor[:, start:end])
    inverse_output = session.motion_rep.inverse(tensor_unnorm, is_normalized=False)
    return inverse_output, session.model_fps, start, end


def motion_to_npz_bytes(session: MotionSession, start: int = 0, end: Optional[int] = None) -> bytes:
    inverse_output, fps, start, end = _motion_tensors(session, start, end)
    prompt_text = session.active_prompt_text(start) or ""

    arrays = {k: np.asarray(v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in inverse_output.items()}
    arrays["fps"] = np.asarray(fps)
    arrays["text"] = np.asarray(prompt_text)
    arrays["start_frame"] = np.asarray(start)
    arrays["end_frame"] = np.asarray(end)

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def motion_to_bvh_text(session: MotionSession, start: int = 0, end: Optional[int] = None) -> str:
    inverse_output, fps, start, end = _motion_tensors(session, start, end)
    exporter = BvhExporter(session.motion_rep.skeleton)
    return exporter.to_bvh_text(
        inverse_output["local_rot_mats"][0],
        inverse_output["root_positions"][0],
        fps,
        character_index=0,
    )
