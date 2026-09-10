# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session store and model loading."""

from __future__ import annotations

import os
import threading
from typing import Optional

import torch

from ardy.model.load_model import load_model, load_text_encoder
from ardy.model.registry import resolve_model_name
from ardy.server.session import MotionSession, new_session_id


class SessionStore:
    def __init__(self, device: str, model_name: Optional[str] = "core"):
        self.device = device
        self._sessions: dict[str, MotionSession] = {}
        self._lock = threading.Lock()
        self._text_encoder = None
        self._model = None
        self._model_name: Optional[str] = None
        self._checkpoints_dir = os.environ.get("CHECKPOINTS_DIR")
        if model_name is not None:
            self.load_model(model_name)

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load_model(self, model_name: str) -> None:
        """Load model weights (called once at server startup)."""
        resolved = resolve_model_name(model_name, checkpoints_dir=self._checkpoints_dir)
        print(f"Loading model '{resolved}' on {self.device}...", flush=True)
        if self._text_encoder is None:
            self._text_encoder = load_text_encoder(mode="auto", device=self.device)
        self._model = load_model(
            resolved,
            device=self.device,
            text_encoder=self._text_encoder,
            checkpoints_dir=self._checkpoints_dir,
        )
        self._model_name = resolved
        print(f"Model '{resolved}' ready.", flush=True)

    def create_session(self) -> MotionSession:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        session_id = new_session_id()
        session = MotionSession(id=session_id, model_name=self._model_name, device=self.device)
        session.model = self._model
        session.motion_rep = self._model.motion_rep
        session.configure_from_model()

        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> MotionSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise KeyError(session_id)
        session.motion_tensor = None
        session.joints_pos = None
        session.joints_rot = None
        session.foot_contacts = None
        session.root_velocities = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def list_session_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())
