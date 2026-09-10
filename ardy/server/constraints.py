# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory constraint storage and model constraint assembly."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from einops import repeat

from ardy.constraints import (
    TYPE_TO_CLASS,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from ardy.root_path import interpolate_root_path


def _to_device_tensor(x, device: str) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device)


@dataclass
class Root2dTrack:
    keyframes: dict[int, np.ndarray] = field(default_factory=dict)
    root_headings: dict[int, float] = field(default_factory=dict)
    dense_path: bool = False
    _cached_t: Optional[np.ndarray] = None
    _cached_path3d: Optional[np.ndarray] = None

    def upsert(self, frame: int, x: float, z: float, heading: Optional[float] = None) -> None:
        self.keyframes[frame] = np.array([x, 0.0, z], dtype=np.float32)
        if heading is not None:
            self.root_headings[frame] = float(heading)
        self._cached_t = None
        self._cached_path3d = None

    def delete(self, frame: int) -> None:
        self.keyframes.pop(frame, None)
        self.root_headings.pop(frame, None)
        self._cached_t = None
        self._cached_path3d = None

    def set_dense_path(self, start_frame: int, end_frame: int, smooth: bool = False) -> np.ndarray:
        if len(self.keyframes) < 1:
            raise ValueError("At least one root2d keyframe is required to interpolate")
        frame_indices = np.array(sorted(self.keyframes.keys()))
        root_pos = np.stack([self.keyframes[f] for f in frame_indices], axis=0)
        t = np.arange(start_frame, end_frame + 1)
        path3d = interpolate_root_path(frame_indices, root_pos, t, smooth=smooth)
        self.dense_path = True
        self._cached_t = t
        self._cached_path3d = path3d
        return path3d

    def get_constraint_info(self) -> dict[str, Any]:
        if not self.dense_path or len(self.keyframes) == 0:
            frame_indices = sorted(self.keyframes.keys())
            if not frame_indices:
                return {"frame_idx": [], "root_pos": None, "root_headings": None}
            root_pos = np.stack([self.keyframes[f] for f in frame_indices], axis=0)
            headings = [self.root_headings.get(f) for f in frame_indices]
            return {
                "frame_idx": frame_indices,
                "root_pos": root_pos,
                "root_headings": headings if any(h is not None for h in headings) else None,
            }

        if self._cached_t is None or self._cached_path3d is None:
            frame_indices = np.array(sorted(self.keyframes.keys()))
            root_pos = np.stack([self.keyframes[f] for f in frame_indices], axis=0)
            t = np.arange(frame_indices[0], frame_indices[-1] + 1)
            path3d = interpolate_root_path(frame_indices, root_pos, t, smooth=False)
            self._cached_t = t
            self._cached_path3d = path3d

        return {
            "frame_idx": self._cached_t.tolist(),
            "root_pos": self._cached_path3d,
            "root_headings": None,
        }

    def crop_and_shift(self, start: int, end: int, shift: int) -> None:
        new_keyframes: dict[int, np.ndarray] = {}
        new_headings: dict[int, float] = {}
        for frame, pos in self.keyframes.items():
            if start <= frame < end:
                new_keyframes[frame - shift] = pos
                if frame in self.root_headings:
                    new_headings[frame - shift] = self.root_headings[frame]
        self.keyframes = new_keyframes
        self.root_headings = new_headings
        self.dense_path = False
        self._cached_t = None
        self._cached_path3d = None

    def shift_after_delete(self, delete_start: int, delete_end: int) -> None:
        delete_count = delete_end - delete_start
        new_keyframes: dict[int, np.ndarray] = {}
        new_headings: dict[int, float] = {}
        for frame, pos in self.keyframes.items():
            if frame < delete_start:
                new_keyframes[frame] = pos
                if frame in self.root_headings:
                    new_headings[frame] = self.root_headings[frame]
            elif frame >= delete_end:
                new_frame = frame - delete_count
                new_keyframes[new_frame] = pos
                if frame in self.root_headings:
                    new_headings[new_frame] = self.root_headings[frame]
        self.keyframes = new_keyframes
        self.root_headings = new_headings
        self.dense_path = False
        self._cached_t = None
        self._cached_path3d = None

    def to_api_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "frame": frame,
                "x": float(pos[0]),
                "z": float(pos[2]),
                "heading": self.root_headings.get(frame),
            }
            for frame, pos in sorted(self.keyframes.items())
        ]


@dataclass
class FullbodyTrack:
    keyframes: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)

    def upsert(self, frame: int, joints_pos: np.ndarray, joints_rot: np.ndarray) -> None:
        self.keyframes[frame] = {
            "joints_pos": np.asarray(joints_pos, dtype=np.float32),
            "joints_rot": np.asarray(joints_rot, dtype=np.float32),
        }

    def delete(self, frame: int) -> None:
        self.keyframes.pop(frame, None)

    def get_constraint_info(self) -> dict[str, Any]:
        if not self.keyframes:
            return {"frame_idx": [], "joints_pos": None, "joints_rot": None}
        frame_indices = sorted(self.keyframes.keys())
        joints_pos = np.stack([self.keyframes[f]["joints_pos"] for f in frame_indices], axis=0)
        joints_rot = np.stack([self.keyframes[f]["joints_rot"] for f in frame_indices], axis=0)
        return {"frame_idx": frame_indices, "joints_pos": joints_pos, "joints_rot": joints_rot}

    def crop_and_shift(self, start: int, end: int, shift: int) -> None:
        new_keyframes: dict[int, dict[str, np.ndarray]] = {}
        for frame, data in self.keyframes.items():
            if start <= frame < end:
                new_keyframes[frame - shift] = data
        self.keyframes = new_keyframes

    def shift_after_delete(self, delete_start: int, delete_end: int) -> None:
        delete_count = delete_end - delete_start
        new_keyframes: dict[int, dict[str, np.ndarray]] = {}
        for frame, data in self.keyframes.items():
            if frame < delete_start:
                new_keyframes[frame] = data
            elif frame >= delete_end:
                new_keyframes[frame - delete_count] = data
        self.keyframes = new_keyframes

    def to_api_dict(self) -> list[dict[str, Any]]:
        return [{"frame": frame, **{k: v.tolist() for k, v in data.items()}} for frame, data in sorted(self.keyframes.items())]


@dataclass
class EndEffectorTrack:
    keyframes: dict[int, dict[str, Any]] = field(default_factory=dict)

    def upsert(
        self,
        frame: int,
        joints_pos: np.ndarray,
        joints_rot: np.ndarray,
        joint_names: list[str],
        end_effector_types: list[str],
    ) -> None:
        self.keyframes[frame] = {
            "joints_pos": np.asarray(joints_pos, dtype=np.float32),
            "joints_rot": np.asarray(joints_rot, dtype=np.float32),
            "joint_names": list(joint_names),
            "end_effector_type": list(end_effector_types),
        }

    def delete(self, frame: int) -> None:
        self.keyframes.pop(frame, None)

    def get_constraint_info(self) -> dict[str, Any]:
        if not self.keyframes:
            return {
                "frame_idx": [],
                "joints_pos": None,
                "joints_rot": None,
                "joint_names": [],
                "end_effector_type": [],
            }
        frame_indices = sorted(self.keyframes.keys())
        joints_pos = np.stack([self.keyframes[f]["joints_pos"] for f in frame_indices], axis=0)
        joints_rot = np.stack([self.keyframes[f]["joints_rot"] for f in frame_indices], axis=0)
        joint_names = [self.keyframes[f]["joint_names"] for f in frame_indices]
        end_effector_type = [self.keyframes[f]["end_effector_type"] for f in frame_indices]
        return {
            "frame_idx": frame_indices,
            "joints_pos": joints_pos,
            "joints_rot": joints_rot,
            "joint_names": joint_names,
            "end_effector_type": end_effector_type,
        }

    def crop_and_shift(self, start: int, end: int, shift: int) -> None:
        new_keyframes: dict[int, dict[str, Any]] = {}
        for frame, data in self.keyframes.items():
            if start <= frame < end:
                new_keyframes[frame - shift] = data
        self.keyframes = new_keyframes

    def shift_after_delete(self, delete_start: int, delete_end: int) -> None:
        delete_count = delete_end - delete_start
        new_keyframes: dict[int, dict[str, Any]] = {}
        for frame, data in self.keyframes.items():
            if frame < delete_start:
                new_keyframes[frame] = data
            elif frame >= delete_end:
                new_keyframes[frame - delete_count] = data
        self.keyframes = new_keyframes

    def to_api_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "frame": frame,
                "joints_pos": data["joints_pos"].tolist(),
                "joints_rot": data["joints_rot"].tolist(),
                "joint_names": data["joint_names"],
                "end_effector_type": data["end_effector_type"],
            }
            for frame, data in sorted(self.keyframes.items())
        ]


@dataclass
class ConstraintStore:
    root2d: Root2dTrack = field(default_factory=Root2dTrack)
    fullbody: FullbodyTrack = field(default_factory=FullbodyTrack)
    end_effector: EndEffectorTrack = field(default_factory=EndEffectorTrack)

    def track_for_type(self, constraint_type: str):
        mapping = {
            "root2d": self.root2d,
            "fullbody": self.fullbody,
            "end_effector": self.end_effector,
        }
        if constraint_type not in mapping:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        return mapping[constraint_type]

    def to_api_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "root2d": self.root2d.to_api_dict(),
            "fullbody": self.fullbody.to_api_dict(),
            "end_effector": self.end_effector.to_api_dict(),
        }

    def crop_and_shift(self, start: int, end: int) -> None:
        shift = start
        self.root2d.crop_and_shift(start, end, shift)
        self.fullbody.crop_and_shift(start, end, shift)
        self.end_effector.crop_and_shift(start, end, shift)

    def shift_after_delete(self, delete_start: int, delete_end: int) -> None:
        self.root2d.shift_after_delete(delete_start, delete_end)
        self.fullbody.shift_after_delete(delete_start, delete_end)
        self.end_effector.shift_after_delete(delete_start, delete_end)


def compute_model_constraints_lst(session, num_frames: int, history_end_idx: int) -> list:
    """Build model constraint objects from the in-memory constraint store."""
    store = session.constraints
    device = session.device
    dense_root_pos_2d = None
    model_constraints = []

    track_specs = [
        ("root2d", store.root2d),
        ("fullbody", store.fullbody),
        ("end_effector", store.end_effector),
    ]

    for track_name, track in track_specs:
        constraint_info = track.get_constraint_info()
        frame_idx = constraint_info["frame_idx"]
        if not frame_idx:
            continue

        valid_info = [(i, fi) for i, fi in enumerate(frame_idx) if fi < num_frames and fi > history_end_idx]
        valid_idx = [i for i, _ in valid_info]
        valid_frame_idx = [fi for _, fi in valid_info]
        if not valid_frame_idx:
            continue

        frame_indices = torch.tensor(valid_frame_idx, device=device)

        if track_name == "root2d":
            root_pos = constraint_info["root_pos"]
            if isinstance(root_pos, np.ndarray):
                root_pos_arr = root_pos[valid_idx]
            else:
                root_pos_arr = root_pos[valid_idx]
            root_pos_2d = _to_device_tensor(root_pos_arr[:, [0, 2]], device)
            model_constraints.append(
                Root2DConstraintSet(session.motion_rep.skeleton, frame_indices, root_pos_2d)
            )
            if store.root2d.dense_path:
                dense_info = store.root2d.get_constraint_info()
                dense_root_pos_2d = _to_device_tensor(dense_info["root_pos"][:, [0, 2]], device)

        elif track_name == "fullbody":
            joints_pos = _to_device_tensor(constraint_info["joints_pos"][valid_idx], device)
            joints_rot = _to_device_tensor(constraint_info["joints_rot"][valid_idx], device)
            root_pos_2d = dense_root_pos_2d[frame_indices] if dense_root_pos_2d is not None else None
            model_constraints.append(
                FullBodyConstraintSet(
                    session.motion_rep.skeleton,
                    frame_indices,
                    joints_pos,
                    joints_rot,
                    root_2d=root_pos_2d,
                )
            )

        elif track_name == "end_effector":
            joints_pos = _to_device_tensor(constraint_info["joints_pos"][valid_idx], device)
            joints_rot = _to_device_tensor(constraint_info["joints_rot"][valid_idx], device)
            end_effector_type_set_lst = [
                constraint_info["end_effector_type"][i] for i in valid_idx
            ]
            cls_idx = defaultdict(list)
            for idx, end_effector_type_set in enumerate(end_effector_type_set_lst):
                for end_effector_type in end_effector_type_set:
                    cls_idx[TYPE_TO_CLASS[end_effector_type]].append(idx)

            for cls, lst_idx in cls_idx.items():
                frame_indices_cls = frame_indices[lst_idx]
                root_pos_2d = None
                if dense_root_pos_2d is not None:
                    root_pos_2d = dense_root_pos_2d[frame_indices_cls]
                model_constraints.append(
                    cls(
                        session.motion_rep.skeleton,
                        frame_indices_cls,
                        joints_pos[lst_idx],
                        joints_rot[lst_idx],
                        root_2d=root_pos_2d,
                    )
                )

    return model_constraints


def compute_constraint_mask(session, num_samples: int, num_frames: int, history_end_idx: int):
    """Compute motion mask and observed motion tensors for generation."""
    model_constraints = compute_model_constraints_lst(session, num_frames, history_end_idx)
    if not model_constraints:
        return None, None

    observed_motion, motion_mask = session.motion_rep.create_conditions_from_constraints(
        model_constraints,
        length=num_frames,
        to_normalize=False,
        device=session.device,
    )
    observed_motion = session.motion_rep.normalize(observed_motion)
    observed_motion = observed_motion * motion_mask
    observed_motion = repeat(observed_motion, "t d -> b t d", b=num_samples)
    motion_mask = repeat(motion_mask, "t d -> b t d", b=num_samples)
    return motion_mask, observed_motion
