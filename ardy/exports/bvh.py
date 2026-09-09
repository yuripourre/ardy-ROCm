# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export ARDY motion tensors to BVH animation files."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ardy.skeleton import SOMASkeleton30, SOMASkeleton77, SkeletonBase
from ardy.tools import to_numpy

METERS_TO_CM = 100.0
END_SITE_LENGTH_CM = 5.0
ROTATION_CHANNEL_NAMES = ("Zrotation", "Yrotation", "Xrotation")
ROOT_POSITION_CHANNEL_NAMES = ("Xposition", "Yposition", "Zposition")
EULER_ORDER = "ZYX"


def _build_children_map(
    bone_parents: Sequence[tuple[str, str | None]],
) -> tuple[str, dict[str, list[str]]]:
    """Return root name and parent→ordered-children map from skeleton hierarchy tuples."""
    children: dict[str, list[str]] = {}
    root_name = None
    for name, parent in bone_parents:
        if parent is None:
            root_name = name
        else:
            children.setdefault(parent, []).append(name)
    if root_name is None:
        raise ValueError("Skeleton hierarchy has no root joint.")
    return root_name, children


def _depth_first_joint_order(root_name: str, children: dict[str, list[str]]) -> list[str]:
    """Depth-first joint order used for BVH hierarchy and channel layout."""
    order: list[str] = []

    def visit(name: str) -> None:
        order.append(name)
        for child_name in children.get(name, []):
            visit(child_name)

    visit(root_name)
    return order


def _compute_offsets_cm(
    bone_parents: Sequence[tuple[str, str | None]],
    neutral_joints: np.ndarray,
    bone_index: dict[str, int],
) -> dict[str, np.ndarray]:
    """Parent-relative OFFSET values in centimeters for each joint."""
    offsets: dict[str, np.ndarray] = {}
    for name, parent in bone_parents:
        idx = bone_index[name]
        if parent is None:
            offsets[name] = neutral_joints[idx] * METERS_TO_CM
        else:
            parent_idx = bone_index[parent]
            offsets[name] = (neutral_joints[idx] - neutral_joints[parent_idx]) * METERS_TO_CM
    return offsets


def _end_site_offset_cm(joint_offset: np.ndarray) -> np.ndarray:
    """Pick a small End Site offset along the bone direction (or +Y if degenerate)."""
    norm = float(np.linalg.norm(joint_offset))
    if norm > 1e-6:
        return joint_offset / norm * END_SITE_LENGTH_CM
    return np.array([0.0, END_SITE_LENGTH_CM, 0.0], dtype=np.float64)


def _rotation_matrices_to_euler_deg(rot_mats: np.ndarray) -> np.ndarray:
    """Convert local rotation matrices ``(..., 3, 3)`` to ZYX Euler angles in degrees."""
    flat = rot_mats.reshape(-1, 3, 3)
    eulers = Rotation.from_matrix(flat).as_euler(EULER_ORDER, degrees=True)
    return eulers.reshape(rot_mats.shape[:-2] + (3,))


def _prepare_motion_for_bvh(
    local_rot_mats: np.ndarray | torch.Tensor,
    root_positions: np.ndarray | torch.Tensor,
    skeleton: SkeletonBase,
    character_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, SkeletonBase]:
    """Normalize batch dims and apply skeleton-specific BVH prep (SOMA expand / t-pose)."""
    local_rot_mats = to_numpy(local_rot_mats)
    root_positions = to_numpy(root_positions)

    if local_rot_mats.ndim == 5:
        local_rot_mats = local_rot_mats[character_index]
        root_positions = root_positions[character_index]
    elif local_rot_mats.ndim != 4 or root_positions.ndim != 2:
        raise ValueError(
            "Expected local_rot_mats shape [T, J, 3, 3] or [B, T, J, 3, 3] and "
            f"root_positions [T, 3] or [B, T, 3]; got {local_rot_mats.shape}, {root_positions.shape}."
        )

    export_skeleton = skeleton
    if isinstance(skeleton, SOMASkeleton30):
        export_skeleton = skeleton.somaskel77
        if local_rot_mats.shape[1] == skeleton.nbjoints:
            local_rot_torch = torch.from_numpy(local_rot_mats).float()
            local_rot_mats = to_numpy(skeleton.to_SOMASkeleton77(local_rot_torch))

    if isinstance(export_skeleton, SOMASkeleton77):
        local_rot_mats, _ = export_skeleton.from_standard_tpose(torch.from_numpy(local_rot_mats).float())
        local_rot_mats = to_numpy(local_rot_mats)

    return local_rot_mats, root_positions, export_skeleton


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _write_hierarchy_block(
    lines: list[str],
    joint_name: str,
    children: dict[str, list[str]],
    offsets: dict[str, np.ndarray],
    root_name: str,
    indent: int,
) -> None:
    """Recursively append HIERARCHY lines for one joint and its descendants."""
    pad = "\t" * indent
    is_root = joint_name == root_name
    joint_type = "ROOT" if is_root else "JOINT"
    lines.append(f"{pad}{joint_type} {joint_name}")
    lines.append(f"{pad}{{")
    lines.append(f"{pad}\tOFFSET {_format_vector(offsets[joint_name])}")
    if is_root:
        channels = list(ROOT_POSITION_CHANNEL_NAMES) + list(ROTATION_CHANNEL_NAMES)
    else:
        channels = list(ROTATION_CHANNEL_NAMES)
    lines.append(f"{pad}\tCHANNELS {len(channels)} {' '.join(channels)}")

    child_names = children.get(joint_name, [])
    for child_name in child_names:
        _write_hierarchy_block(lines, child_name, children, offsets, root_name, indent + 1)

    if not child_names:
        end_offset = _end_site_offset_cm(offsets[joint_name])
        lines.append(f"{pad}\tEnd Site")
        lines.append(f"{pad}\t{{")
        lines.append(f"{pad}\t\tOFFSET {_format_vector(end_offset)}")
        lines.append(f"{pad}\t}}")

    lines.append(f"{pad}}}")


def _build_motion_channels(
    local_rot_mats: np.ndarray,
    root_positions: np.ndarray,
    joint_order: list[str],
    bone_index: dict[str, int],
    root_name: str,
) -> np.ndarray:
    """Stack per-frame channel values in BVH joint order."""
    num_frames = local_rot_mats.shape[0]
    eulers_deg = _rotation_matrices_to_euler_deg(local_rot_mats)
    root_cm = root_positions * METERS_TO_CM

    channels_per_frame = 0
    for joint_name in joint_order:
        channels_per_frame += 6 if joint_name == root_name else 3

    frames = np.zeros((num_frames, channels_per_frame), dtype=np.float64)
    for frame_idx in range(num_frames):
        col = 0
        for joint_name in joint_order:
            joint_idx = bone_index[joint_name]
            if joint_name == root_name:
                frames[frame_idx, col : col + 3] = root_cm[frame_idx]
                col += 3
                frames[frame_idx, col : col + 3] = eulers_deg[frame_idx, joint_idx]
                col += 3
            else:
                frames[frame_idx, col : col + 3] = eulers_deg[frame_idx, joint_idx]
                col += 3
    return frames


class BvhExporter:
    """Convert ARDY local rotations and root translation into BVH files."""

    def __init__(self, skeleton: SkeletonBase):
        self.skeleton = skeleton

    def to_bvh_text(
        self,
        local_rot_mats: np.ndarray | torch.Tensor,
        root_positions: np.ndarray | torch.Tensor,
        fps: float,
        character_index: int = 0,
    ) -> str:
        """Return a complete BVH file as a string."""
        local_rot_mats, root_positions, export_skeleton = _prepare_motion_for_bvh(
            local_rot_mats,
            root_positions,
            self.skeleton,
            character_index=character_index,
        )

        bone_parents = export_skeleton.bone_order_names_with_parents
        bone_index = export_skeleton.bone_index
        neutral_joints = to_numpy(export_skeleton.neutral_joints)

        root_name, children = _build_children_map(bone_parents)
        joint_order = _depth_first_joint_order(root_name, children)
        offsets = _compute_offsets_cm(bone_parents, neutral_joints, bone_index)

        lines: list[str] = ["HIERARCHY"]
        _write_hierarchy_block(lines, root_name, children, offsets, root_name, indent=0)

        frame_time = 1.0 / float(fps)
        motion = _build_motion_channels(local_rot_mats, root_positions, joint_order, bone_index, root_name)
        lines.append("MOTION")
        lines.append(f"Frames: {motion.shape[0]}")
        lines.append(f"Frame Time: {frame_time:.6f}")
        for frame_idx in range(motion.shape[0]):
            lines.append(" ".join(f"{value:.6f}" for value in motion[frame_idx]))
        return "\n".join(lines) + "\n"

    def save_bvh(
        self,
        local_rot_mats: np.ndarray | torch.Tensor,
        root_positions: np.ndarray | torch.Tensor,
        bvh_path: str,
        fps: float,
        character_index: int = 0,
    ) -> None:
        """Write a BVH file to disk."""
        bvh_text = self.to_bvh_text(
            local_rot_mats,
            root_positions,
            fps,
            character_index=character_index,
        )
        parent = os.path.dirname(bvh_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(bvh_path, "w", encoding="utf-8") as f:
            f.write(bvh_text)


def write_bvh(
    path: str,
    local_rot_mats: np.ndarray | torch.Tensor,
    root_positions: np.ndarray | torch.Tensor,
    fps: float,
    skeleton: SkeletonBase,
    character_index: int = 0,
) -> None:
    """Write ``local_rot_mats`` and ``root_positions`` to a BVH file.

    Args:
        path: Output ``.bvh`` path.
        local_rot_mats: ``[T, J, 3, 3]`` or ``[B, T, J, 3, 3]`` local joint rotations.
        root_positions: ``[T, 3]`` or ``[B, T, 3]`` root translations in meters.
        fps: Playback rate for the ``Frame Time`` header.
        skeleton: Source skeleton (SOMA30 is expanded to 77 joints automatically).
        character_index: Batch index when motion tensors include a leading batch dimension.
    """
    BvhExporter(skeleton).save_bvh(
        local_rot_mats,
        root_positions,
        path,
        fps,
        character_index=character_index,
    )
