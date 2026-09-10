# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import numpy as np

from ardy.exports.bvh import BvhExporter, write_bvh
from ardy.skeleton import CoreSkeleton27, G1Skeleton34, SOMASkeleton30, SOMASkeleton77
from ardy.skeleton.bvh import Bvh


def _identity_motion(num_frames: int, num_joints: int) -> tuple[np.ndarray, np.ndarray]:
    local_rot_mats = np.tile(np.eye(3), (num_frames, num_joints, 1, 1))
    root_positions = np.zeros((num_frames, 3), dtype=np.float64)
    return local_rot_mats, root_positions


def _parse_bvh_header(path: Path) -> tuple[int, float, int]:
    text = path.read_text(encoding="utf-8")
    mocap = Bvh(text, backend="np")
    frames_line = next(line for line in text.splitlines() if line.startswith("Frames:"))
    num_frames = int(frames_line.split(":")[1].strip())
    channel_count = mocap.np_data_array.shape[1]
    return num_frames, mocap.frame_time, channel_count


def test_write_core_skeleton_bvh_structure():
    skeleton = CoreSkeleton27()
    num_frames = 4
    fps = 20.0
    local_rot_mats, root_positions = _identity_motion(num_frames, skeleton.nbjoints)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "motion.bvh"
        write_bvh(str(path), local_rot_mats, root_positions, fps, skeleton)

        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.startswith("HIERARCHY")
        assert "ROOT Hips" in text
        assert "MOTION" in text

        parsed_frames, frame_time, channel_count = _parse_bvh_header(path)
        assert parsed_frames == num_frames
        assert abs(frame_time - 1.0 / fps) < 1e-6
        expected_channels = 6 + (skeleton.nbjoints - 1) * 3
        assert channel_count == expected_channels
        motion_data = Bvh(text, backend="np").np_data_array
        assert np.isfinite(motion_data).all()


def test_write_g1_skeleton_bvh_structure():
    skeleton = G1Skeleton34()
    num_frames = 3
    local_rot_mats, root_positions = _identity_motion(num_frames, skeleton.nbjoints)

    exporter = BvhExporter(skeleton)
    text = exporter.to_bvh_text(local_rot_mats, root_positions, fps=25.0)

    assert "ROOT pelvis_skel" in text
    assert "left_hip_pitch_skel" in text
    motion_values = np.array([float(x) for x in text.split("MOTION")[-1].splitlines()[-num_frames:][0].split()])
    assert motion_values.shape[0] == 6 + (skeleton.nbjoints - 1) * 3


def test_soma30_expands_to_77_joints_in_bvh():
    skeleton = SOMASkeleton30()
    num_frames = 2
    local_rot_mats, root_positions = _identity_motion(num_frames, skeleton.nbjoints)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "soma.bvh"
        write_bvh(str(path), local_rot_mats, root_positions, fps=30.0, skeleton=skeleton)
        _, _, channel_count = _parse_bvh_header(path)
        soma77 = SOMASkeleton77()
        expected_channels = 6 + (soma77.nbjoints - 1) * 3
        assert channel_count == expected_channels


def test_batched_motion_exports_selected_character():
    skeleton = CoreSkeleton27()
    num_frames = 2
    local_rot_mats = np.tile(np.eye(3), (2, num_frames, skeleton.nbjoints, 1, 1))
    root_positions = np.zeros((2, num_frames, 3), dtype=np.float64)
    root_positions[1, :, 0] = 1.0

    exporter = BvhExporter(skeleton)
    text = exporter.to_bvh_text(
        local_rot_mats,
        root_positions,
        fps=20.0,
        character_index=1,
    )
    first_frame = np.array([float(x) for x in text.split("Frame Time:")[-1].strip().splitlines()[1].split()])
    assert abs(first_frame[0] - 100.0) < 1e-3  # 1 m -> 100 cm on character 1
