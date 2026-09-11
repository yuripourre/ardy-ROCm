# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from interactive_demo.accel import (
    acceleration_options,
    default_compile_model,
    initial_acceleration_mode,
    is_tensorrt_usable,
)


def test_no_cuda_skips_tensorrt():
    with patch("interactive_demo.accel.is_rocm_torch", return_value=False):
        with patch("interactive_demo.accel.is_cuda_usable", return_value=False):
            assert is_tensorrt_usable() is False
            assert default_compile_model() is False
            assert acceleration_options() == ["None", "torch.compile"]
            assert initial_acceleration_mode(True) == "None"
            assert initial_acceleration_mode(False) == "None"


def test_rocm_prefers_torch_compile():
    with patch("interactive_demo.accel.is_rocm_torch", return_value=True):
        with patch("interactive_demo.accel.is_cuda_usable", return_value=True):
            with patch("interactive_demo.accel.is_tensorrt_usable", return_value=False):
                assert default_compile_model() is True
                assert acceleration_options() == ["None", "torch.compile"]
                assert initial_acceleration_mode(True) == "torch.compile"
