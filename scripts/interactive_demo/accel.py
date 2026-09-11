# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceleration backend helpers for the interactive demo."""

from __future__ import annotations

import torch


def is_rocm_torch() -> bool:
    return bool(getattr(torch.version, "hip", None))


def is_cuda_usable() -> bool:
    return torch.cuda.is_available()


def is_tensorrt_usable() -> bool:
    """True when ONNX-TRT acceleration can run (NVIDIA CUDA + tensorrt installed)."""
    if is_rocm_torch() or not is_cuda_usable():
        return False
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return False
    return True


def default_compile_model() -> bool:
    """Whether startup should prefer an accelerated backend over plain PyTorch."""
    if is_rocm_torch():
        return True
    return is_tensorrt_usable()


def acceleration_options() -> list[str]:
    if is_tensorrt_usable():
        return ["None", "ONNX-TRT (fp16)", "ONNX-TRT (fp32)", "torch.compile"]
    return ["None", "torch.compile"]


def initial_acceleration_mode(compile_model: bool) -> str:
    if not compile_model:
        return "None"
    if is_rocm_torch():
        return "torch.compile"
    if is_tensorrt_usable():
        return "ONNX-TRT (fp16)"
    if is_cuda_usable():
        return "torch.compile"
    return "None"
