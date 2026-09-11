# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disable broken torchvision before transformers/peft import vision utilities."""

from __future__ import annotations

import importlib.util


def disable_broken_torchvision() -> None:
    """Tell transformers to ignore torchvision when the wheel is installed but unusable."""
    if importlib.util.find_spec("torchvision") is None:
        return
    try:
        import torchvision  # noqa: F401
    except Exception:
        from transformers.utils import import_utils

        import_utils.is_torchvision_available.cache_clear()
        import_utils.is_torchvision_available = lambda: False  # type: ignore[method-assign]
