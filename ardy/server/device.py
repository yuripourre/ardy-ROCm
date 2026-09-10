# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device selection for the API server."""

import torch


def resolve_device() -> str:
    """Pick the inference device once at server startup."""
    return "cuda" if torch.cuda.is_available() else "cpu"
