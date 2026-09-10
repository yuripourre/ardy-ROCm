#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper — see the dedicated examples:
#   ./examples/api_generate_bvh_single.sh   one prompt + frame count
#   ./examples/api_generate_bvh_multi.sh    multiple prompts concatenated

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/api_generate_bvh_single.sh" "$@"
