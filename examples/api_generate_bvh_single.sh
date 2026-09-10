#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot BVH generation: single prompt + frame count.
#
# POST /v1/generate/bvh
#   {"prompt": "...", "frames": 80}
#
# Prerequisites:
#   pip install -e ".[api]"
#   ./api.sh
#
# Usage:
#   ./examples/api_generate_bvh_single.sh
#   PROMPT="A person jumps." FRAMES=120 ./examples/api_generate_bvh_single.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/api_lib.sh
source "${SCRIPT_DIR}/api_lib.sh"

HOST="${HOST:-http://127.0.0.1:2334}"
API="${HOST}/v1"
PROMPT="${PROMPT:-A person walks forward.}"
FRAMES="${FRAMES:-80}"
OUTPUT_BVH="${OUTPUT_BVH:-outputs/api_generate_bvh_single.bvh}"

build_request_json() {
  python3 - "${PROMPT}" "${FRAMES}" <<'PY'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "frames": int(sys.argv[2])}))
PY
}

wait_for_server "${HOST}"

echo "Generating ${FRAMES} frames: ${PROMPT}"
mkdir -p "$(dirname "${OUTPUT_BVH}")"
curl -sf -X POST "${API}/generate/bvh" \
  -H "Content-Type: application/json" \
  --max-time "${API_CURL_MAX_TIME}" \
  -d "$(build_request_json)" \
  -o "${OUTPUT_BVH}"
echo "Done. BVH written to ${OUTPUT_BVH}"
