#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot BVH generation: multiple prompts concatenated by the server.
#
# POST /v1/generate/bvh
#   {"segments": [
#     {"prompt": "...", "frames": 40},
#     {"prompt": "...", "frames": 40},
#     {"prompt": "...", "frames": 40}
#   ]}
#
# Three prompts, 40 frames each (3 × 40 = 120 total):
#   frames  0-39  → walks forward
#   frames 40-79  → turns left
#   frames 80-119 → waves
#
# Prerequisites:
#   pip install -e ".[api]"
#   ./api.sh
#
# Usage:
#   ./examples/api_generate_bvh_multi.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/api_lib.sh
source "${SCRIPT_DIR}/api_lib.sh"

HOST="${HOST:-http://127.0.0.1:2334}"
API="${HOST}/v1"
OUTPUT_BVH="${OUTPUT_BVH:-outputs/api_generate_bvh_multi.bvh}"
SEGMENT_FRAMES=40
TARGET_FRAMES=120

PROMPT_1='A person walks forward at a steady pace.'
PROMPT_2='A person turns to the left while walking.'
PROMPT_3='A person waves with both hands cheerfully.'

build_request_json() {
  python3 - "${SEGMENT_FRAMES}" "${PROMPT_1}" "${PROMPT_2}" "${PROMPT_3}" <<'PY'
import json, sys
seg_len = int(sys.argv[1])
texts = [sys.argv[2], sys.argv[3], sys.argv[4]]
segments = []
for text in texts:
    segments.append({"prompt": text, "frames": seg_len})
print(json.dumps({"segments": segments}))
PY
}

wait_for_server "${HOST}"

echo "Generating ${TARGET_FRAMES} frames (3 × ${SEGMENT_FRAMES}):"
echo "  0-$((SEGMENT_FRAMES - 1)): ${PROMPT_1}"
echo "  ${SEGMENT_FRAMES}-$((SEGMENT_FRAMES * 2 - 1)): ${PROMPT_2}"
echo "  $((SEGMENT_FRAMES * 2))-$((TARGET_FRAMES - 1)): ${PROMPT_3}"

mkdir -p "$(dirname "${OUTPUT_BVH}")"
curl -sf -X POST "${API}/generate/bvh" \
  -H "Content-Type: application/json" \
  --max-time "${API_CURL_MAX_TIME}" \
  -d "$(build_request_json)" \
  -o "${OUTPUT_BVH}"
echo "Done. BVH written to ${OUTPUT_BVH}"
