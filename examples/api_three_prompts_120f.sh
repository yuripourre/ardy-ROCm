#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build a 120-frame motion clip using the ARDY REST API with curl only.
#
# Three text prompts, 40 frames each (3 × 40 = 120):
#   frames  0-39  → prompt 1  "A person walks forward"
#   frames 40-79  → prompt 2  "A person turns to the left"
#   frames 80-119 → prompt 3  "A person waves with both hands"
#
# Prerequisites:
#   pip install -e ".[api]"
#   ./api.sh                                # in another terminal
#
# Usage:
#   ./examples/api_three_prompts_120f.sh
#   HOST=http://192.168.1.10:2334 ./examples/api_three_prompts_120f.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=examples/api_lib.sh
source "${SCRIPT_DIR}/api_lib.sh"

HOST="${HOST:-http://127.0.0.1:2334}"
API="${HOST}/v1"
OUTPUT_BVH="${OUTPUT_BVH:-outputs/api_three_prompts_120f.bvh}"
SEGMENT_LEN=40
TARGET_FRAMES=120

PROMPT_1='A person walks forward at a steady pace.'
PROMPT_2='A person turns to the left while walking.'
PROMPT_3='A person waves with both hands cheerfully.'

frame_count() {
  curl -s "${API}/sessions/${SESSION_ID}" | json_field "['frame_count']"
}

trim_to() {
  local keep="$1"
  local total
  total="$(frame_count)"
  if [[ "${total}" -gt "${keep}" ]]; then
    echo "  Trimming frames ${keep}-${total} (keeping 0-$((keep - 1)))"
    curl -sf -X DELETE "${API}/sessions/${SESSION_ID}/motion/frames" \
      -H "Content-Type: application/json" \
      -d "{\"start\": ${keep}, \"end\": ${total}}" >/dev/null
  fi
}

generate_step() {
  local body="${1:-"{}"}"
  local response
  response="$(api_post_json "${API}/sessions/${SESSION_ID}/generate/step" "${body}")"
  echo "${response}" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'    frames {r[\"start_frame\"]}-{r[\"end_frame\"]}')"
}

restart_from_frame() {
  local frame="$1"
  local response
  response="$(api_post_json "${API}/sessions/${SESSION_ID}/generate/restart_from" "{\"frame\": ${frame}}")"
  echo "${response}" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'    frames {r[\"start_frame\"]}-{r[\"end_frame\"]}')"
}

generate_until() {
  local target="$1"
  local body="${2:-"{}"}"
  local steps=0
  while [[ "$(frame_count)" -lt "${target}" ]]; do
    local count
    count="$(frame_count)"
    generate_step "${body}"
    steps=$((steps + 1))
    if [[ "${count}" -eq "$(frame_count)" ]]; then
      echo "ERROR: generate/step did not add frames (stuck at ${count}, target ${target})" >&2
      exit 1
    fi
    if [[ "${steps}" -ge "${GENERATE_MAX_STEPS}" ]]; then
      echo "ERROR: generate_until exceeded ${GENERATE_MAX_STEPS} steps (frame_count=$(frame_count), target=${target})" >&2
      exit 1
    fi
  done
}

build_full_prompts_json() {
  python3 - "${SEGMENT_LEN}" "${PROMPT_1}" "${PROMPT_2}" "${PROMPT_3}" <<'PY'
import json, sys
seg_len = int(sys.argv[1])
texts = [sys.argv[2], sys.argv[3], sys.argv[4]]
prompts = []
for seg, text in enumerate(texts):
    start = seg * seg_len
    end = start + seg_len - 1
    prompts.append({"text": text, "start_frame": start, "end_frame": end})
print(json.dumps(prompts))
PY
}

wait_for_server "${HOST}"

echo "Creating session..."
SESSION_ID="$(
  curl -sf -X POST "${API}/sessions" | json_field "['id']"
)"
echo "Session: ${SESSION_ID}"

echo "Setting full 3-prompt timeline..."
api_put_json "${API}/sessions/${SESSION_ID}/prompts" "$(build_full_prompts_json)" >/dev/null

NUM_SEGMENTS=$((TARGET_FRAMES / SEGMENT_LEN))

for seg in $(seq 0 $((NUM_SEGMENTS - 1))); do
  seg_start=$((seg * SEGMENT_LEN))
  seg_end_frame=$((seg_start + SEGMENT_LEN - 1))
  target_frames=$((seg_end_frame + 1))

  echo ""
  echo "=== Segment ${seg}: frames ${seg_start}-${seg_end_frame} ==="

  if [[ "${seg}" -eq 0 ]]; then
    echo "  Generating..."
    generate_until "${target_frames}"
  else
    restart_frame=$((seg_start - 1))
    echo "  Restarting from frame ${restart_frame}..."
    restart_from_frame "${restart_frame}"
    echo "  Extending to frame ${target_frames}..."
    generate_until "${target_frames}"
  fi

  echo "  Frame count: $(frame_count)"
done

trim_to "${TARGET_FRAMES}"

FINAL_FRAMES="$(frame_count)"
echo ""
echo "Final frame count: ${FINAL_FRAMES}"

mkdir -p "$(dirname "${OUTPUT_BVH}")"
echo "Exporting BVH to ${OUTPUT_BVH}..."
curl -sf "${API}/sessions/${SESSION_ID}/export/bvh" -o "${OUTPUT_BVH}"
echo "Done. BVH written to ${OUTPUT_BVH}"

echo "Cleaning up session ${SESSION_ID}..."
curl -sf -X DELETE "${API}/sessions/${SESSION_ID}" >/dev/null
echo "Finished."
