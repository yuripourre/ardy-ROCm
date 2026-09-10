# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared curl helpers for ARDY API example scripts.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/api_lib.sh"

# shellcheck disable=SC2034
API_CURL_MAX_TIME="${API_CURL_MAX_TIME:-600}"
GENERATE_MAX_STEPS="${GENERATE_MAX_STEPS:-50}"

json_field() {
  python3 -c "import json,sys; print(json.load(sys.stdin)$1)"
}

api_post_json() {
  local url="$1"
  local body="${2:-"{}"}"
  local tmp http_code
  tmp="$(mktemp)"
  http_code="$(
    curl -sS -o "${tmp}" -w "%{http_code}" --max-time "${API_CURL_MAX_TIME}" \
      -X POST "${url}" \
      -H "Content-Type: application/json" \
      -d "${body}"
  )"
  if [[ "${http_code}" -lt 200 || "${http_code}" -ge 300 ]]; then
    echo "ERROR: POST ${url} failed (HTTP ${http_code})" >&2
    if [[ -s "${tmp}" ]]; then
      cat "${tmp}" >&2
      echo >&2
    fi
    rm -f "${tmp}"
    return 1
  fi
  cat "${tmp}"
  rm -f "${tmp}"
}

api_put_json() {
  local url="$1"
  local body="$2"
  local tmp http_code
  tmp="$(mktemp)"
  http_code="$(
    curl -sS -o "${tmp}" -w "%{http_code}" --max-time 60 \
      -X PUT "${url}" \
      -H "Content-Type: application/json" \
      -d "${body}"
  )"
  if [[ "${http_code}" -lt 200 || "${http_code}" -ge 300 ]]; then
    echo "ERROR: PUT ${url} failed (HTTP ${http_code})" >&2
    if [[ -s "${tmp}" ]]; then
      cat "${tmp}" >&2
      echo >&2
    fi
    rm -f "${tmp}"
    return 1
  fi
  cat "${tmp}"
  rm -f "${tmp}"
}

wait_for_server() {
  local host="$1"
  echo "Waiting for ARDY API at ${host}..."
  for _ in $(seq 1 60); do
    if curl -sf "${host}/v1/health" >/dev/null 2>&1; then
      echo "Server is up ($(curl -s "${host}/v1/health" | json_field "['device']"))"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: API server not reachable. Start it with: ./api.sh" >&2
  return 1
}
