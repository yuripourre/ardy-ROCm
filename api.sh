#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
else
  echo "Missing $ROOT/.venv — create it and install deps first." >&2
  exit 1
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

TEXT_ENCODER_PORT="${TEXT_ENCODER_PORT:-9550}"
TEXT_ENCODER_URL="${TEXT_ENCODER_URL:-http://127.0.0.1:${TEXT_ENCODER_PORT}/}"

text_encoder_reachable() {
  curl -sf "${TEXT_ENCODER_URL}" >/dev/null 2>&1
}

start_text_encoder_if_needed() {
  if [[ "${START_TEXT_ENCODER:-1}" == "0" ]]; then
    return
  fi
  if text_encoder_reachable; then
    echo "Text encoder service already running at ${TEXT_ENCODER_URL}"
    return
  fi
  echo "Text encoder not reachable — starting local service on port ${TEXT_ENCODER_PORT}..."
  python scripts/run_text_encoder_server.py --port "${TEXT_ENCODER_PORT}" &
  local encoder_pid=$!
  for _ in $(seq 1 180); do
    if text_encoder_reachable; then
      echo "Text encoder ready (pid ${encoder_pid})."
      return
    fi
    if ! kill -0 "${encoder_pid}" 2>/dev/null; then
      echo "ERROR: Text encoder process exited before becoming ready." >&2
      exit 1
    fi
    sleep 2
  done
  echo "ERROR: Timed out waiting for text encoder at ${TEXT_ENCODER_URL}" >&2
  exit 1
}

start_text_encoder_if_needed

exec python scripts/run_api_server.py --model "${MODEL:-core}" "$@"
