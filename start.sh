#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Prefer the project venv (where viser / ardy / ROCm torch are installed).
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
else
  echo "Missing $ROOT/.venv — create it and install deps first." >&2
  exit 1
fi

# Use the discrete AMD GPU (skip the iGPU) when ROCm is available.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# CUDA PyTorch without a working NVIDIA GPU cannot build TensorRT engines (CUDA error 35).
EXTRA_ARGS=()
if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  EXTRA_ARGS+=(--no-compile)
fi

exec python scripts/run_demo.py "${EXTRA_ARGS[@]}" "$@"
