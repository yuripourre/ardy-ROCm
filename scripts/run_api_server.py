# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless REST API server for ARDY motion generation.

Examples:
    python scripts/run_api_server.py
    python scripts/run_api_server.py --host 0.0.0.0 --port 2334 --model core
"""

import argparse

import uvicorn

from ardy.model import DEFAULT_MODEL
from ardy.server.app import create_app
from ardy.server.device import resolve_device
from ardy.server.store import SessionStore


def parse_args():
    parser = argparse.ArgumentParser(description="ARDY REST API server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2334)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model nickname to load at startup (default: core).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device()
    print(f"ARDY API server using device: {device}", flush=True)
    store = SessionStore(device=device, model_name=args.model)
    app = create_app(store)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
