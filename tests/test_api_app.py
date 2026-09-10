# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient

from ardy.server.app import create_app
from ardy.server.store import SessionStore


def test_health_and_models():
    store = SessionStore(device="cpu", model_name=None)
    client = TestClient(create_app(store))
    health = client.get("/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["device"] == "cpu"
    assert body["ready"] is False
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert "core" in models.json()["models"]


def test_generate_bvh_not_ready():
    store = SessionStore(device="cpu", model_name=None)
    client = TestClient(create_app(store))
    response = client.post(
        "/v1/generate/bvh",
        json={"prompt": "A person walks.", "frames": 40},
    )
    assert response.status_code == 503


def test_generate_bvh_validation():
    store = SessionStore(device="cpu", model_name=None)
    client = TestClient(create_app(store))
    assert client.post("/v1/generate/bvh", json={"prompt": "", "frames": 40}).status_code == 422
    assert client.post("/v1/generate/bvh", json={"prompt": "walk", "frames": 0}).status_code == 422
    assert client.post("/v1/generate/bvh", json={"segments": []}).status_code == 422
    assert (
        client.post(
            "/v1/generate/bvh",
            json={
                "segments": [{"prompt": "walk", "frames": 40}],
                "prompt": "walk",
                "frames": 40,
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/generate/bvh",
            json={
                "segments": [
                    {"prompt": "walk", "frames": 1500},
                    {"prompt": "turn", "frames": 600},
                ]
            },
        ).status_code
        == 422
    )


def test_generate_bvh_segments_not_ready():
    store = SessionStore(device="cpu", model_name=None)
    client = TestClient(create_app(store))
    response = client.post(
        "/v1/generate/bvh",
        json={
            "segments": [
                {"prompt": "A person walks.", "frames": 40},
                {"prompt": "A person turns.", "frames": 40},
            ]
        },
    )
    assert response.status_code == 503
