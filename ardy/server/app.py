# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for the ARDY REST API."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response

from ardy.model.registry import AVAILABLE_MODELS
from ardy.server.generation import (
    delete_motion_frames,
    generate_bvh_from_segments,
    generate_step,
    preserve_motion_frames,
    restart_from,
)
from ardy.server.motion_io import motion_to_bvh_text, motion_to_npz_bytes
from ardy.server.schemas import (
    ConstraintsResponse,
    EndEffectorKeyframeRequest,
    FrameRangeRequest,
    FullbodyKeyframeRequest,
    GenerateBvhRequest,
    GenerateStepRequest,
    GenerateStepResponse,
    HealthResponse,
    InterpolateRoot2dRequest,
    ModelsResponse,
    PromptInput,
    PromptResponse,
    RestartFromRequest,
    Root2dKeyframeRequest,
    SessionInfoResponse,
    SessionResponse,
)
from ardy.server.session import PromptSegment
from ardy.server.store import SessionStore


def create_app(store: SessionStore) -> FastAPI:
    app = FastAPI(title="ARDY API", version="1.0.0")

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if store.is_ready else "loading",
            device=store.device,
            model=store.model_name,
            ready=store.is_ready,
        )

    @app.get("/v1/models", response_model=ModelsResponse)
    def list_models() -> ModelsResponse:
        nicknames = sorted({name for name in AVAILABLE_MODELS if not name.startswith("ARDY-")})
        return ModelsResponse(models=nicknames)

    @app.post("/v1/generate/bvh")
    def generate_bvh(body: GenerateBvhRequest) -> Response:
        if not store.is_ready:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

        cfg_weight = None
        if body.cfg_weight is not None:
            if len(body.cfg_weight) == 1:
                cfg_weight = (body.cfg_weight[0], body.cfg_weight[0])
            elif len(body.cfg_weight) == 2:
                cfg_weight = (body.cfg_weight[0], body.cfg_weight[1])
            else:
                raise HTTPException(status_code=400, detail="cfg_weight expects 1 or 2 floats")

        session = store.create_session()
        try:
            with session.generation_lock:
                try:
                    segments = [(seg.prompt, seg.frames) for seg in body.resolved_segments]
                    bvh_text = generate_bvh_from_segments(
                        session,
                        segments=segments,
                        diffusion_steps=body.diffusion_steps,
                        cfg_weight=cfg_weight,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except RuntimeError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            try:
                store.delete(session.id)
            except KeyError:
                pass

        return Response(
            content=bvh_text,
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="motion.bvh"'},
        )

    @app.post("/v1/sessions", response_model=SessionResponse, status_code=201)
    def create_session() -> SessionResponse:
        if not store.is_ready:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")
        try:
            session = store.create_session()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SessionResponse(
            id=session.id,
            model=session.model_name,
            fps=session.model_fps,
            horizon=session.gen_horizon_len,
            skeleton=session.skeleton_name,
            frame_count=session.frame_count,
        )

    def _get_session(session_id: str):
        try:
            return store.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

    @app.get("/v1/sessions/{session_id}", response_model=SessionInfoResponse)
    def get_session(session_id: str) -> SessionInfoResponse:
        session = _get_session(session_id)
        return SessionInfoResponse(
            id=session.id,
            model=session.model_name,
            fps=session.model_fps,
            horizon=session.gen_horizon_len,
            skeleton=session.skeleton_name,
            frame_count=session.frame_count,
        )

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        try:
            store.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return Response(status_code=204)

    @app.get("/v1/sessions/{session_id}/prompts", response_model=list[PromptResponse])
    def list_prompts(session_id: str) -> list[PromptResponse]:
        session = _get_session(session_id)
        return [
            PromptResponse(
                id=p.id,
                text=p.text,
                start_frame=p.start_frame,
                end_frame=p.end_frame,
            )
            for p in session.prompts
        ]

    @app.put("/v1/sessions/{session_id}/prompts", response_model=list[PromptResponse])
    def replace_prompts(session_id: str, body: list[PromptInput]) -> list[PromptResponse]:
        session = _get_session(session_id)
        session.prompts = [
            PromptSegment(
                id=prompt.id or uuid.uuid4().hex,
                text=prompt.text,
                start_frame=prompt.start_frame,
                end_frame=prompt.end_frame,
            )
            for prompt in body
        ]
        session.text_embedding = None
        return list_prompts(session_id)

    @app.patch("/v1/sessions/{session_id}/prompts/{prompt_id}", response_model=PromptResponse)
    def patch_prompt(session_id: str, prompt_id: str, body: PromptInput) -> PromptResponse:
        session = _get_session(session_id)
        for prompt in session.prompts:
            if prompt.id == prompt_id:
                prompt.text = body.text
                prompt.start_frame = body.start_frame
                prompt.end_frame = body.end_frame
                session.text_embedding = None
                return PromptResponse(
                    id=prompt.id,
                    text=prompt.text,
                    start_frame=prompt.start_frame,
                    end_frame=prompt.end_frame,
                )
        raise HTTPException(status_code=404, detail="Prompt not found")

    @app.get("/v1/sessions/{session_id}/constraints", response_model=ConstraintsResponse)
    def list_constraints(session_id: str) -> ConstraintsResponse:
        session = _get_session(session_id)
        data = session.constraints.to_api_dict()
        return ConstraintsResponse(**data)

    @app.put("/v1/sessions/{session_id}/constraints/{constraint_type}/keyframes/{frame}")
    def upsert_keyframe(
        session_id: str,
        constraint_type: str,
        frame: int,
        body: dict,
    ) -> dict:
        session = _get_session(session_id)
        try:
            if constraint_type == "root2d":
                req = Root2dKeyframeRequest(**body)
                session.constraints.root2d.upsert(frame, req.x, req.z, req.heading)
            elif constraint_type == "fullbody":
                req = FullbodyKeyframeRequest(**body)
                session.constraints.fullbody.upsert(frame, np.array(req.joints_pos), np.array(req.joints_rot))
            elif constraint_type == "end_effector":
                req = EndEffectorKeyframeRequest(**body)
                session.constraints.end_effector.upsert(
                    frame,
                    np.array(req.joints_pos),
                    np.array(req.joints_rot),
                    req.joint_names,
                    req.end_effector_type,
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown constraint type: {constraint_type}")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"frame": frame, "type": constraint_type}

    @app.delete("/v1/sessions/{session_id}/constraints/{constraint_type}/keyframes/{frame}", status_code=204)
    def delete_keyframe(session_id: str, constraint_type: str, frame: int) -> Response:
        session = _get_session(session_id)
        try:
            session.constraints.track_for_type(constraint_type).delete(frame)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.post("/v1/sessions/{session_id}/constraints/root2d/interpolate")
    def interpolate_root2d(session_id: str, body: InterpolateRoot2dRequest) -> dict:
        session = _get_session(session_id)
        if body.end_frame < body.start_frame:
            raise HTTPException(status_code=400, detail="end_frame must be >= start_frame")
        try:
            path = session.constraints.root2d.set_dense_path(body.start_frame, body.end_frame, smooth=body.smooth)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "start_frame": body.start_frame,
            "end_frame": body.end_frame,
            "num_frames": int(path.shape[0]),
        }

    @app.post("/v1/sessions/{session_id}/generate/step", response_model=GenerateStepResponse)
    def run_generate_step(session_id: str, body: GenerateStepRequest) -> GenerateStepResponse:
        session = _get_session(session_id)
        cfg_weight = None
        if body.cfg_weight is not None:
            if len(body.cfg_weight) == 1:
                cfg_weight = (body.cfg_weight[0], body.cfg_weight[0])
            elif len(body.cfg_weight) == 2:
                cfg_weight = (body.cfg_weight[0], body.cfg_weight[1])
            else:
                raise HTTPException(status_code=400, detail="cfg_weight expects 1 or 2 floats")

        with session.generation_lock:
            try:
                start_frame, end_frame = generate_step(
                    session,
                    diffusion_steps=body.diffusion_steps,
                    cfg_weight=cfg_weight,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        return GenerateStepResponse(start_frame=start_frame, end_frame=end_frame)

    @app.post("/v1/sessions/{session_id}/generate/restart_from")
    def run_restart_from(session_id: str, body: RestartFromRequest) -> dict:
        session = _get_session(session_id)
        with session.generation_lock:
            try:
                restart_from(session, body.frame)
                start_frame, end_frame = generate_step(session)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"frame": body.frame, "start_frame": start_frame, "end_frame": end_frame}

    @app.get("/v1/sessions/{session_id}/motion")
    def get_motion(
        session_id: str,
        start: int = Query(0, ge=0),
        end: Optional[int] = Query(None, ge=1),
    ) -> Response:
        session = _get_session(session_id)
        try:
            data = motion_to_npz_bytes(session, start=start, end=end)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="application/octet-stream")

    @app.delete("/v1/sessions/{session_id}/motion/frames", status_code=204)
    def remove_motion_frames(session_id: str, body: FrameRangeRequest) -> Response:
        session = _get_session(session_id)
        try:
            delete_motion_frames(session, body.start, body.end)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.post("/v1/sessions/{session_id}/motion/preserve")
    def preserve_frames(session_id: str, body: FrameRangeRequest) -> dict:
        session = _get_session(session_id)
        try:
            preserve_motion_frames(session, body.start, body.end)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"frame_count": session.frame_count}

    @app.get("/v1/sessions/{session_id}/export/npz")
    def export_npz(session_id: str) -> Response:
        session = _get_session(session_id)
        try:
            data = motion_to_npz_bytes(session)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="application/octet-stream")

    @app.get("/v1/sessions/{session_id}/export/bvh")
    def export_bvh(session_id: str) -> Response:
        session = _get_session(session_id)
        try:
            text = motion_to_bvh_text(session)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=text, media_type="text/plain")

    return app
