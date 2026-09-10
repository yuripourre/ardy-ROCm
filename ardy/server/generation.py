# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autoregressive generation for API sessions."""

from __future__ import annotations

import time
from typing import Optional

import torch

from ardy.postprocess import post_process_motion
from ardy.server.constraints import compute_constraint_mask, compute_model_constraints_lst
from ardy.server.motion_io import motion_to_bvh_text
from ardy.server.session import INFINITE_FRAME_IDX, MotionSession, PromptSegment
from ardy.server.window_budget import compute_window_num_frames


NUM_SAMPLES = 1


def _get_history_motion(session: MotionSession):
    frame_idx = session.frame_idx
    replan_buffer_size = session.replan_buffer_size
    history_crop_length = session.history_crop_length
    motion_tensor = session.motion_tensor

    cur_motion_len = motion_tensor.shape[1] if motion_tensor is not None else 0
    history_end_idx = min(cur_motion_len - 1, frame_idx + replan_buffer_size)
    if cur_motion_len >= session.num_frames_per_token:
        history_end_idx = max(history_end_idx, session.num_frames_per_token - 1)
    history_length = min(history_end_idx + 1, history_crop_length)
    history_length = history_length // session.num_frames_per_token * session.num_frames_per_token
    history_start_idx = max(0, history_end_idx - history_length + 1)

    history_motion_tensor = None
    if motion_tensor is not None and history_start_idx <= history_end_idx:
        history_motion_tensor = motion_tensor[:, history_start_idx : history_end_idx + 1]

    return history_motion_tensor, history_start_idx, history_end_idx, history_length


def _constraint_indices(session: MotionSession) -> list[int]:
    indices: list[int] = []
    for track in (
        session.constraints.root2d,
        session.constraints.fullbody,
        session.constraints.end_effector,
    ):
        info = track.get_constraint_info()
        indices.extend(info["frame_idx"])
    return indices


def encode_active_prompt(session: MotionSession, text_encoder, generation_frame: int) -> None:
    prompt_text = session.active_prompt_text(generation_frame)
    if prompt_text is None:
        raise ValueError(f"No prompt covers frame {generation_frame}")
    text_feat, _ = text_encoder([prompt_text])
    session.text_embedding = text_feat.to(session.device)


def _generation_start_frame(session: MotionSession, history_end_idx: int) -> int:
    if session.motion_tensor is not None and session.motion_tensor.shape[1] > 0:
        return history_end_idx + 1
    return session.frame_idx


def generate_step(
    session: MotionSession,
    diffusion_steps: Optional[int] = None,
    cfg_weight: Optional[tuple[float, float]] = None,
) -> tuple[int, int]:
    """Run one autoregressive generation step. Returns ``(start_frame, end_frame)``."""
    if session.model is None:
        raise RuntimeError("Model is not loaded for this session")

    start_time = time.time()
    history_motion_tensor, history_start_idx, history_end_idx, history_length = _get_history_motion(session)

    generation_frame = _generation_start_frame(session, history_end_idx)
    encode_active_prompt(session, session.model.text_encoder, generation_frame)

    text_feat = session.text_embedding.repeat(NUM_SAMPLES, 1, 1)
    text_pad_mask = torch.ones(text_feat.shape[0], text_feat.shape[1], device=session.device, dtype=torch.bool)

    motion_mask = None
    observed_motion = None

    all_constraint_indices = _constraint_indices(session)
    has_valid_timeline_constraints = (
        len(all_constraint_indices) > 0 and max(all_constraint_indices) > history_end_idx
    )

    num_frames = compute_window_num_frames(
        history_length=history_length,
        gen_horizon_len=session.gen_horizon_len,
        num_frames_per_token=session.num_frames_per_token,
        max_window_len=session.max_window_len,
        history_start_idx=history_start_idx,
        max_constraint_idx=(max(all_constraint_indices) if has_valid_timeline_constraints else None),
        future_crop_length=session.future_crop_length,
    )

    if has_valid_timeline_constraints:
        motion_mask, observed_motion = compute_constraint_mask(
            session,
            NUM_SAMPLES,
            num_frames=num_frames + history_start_idx,
            history_end_idx=history_end_idx,
        )
        if motion_mask is not None and observed_motion is not None:
            motion_mask = motion_mask[:, history_start_idx:]
            observed_motion = observed_motion[:, history_start_idx:]
            motion_mask[:, :history_length] = 0.0
            observed_motion[:, :history_length] = 0.0

    denoising_steps = diffusion_steps if diffusion_steps is not None else session.diffusion_steps
    if cfg_weight is None:
        cfg_weight = (session.cfg_text_weight, session.cfg_constraint_weight)

    if history_motion_tensor is None:
        init_global_translation = (
            torch.zeros((NUM_SAMPLES, session.motion_rep.nfeats_dict["root_pos"]), device=session.device)
            if session.init_global_translation is None
            else session.init_global_translation.unsqueeze(0).repeat(NUM_SAMPLES, 1)
        )
        init_first_heading_angle = torch.full(
            (NUM_SAMPLES,),
            session.init_first_heading_angle,
            dtype=torch.float32,
            device=session.device,
        )
    else:
        init_global_translation = None
        init_first_heading_angle = None

    samples = session.model.autoregressive_step(
        num_frames=num_frames,
        num_denoising_steps=denoising_steps,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        cfg_weight=cfg_weight,
        texts=None,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        init_history_sequence=history_motion_tensor,
        init_global_translation=init_global_translation,
        init_first_heading_angle=init_first_heading_angle,
    )

    samples_unnormalized = session.motion_rep.unnormalize(samples)
    pred_joints_output = session.motion_rep.inverse(samples_unnormalized, is_normalized=False)

    joints_pos = pred_joints_output["posed_joints"]
    joints_rot = pred_joints_output["global_rot_mats"]
    foot_contacts = pred_joints_output.get("foot_contacts")
    if foot_contacts is None:
        foot_contacts = torch.zeros(
            (NUM_SAMPLES, history_length + session.gen_horizon_len, 2),
            device=session.device,
        )

    use_postprocess = session.enable_postprocess and "g1" not in session.model_name.lower()
    if use_postprocess:
        model_constraints = compute_model_constraints_lst(
            session,
            num_frames=session.gen_horizon_len + history_length + history_start_idx,
            history_end_idx=history_end_idx,
        )
        if model_constraints:
            local_rot_mats = pred_joints_output["local_rot_mats"]
            root_positions = pred_joints_output["root_positions"]
            for constraint in model_constraints:
                constraint.frame_indices = constraint.frame_indices - history_start_idx - history_length
            corrected_output = post_process_motion(
                local_rot_mats[:, history_length:],
                root_positions[:, history_length:],
                foot_contacts[:, history_length:],
                session.motion_rep.skeleton,
                constraint_lst=model_constraints,
                contact_threshold=session.postprocess_contact_threshold,
                root_margin=session.postprocess_root_margin,
            )
            joints_pos[:, history_length:] = corrected_output["posed_joints"]
            joints_rot[:, history_length:] = corrected_output["global_rot_mats"]
            corrected_tensor_unnormalized = session.motion_rep(
                local_joint_rots=corrected_output["local_rot_mats"],
                root_positions=corrected_output["root_positions"],
                to_normalize=False,
            )
            corrected_tensor_normalized = session.motion_rep.normalize(corrected_tensor_unnormalized)
            samples_unnormalized[:, history_length:] = corrected_tensor_unnormalized
            samples[:, history_length:] = corrected_tensor_normalized
            foot_contacts[:, history_length:] = corrected_tensor_unnormalized[
                :, :, session.motion_rep.slice_dict["foot_contacts"]
            ]

    joint_velocities = samples_unnormalized[:, :, session.motion_rep.slice_dict["velocities"]]
    joint_velocities = joint_velocities.reshape(
        NUM_SAMPLES,
        history_length + session.gen_horizon_len,
        session.motion_rep.skeleton.nbjoints,
        3,
    )
    root_velocities = joint_velocities[:, :, session.motion_rep.skeleton.root_idx, :]

    with session.motion_tensor_lock:
        if session.motion_tensor is None:
            session.motion_tensor = samples.clone()
            session.joints_pos = joints_pos.clone()
            session.joints_rot = joints_rot.clone()
            session.foot_contacts = foot_contacts.clone()
            session.root_velocities = root_velocities.clone()
            start_frame = 0
        else:
            start_frame = history_end_idx + 1
            session.motion_tensor = torch.cat(
                [session.motion_tensor[:, : history_end_idx + 1], samples[:, history_length:]],
                dim=1,
            )
            session.joints_pos = torch.cat(
                [session.joints_pos[:, : history_end_idx + 1], joints_pos[:, history_length:]],
                dim=1,
            )
            session.joints_rot = torch.cat(
                [session.joints_rot[:, : history_end_idx + 1], joints_rot[:, history_length:]],
                dim=1,
            )
            session.foot_contacts = torch.cat(
                [session.foot_contacts[:, : history_end_idx + 1], foot_contacts[:, history_length:]],
                dim=1,
            )
            session.root_velocities = torch.cat(
                [session.root_velocities[:, : history_end_idx + 1], root_velocities[:, history_length:]],
                dim=1,
            )

        session.max_frame_idx = session.motion_tensor.shape[1] - 1
        end_frame = session.max_frame_idx
        # Advance playhead so the next step appends after the latest frame (API has no playback loop).
        session.frame_idx = session.max_frame_idx

    print(f"Generate step time: {time.time() - start_time:.3f}s, frames {start_frame}-{end_frame}")
    return start_frame, end_frame


def restart_from(session: MotionSession, frame: int) -> None:
    """Preserve frames ``0..frame``, discard the rest, and prepare for replanning."""
    if session.model is None:
        raise RuntimeError("Model is not loaded for this session")
    if frame < 0:
        raise ValueError("frame must be >= 0")

    keep_end = frame + 1
    with session.motion_tensor_lock:
        if session.motion_tensor is not None:
            session.motion_tensor = session.motion_tensor[:, :keep_end]
        if session.joints_pos is not None:
            session.joints_pos = session.joints_pos[:, :keep_end]
        if session.joints_rot is not None:
            session.joints_rot = session.joints_rot[:, :keep_end]
        if session.foot_contacts is not None:
            session.foot_contacts = session.foot_contacts[:, :keep_end]
        if session.root_velocities is not None:
            session.root_velocities = session.root_velocities[:, :keep_end]

    session.max_frame_idx = frame if keep_end > 0 else -1
    session.frame_idx = frame
    session.text_embedding = None

    has_upcoming_prompt = any(p.start_frame == frame + 1 for p in session.prompts)
    for prompt in session.prompts:
        if prompt.covers_frame(frame) and has_upcoming_prompt:
            prompt.end_frame = frame

    for track in (
        session.constraints.root2d,
        session.constraints.fullbody,
        session.constraints.end_effector,
    ):
        info = track.get_constraint_info()
        for fi in list(info["frame_idx"]):
            if fi > frame:
                track.delete(fi)


def delete_motion_frames(session: MotionSession, start: int, end: int) -> None:
    """Delete frames in ``[start, end)`` and reindex constraints."""
    if start < 0 or end <= start:
        raise ValueError("Invalid frame range: require 0 <= start < end")
    if session.motion_tensor is None:
        raise RuntimeError("No motion to edit")

    total_frames = session.motion_tensor.shape[1]
    if end > total_frames:
        raise ValueError(f"end frame {end} exceeds motion length {total_frames}")

    with session.motion_tensor_lock:
        keep_before = session.motion_tensor[:, :start]
        keep_after = session.motion_tensor[:, end:]
        session.motion_tensor = torch.cat([keep_before, keep_after], dim=1)

        session.joints_pos = torch.cat([session.joints_pos[:, :start], session.joints_pos[:, end:]], dim=1)
        session.joints_rot = torch.cat([session.joints_rot[:, :start], session.joints_rot[:, end:]], dim=1)
        session.foot_contacts = torch.cat(
            [session.foot_contacts[:, :start], session.foot_contacts[:, end:]],
            dim=1,
        )
        session.root_velocities = torch.cat(
            [session.root_velocities[:, :start], session.root_velocities[:, end:]],
            dim=1,
        )
        session.max_frame_idx = session.motion_tensor.shape[1] - 1

    session.constraints.shift_after_delete(start, end)

    delete_count = end - start
    new_prompts: list[PromptSegment] = []
    for prompt in session.prompts:
        prompt_end = prompt.end_frame if prompt.end_frame is not None else INFINITE_FRAME_IDX
        if prompt.start_frame >= end:
            prompt.start_frame -= delete_count
            if prompt.end_frame is not None:
                prompt.end_frame -= delete_count
            new_prompts.append(prompt)
        elif prompt_end < start:
            new_prompts.append(prompt)
        elif prompt.start_frame < start and prompt_end >= end:
            if prompt.end_frame is not None:
                prompt.end_frame -= delete_count
            new_prompts.append(prompt)
    session.prompts = new_prompts

    if session.frame_idx >= end:
        session.frame_idx -= delete_count
    elif session.frame_idx >= start:
        session.frame_idx = start


def preserve_motion_frames(session: MotionSession, start: int, end: int) -> None:
    """Keep only frames in ``[start, end)`` and shift indices to start at 0."""
    if start < 0 or end <= start:
        raise ValueError("Invalid frame range: require 0 <= start < end")
    if session.motion_tensor is None:
        raise RuntimeError("No motion to edit")
    if end > session.motion_tensor.shape[1]:
        raise ValueError(f"end frame {end} exceeds motion length {session.motion_tensor.shape[1]}")

    with session.motion_tensor_lock:
        session.motion_tensor = session.motion_tensor[:, start:end].clone()
        session.joints_pos = session.joints_pos[:, start:end].clone()
        session.joints_rot = session.joints_rot[:, start:end].clone()
        session.foot_contacts = session.foot_contacts[:, start:end].clone()
        session.root_velocities = session.root_velocities[:, start:end].clone()
        session.max_frame_idx = session.motion_tensor.shape[1] - 1

    session.constraints.crop_and_shift(start, end)

    shift = start
    new_prompts: list[PromptSegment] = []
    for prompt in session.prompts:
        prompt_end = prompt.end_frame if prompt.end_frame is not None else INFINITE_FRAME_IDX
        if prompt_end < start or prompt.start_frame >= end:
            continue
        new_start = max(0, prompt.start_frame - shift)
        if prompt.end_frame is None:
            new_end = None
        else:
            new_end = min(end - shift - 1, prompt.end_frame - shift)
        new_prompts.append(
            PromptSegment(id=prompt.id, text=prompt.text, start_frame=new_start, end_frame=new_end)
        )
    session.prompts = new_prompts
    session.frame_idx = max(0, session.frame_idx - shift)


def _generate_until(
    session: MotionSession,
    target_frames: int,
    diffusion_steps: Optional[int] = None,
    cfg_weight: Optional[tuple[float, float]] = None,
) -> None:
    max_steps = target_frames // session.num_frames_per_token + 10
    steps = 0
    while session.frame_count < target_frames:
        if steps >= max_steps:
            raise RuntimeError(
                f"Generation did not reach {target_frames} frames after {max_steps} steps "
                f"(frame_count={session.frame_count})"
            )
        count = session.frame_count
        generate_step(session, diffusion_steps=diffusion_steps, cfg_weight=cfg_weight)
        steps += 1
        if count == session.frame_count:
            raise RuntimeError(
                f"generate_step did not add frames (stuck at {count}, target={target_frames})"
            )


def _build_segment_prompts(segments: list[tuple[str, int]]) -> list[PromptSegment]:
    prompts: list[PromptSegment] = []
    seg_start = 0
    for index, (text, length) in enumerate(segments):
        seg_end = seg_start + length - 1
        prompts.append(
            PromptSegment(
                id=f"seg{index}",
                text=text,
                start_frame=seg_start,
                end_frame=seg_end,
            )
        )
        seg_start = seg_end + 1
    return prompts


def generate_bvh_from_segments(
    session: MotionSession,
    segments: list[tuple[str, int]],
    diffusion_steps: Optional[int] = None,
    cfg_weight: Optional[tuple[float, float]] = None,
) -> str:
    """Generate motion for one or more prompt segments and return BVH text."""
    if not segments:
        raise ValueError("segments must not be empty")

    total_frames = sum(length for _, length in segments)
    session.prompts = _build_segment_prompts(segments)
    session.text_embedding = None

    seg_start = 0
    for index, (_, length) in enumerate(segments):
        target_frames = seg_start + length
        if index == 0:
            _generate_until(session, target_frames, diffusion_steps, cfg_weight)
        else:
            restart_from(session, seg_start - 1)
            _generate_until(session, target_frames, diffusion_steps, cfg_weight)
        seg_start = target_frames

    if session.frame_count > total_frames:
        delete_motion_frames(session, total_frames, session.frame_count)

    return motion_to_bvh_text(session)


def generate_bvh_from_prompt(
    session: MotionSession,
    prompt: str,
    frames: int,
    diffusion_steps: Optional[int] = None,
    cfg_weight: Optional[tuple[float, float]] = None,
) -> str:
    """Generate motion for a single prompt and return BVH text."""
    return generate_bvh_from_segments(
        session,
        [(prompt, frames)],
        diffusion_steps=diffusion_steps,
        cfg_weight=cfg_weight,
    )
