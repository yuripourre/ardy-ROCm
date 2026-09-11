# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from ardy.motion_resample import (
    append_cycle_blend_frames,
    resample_local_motion,
    resolve_animation_end_frame,
    resolve_next_prompt_start_from_spans,
    resolve_prompt_source_span,
    resolve_resized_clip_length,
    resolve_playhead_after_delete,
    resolve_restart_from_now_generate_start,
    resolve_restart_from_now_keep_end,
    resolve_restart_prompt_text,
    should_skip_generation_at_limit,
)

from .common import *  # noqa: F401,F403
from .window_budget import compute_window_num_frames


class GenerationMixin:
    def _apply_generation_seed(self, session: ClientSession) -> int:
        """Apply fixed or random seed before a new generation run."""
        gui = session.gui_elements
        if gui.gui_randomize_seed_checkbox.value:
            seed = random.randint(0, 2**31 - 1)
            gui.gui_seed.value = seed
        else:
            seed = int(gui.gui_seed.value)
        seed_everything(seed)
        return seed

    def _is_loop_cycle_active(self, session: ClientSession) -> bool:
        """True when Loop Cycle is enabled."""
        return session.gui_elements.gui_loop_cycle_checkbox.value

    def sync_loop_gui_controls(self, client_id: int) -> None:
        """Keep Loop folder controls in sync with session motion state."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        gui = session.gui_elements
        has_loop = gui.gui_loop_cycle_checkbox.value
        gui.gui_loop_blend_frames.disabled = not has_loop
        gui.gui_model_loop_transition_checkbox.disabled = not has_loop
        has_motion = session.max_frame_idx >= 0
        gui.gui_apply_loop_button.disabled = not (has_loop and has_motion)

    def _resolve_loop_content_frames(self, session: ClientSession) -> Optional[int]:
        """Return N content frames for loop closing (Animation Frames cap or full clip)."""
        if session.motion_tensor is None:
            return None
        target_end = self._resolve_animation_end_frame(session)
        if target_end is not None:
            return target_end + 1
        return session.motion_tensor.shape[1]

    def _resolve_loop_blend_frames(self, session: ClientSession) -> int:
        """Return K blend frames when Loop Cycle is on, else 0."""
        if not self._is_loop_cycle_active(session):
            return 0
        return max(MIN_LOOP_BLEND_FRAMES, int(session.gui_elements.gui_loop_blend_frames.value))

    def _resolve_animation_end_frame(self, session: ClientSession) -> Optional[int]:
        """Return the last allowed frame index when Animation Frames is set, else None."""
        return resolve_animation_end_frame(
            session.animation_limit_base_frame,
            int(session.gui_elements.gui_constraint_num_frames.value),
            session.target_animation_end_frame,
        )

    def _resolve_effective_end_frame(self, session: ClientSession) -> Optional[int]:
        """Timeline/playback end frame, including loop blend extension when applied."""
        if self._is_loop_cycle_active(session) and session.loop_content_frame_count is not None:
            return session.max_frame_idx
        target_end = self._resolve_animation_end_frame(session)
        if target_end is not None:
            return target_end
        return None

    def _resample_session_motion_to_length(self, session: ClientSession, target_frames: int) -> None:
        """Time-warp the full session clip to ``target_frames`` (SLERP + linear root)."""
        if session.motion_tensor is None or session.motion_rep is None:
            return
        source_frames = session.motion_tensor.shape[1]
        if source_frames == target_frames:
            session.max_frame_idx = target_frames - 1
            session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx
            return

        inverse_out = session.motion_rep.inverse(session.motion_tensor, is_normalized=True)
        local_rot_mats = inverse_out["local_rot_mats"]
        root_positions = inverse_out["root_positions"]
        num_samples = local_rot_mats.shape[0]
        resampled_rots = []
        resampled_roots = []
        for sample_idx in range(num_samples):
            rots, root = resample_local_motion(
                local_rot_mats[sample_idx],
                root_positions[sample_idx],
                target_frames,
            )
            resampled_rots.append(rots)
            resampled_roots.append(root)
        local_rot_mats = torch.stack(resampled_rots, dim=0)
        root_positions = torch.stack(resampled_roots, dim=0)
        lengths = torch.full(
            (num_samples,),
            target_frames,
            device=local_rot_mats.device,
            dtype=torch.long,
        )
        feats_unnorm = session.motion_rep(
            local_rot_mats,
            root_positions,
            to_normalize=False,
            lengths=lengths,
        )
        feats_norm = session.motion_rep.normalize(feats_unnorm)
        decoded = session.motion_rep.inverse(feats_unnorm, is_normalized=False)
        joint_velocities = feats_unnorm[:, :, session.motion_rep.slice_dict["velocities"]]
        joint_velocities = joint_velocities.reshape(
            num_samples,
            target_frames,
            session.motion_rep.skeleton.nbjoints,
            3,
        )
        root_velocities = joint_velocities[:, :, session.motion_rep.skeleton.root_idx, :]

        with session.motion_tensor_lock:
            session.motion_tensor = feats_norm
            session.joints_pos = decoded["posed_joints"]
            session.joints_rot = decoded["global_rot_mats"]
            session.foot_contacts = decoded["foot_contacts"]
            session.root_velocities = root_velocities
            session.max_frame_idx = target_frames - 1
        session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx
        print(f"Resampled generated motion from {source_frames} to {target_frames} frames")

    def _reencode_session_motion(
        self,
        session: ClientSession,
        local_rot_mats: torch.Tensor,
        root_positions: torch.Tensor,
        target_frames: int,
    ) -> None:
        """Encode local pose tensors back into session motion state."""
        num_samples = local_rot_mats.shape[0]
        lengths = torch.full(
            (num_samples,),
            target_frames,
            device=local_rot_mats.device,
            dtype=torch.long,
        )
        feats_unnorm = session.motion_rep(
            local_rot_mats,
            root_positions,
            to_normalize=False,
            lengths=lengths,
        )
        feats_norm = session.motion_rep.normalize(feats_unnorm)
        decoded = session.motion_rep.inverse(feats_unnorm, is_normalized=False)
        joint_velocities = feats_unnorm[:, :, session.motion_rep.slice_dict["velocities"]]
        joint_velocities = joint_velocities.reshape(
            num_samples,
            target_frames,
            session.motion_rep.skeleton.nbjoints,
            3,
        )
        root_velocities = joint_velocities[:, :, session.motion_rep.skeleton.root_idx, :]

        with session.motion_tensor_lock:
            session.motion_tensor = feats_norm
            session.joints_pos = decoded["posed_joints"]
            session.joints_rot = decoded["global_rot_mats"]
            session.foot_contacts = decoded["foot_contacts"]
            session.root_velocities = root_velocities
            session.max_frame_idx = target_frames - 1
        session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx

    def _repeat_local_pose(
        self,
        local_rot_mats: torch.Tensor,
        root_positions: torch.Tensor,
        frame_idx: int,
        frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat a single pose for ``frame_count`` frames."""
        return (
            local_rot_mats[frame_idx].unsqueeze(0).expand(frame_count, -1, -1, -1).clone(),
            root_positions[frame_idx].unsqueeze(0).expand(frame_count, -1).clone(),
        )

    def _resample_session_motion_span(
        self,
        session: ClientSession,
        prompt_id: str,
        old_start: int,
        old_end: int,
        new_start: int,
        new_end: int,
    ) -> bool:
        """Time-warp this prompt's generated motion to the new bar length.

        The visual bar is often longer than the clip (open-ended padding). Source
        duration is the real motion owned by this prompt, not the padded bar.
        Frames after the bar that are not a later prompt are dropped.
        """
        if session.motion_rep is None:
            return False

        new_len = new_end - new_start
        if new_len < 1 or new_start < 0:
            return False

        with session.motion_tensor_lock:
            if session.motion_tensor is None:
                return False
            motion_snapshot = session.motion_tensor.detach().clone()

        source_frames = motion_snapshot.shape[1]
        if source_frames < 1:
            return False

        prompt_spans: dict[str, tuple[int, int]] = {}
        loop_uuid = None
        if session.timeline_data is not None:
            prompt_spans = session.timeline_data.get("prompt_spans", {})
            loop_uuid = session.timeline_data.get("loop_prompt_uuid")
        next_prompt_start = resolve_next_prompt_start_from_spans(
            prompt_spans, prompt_id, old_start, source_frames, loop_uuid
        )
        has_later_prompt = next_prompt_start is not None
        source_start, source_end = resolve_prompt_source_span(
            old_start,
            source_frames,
            next_prompt_start if has_later_prompt else None,
        )
        source_len = source_end - source_start

        inverse_out = session.motion_rep.inverse(motion_snapshot, is_normalized=True)
        local_rot_mats = inverse_out["local_rot_mats"]
        root_positions = inverse_out["root_positions"]
        num_samples = local_rot_mats.shape[0]

        hold_frame = min(max(0, source_start - 1), source_frames - 1)
        if source_len < 1:
            hold_frame = min(max(0, new_start - 1), source_frames - 1)

        resampled_segment_rots = []
        resampled_segment_roots = []
        for sample_idx in range(num_samples):
            if source_len < 1:
                segment_rots, segment_root = self._repeat_local_pose(
                    local_rot_mats[sample_idx],
                    root_positions[sample_idx],
                    hold_frame,
                    new_len,
                )
            elif source_len == new_len:
                segment_rots = local_rot_mats[sample_idx, source_start:source_end]
                segment_root = root_positions[sample_idx, source_start:source_end]
            else:
                segment_rots, segment_root = resample_local_motion(
                    local_rot_mats[sample_idx, source_start:source_end],
                    root_positions[sample_idx, source_start:source_end],
                    new_len,
                )
            resampled_segment_rots.append(segment_rots)
            resampled_segment_roots.append(segment_root)

        segment_rots = torch.stack(resampled_segment_rots, dim=0)
        segment_roots = torch.stack(resampled_segment_roots, dim=0)

        prefix_len = min(new_start, source_start, source_frames)
        prefix_rots = local_rot_mats[:, :prefix_len]
        prefix_roots = root_positions[:, :prefix_len]
        if new_start > prefix_len:
            pad_len = new_start - prefix_len
            pad_hold_frame = max(0, prefix_len - 1) if prefix_len > 0 else hold_frame
            pad_rots = []
            pad_roots = []
            for sample_idx in range(num_samples):
                pad_rot, pad_root = self._repeat_local_pose(
                    local_rot_mats[sample_idx],
                    root_positions[sample_idx],
                    pad_hold_frame,
                    pad_len,
                )
                pad_rots.append(pad_rot)
                pad_roots.append(pad_root)
            prefix_rots = torch.cat([prefix_rots, torch.stack(pad_rots, dim=0)], dim=1)
            prefix_roots = torch.cat([prefix_roots, torch.stack(pad_roots, dim=0)], dim=1)

        if has_later_prompt:
            suffix_rots = local_rot_mats[:, source_end:]
            suffix_roots = root_positions[:, source_end:]
            new_local_rot_mats = torch.cat([prefix_rots, segment_rots, suffix_rots], dim=1)
            new_root_positions = torch.cat([prefix_roots, segment_roots, suffix_roots], dim=1)
            target_frames = new_local_rot_mats.shape[1]
        else:
            new_local_rot_mats = torch.cat([prefix_rots, segment_rots], dim=1)
            new_root_positions = torch.cat([prefix_roots, segment_roots], dim=1)
            target_frames = resolve_resized_clip_length(
                source_frames,
                source_start,
                source_end,
                new_start,
                new_end,
                has_later_prompt=False,
            )
            new_local_rot_mats = new_local_rot_mats[:, :target_frames]
            new_root_positions = new_root_positions[:, :target_frames]

        self._reencode_session_motion(session, new_local_rot_mats, new_root_positions, target_frames)
        print(
            f"Resampled segment [{old_start}:{old_end}) (motion [{source_start}:{source_end})) "
            f"-> [{new_start}:{new_end}), clip length {source_frames} -> {target_frames}"
        )
        return True

    def _on_timeline_prompt_resize(self, client_id: int, prompt_id: str) -> None:
        """Interpolate motion when a timeline prompt region is resized."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client
        if session.timeline_data is None:
            return
        if not hasattr(client, "timeline"):
            return

        with session.replan_lock:
            with session.motion_tensor_lock:
                has_motion = session.motion_tensor is not None
            if not has_motion:
                return

            session.playing = False

            prompt = client.timeline._prompts.get(prompt_id)
            if prompt is None:
                return

            prompt_spans = session.timeline_data.get("prompt_spans", {})
            old_span = prompt_spans.get(prompt_id)
            if old_span is None:
                self._sync_prompt_spans(session, client)
                return

            old_start, old_end = old_span
            new_start = int(prompt.start_frame)
            new_end = int(prompt.end_frame)
            if (old_start, old_end) == (new_start, new_end):
                return

            loop_uuid = session.timeline_data.get("loop_prompt_uuid")
            if not self._resample_session_motion_span(
                session, prompt_id, old_start, old_end, new_start, new_end
            ):
                print(
                    f"Prompt resize ignored for '{prompt_id}': "
                    f"could not resample [{old_start}:{old_end}) -> [{new_start}:{new_end})"
                )
                return

            if prompt_id == loop_uuid:
                session.gui_elements.gui_loop_blend_frames.value = max(
                    MIN_LOOP_BLEND_FRAMES, new_end - new_start
                )
            elif session.loop_content_frame_count is not None:
                blend_frames = self._resolve_loop_blend_frames(session)
                session.loop_content_frame_count = max(session.max_frame_idx + 1 - blend_frames, 1)

            session.timeline_data["user_prompt_layout"] = True
            self._sync_prompt_spans(session, client)

            clip_last_frame = session.max_frame_idx
            session.target_animation_end_frame = clip_last_frame
            gui_frames = int(session.gui_elements.gui_constraint_num_frames.value)
            if gui_frames > 0:
                session.gui_elements.gui_constraint_num_frames.value = clip_last_frame + 1
                session.animation_limit_base_frame = 0

            self._refresh_timeline_display(client_id)
            self.set_frame(client_id, min(session.frame_idx, clip_last_frame))
            self.sync_loop_gui_controls(client_id)

    def _delete_session_motion_frames(
        self,
        session: ClientSession,
        delete_start: int,
        delete_end: int,
    ) -> bool:
        """Remove motion frames ``[delete_start, delete_end)`` and shorten the clip."""
        if delete_end <= delete_start or delete_start < 0:
            return False
        with session.motion_tensor_lock:
            if session.motion_tensor is None:
                return False
            source_frames = session.motion_tensor.shape[1]
            if delete_end > source_frames:
                delete_end = source_frames
            if delete_end <= delete_start:
                return False

            keep_before = session.motion_tensor[:, :delete_start]
            keep_after = session.motion_tensor[:, delete_end:]
            session.motion_tensor = torch.cat([keep_before, keep_after], dim=1)

            if session.joints_pos is not None:
                session.joints_pos = torch.cat(
                    [session.joints_pos[:, :delete_start], session.joints_pos[:, delete_end:]],
                    dim=1,
                )
            if session.joints_rot is not None:
                session.joints_rot = torch.cat(
                    [session.joints_rot[:, :delete_start], session.joints_rot[:, delete_end:]],
                    dim=1,
                )
            if session.foot_contacts is not None:
                session.foot_contacts = torch.cat(
                    [session.foot_contacts[:, :delete_start], session.foot_contacts[:, delete_end:]],
                    dim=1,
                )
            if session.root_velocities is not None:
                session.root_velocities = torch.cat(
                    [
                        session.root_velocities[:, :delete_start],
                        session.root_velocities[:, delete_end:],
                    ],
                    dim=1,
                )

        session.max_frame_idx = session.motion_tensor.shape[1] - 1
        session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx
        return True

    def _on_timeline_prompt_delete(self, client_id: int, prompt_id: str) -> None:
        """Remove a deleted prompt's motion span and repack the clip."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client
        if session.timeline_data is None or not hasattr(client, "timeline"):
            return

        with session.replan_lock:
            with session.motion_tensor_lock:
                has_motion = session.motion_tensor is not None
            if not has_motion:
                return

            prompt_spans = session.timeline_data.get("prompt_spans", {})
            old_span = prompt_spans.get(prompt_id)
            if old_span is None:
                self._sync_prompt_spans(session, client)
                return

            delete_start, delete_end = int(old_span[0]), int(old_span[1])
            if delete_end <= delete_start:
                return

            session.playing = False
            playhead_before = session.frame_idx

            if not self._delete_session_motion_frames(session, delete_start, delete_end):
                return

            self._remove_constraints_in_frame_range(client_id, delete_start, delete_end)

            loop_uuid = session.timeline_data.get("loop_prompt_uuid")
            if prompt_id == loop_uuid:
                session.timeline_data["loop_prompt_uuid"] = None
                session.loop_content_frame_count = None
            elif session.loop_content_frame_count is not None:
                blend_frames = self._resolve_loop_blend_frames(session)
                session.loop_content_frame_count = max(session.max_frame_idx + 1 - blend_frames, 1)

            prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])
            if prompt_id in prompt_uuid_list:
                prompt_uuid_list.remove(prompt_id)
            prompt_spans.pop(prompt_id, None)

            session.timeline_data["user_prompt_layout"] = True
            self._sync_prompt_spans(session, client)

            clip_last_frame = session.max_frame_idx
            if clip_last_frame < 0:
                session.target_animation_end_frame = None
                session.animation_limit_base_frame = 0
            else:
                session.target_animation_end_frame = clip_last_frame
                gui_frames = int(session.gui_elements.gui_constraint_num_frames.value)
                if gui_frames > 0:
                    session.gui_elements.gui_constraint_num_frames.value = clip_last_frame + 1
                    session.animation_limit_base_frame = 0

            new_playhead = resolve_playhead_after_delete(
                playhead_before, delete_start, delete_end
            )
            new_playhead = min(new_playhead, clip_last_frame) if clip_last_frame >= 0 else 0
            self._refresh_timeline_display(client_id)
            self.set_frame(client_id, max(0, new_playhead))
            self.sync_loop_gui_controls(client_id)
            print(
                f"Deleted prompt '{prompt_id}' motion [{delete_start}:{delete_end}), "
                f"clip length -> {clip_last_frame + 1}"
            )

    def _trim_session_motion_to_frame_count(self, session: ClientSession, frame_count: int) -> None:
        """Crop all motion tensors to ``frame_count`` frames."""
        with session.motion_tensor_lock:
            if session.motion_tensor is not None:
                session.motion_tensor = session.motion_tensor[:, :frame_count]
            if session.joints_pos is not None:
                session.joints_pos = session.joints_pos[:, :frame_count]
            if session.joints_rot is not None:
                session.joints_rot = session.joints_rot[:, :frame_count]
            if session.foot_contacts is not None:
                session.foot_contacts = session.foot_contacts[:, :frame_count]
            if session.root_velocities is not None:
                session.root_velocities = session.root_velocities[:, :frame_count]
        session.max_frame_idx = frame_count - 1
        session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx

    def _loop_cycle_content_frames(self, session: ClientSession) -> Optional[int]:
        """Return N content frame count when loop closing can run, else None."""
        if not self._is_loop_cycle_active(session):
            return None
        if session.motion_tensor is None or session.motion_rep is None:
            return None
        content_frames = self._resolve_loop_content_frames(session)
        if content_frames is None:
            return None
        if session.motion_tensor.shape[1] != content_frames:
            return None
        return content_frames

    def _commit_loop_cycle_motion(self, session: ClientSession, motion_tensor: torch.Tensor, output_frames: int) -> None:
        """Re-decode motion tensor and update session state after loop closing."""
        num_samples = motion_tensor.shape[0]
        feats_unnorm = session.motion_rep.unnormalize(motion_tensor)
        decoded = session.motion_rep.inverse(feats_unnorm, is_normalized=False)
        joint_velocities = feats_unnorm[:, :, session.motion_rep.slice_dict["velocities"]]
        joint_velocities = joint_velocities.reshape(
            num_samples,
            output_frames,
            session.motion_rep.skeleton.nbjoints,
            3,
        )
        root_velocities = joint_velocities[:, :, session.motion_rep.skeleton.root_idx, :]

        with session.motion_tensor_lock:
            session.motion_tensor = motion_tensor
            session.joints_pos = decoded["posed_joints"]
            session.joints_rot = decoded["global_rot_mats"]
            session.foot_contacts = decoded["foot_contacts"]
            session.root_velocities = root_velocities
            session.max_frame_idx = output_frames - 1
        session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx

    def _append_loop_cycle_blend(self, session: ClientSession) -> None:
        """Append loop-closing frames using SLERP or model generation."""
        if session.gui_elements.gui_model_loop_transition_checkbox.value:
            self._generate_loop_transition_model(session)
        else:
            self._append_loop_cycle_blend_slerp(session)

    def _append_loop_cycle_blend_slerp(self, session: ClientSession) -> None:
        """Append K SLERP blend frames from last toward first, drop duplicate, re-encode motion."""
        content_frames = self._loop_cycle_content_frames(session)
        if content_frames is None:
            return

        blend_frames = self._resolve_loop_blend_frames(session)
        if blend_frames < 1:
            return

        inverse_out = session.motion_rep.inverse(session.motion_tensor, is_normalized=True)
        local_rot_mats = inverse_out["local_rot_mats"]
        root_positions = inverse_out["root_positions"]
        num_samples = local_rot_mats.shape[0]

        blended_rots = []
        blended_roots = []
        for sample_idx in range(num_samples):
            rots, root = append_cycle_blend_frames(
                local_rot_mats[sample_idx],
                root_positions[sample_idx],
                blend_frames,
            )
            blended_rots.append(rots)
            blended_roots.append(root)
        local_rot_mats = torch.stack(blended_rots, dim=0)
        root_positions = torch.stack(blended_roots, dim=0)
        output_frames = content_frames + blend_frames
        lengths = torch.full(
            (num_samples,),
            output_frames,
            device=local_rot_mats.device,
            dtype=torch.long,
        )
        feats_unnorm = session.motion_rep(
            local_rot_mats,
            root_positions,
            to_normalize=False,
            lengths=lengths,
        )
        feats_norm = session.motion_rep.normalize(feats_unnorm)
        self._commit_loop_cycle_motion(session, feats_norm, output_frames)
        print(
            f"Loop cycle: appended {blend_frames} SLERP closing frames, "
            f"clip is {output_frames} frames."
        )

    def _generate_loop_transition_model(self, session: ClientSession) -> None:
        """Generate K closing frames with the model, constrained to end at frame 0."""
        content_frames = self._loop_cycle_content_frames(session)
        if content_frames is None:
            return
        if session.model is None or session.text_embedding is None:
            return

        blend_frames = self._resolve_loop_blend_frames(session)
        if blend_frames < 1:
            return
        if blend_frames > session.gen_horizon_len:
            print(
                f"Loop cycle: blend frames {blend_frames} exceeds gen_horizon_len "
                f"{session.gen_horizon_len}, clamping."
            )
            blend_frames = session.gen_horizon_len

        history_start_idx = 0
        history_motion_tensor = session.motion_tensor[:, :content_frames]
        token = session.num_frames_per_token
        padded_history_len = history_motion_tensor.shape[1]
        if padded_history_len % token != 0:
            pad_frames = token - (padded_history_len % token)
            last_frame = history_motion_tensor[:, -1:, :].expand(-1, pad_frames, -1)
            history_motion_tensor = torch.cat([history_motion_tensor, last_frame], dim=1)
            padded_history_len = history_motion_tensor.shape[1]

        history_length = padded_history_len
        close_frame_idx = padded_history_len + blend_frames

        num_samples = session.gui_elements.gui_num_samples.value
        text_feat = session.text_embedding.repeat(num_samples, 1, 1)
        text_pad_mask = torch.ones(text_feat.shape[0], text_feat.shape[1], device=self.device, dtype=torch.bool)

        joints_pos_0 = session.joints_pos[0, 0].detach().cpu()
        joints_rot_0 = session.joints_rot[0, 0].detach().cpu()
        model_constraints = [
            FullBodyConstraintSet(
                session.motion_rep.skeleton,
                torch.tensor([close_frame_idx]),
                joints_pos_0.unsqueeze(0),
                joints_rot_0.unsqueeze(0),
            )
        ]

        num_frames = compute_window_num_frames(
            history_length=history_length,
            gen_horizon_len=session.gen_horizon_len,
            num_frames_per_token=session.num_frames_per_token,
            max_window_len=session.max_window_len,
            history_start_idx=history_start_idx,
            max_constraint_idx=close_frame_idx,
            future_crop_length=session.gui_elements.gui_future_crop_length.value,
        )

        observed_motion, motion_mask = session.motion_rep.create_conditions_from_constraints(
            model_constraints,
            length=num_frames,
            to_normalize=False,
            device=self.device,
        )
        observed_motion = session.motion_rep.normalize(observed_motion)
        observed_motion = observed_motion * motion_mask
        observed_motion = repeat(observed_motion, "t d -> b t d", b=num_samples)
        motion_mask = repeat(motion_mask, "t d -> b t d", b=num_samples)
        motion_mask = motion_mask[:, history_start_idx:]
        observed_motion = observed_motion[:, history_start_idx:]
        motion_mask[:, :history_length] = 0.0
        observed_motion[:, :history_length] = 0.0

        samples = session.model.autoregressive_step(
            num_frames=num_frames,
            num_denoising_steps=session.gui_elements.gui_diffusion_steps_slider.value,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=(
                session.gui_elements.gui_cfg_text_weight.value,
                session.gui_elements.gui_cfg_constraint_weight.value,
            ),
            texts=None,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            init_history_sequence=history_motion_tensor,
            init_global_translation=None,
            init_first_heading_angle=None,
        )

        closing_motion = samples[:, padded_history_len : padded_history_len + blend_frames]
        combined_motion = torch.cat([session.motion_tensor[:, :content_frames], closing_motion], dim=1)
        output_frames = content_frames + blend_frames
        self._commit_loop_cycle_motion(session, combined_motion, output_frames)
        print(
            f"Loop cycle: model generated {blend_frames} closing frames, "
            f"clip is {output_frames} frames."
        )

    def apply_loop_cycle(self, client_id: int) -> None:
        """Close the current clip by appending loop transition frames (manual trigger)."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        if not self._is_loop_cycle_active(session):
            return
        if session.model is None or session.motion_tensor is None or session.motion_rep is None:
            return

        if session.loop_content_frame_count is not None:
            self._trim_session_motion_to_frame_count(session, session.loop_content_frame_count)

        self._remove_loop_prompt_region(session, session.client)

        content_frames = self._resolve_loop_content_frames(session)
        if content_frames is None or content_frames < 1:
            return

        if session.motion_tensor.shape[1] != content_frames:
            self._resample_session_motion_to_length(session, content_frames)

        self._append_loop_cycle_blend(session)
        session.loop_content_frame_count = content_frames

        effective_end = self._resolve_effective_end_frame(session)
        frame_idx = session.frame_idx
        if effective_end is not None:
            frame_idx = min(frame_idx, effective_end)
        self._refresh_timeline_display(client_id)
        self.set_frame(client_id, frame_idx)
        self.sync_loop_gui_controls(client_id)

    def _apply_animation_frame_limit(self, client_id: int, session: ClientSession) -> None:
        """Clamp generated motion to the Animation Frames length and refresh the timeline."""
        target_end = self._resolve_animation_end_frame(session)
        if target_end is None:
            return
        target_frames = target_end + 1
        if session.motion_tensor is not None and session.motion_tensor.shape[1] != target_frames:
            base_frame = session.animation_limit_base_frame
            if base_frame > 0:
                source_frames = session.motion_tensor.shape[1]
                self._resample_session_motion_span(
                    session,
                    "",
                    base_frame,
                    source_frames,
                    base_frame,
                    target_frames,
                )
            else:
                self._resample_session_motion_to_length(session, target_frames)
            session.loop_content_frame_count = None
        effective_end = self._resolve_effective_end_frame(session)
        if effective_end is not None and session.frame_idx > effective_end:
            self.set_frame(client_id, effective_end)
        else:
            self._refresh_timeline_display(client_id)
        self.sync_loop_gui_controls(client_id)

    def restart(self, client_id: int, clear_animation_limit: Optional[bool] = None):
        """Restart the demo for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        self._apply_generation_seed(session)

        playing = session.playing
        session.playing = False
        self.clear_motions(client_id)
        session.max_frame_idx = -1
        session.frame_idx = 0
        session.loop_content_frame_count = None
        session.animation_limit_base_frame = 0
        if clear_animation_limit is None:
            clear_animation_limit = int(session.gui_elements.gui_constraint_num_frames.value) <= 0
        if clear_animation_limit:
            session.target_animation_end_frame = None

        # Reset camera state for smooth transitions
        session.camera_position = None
        session.camera_look_at = None
        session.camera_forward_direction = None
        session.camera_position_buffer.clear()
        session.camera_last_update_frame = -1

        self.clear_timeline_prompts(client_id)
        generate_prompt = self._sync_generate_prompt_to_text_tab(session)
        restart_prompt = resolve_restart_prompt_text(
            generate_prompt,
            session.gui_elements.gui_prompt_text.value,
        )
        self.on_text_prompt_update(
            client_id,
            trigger_replan=False,
            initial_prompt=True,
            show_notification=False,
            prompt_text=restart_prompt,
        )
        self._refresh_timeline_display(client_id)

        self._generate_step(client_id)

        session.playing = playing
        self.set_frame(client_id, 0)

    def restart_from_now(self, client_id: int):
        """Restart generation from the current frame, clearing all motions after it."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        if session.model is None:
            print(f"Model not loaded for client {client_id}!")
            return

        current_frame = session.frame_idx
        if current_frame <= 0:
            # Regenerating from the first frame is a full restart (no leftover 1-frame region).
            self.restart(client_id)
            return

        self._apply_generation_seed(session)

        playing = session.playing
        session.playing = False

        with session.replan_lock:
            # Keep frames through the playhead; regenerate starting at the next frame.
            keep_end = resolve_restart_from_now_keep_end(current_frame)
            generate_start = resolve_restart_from_now_generate_start(current_frame)
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

            session.max_frame_idx = current_frame
            session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx

            if session.loop_content_frame_count is not None:
                session.loop_content_frame_count = None

            session.target_animation_end_frame = None
            if session.timeline_data is not None:
                session.timeline_data["user_prompt_layout"] = False

            self._remove_constraints_from_frame(client_id, generate_start)

            if session.timeline_data is not None:
                self._remove_loop_prompt_region(session, session.client)

            generate_prompt = self._sync_generate_prompt_to_text_tab(session)
            restart_prompt = resolve_restart_prompt_text(
                generate_prompt,
                session.gui_elements.gui_prompt_text.value,
            )
            self.on_text_prompt_update(
                client_id,
                trigger_replan=False,
                show_notification=False,
                prompt_text=restart_prompt,
            )

            print(
                f"[Restart From Now] Cleared motion after frame {current_frame}, "
                f"generating from frame {generate_start}"
            )

            session.skip_animation_frame_limit = True
            try:
                self._generate_step(client_id)
            finally:
                session.skip_animation_frame_limit = False

            gui_frames = int(session.gui_elements.gui_constraint_num_frames.value)
            if gui_frames > 0 and session.motion_tensor is not None:
                source_end = session.motion_tensor.shape[1]
                self._resample_session_motion_span(
                    session,
                    "",
                    generate_start,
                    source_end,
                    generate_start,
                    generate_start + gui_frames,
                )
                session.animation_limit_base_frame = generate_start
                session.target_animation_end_frame = generate_start + gui_frames - 1

            self._refresh_timeline_display(client_id)
            self.set_frame(client_id, current_frame)

        session.playing = playing

    def on_replan_trigger(self, client_id: int, skip_if_busy: bool = False):
        """Called when approaching end of timeline or when prompt changes.

        With skip_if_busy=True the trigger is dropped when a replan is already running, instead of
        queuing behind it. Used by the per-frame auto-replan check, which would otherwise pile up
        redundant generations (it re-fires on the next frame anyway).
        """
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        if skip_if_busy:
            if not session.replan_lock.acquire(blocking=False):
                return
            try:
                self._generate_step(client_id)
            finally:
                session.replan_lock.release()
        else:
            with session.replan_lock:
                self._generate_step(client_id)

    def _get_history_motion(self, session: ClientSession):
        """Get history motion for autoregressive generation."""
        frame_idx = session.frame_idx
        replan_buffer_size = session.gui_elements.gui_replan_buffer_size.value
        history_crop_length = session.gui_elements.gui_history_crop_length.value
        motion_tensor = session.motion_tensor

        cur_motion_len = motion_tensor.shape[1] if motion_tensor is not None else 0
        history_end_idx = min(cur_motion_len - 1, frame_idx + replan_buffer_size)
        if (
            cur_motion_len >= session.num_frames_per_token
        ):  # if there are history frames, ensure history end idx is at least num_frames_per_token - 1
            history_end_idx = max(history_end_idx, session.num_frames_per_token - 1)
        history_length = min(history_end_idx + 1, history_crop_length)
        history_length = history_length // session.num_frames_per_token * session.num_frames_per_token
        history_start_idx = max(0, history_end_idx - history_length + 1)

        history_motion_tensor = None
        if motion_tensor is not None and history_start_idx <= history_end_idx:
            history_motion_tensor = motion_tensor[:, history_start_idx : history_end_idx + 1]

        return history_motion_tensor, history_start_idx, history_end_idx, history_length

    def _generate_step(self, client_id: int):
        """One autoregressive generation step."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        if session.model is None:
            print(f"Model not loaded for client {client_id}!")
            return

        if should_skip_generation_at_limit(
            session.max_frame_idx,
            self._resolve_animation_end_frame(session),
            session.skip_animation_frame_limit,
        ):
            return

        start_time = time.time()

        history_motion_tensor, history_start_idx, history_end_idx, history_length = self._get_history_motion(session)
        print(
            f"Generate with frame idx {session.frame_idx}, history start: {history_start_idx}, end: {history_end_idx}, length: {history_length}"
        )

        num_samples = session.gui_elements.gui_num_samples.value
        text_feat = session.text_embedding.repeat(num_samples, 1, 1)
        text_pad_mask = torch.ones(text_feat.shape[0], text_feat.shape[1], device=self.device, dtype=torch.bool)

        motion_mask = None
        observed_motion = None

        # Check if we have timeline constraints (including waypoints)
        constraint_idx_list = [c.get_constraint_info()["frame_idx"] for c in session.constraints.values()]
        # merge all constraint indices into a single list
        all_constraint_indices = [idx for sublist in constraint_idx_list for idx in sublist]
        has_valid_timeline_constraints = (
            len(all_constraint_indices) > 0 and max(all_constraint_indices) > history_end_idx
        )

        # number of frames of the visible sequence to the model
        num_frames = compute_window_num_frames(
            history_length=history_length,
            gen_horizon_len=session.gen_horizon_len,
            num_frames_per_token=session.num_frames_per_token,
            max_window_len=session.max_window_len,
            history_start_idx=history_start_idx,
            max_constraint_idx=(max(all_constraint_indices) if has_valid_timeline_constraints else None),
            future_crop_length=session.gui_elements.gui_future_crop_length.value,
        )

        # Process timeline constraints
        if has_valid_timeline_constraints:
            motion_mask, observed_motion = self.compute_constraint_mask(
                session,
                num_samples,
                num_frames=num_frames + history_start_idx,
                history_end_idx=history_end_idx,
            )

            if motion_mask is not None and observed_motion is not None:
                motion_mask = motion_mask[:, history_start_idx:]
                observed_motion = observed_motion[:, history_start_idx:]
                motion_mask[:, :history_length] = 0.0  # disable history frames constraints
                observed_motion[:, :history_length] = 0.0

        # if motion_mask is not None and observed_motion is not None:
        #     print(f"motion mask non zero: {(motion_mask != 0.0).sum()}, observed motion non zero: {(observed_motion != 0.0).sum()}")
        #     # motion_mask_by_dim = motion_mask.any(dim=(0, 1))
        #     # print(f"Nonzero dimensions: {motion_mask_by_dim.nonzero()}")
        #     observed_motion_by_dim = observed_motion.any(dim=(0, 1))
        #     print(f"Observed motion nonzero dimensions: {observed_motion_by_dim.nonzero().squeeze()}")

        print(f"Num frames: {num_frames}")

        if history_motion_tensor is None:
            num_samples = session.gui_elements.gui_num_samples.value
            init_global_translation = (
                torch.from_numpy(session.init_global_translation)
                .to(dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(num_samples, 1)
            )
            init_first_heading_angle = (
                torch.ones(num_samples, dtype=torch.float32, device=self.device) * session.init_first_heading_angle
            )
        else:
            init_global_translation = None
            init_first_heading_angle = None

        # Generate motion
        samples = session.model.autoregressive_step(
            num_frames=num_frames,
            num_denoising_steps=session.gui_elements.gui_diffusion_steps_slider.value,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=(
                session.gui_elements.gui_cfg_text_weight.value,
                session.gui_elements.gui_cfg_constraint_weight.value,
            ),
            texts=None,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            init_history_sequence=history_motion_tensor,
            init_global_translation=init_global_translation,
            init_first_heading_angle=init_first_heading_angle,
        )

        # Convert to joints
        samples_unnormalized = session.motion_rep.unnormalize(samples)
        pred_joints_output = session.motion_rep.inverse(
            samples_unnormalized,
            is_normalized=False,
        )

        joints_pos = pred_joints_output["posed_joints"]
        joints_rot = pred_joints_output["global_rot_mats"]
        foot_contacts = pred_joints_output.get("foot_contacts")

        # Apply post-processing if enabled
        if session.gui_elements.gui_enable_postprocess_checkbox.value:
            postprocess_start_time = time.time()

            # Get constraints in the generation horizon
            model_constraints = self.compute_model_constraints_lst(
                session,
                num_frames=session.gen_horizon_len + history_length + history_start_idx,
                history_end_idx=history_end_idx,
            )

            #  check if the model_constraints is not empty
            if len(model_constraints) > 0:
                # Get local rotations and root positions from pred_joints_output
                local_rot_mats = pred_joints_output["local_rot_mats"]  # [B, T, J, 3, 3]
                root_positions = pred_joints_output["root_positions"]  # [B, T, 3]

                # subtract history_end_idx from the model_constraints frame indices
                for constraint in model_constraints:
                    constraint.frame_indices = constraint.frame_indices - history_start_idx - history_length

                # Apply post-processing to generation horizon frames
                corrected_output = post_process_motion(
                    local_rot_mats[:, history_length:],
                    root_positions[:, history_length:],
                    foot_contacts[:, history_length:],
                    session.motion_rep.skeleton,
                    constraint_lst=model_constraints if model_constraints else None,
                    contact_threshold=session.gui_elements.gui_postprocess_contact_threshold.value,
                    root_margin=session.gui_elements.gui_postprocess_root_margin.value,
                )

                # calculate corrected motion_tensor, joints_pos, joints_rot, foot_contacts
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

            postprocess_end_time = time.time()
            print(
                f"[PostProcess] Motion correction applied in {postprocess_end_time - postprocess_start_time:.4f} seconds"
            )

        # Extract root velocities from motion representation
        joint_velocities = samples_unnormalized[:, :, session.motion_rep.slice_dict["velocities"]]
        joint_velocities = joint_velocities.reshape(
            num_samples,
            history_length + session.gen_horizon_len,
            session.motion_rep.skeleton.nbjoints,
            3,
        )
        root_velocities = joint_velocities[:, :, session.motion_rep.skeleton.root_idx, :]

        # Update motion data
        with session.motion_tensor_lock:
            if session.motion_tensor is None:
                session.motion_tensor = samples.clone()
                session.joints_pos = joints_pos.clone()
                session.joints_rot = joints_rot.clone()
                session.foot_contacts = foot_contacts.clone()
                session.root_velocities = root_velocities.clone()
                for i in range(num_samples):
                    self.add_character(client_id, session.motion_rep.skeleton, i)
            else:
                session.motion_tensor = torch.cat(
                    [
                        session.motion_tensor[:, : history_end_idx + 1],
                        samples[:, history_length:],
                    ],
                    dim=1,
                )
                session.joints_pos = torch.cat(
                    [
                        session.joints_pos[:, : history_end_idx + 1],
                        joints_pos[:, history_length:],
                    ],
                    dim=1,
                )
                session.joints_rot = torch.cat(
                    [
                        session.joints_rot[:, : history_end_idx + 1],
                        joints_rot[:, history_length:],
                    ],
                    dim=1,
                )
                session.foot_contacts = torch.cat(
                    [
                        session.foot_contacts[:, : history_end_idx + 1],
                        foot_contacts[:, history_length:],
                    ],
                    dim=1,
                )
                session.root_velocities = torch.cat(
                    [
                        session.root_velocities[:, : history_end_idx + 1],
                        root_velocities[:, history_length:],
                    ],
                    dim=1,
                )

            # Update timeline
            session.max_frame_idx = session.motion_tensor.shape[1] - 1

        if not session.skip_animation_frame_limit:
            self._apply_animation_frame_limit(client_id, session)

        # Update frame index input max value
        session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx

        end_time = time.time()
        print(f"Generate step time: {end_time - start_time} seconds")
