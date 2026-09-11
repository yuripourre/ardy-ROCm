# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from ardy.motion_resample import clamp_playhead_frame, should_pause_playback_at_clip_end

from .common import *  # noqa: F401,F403


class PlaybackMixin:
    def run_client_playback(self, client_id: int):
        """Playback loop for a specific client."""
        print(f"Starting playback loop for client {client_id}")

        elapsed_history = []

        while True:
            # Check if client is still active and should continue
            if not self.client_active(client_id):
                print(f"Client {client_id} no longer active, stopping playback loop")
                break

            session = self.client_sessions[client_id]
            if session.stop_playback:
                print(f"Stop signal received for client {client_id}")
                break

            pending_resize = session.pending_prompt_resize
            if pending_resize is not None:
                session.pending_prompt_resize = None
                self._on_timeline_prompt_resize(client_id, pending_resize)

            last_update_time = time.time()

            # Update frame if playing
            if session.playing:
                playback_end = self._clamp_playhead_frame(session, session.max_frame_idx)
                at_clip_end = session.frame_idx >= playback_end
                effective_end = self._resolve_effective_end_frame(session)
                auto_replan = session.gui_elements.gui_enable_auto_replan_checkbox.value
                if not at_clip_end:
                    session.frame_idx += 1
                    self.set_frame(client_id, session.frame_idx)
                elif should_pause_playback_at_clip_end(
                    at_clip_end=True,
                    animation_end=effective_end,
                    auto_replan=auto_replan,
                ):
                    self._set_playing(session, False)
                else:
                    # Hold last frame; retrigger auto-replan while waiting for more motion.
                    self.set_frame(client_id, session.frame_idx)

            # Sleep to maintain target FPS (using model's native FPS)
            time_remaining = max(0, 1.0 / session.model_fps - (time.time() - last_update_time))
            time.sleep(time_remaining)

            # Track moving average of actual fps
            elapsed = time.time() - last_update_time
            elapsed_history.append(elapsed)
            if len(elapsed_history) > 10:
                elapsed_history.pop(0)

            if self.client_active(client_id):
                session.gui_elements.gui_actual_fps.value = 1.0 / (sum(elapsed_history) / len(elapsed_history))

        print(f"Playback loop ended for client {client_id}")

    def run(self):
        """Main dummy loop to keep server alive."""
        print("Main server loop started")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Server shutting down...")
            # Signal all playback threads to stop
            for session in self.client_sessions.values():
                session.stop_playback = True

    def _sync_playback_step_buttons(self, session: ClientSession) -> None:
        """Enable/disable step and jump buttons from playhead position."""
        gui = session.gui_elements
        at_start = session.frame_idx <= 0
        playback_end = self._clamp_playhead_frame(session, session.max_frame_idx)
        at_end = session.max_frame_idx < 0 or session.frame_idx >= playback_end
        if session.playing:
            gui.gui_next_frame_button.disabled = True
            gui.gui_prev_frame_button.disabled = True
            gui.gui_first_frame_button.disabled = True
            gui.gui_last_frame_button.disabled = True
            return
        gui.gui_prev_frame_button.disabled = at_start
        gui.gui_first_frame_button.disabled = at_start
        gui.gui_next_frame_button.disabled = at_end
        gui.gui_last_frame_button.disabled = at_end

    def _set_playing(self, session: ClientSession, playing: bool) -> None:
        """Set playback state and keep Play/Pause plus step buttons in sync."""
        session.playing = playing
        session.gui_elements.gui_play_pause_button.label = "Pause" if playing else "Play"
        self._sync_playback_step_buttons(session)

    def _timeline_frame_range(self, session: ClientSession, frame_idx: int) -> tuple[int, int]:
        """Visible timeline span, always including frame 0 so the playhead can scrub backward."""
        target_end = self._resolve_effective_end_frame(session)
        clip_end = max(session.max_frame_idx, frame_idx, 0)
        if target_end is not None:
            return 0, max(target_end, clip_end)
        return 0, clip_end + TIMELINE_WINDOW_AFTER

    def _clamp_playhead_frame(self, session: ClientSession, frame_idx: int) -> int:
        """Keep the playhead on a generated frame inside the active clip/bar."""
        return clamp_playhead_frame(
            frame_idx,
            session.max_frame_idx,
            self._resolve_effective_end_frame(session),
        )

    def _apply_timeline_window(
        self,
        session: ClientSession,
        client: viser.ClientHandle,
        frame_idx: int,
        *,
        set_playhead: bool,
    ) -> None:
        """Update the visible timeline span only when it actually changes."""
        if not hasattr(client, "timeline"):
            return
        window_start, window_end = self._timeline_frame_range(session, frame_idx)
        last_range = None
        if session.timeline_data is not None:
            last_range = session.timeline_data.get("visible_frame_range")
        if last_range != (window_start, window_end):
            client.timeline.set_frame_range(start_frame=window_start, end_frame=window_end)
            if session.timeline_data is not None:
                session.timeline_data["visible_frame_range"] = (window_start, window_end)
        if set_playhead:
            client.timeline.set_current_frame(frame_idx)

    def _resolve_content_end_frame(self, session: ClientSession) -> Optional[int]:
        """Last frame index of original content (excludes loop transition segment)."""
        if session.loop_content_frame_count is not None:
            return session.loop_content_frame_count - 1
        return self._resolve_animation_end_frame(session)

    def _sync_prompt_spans(self, session: ClientSession, client: viser.ClientHandle) -> None:
        """Record each timeline prompt's (start, end) for resize/move handling."""
        if session.timeline_data is None or not hasattr(client, "timeline"):
            return
        spans = session.timeline_data.setdefault("prompt_spans", {})
        for prompt_uuid in session.timeline_data.get("prompt_uuid_list", []):
            prompt = client.timeline._prompts.get(prompt_uuid)
            if prompt is not None:
                spans[prompt_uuid] = (int(prompt.start_frame), int(prompt.end_frame))
        loop_uuid = session.timeline_data.get("loop_prompt_uuid")
        if loop_uuid is not None:
            loop_prompt = client.timeline._prompts.get(loop_uuid)
            if loop_prompt is not None:
                spans[loop_uuid] = (int(loop_prompt.start_frame), int(loop_prompt.end_frame))

    def _prompt_end_frame(self, session: ClientSession) -> int:
        """Prompt bar end for content prompts: content boundary, animation cap, or visible window."""
        content_end = self._resolve_content_end_frame(session)
        if content_end is not None:
            return content_end
        # Unbounded clips use the visible window, not INFINITE_FRAME_IDX (99999 frames ≈ 5000s at 20fps).
        return max(session.max_frame_idx, 0) + TIMELINE_WINDOW_AFTER

    def _loop_transition_prompt_text(self, session: ClientSession) -> str:
        if session.gui_elements.gui_model_loop_transition_checkbox.value:
            return "Loop transition (model)"
        return "Loop transition"

    def _remove_loop_prompt_region(self, session: ClientSession, client: viser.ClientHandle) -> None:
        """Remove the loop transition prompt segment from the timeline."""
        if session.timeline_data is None:
            return
        loop_uuid = session.timeline_data.get("loop_prompt_uuid")
        if loop_uuid is None:
            return
        try:
            client.timeline.remove_prompt(loop_uuid)
        except (AttributeError, Exception) as e:
            print(f"Could not remove loop prompt: {e}")
        session.timeline_data["loop_prompt_uuid"] = None

    def _sync_loop_prompt_region(self, session: ClientSession, client: viser.ClientHandle) -> None:
        """Create or update a separate prompt bar for loop transition frames N..N+K-1."""
        if session.timeline_data is None:
            return

        content_frames = session.loop_content_frame_count
        if content_frames is None:
            self._remove_loop_prompt_region(session, client)
            return

        blend_frames = self._resolve_loop_blend_frames(session)
        if blend_frames < 1:
            self._remove_loop_prompt_region(session, client)
            return

        loop_start = content_frames
        loop_end = content_frames + blend_frames - 1
        prompt_text = self._loop_transition_prompt_text(session)

        self._remove_loop_prompt_region(session, client)
        try:
            loop_uuid = client.timeline.add_prompt(
                text=prompt_text,
                start_frame=loop_start,
                end_frame=loop_end,
                color=LOOP_TRANSITION_PROMPT_COLOR,
            )
            session.timeline_data["loop_prompt_uuid"] = loop_uuid
            self._sync_prompt_spans(session, client)
        except (AttributeError, Exception) as e:
            print(f"Could not sync loop prompt region: {e}")

    def _refresh_timeline_display(self, client_id: int) -> None:
        """Update timeline frame range and prompt spans after clip length changes."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client
        if not hasattr(client, "timeline"):
            return

        prompt_end = self._prompt_end_frame(session)
        if session.timeline_data is not None:
            prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])
            user_layout = session.timeline_data.get("user_prompt_layout", False)
            if prompt_uuid_list and not user_layout:
                try:
                    if session.loop_content_frame_count is not None:
                        for prompt_uuid in prompt_uuid_list:
                            prompt = client.timeline._prompts.get(prompt_uuid)
                            if prompt is None:
                                continue
                            if prompt.start_frame <= prompt_end and prompt.end_frame > prompt_end:
                                client.timeline.update_prompt(prompt_uuid, end_frame=prompt_end)
                    else:
                        client.timeline.update_prompt(prompt_uuid_list[-1], end_frame=prompt_end)
                except (AttributeError, Exception) as e:
                    print(f"Could not update timeline prompt end frame: {e}")
            self._sync_prompt_spans(session, client)
            self._sync_loop_prompt_region(session, client)

        frame_idx = session.frame_idx
        try:
            self._apply_timeline_window(session, client, frame_idx, set_playhead=True)
            client.flush()
        except (AttributeError, Exception) as e:
            print(f"Could not refresh timeline display: {e}")

    def step_frame(self, client_id: int, delta: int) -> None:
        """Step the playhead by ``delta`` frames, clamped to the motion range."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        if session.max_frame_idx < 0:
            return
        playback_end = self._clamp_playhead_frame(session, session.max_frame_idx)
        new_frame = max(0, min(session.frame_idx + delta, playback_end))
        if new_frame == session.frame_idx:
            return
        self._set_playing(session, False)
        self.set_frame(client_id, new_frame)

    def go_first_frame(self, client_id: int) -> None:
        """Jump to frame 0 (same as the First Frame button)."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        if session.max_frame_idx < 0:
            return
        self._set_playing(session, False)
        self.set_frame(client_id, 0)

    def go_last_frame(self, client_id: int) -> None:
        """Jump to the last playable frame (same as the Last Frame button)."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        if session.max_frame_idx < 0:
            return
        self._set_playing(session, False)
        self.set_frame(client_id, self._clamp_playhead_frame(session, session.max_frame_idx))

    def set_frame(self, client_id: int, frame_idx: int, trigger_by_gui_timeline: bool = False):
        """Set the current frame for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        frame_idx = self._clamp_playhead_frame(session, frame_idx)
        session.frame_idx = frame_idx
        if hasattr(client, "timeline"):
            try:
                self._apply_timeline_window(
                    session,
                    client,
                    frame_idx,
                    set_playhead=not trigger_by_gui_timeline,
                )
            except (AttributeError, Exception) as e:
                print(f"Could not update timeline frame: {e}")

        session.cur_time = frame_idx / session.model_fps
        gui = session.gui_elements
        if gui.gui_current_time.value != session.cur_time:
            gui.gui_current_time.value = session.cur_time
        if gui.gui_frame_idx_input.value != frame_idx:
            gui.gui_frame_idx_input.value = frame_idx
        self._sync_playback_step_buttons(session)

        # Check if approaching end of timeline
        thresh = session.gui_elements.gui_replan_trigger_thresh.value
        enable_auto_replan = session.gui_elements.gui_enable_auto_replan_checkbox.value
        target_end = self._resolve_animation_end_frame(session)
        at_animation_limit = target_end is not None and session.max_frame_idx >= target_end
        if (
            not trigger_by_gui_timeline
            and enable_auto_replan
            and not at_animation_limit
            and session.max_frame_idx - frame_idx <= thresh
        ):
            # Cheap pre-check to avoid spawning a thread while a replan runs;
            # skip_if_busy makes the trigger drop atomically if another thread
            # won the race between this check and the lock acquisition.
            if not session.replan_lock.locked():
                threading.Thread(
                    target=self.on_replan_trigger,
                    args=(client_id,),
                    kwargs={"skip_if_busy": True},
                    daemon=True,
                ).start()

        # Update constraint visibility - show constraints that have been reached (frame_idx <= current frame)
        show_hand_orientations = session.gui_elements.gui_viz_hand_orientations_checkbox.value
        hide_distant_constraints = session.gui_elements.gui_viz_hide_distant_constraints_checkbox.value

        # Calculate max visible future frame if hiding distant constraints
        max_future_frame = float("inf")
        if hide_distant_constraints:
            future_crop = session.gui_elements.gui_future_crop_length.value
            gen_horizon = session.gen_horizon_len
            max_future_frame = frame_idx + future_crop + gen_horizon

        for track_name, constraint in session.constraints.items():
            for constraint_frame_idx in constraint.scene_elements.keys():
                # Basic visibility: constraint is in the future (not yet reached)
                visibility = constraint_frame_idx >= frame_idx

                # Additionally hide if too far in the future
                if hide_distant_constraints and constraint_frame_idx > max_future_frame:
                    visibility = False

                # Pass show_rotation_axes parameter for End-Effectors constraints
                if track_name == "End-Effectors":
                    constraint.set_keyframe_visibility(
                        constraint_frame_idx,
                        visibility,
                        show_rotation_axes=show_hand_orientations,
                    )
                else:
                    constraint.set_keyframe_visibility(constraint_frame_idx, visibility)
            # Update interval labels visibility once per constraint track
            if track_name == "2D Root":
                constraint.set_interval_labels_visibility(frame_idx)

        # Update character poses and get root position for target velocity arrow
        # Note: Each character has its own actual velocity arrow (blue) shown on their skeleton
        # We also have ONE shared target velocity arrow (orange) for the entire session
        root_pos_for_target = None
        with session.characters_lock:
            with session.motion_tensor_lock:
                max_frame_idx = session.max_frame_idx
                joints_pos = session.joints_pos
                joints_rot = session.joints_rot
                root_velocities = session.root_velocities
                foot_contacts = session.foot_contacts
                if frame_idx >= 0 and frame_idx <= max_frame_idx and joints_pos is not None and joints_rot is not None:
                    for character_idx, character in enumerate(session.characters.values()):
                        root_velocity = None
                        if root_velocities is not None:
                            root_velocity = root_velocities[character_idx, frame_idx]

                        frame_foot_contacts = (
                            foot_contacts[character_idx, frame_idx] > 0.5
                            if foot_contacts is not None
                            else None
                        )
                        character.set_pose(
                            joints_pos[character_idx, frame_idx],
                            joints_rot[character_idx, frame_idx],
                            foot_contacts=frame_foot_contacts,
                            root_velocity=root_velocity,
                        )

                        if character_idx == 0 and root_pos_for_target is None:
                            root_pos_for_target = character.skeleton_mesh.cur_joints_pos[character.skeleton.root_idx]

        # Update reference motion character
        if (
            session.ref_character is not None
            and session.ref_joints_pos is not None
            and session.gui_elements.gui_viz_ref_motion_checkbox.value
        ):
            ref_frame = min(frame_idx, session.ref_joints_pos.shape[0] - 1)
            if ref_frame >= 0:
                session.ref_character.set_pose(
                    session.ref_joints_pos[ref_frame],
                    session.ref_joints_rot[ref_frame],
                )

        # Update hand gizmos if enabled
        if session.gui_elements.gui_viz_hand_orientations_checkbox.value:
            if not session.hand_gizmos:
                self.create_hand_gizmos(client_id)
            self.update_hand_gizmos(client_id, frame_idx)
        elif session.hand_gizmos:
            # Hide hand gizmos if checkbox is off
            self.set_hand_gizmos_visibility(client_id, False)

        # Update target velocity arrow visualization (orange, user-specified target)
        if session.target_velocity_arrow is not None:
            gui_elements = session.gui_elements
            # Show target velocity arrow if enabled and we have a valid root position
            if gui_elements.gui_use_target_velocity_checkbox.value and root_pos_for_target is not None:
                # Get target velocity from GUI (x, z)
                target_vel_xz = gui_elements.gui_target_root_velocity.value
                target_velocity = np.array([target_vel_xz[0], 0.0, target_vel_xz[1]])

                # Update target velocity arrow
                session.target_velocity_arrow.update(
                    root_velocity=target_velocity,
                    root_pos=root_pos_for_target,
                    visible=True,
                )

                # Predict future root positions and update 2D root constraints
                if frame_idx % TARGET_VELOCITY_UPDATE_INTERVAL == 0:
                    self._update_root_constraints_from_target_velocity(client_id, frame_idx, target_velocity)
            else:
                # Hide target velocity arrow if disabled or no character available
                session.target_velocity_arrow.set_visibility(False)

        # Update camera
        if session.gui_elements.gui_viz_auto_camera_checkbox.value:
            self.update_camera_follow(client_id, frame_idx)
