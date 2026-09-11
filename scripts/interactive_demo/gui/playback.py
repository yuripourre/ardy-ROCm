# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: Playback tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403


class GuiPlaybackMixin:
    def _build_playback_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("Playback", viser.Icon.PLAYER_PLAY):
            g.gui_play_pause_button = client.gui.add_button("Play", disabled=False)
            g.gui_first_frame_button = client.gui.add_button(
                "First Frame",
                disabled=True,
                icon=viser.Icon.PLAYER_SKIP_BACK,
            )
            g.gui_prev_frame_button = client.gui.add_button(
                "Prev Frame", disabled=True, icon=viser.Icon.PLAYER_TRACK_PREV_FILLED
            )
            g.gui_next_frame_button = client.gui.add_button(
                "Next Frame", disabled=False, icon=viser.Icon.PLAYER_TRACK_NEXT_FILLED
            )
            g.gui_last_frame_button = client.gui.add_button(
                "Last Frame",
                disabled=True,
                icon=viser.Icon.PLAYER_SKIP_FORWARD,
            )
            g.gui_actual_fps = client.gui.add_number("Actual FPS", initial_value=30.0, step=0.0001, disabled=True)
            g.gui_current_time = client.gui.add_number("Current Time (s)", initial_value=0.0, step=0.01, disabled=True)
            g.gui_model_fps = client.gui.add_number("Native FPS", initial_value=30, disabled=True)
            g.gui_frame_idx_input = client.gui.add_number("Frame Index", initial_value=0, min=0, max=199, step=1)
            g.gui_enable_auto_replan_checkbox = client.gui.add_checkbox("Enable Auto Replan", initial_value=True)

            @g.gui_frame_idx_input.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                target_frame = int(g.gui_frame_idx_input.value)
                # Clamp to valid range
                target_frame = max(0, min(target_frame, session.max_frame_idx))
                if target_frame != session.frame_idx:
                    self.set_frame(client_id, target_frame)

            @g.gui_play_pause_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                self._set_playing(session, not session.playing)

            @g.gui_next_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                self.step_frame(client_id, 1)

            @g.gui_prev_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                self.step_frame(client_id, -1)

            @g.gui_first_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.max_frame_idx < 0:
                    return
                self._set_playing(session, False)
                self.set_frame(client_id, 0)

            @g.gui_last_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.max_frame_idx < 0:
                    return
                self._set_playing(session, False)
                self.set_frame(client_id, self._clamp_playhead_frame(session, session.max_frame_idx))

        #
        # Text tab
        #
