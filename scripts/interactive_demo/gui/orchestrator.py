# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI orchestrator (per-tab builders live in gui_*.py)."""

from types import SimpleNamespace

from ..common import *  # noqa: F401,F403
from .instructions import INSTRUCTIONS_TAB_MD


class GuiMixin:
    def create_gui(self, client: viser.ClientHandle, constraint_tracks: dict):
        """Create GUI elements for a client."""
        g = SimpleNamespace()
        client_id = client.client_id

        # Setup timeline (check if available)
        timeline = None
        timeline_available = hasattr(client, "timeline")

        default_prompt = DEFAULT_PROMPT
        uuid = None

        if timeline_available:
            try:
                print("Setting up timeline")
                timeline = client.timeline
                uuid = timeline.add_prompt(
                    text=default_prompt,
                    start_frame=0,
                    end_frame=TIMELINE_WINDOW_AFTER,
                    color=self.get_prompt_color(0),
                )
                timeline.set_visible(True)
                timeline.set_current_frame(0)
                # Set initial window: 20 frames before (0 in this case) + 200 frames after
                timeline.set_frame_range(start_frame=0, end_frame=TIMELINE_WINDOW_AFTER)
            except Exception as e:
                print(f"Warning: Could not setup timeline: {e}")
                timeline_available = False
        else:
            print("=" * 60)
            print("⚠️  TIMELINE FEATURE NOT AVAILABLE")
            print("=" * 60)
            print("Your viser version doesn't support timeline features.")
            print("The demo will work with waypoints and keyframes only.")
            print("To enable timeline: install viser from source with timeline support")
            print("=" * 60)

        # Create timeline tracks (only if timeline is available)
        timeline_tracks = {}
        if timeline_available and timeline is not None:
            fullbody_id = timeline.add_track(
                "Full-Body",
                track_type="keyframe",
                color=(219, 148, 86),
                height_scale=0.5,
            )
            timeline_tracks[fullbody_id] = {"name": "Full-Body"}

            root2d_id = timeline.add_track(
                "2D Root",
                track_type="keyframe",
                color=(150, 100, 200),
                height_scale=0.5,
            )
            timeline_tracks[root2d_id] = {"name": "2D Root"}

            lefthand_id = timeline.add_track(
                "Left Hand",
                track_type="keyframe",
                color=(100, 200, 150),
                height_scale=0.5,
            )
            timeline_tracks[lefthand_id] = {"name": "Left Hand"}

            righthand_id = timeline.add_track(
                "Right Hand",
                track_type="keyframe",
                color=(200, 100, 150),
                height_scale=0.5,
            )
            timeline_tracks[righthand_id] = {"name": "Right Hand"}

            leftfoot_id = timeline.add_track(
                "Left Foot",
                track_type="keyframe",
                color=(219, 148, 86),
                height_scale=0.5,
            )
            timeline_tracks[leftfoot_id] = {"name": "Left Foot"}

            rightfoot_id = timeline.add_track(
                "Right Foot",
                track_type="keyframe",
                color=(150, 100, 200),
                height_scale=0.5,
            )
            timeline_tracks[rightfoot_id] = {"name": "Right Foot"}

        # Setup timeline data
        timeline_data = {
            "tracks": timeline_tracks,
            "tracks_ids": {val["name"]: key for key, val in timeline_tracks.items()},
            "keyframes": {},
            "intervals": {},
            "keyframe_update_lock": threading.Lock(),
            "keyframe_move_timers": {},
            "pending_keyframe_moves": {},
            "prompt_uuid_list": [uuid] if uuid is not None else [],
            "prompt_counter": 1,  # Counter for prompt colors (starts at 1 since initial prompt is 0)
            "prompt_spans": {uuid: (0, TIMELINE_WINDOW_AFTER)} if uuid is not None else {},
            "user_prompt_layout": False,
            "loop_prompt_uuid": None,
            "visible_frame_range": None,
        }

        # Active prompt label
        g.gui_active_prompt_label = client.gui.add_markdown("**Active Prompt:** A person is walking.")

        tab_group = client.gui.add_tab_group()

        #
        # Playback tab
        #
        self._build_playback_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_text_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_generate_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_visualize_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_model_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_io_tab(client, client_id, tab_group, g, timeline, default_prompt)

        #
        # Instructions tab
        #
        with tab_group.add_tab("Instructions", viser.Icon.INFO_CIRCLE):
            client.gui.add_markdown(INSTRUCTIONS_TAB_MD)

        gui_elements = GuiElements(**{f: getattr(g, f) for f in GuiElements.__dataclass_fields__})
        return gui_elements, timeline_tracks, timeline_data

    def _close_timeline_prompts_before_frame(
        self,
        session: ClientSession,
        client: viser.ClientHandle,
        first_new_frame: int,
    ) -> None:
        """Clamp prompts so they end before ``first_new_frame``; drop 0- and 1-frame stubs."""
        if session.timeline_data is None or not hasattr(client, "timeline"):
            return

        prior_end = first_new_frame - 1
        prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])
        kept_uuids = []
        for prompt_uuid in prompt_uuid_list:
            try:
                prompt = client.timeline._prompts.get(prompt_uuid)
                if prompt is None:
                    continue
                if prompt.start_frame >= first_new_frame or prior_end < 0:
                    client.timeline.remove_prompt(prompt_uuid)
                    print(
                        f"Removed prompt '{prompt.text}' (start={prompt.start_frame}) "
                        f"before new region at frame {first_new_frame}"
                    )
                    continue
                new_end = min(int(prompt.end_frame), prior_end)
                # Inclusive span of 1 frame (start == end) is a stub — don't keep it.
                if new_end <= prompt.start_frame:
                    client.timeline.remove_prompt(prompt_uuid)
                    print(
                        f"Removed 1-frame prompt stub '{prompt.text}' "
                        f"({prompt.start_frame}-{new_end})"
                    )
                    continue
                if new_end != prompt.end_frame:
                    client.timeline.update_prompt(prompt_uuid, end_frame=new_end)
                    print(f"Updated prompt '{prompt_uuid}' to end at frame {new_end}")
                kept_uuids.append(prompt_uuid)
            except Exception as e:
                print(f"Could not close prompt before frame {first_new_frame}: {e}")

        session.timeline_data["prompt_uuid_list"] = kept_uuids
        session.timeline_data["prompt_counter"] = len(kept_uuids)
        self._sync_prompt_spans(session, client)

    def _sync_generate_prompt_to_text_tab(self, session: ClientSession) -> None:
        session.gui_elements.gui_prompt_text.value = (
            session.gui_elements.gui_generate_prompt_text.value
        )

    def on_text_prompt_update(
        self,
        client_id: int,
        trigger_replan: bool = True,
        initial_prompt: bool = False,
        show_notification: bool = True,
        segment_start_at_playhead: bool = False,
    ):
        """Update text embedding when prompt changes and update timeline prompts."""
        start_time = time.time()
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        if session.model is None:
            return

        text_prompt = session.gui_elements.gui_prompt_text.value
        text_feat, _ = session.model.text_encoder([text_prompt])
        session.text_embedding = text_feat.to(self.device)

        session.gui_elements.gui_active_prompt_label.content = f"**Active Prompt:** {text_prompt}"

        current_frame = max(0, session.frame_idx)
        if segment_start_at_playhead:
            new_segment_start = current_frame
        elif initial_prompt:
            new_segment_start = 0
        else:
            new_segment_start = current_frame + 1

        if session.timeline_data is not None and hasattr(client, "timeline"):
            if segment_start_at_playhead or not initial_prompt:
                self._close_timeline_prompts_before_frame(session, client, new_segment_start)
            prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])
            try:
                prompt_counter = session.timeline_data.get("prompt_counter", 1)
                prompt_color = self.get_prompt_color(prompt_counter)

                new_uuid = client.timeline.add_prompt(
                    text=text_prompt,
                    start_frame=new_segment_start,
                    end_frame=self._prompt_end_frame(session),
                    color=prompt_color,
                )
                prompt_uuid_list.append(new_uuid)
                session.timeline_data["prompt_counter"] = prompt_counter + 1
                self._sync_prompt_spans(session, client)
                print(
                    f"Added new prompt '{new_uuid}' starting at frame {new_segment_start}: "
                    f"'{text_prompt}' (color: {prompt_color})"
                )
            except (AttributeError, Exception) as e:
                print(f"Could not add new prompt to timeline: {e}")

        if show_notification:
            session.client.add_notification(
                title="Text prompt updated",
                body=f"New prompt starts at frame {new_segment_start}",
                auto_close_seconds=3.0,
                color="blue",
            )

        end_time = time.time()
        print(f"Time taken to update text prompt: {end_time - start_time} seconds")

        if trigger_replan:
            threading.Thread(target=self.on_replan_trigger, args=(client_id,), daemon=True).start()
