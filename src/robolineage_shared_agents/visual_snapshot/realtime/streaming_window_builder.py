from __future__ import annotations

from pathlib import Path

from ..keyframe_selector import KeyframeSelector
from ..types import ActionDerivedSignal, ActionEvent, ActionGuidedWindow, FrameActionRecord
from .frame_ring_buffer import FrameRingBuffer


class StreamingWindowBuilder:
    """Build ActionGuidedWindow instances from live frame/action buffers."""

    def __init__(
        self,
        frame_buffer: FrameRingBuffer,
        context_frames: int = 15,
        max_keyframes: int = 3,
        camera_name: str = "camera_h",
    ):
        self.frame_buffer = frame_buffer
        self.context_frames = max(1, context_frames)
        self.selector = KeyframeSelector(context_frames=context_frames, max_keyframes=max_keyframes)
        self.camera_name = camera_name

    def build(
        self,
        event: ActionEvent,
        records: list[FrameActionRecord],
        signals: list[ActionDerivedSignal],
        rollout_id: str = "live",
    ) -> ActionGuidedWindow:
        record_map = {record.frame_index: record for record in records}
        signal_map = {signal.frame_index: signal for signal in signals}
        first_frame = records[0].frame_index
        last_frame = records[-1].frame_index
        start_frame = max(first_frame, event.anchor_frame - self.context_frames)
        end_frame = min(last_frame, event.anchor_frame + self.context_frames)
        frame_ids = list(range(start_frame, end_frame + 1))
        action_rows = [record_map[frame_id] for frame_id in frame_ids if frame_id in record_map]
        action_signals = [signal_map[frame_id] for frame_id in frame_ids if frame_id in signal_map]
        keyframe_ids = self.selector.select(event, records, signals)
        color_frames = self._get_keyframes_with_fallback(keyframe_ids)

        anchor_record = record_map[event.anchor_frame]
        return ActionGuidedWindow(
            rollout_id=rollout_id,
            frame_ids=frame_ids,
            timestamps=[row.timestamp_sec for row in action_rows],
            camera_name=self.camera_name,
            color_frames=color_frames,
            depth_frames=[],
            end_frame_id=event.anchor_frame,
            end_timestamp=anchor_record.timestamp_sec,
            anchor_frame_id=event.anchor_frame,
            event_type=event.event_type,
            keyframe_ids=keyframe_ids,
            action_summary=self._build_summary(action_rows, action_signals, event),
            event_details=dict(event.details),
            source_video_name=Path("live_stream").name,
        )

    def _get_keyframes_with_fallback(self, keyframe_ids: list[int]) -> list:
        """Retrieve keyframes; if a frame is missing, use the latest available frame."""
        color_frames = []
        latest_available = self.frame_buffer.latest_frame_index()
        for frame_id in keyframe_ids:
            frame = self.frame_buffer.get_rgb(frame_id)
            if frame is None and latest_available is not None:
                frame = self.frame_buffer.get_rgb(latest_available)
            color_frames.append(frame)
        return color_frames

    @staticmethod
    def _build_summary(
        action_rows: list[FrameActionRecord],
        action_signals: list[ActionDerivedSignal],
        event: ActionEvent,
    ) -> dict[str, object]:
        first_row = action_rows[0]
        last_row = action_rows[-1]
        first_signal = action_signals[0]
        last_signal = action_signals[-1]

        mean_motion = sum(signal.motion_energy_avg for signal in action_signals) / len(action_signals)
        max_motion = max(signal.motion_energy_avg for signal in action_signals)
        still_fraction = sum(1 for signal in action_signals if signal.is_still) / len(action_signals)
        position_delta = tuple(round(last_row.eef_xyz[i] - first_row.eef_xyz[i], 6) for i in range(3))
        rotation_delta = tuple(round(last_row.eef_rxyz[i] - first_row.eef_rxyz[i], 6) for i in range(3))

        return {
            "event_type": event.event_type,
            "frame_range": [first_row.frame_index, last_row.frame_index],
            "time_range_sec": [round(first_row.timestamp_sec, 4), round(last_row.timestamp_sec, 4)],
            "gripper_before": round(first_row.gripper, 6),
            "gripper_after": round(last_row.gripper, 6),
            "gripper_state_before": first_signal.gripper_state,
            "gripper_state_after": last_signal.gripper_state,
            "mean_motion_energy": round(mean_motion, 6),
            "max_motion_energy": round(max_motion, 6),
            "still_fraction": round(still_fraction, 3),
            "position_delta": position_delta,
            "rotation_delta": rotation_delta,
        }
