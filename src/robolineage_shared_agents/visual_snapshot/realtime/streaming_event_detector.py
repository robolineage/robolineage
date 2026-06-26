from __future__ import annotations

from typing import Optional

from ..types import ActionDerivedSignal, ActionEvent

# Internal implementation note.
_RATE_LIMITED_EVENTS = frozenset({"still_start", "motion_resume", "heartbeat"})


class StreamingEventDetector:
    """Incremental action event detector for streaming ActionDerivedSignal rows."""

    def __init__(
        self,
        still_min_frames: int = 15,
        heartbeat_interval: float = 5.0,
        periodic_interval_sec: float = 2.0,
        motion_resume_threshold: float = 8e-4,
        min_same_event_interval: float = 3.0,
    ):
        self.still_min_frames = max(1, still_min_frames)
        self.heartbeat_interval = heartbeat_interval
        self.periodic_interval_sec = max(0.0, periodic_interval_sec)
        self.motion_resume_threshold = motion_resume_threshold
        self.min_same_event_interval = min_same_event_interval
        self._still_count = 0
        self._still_emitted = False
        self._last_event_ts: Optional[float] = None
        self._last_periodic_ts: Optional[float] = None
        self._last_ts_by_type: dict[str, float] = {}

    def feed(self, signal: ActionDerivedSignal) -> list[ActionEvent]:
        events: list[ActionEvent] = []
        if self._last_periodic_ts is None:
            self._last_periodic_ts = signal.timestamp_sec

        if signal.gripper_edge == "closing_edge":
            events.append(
                ActionEvent(
                    event_type="gripper_close",
                    anchor_frame=signal.frame_index,
                    timestamp_sec=signal.timestamp_sec,
                    confidence=1.0,
                    details={"gripper_state": signal.gripper_state},
                )
            )
        elif signal.gripper_edge == "opening_edge":
            events.append(
                ActionEvent(
                    event_type="gripper_open",
                    anchor_frame=signal.frame_index,
                    timestamp_sec=signal.timestamp_sec,
                    confidence=1.0,
                    details={"gripper_state": signal.gripper_state},
                )
            )

        if signal.is_still:
            self._still_count += 1
            if self._still_count >= self.still_min_frames and not self._still_emitted:
                if self._allow("still_start", signal.timestamp_sec):
                    events.append(
                        ActionEvent(
                            event_type="still_start",
                            anchor_frame=signal.frame_index,
                            timestamp_sec=signal.timestamp_sec,
                            confidence=0.85,
                            details={"still_frames": self._still_count},
                        )
                    )
                self._still_emitted = True
        else:
            if self._still_emitted and signal.motion_energy_avg >= self.motion_resume_threshold:
                if self._allow("motion_resume", signal.timestamp_sec):
                    events.append(
                        ActionEvent(
                            event_type="motion_resume",
                            anchor_frame=signal.frame_index,
                            timestamp_sec=signal.timestamp_sec,
                            confidence=0.7,
                            details={"from_still_frames": self._still_count},
                        )
                    )
            self._still_count = 0
            self._still_emitted = False

        if not events and self.periodic_interval_sec > 0:
            assert self._last_periodic_ts is not None
            if signal.timestamp_sec - self._last_periodic_ts >= self.periodic_interval_sec:
                events.append(
                    ActionEvent(
                        event_type="periodic_sample",
                        anchor_frame=signal.frame_index,
                        timestamp_sec=signal.timestamp_sec,
                        confidence=0.45,
                        details={"interval_sec": self.periodic_interval_sec},
                    )
                )

        if not events and self.heartbeat_interval > 0:
            if self._last_event_ts is None:
                self._last_event_ts = signal.timestamp_sec
            elif signal.timestamp_sec - self._last_event_ts >= self.heartbeat_interval:
                if self._allow("heartbeat", signal.timestamp_sec):
                    events.append(
                        ActionEvent(
                            event_type="heartbeat",
                            anchor_frame=signal.frame_index,
                            timestamp_sec=signal.timestamp_sec,
                            confidence=0.5,
                        )
                    )

        if events:
            self._last_event_ts = events[-1].timestamp_sec
            self._last_periodic_ts = events[-1].timestamp_sec
            for ev in events:
                if ev.event_type in _RATE_LIMITED_EVENTS:
                    self._last_ts_by_type[ev.event_type] = ev.timestamp_sec
        return events

    def _allow(self, event_type: str, ts: float) -> bool:
        """Return True if enough time has passed since the last same-type event."""
        if event_type not in _RATE_LIMITED_EVENTS:
            return True
        last = self._last_ts_by_type.get(event_type)
        return last is None or ts - last >= self.min_same_event_interval
