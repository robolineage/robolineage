"""Incremental equivalent of ActionSignalBuilder for realtime records."""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

from .types import RealtimeActionRecord
from ..types import ActionDerivedSignal, FrameActionRecord


class StreamingSignalBuilder:
    """Incremental equivalent of ActionSignalBuilder for RealtimeActionRecord streams."""

    def __init__(
        self,
        gripper_close_threshold: float = -1.0,
        rotation_weight: float = 0.2,
        smoothing_window: int = 10,
        still_threshold: float = 3e-4,
    ):
        self.gripper_close_threshold = gripper_close_threshold
        self.rotation_weight = rotation_weight
        self.smoothing_window = max(1, smoothing_window)
        self.still_threshold = still_threshold
        self._motion_window: deque[float] = deque(maxlen=self.smoothing_window)
        self._prev_record: Optional[FrameActionRecord] = None
        self._prev_state: Optional[str] = None
        self.records: list[FrameActionRecord] = []
        self.signals: list[ActionDerivedSignal] = []

    def feed(self, msg: RealtimeActionRecord) -> tuple[FrameActionRecord, ActionDerivedSignal]:
        record = FrameActionRecord(
            episode="live",
            frame_index=msg.frame_index,
            timestamp_sec=msg.host_mono_ns / 1_000_000_000,
            mp4_file="",
            hdf5_file="",
            eef_xyz=msg.eef_xyz,
            eef_rxyz=msg.eef_rxyz,
            gripper=msg.gripper,
        )

        state = self._gripper_state(record.gripper)
        if self._prev_record is None:
            translation_speed = 0.0
            rotation_speed = 0.0
            edge = "none"
        else:
            translation_speed = self._l2_distance(record.eef_xyz, self._prev_record.eef_xyz)
            rotation_speed = self._l2_distance(record.eef_rxyz, self._prev_record.eef_rxyz)
            if self._prev_state == "open" and state == "closed":
                edge = "closing_edge"
            elif self._prev_state == "closed" and state == "open":
                edge = "opening_edge"
            else:
                edge = "none"

        motion_energy = translation_speed + self.rotation_weight * rotation_speed
        self._motion_window.append(motion_energy)
        motion_energy_avg = sum(self._motion_window) / len(self._motion_window)
        signal = ActionDerivedSignal(
            frame_index=record.frame_index,
            timestamp_sec=record.timestamp_sec,
            gripper_state=state,  # type: ignore[arg-type]
            gripper_edge=edge,  # type: ignore[arg-type]
            translation_speed=translation_speed,
            rotation_speed=rotation_speed,
            motion_energy=motion_energy,
            motion_energy_avg=motion_energy_avg,
            is_still=motion_energy_avg < self.still_threshold,
        )

        self._prev_record = record
        self._prev_state = state
        self.records.append(record)
        self.signals.append(signal)
        return record, signal

    def _gripper_state(self, gripper_value: float) -> str:
        return "closed" if gripper_value <= self.gripper_close_threshold else "open"

    @staticmethod
    def _l2_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
        return math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(3)))
