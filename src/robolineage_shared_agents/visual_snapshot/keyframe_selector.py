from __future__ import annotations

from .types import ActionDerivedSignal, ActionEvent, FrameActionRecord


class KeyframeSelector:
    def __init__(self, context_frames: int = 15, max_keyframes: int = 3):
        self.context_frames = max(1, context_frames)
        self.max_keyframes = max(1, max_keyframes)

    def select(
        self,
        event: ActionEvent,
        records: list[FrameActionRecord],
        signals: list[ActionDerivedSignal],
    ) -> list[int]:
        if not records:
            return []

        first_frame = records[0].frame_index
        last_frame = records[-1].frame_index
        start_frame = max(first_frame, event.anchor_frame - self.context_frames)
        end_frame = min(last_frame, event.anchor_frame + self.context_frames)

        candidates = [
            start_frame,
            max(first_frame, event.anchor_frame - max(1, self.context_frames // 2)),
            event.anchor_frame,
            min(last_frame, event.anchor_frame + max(1, self.context_frames // 2)),
            end_frame,
        ]

        peak_frame = self._motion_peak_frame(start_frame, end_frame, signals)
        if peak_frame is not None:
            candidates.append(peak_frame)

        unique = sorted(set(candidates))
        if len(unique) <= self.max_keyframes:
            return unique

        if self.max_keyframes == 1:
            return [event.anchor_frame]

        step = (len(unique) - 1) / (self.max_keyframes - 1)
        return [unique[round(i * step)] for i in range(self.max_keyframes)]

    @staticmethod
    def _motion_peak_frame(
        start_frame: int,
        end_frame: int,
        signals: list[ActionDerivedSignal],
    ) -> int | None:
        peak_signal = None
        for signal in signals:
            if start_frame <= signal.frame_index <= end_frame:
                if peak_signal is None or signal.motion_energy_avg > peak_signal.motion_energy_avg:
                    peak_signal = signal
        return peak_signal.frame_index if peak_signal is not None else None
