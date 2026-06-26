from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .types import RolloutMemoryContext


@dataclass
class PhaseDecisionRecord:
    frame_id: int
    timestamp: float
    event_type: str
    visual_phase: str
    final_phase: str
    confidence: float
    risk_level: str
    progress: str
    action_hint_phase: str
    fusion_reason: str
    state_reason: str


class RolloutMemory:
    """Short-horizon online memory for one rollout.

    The post-rollout annotation line will eventually do full sequence review.
    This memory is deliberately lightweight and exists only to make online VSA
    judgments temporally aware.
    """

    def __init__(self, max_entries: int = 8):
        self._records: deque[PhaseDecisionRecord] = deque(maxlen=max_entries)
        self._phase_first_seen_frames: dict[str, int] = {}
        self._phase_confidence: dict[str, float] = {}

    def reset(self) -> None:
        self._records.clear()
        self._phase_first_seen_frames.clear()
        self._phase_confidence.clear()

    def context(self) -> RolloutMemoryContext:
        records = list(self._records)
        if not records:
            return RolloutMemoryContext()

        recent_final = [record.final_phase for record in records[-5:]]
        recent_visual = [record.visual_phase for record in records[-5:]]
        recent_events = [
            f"{record.event_type}@{record.frame_id}"
            for record in records[-5:]
            if record.event_type
        ]
        last = records[-1]
        summary_parts = [
            f"last_confirmed_phase={last.final_phase}",
            "recent_final_phases=" + " -> ".join(recent_final),
        ]
        if recent_visual:
            summary_parts.append("recent_visual_suggestions=" + " -> ".join(recent_visual))
        if recent_events:
            summary_parts.append("recent_events=" + ", ".join(recent_events))

        return RolloutMemoryContext(
            last_confirmed_phase=last.final_phase,
            recent_final_phases=recent_final,
            recent_visual_phases=recent_visual,
            recent_events=recent_events,
            phase_first_seen_frames=dict(self._phase_first_seen_frames),
            phase_confidence=dict(self._phase_confidence),
            summary="; ".join(summary_parts),
        )

    def add(self, record: PhaseDecisionRecord) -> None:
        self._records.append(record)
        if record.final_phase and record.final_phase not in self._phase_first_seen_frames:
            self._phase_first_seen_frames[record.final_phase] = record.frame_id
        if record.final_phase:
            self._phase_confidence[record.final_phase] = max(
                float(record.confidence),
                self._phase_confidence.get(record.final_phase, 0.0),
            )
