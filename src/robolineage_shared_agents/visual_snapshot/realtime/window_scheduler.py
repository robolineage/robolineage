from __future__ import annotations

from ..types import ActionEvent


_COVERAGE_EVENTS = {"periodic_sample", "heartbeat"}
_GRIPPER_EVENTS = {"gripper_open", "gripper_close"}
_CONTACT_MOTION_EVENTS = {"still_start", "motion_resume"}
_PROTECTED_EVENTS = {"sequence_start", "final_observation"}
_PRIORITY = {
    "sequence_start": 80,
    "gripper_burst": 75,
    "contact_transition": 74,
    "gripper_close": 70,
    "gripper_open": 70,
    "motion_resume": 60,
    "still_start": 55,
    "final_observation": 90,
    "periodic_sample": 10,
    "heartbeat": 5,
}


class WindowScheduler:
    """Coalesce realtime VSA trigger candidates before VLM window building."""

    def __init__(self, *, context_frames: int = 15, merge_window_sec: float = 1.0) -> None:
        self.context_frames = max(0, int(context_frames))
        self.merge_window_sec = max(0.0, float(merge_window_sec))
        self._pending_events: list[ActionEvent] = []

    def coalesce(self, events: list[ActionEvent]) -> list[ActionEvent]:
        if len(events) < 2:
            return events

        return [self._represent(cluster) for cluster in self._clusters(events)]

    def schedule(
        self,
        events: list[ActionEvent],
        *,
        watermark_timestamp: float | None,
        force: bool = False,
    ) -> list[ActionEvent]:
        """Stateful online scheduler.

        ``coalesce`` is intentionally stateless for tests/offline callers.
        Realtime dispatch uses this method so an event released in one batch can
        wait briefly for a nearby gripper/contact edge released by the next
        batch.
        """
        if events:
            self._pending_events.extend(events)
        if not self._pending_events:
            return []

        clusters = self._clusters(self._pending_events)
        ready: list[ActionEvent] = []
        pending: list[ActionEvent] = []
        blocked = False
        for cluster in clusters:
            if not blocked and (force or self._cluster_ready(cluster, watermark_timestamp)):
                ready.append(self._represent(cluster))
            else:
                blocked = True
                pending.extend(cluster)
        self._pending_events = pending
        return ready

    def _clusters(self, events: list[ActionEvent]) -> list[list[ActionEvent]]:
        ordered = sorted(events, key=lambda item: (item.timestamp_sec, item.anchor_frame, item.event_type))
        clusters: list[list[ActionEvent]] = []
        for event in ordered:
            if not clusters or not self._belongs_to_cluster(event, clusters[-1]):
                clusters.append([event])
            else:
                clusters[-1].append(event)
        return clusters

    def _cluster_ready(self, cluster: list[ActionEvent], watermark_timestamp: float | None) -> bool:
        if any(event.event_type in _PROTECTED_EVENTS for event in cluster):
            return True
        if watermark_timestamp is None:
            return False
        latest_event_ts = max(event.timestamp_sec for event in cluster)
        return watermark_timestamp - latest_event_ts >= self.merge_window_sec

    def _belongs_to_cluster(self, event: ActionEvent, cluster: list[ActionEvent]) -> bool:
        if event.event_type in _PROTECTED_EVENTS:
            return False
        if any(item.event_type in _PROTECTED_EVENTS for item in cluster):
            return False
        return any(self._events_overlap(event, item) for item in cluster)

    def _events_overlap(self, left: ActionEvent, right: ActionEvent) -> bool:
        if abs(left.timestamp_sec - right.timestamp_sec) <= self.merge_window_sec:
            return True
        left_range = self._frame_range(left)
        right_range = self._frame_range(right)
        return left_range[0] <= right_range[1] and right_range[0] <= left_range[1]

    def _frame_range(self, event: ActionEvent) -> tuple[int, int]:
        return (
            event.anchor_frame - self.context_frames,
            event.anchor_frame + self.context_frames,
        )

    def _represent(self, cluster: list[ActionEvent]) -> ActionEvent:
        if len(cluster) == 1:
            return cluster[0]

        source_types = [event.event_type for event in cluster if event.event_type not in _COVERAGE_EVENTS]
        covered_events = [event for event in cluster if event.event_type in _COVERAGE_EVENTS]
        covered_types = [event.event_type for event in covered_events]
        source_type_set = set(source_types)
        if source_type_set.intersection(_GRIPPER_EVENTS) and source_type_set.intersection(_CONTACT_MOTION_EVENTS):
            event_type = "contact_transition"
            representative = max(cluster, key=lambda event: (event.anchor_frame, event.timestamp_sec))
        elif _GRIPPER_EVENTS.issubset(source_type_set):
            event_type = "gripper_burst"
            representative = max(cluster, key=lambda event: (event.anchor_frame, event.timestamp_sec))
        else:
            representative = max(
                cluster,
                key=lambda event: (
                    _PRIORITY.get(event.event_type, 50),
                    event.confidence,
                    event.anchor_frame,
                    event.timestamp_sec,
                ),
            )
            event_type = representative.event_type

        details = dict(representative.details)
        if source_types:
            details["source_event_types"] = _dedupe(source_types)
        if covered_types:
            details["covered_event_types"] = _dedupe(covered_types)
            details["covered_events"] = [
                {
                    "event_type": event.event_type,
                    "anchor_frame": event.anchor_frame,
                    "timestamp_sec": event.timestamp_sec,
                    "reason": f"coalesced_with_{event_type}",
                }
                for event in covered_events
            ]
        details["scheduler_coalesced_count"] = len(cluster)
        details["scheduler_anchor_frames"] = [event.anchor_frame for event in cluster]

        return ActionEvent(
            event_type=event_type,
            anchor_frame=representative.anchor_frame,
            timestamp_sec=representative.timestamp_sec,
            confidence=max(event.confidence for event in cluster),
            details=details,
        )


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
