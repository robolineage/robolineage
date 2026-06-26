"""Feedback event dispatcher for VSA snapshot output."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from robolineage_contracts.agents import SnapshotAssessment
from robolineage_contracts.session import EventEnvelope, EventSource, FeedbackEventName, SessionState

from robolineage_session.events import EventLogger
from robolineage_session.session import Session


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_envelope(
    *,
    event: str,
    rollout_id: str | None,
    source: EventSource,
    payload: dict | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event=event,
        event_id=str(uuid.uuid4()),
        timestamp=utc_now_iso(),
        rollout_id=rollout_id,
        source=source,
        payload=payload or {},
    )


class FeedbackDispatcher:
    def __init__(
        self,
        *,
        session: Session,
        logger: EventLogger,
        broadcast: Callable[[EventEnvelope], None],
    ) -> None:
        self.session = session
        self.logger = logger
        self.broadcast = broadcast

    def _emit(self, env: EventEnvelope) -> None:
        self.logger.append(env)
        self.broadcast(env)

    def on_snapshot(self, snapshot: SnapshotAssessment) -> None:
        base_payload = {
            "frame_id": snapshot.frame_id,
            "timestamp": snapshot.timestamp,
            "progress": snapshot.progress,
            "risk_level": snapshot.risk_level,
            "phase": snapshot.phase,
            "confidence": snapshot.confidence,
            "needs_review": snapshot.needs_review,
        }
        self._emit(make_envelope(
            event=FeedbackEventName.ASSESSMENT_UPDATED.value,
            rollout_id=self.session.rollout_id,
            source=EventSource.VSA,
            payload=base_payload,
        ))

        if snapshot.risk_level == "high" or snapshot.imminent_failure:
            self._emit(make_envelope(
                event=FeedbackEventName.RISK_ALERT.value,
                rollout_id=self.session.rollout_id,
                source=EventSource.VSA,
                payload={**base_payload, "imminent_failure": snapshot.imminent_failure},
            ))

        vlm_error = None
        if isinstance(snapshot.vlm_meta, dict):
            vlm_error = snapshot.vlm_meta.get("error")
        if vlm_error:
            self._emit(make_envelope(
                event=FeedbackEventName.VLM_FAILURE.value,
                rollout_id=self.session.rollout_id,
                source=EventSource.VSA,
                payload={**base_payload, "error": vlm_error},
            ))
            if self.session.state == SessionState.COLLECTING:
                self.session.state_machine.transition(SessionState.PAUSED)
