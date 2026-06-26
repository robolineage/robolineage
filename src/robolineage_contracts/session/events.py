"""ControlEvent / FeedbackEvent / EventEnvelope — UX ↔ AR ↔ VSA communication contract.

Source of truth: the session event contract (error codes).

The wire envelope (`EventEnvelope`) is the same JSON shape regardless of
direction or source. Two enum families distinguish kind:

  - **ControlEvent** (UX → AR) — 6 events that drive the state machine
    (START_COLLECTING / PAUSE / RESUME / STOP / SUBMIT / DISCARD)
  - **FeedbackEvent** (AR → UX, VSA → UX) — 6 events that report state
    changes or VLM activity (SESSION_OPENED / SESSION_CLOSED / FRAME_DROPPED /
    ASSESSMENT_UPDATED / RISK_ALERT / VLM_FAILURE)

Plus `ErrorCode` for the `event="ERROR"` envelope (see §5) and `EventSource`
for the `source` field of every envelope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ── Event-name enums ─────────────────────────────────────────────────────

class ControlEventName(str, Enum):
    """UX → AR control events. Drive the session state machine."""
    START_COLLECTING = "START_COLLECTING"
    PAUSE_COLLECTING = "PAUSE_COLLECTING"
    RESUME_COLLECTING = "RESUME_COLLECTING"
    STOP_COLLECTING = "STOP_COLLECTING"
    SUBMIT_ROLLOUT = "SUBMIT_ROLLOUT"
    DISCARD_ROLLOUT = "DISCARD_ROLLOUT"


class FeedbackEventName(str, Enum):
    """AR → UX / VSA → UX feedback events."""
    SESSION_OPENED = "SESSION_OPENED"
    SESSION_CLOSED = "SESSION_CLOSED"
    FRAME_DROPPED = "FRAME_DROPPED"
    ASSESSMENT_UPDATED = "ASSESSMENT_UPDATED"
    RISK_ALERT = "RISK_ALERT"
    VLM_FAILURE = "VLM_FAILURE"


# ── Source / error code ──────────────────────────────────────────────────

class EventSource(str, Enum):
    """Who emitted this event. Goes in `EventEnvelope.source`."""
    UX = "ux"
    AR = "ar"
    POLICY = "policy"
    VSA = "vsa"


class ErrorCode(str, Enum):
    """Errors that ride on `event="ERROR"` envelopes (see §5).

    Shared namespace with doc/agent contract.md (WP5 maintains the union);
    H mirrors only the entries actually emitted by the AR / VSA path here.
    """
    E_SESSION_OPEN_FAILED = "E_SESSION_OPEN_FAILED"
    E_SESSION_CLOSE_FAILED = "E_SESSION_CLOSE_FAILED"
    E_ILLEGAL_STATE_TRANSITION = "E_ILLEGAL_STATE_TRANSITION"
    E_VLM_TIMEOUT = "E_VLM_TIMEOUT"


# ── Envelope ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EventEnvelope:
    """Standard JSON shape used by all session events (§2).

    Attributes:
        event: event name (a `ControlEventName` / `FeedbackEventName` value, or
               the string "ERROR" for error envelopes)
        event_id: uuid v4
        timestamp: ISO8601 with millisecond precision (e.g. "...10:00:00.123Z")
        rollout_id: UUID of the rollout this event refers to. Allowed to be
                   None for events that fire before SESSION_OPENED (e.g. an
                   OPEN-side error).
        source: which subsystem emitted the event
        payload: free-form dict per event-type (validated by event-specific
                 helpers in session runtime; contracts does not enforce shape here so the
                 envelope stays forward-compatible).
    """
    event: str
    event_id: str
    timestamp: str
    rollout_id: Optional[str]
    source: EventSource
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event:
            raise ValueError("event must be a non-empty string")
        if not self.event_id:
            raise ValueError("event_id must be a non-empty string (uuid)")
        if not self.timestamp:
            raise ValueError("timestamp must be a non-empty string (ISO8601)")
