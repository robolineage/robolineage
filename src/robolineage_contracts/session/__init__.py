"""Session state-machine + event envelope contracts (H5, v0.1.3).

Used by:
  Session service + AR                   — primary consumer
  Master Agent                           — reads SessionState to know when
                                             a rollout is settled

Public:
    # state.py
    SessionState                              — five-state enum
    ALLOWED_TRANSITIONS                       — Mapping[before, frozenset[after]]
    validate_transition(before, after)        — pure stateless helper

    # events.py
    ControlEventName, FeedbackEventName       — event-name enums
    EventSource                                — emitter class
    ErrorCode                                  — `event="ERROR"` error code
    EventEnvelope                              — generic wire envelope dataclass
"""
from robolineage_contracts.session.events import (
    ControlEventName,
    ErrorCode,
    EventEnvelope,
    EventSource,
    FeedbackEventName,
)
from robolineage_contracts.session.state import (
    ALLOWED_TRANSITIONS,
    SessionState,
    validate_transition,
)

__all__ = [
    # state
    "SessionState",
    "ALLOWED_TRANSITIONS",
    "validate_transition",
    # events
    "ControlEventName",
    "FeedbackEventName",
    "EventSource",
    "ErrorCode",
    "EventEnvelope",
]
