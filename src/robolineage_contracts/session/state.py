"""SessionState contract and legal transition table.

Source of truth: the session-state contract.

The runtime state machine is implemented in `src/robolineage_session/state_machine.py`.
This module only exposes the enum, transition table, and pure validation helper.
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping


class SessionState(str, Enum):
    """Five lifecycle states for a collection or evaluation session."""
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    PAUSED = "PAUSED"
    REVIEWING = "REVIEWING"
    SUBMITTED = "SUBMITTED"


# Allowed transitions per the session-state contract.
#
# Reading the table:
#   ALLOWED_TRANSITIONS[before] = frozenset of states that may follow `before`.
# A transition (before, after) is legal iff `after in ALLOWED_TRANSITIONS[before]`.
ALLOWED_TRANSITIONS: Mapping[SessionState, frozenset[SessionState]] = {
    SessionState.IDLE: frozenset({SessionState.COLLECTING}),
    SessionState.COLLECTING: frozenset({SessionState.PAUSED, SessionState.REVIEWING}),
    SessionState.PAUSED: frozenset({SessionState.COLLECTING, SessionState.REVIEWING}),
    SessionState.REVIEWING: frozenset({SessionState.SUBMITTED, SessionState.COLLECTING}),
    SessionState.SUBMITTED: frozenset({SessionState.IDLE}),
}


def validate_transition(before: SessionState, after: SessionState) -> bool:
    """Return True iff transitioning from `before` to `after` is legal.

    Pure / stateless / no exceptions. Session runtime wraps this in a stateful
    `StateMachine` class that raises `IllegalStateTransition` on violation.
    """
    return after in ALLOWED_TRANSITIONS[before]
