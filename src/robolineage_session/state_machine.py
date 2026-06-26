"""Stateful implementation of the RoboLineage session transition table.

The source of truth for legal transitions is `robolineage_contracts.session`; this
module owns only runtime state, locking, and the business exception raised on
illegal transitions.
"""
from __future__ import annotations

import threading

from robolineage_contracts.session import ErrorCode, SessionState, validate_transition


class IllegalStateTransition(Exception):
    """Raised when a session transition is not allowed by the contract table."""

    error_code = ErrorCode.E_ILLEGAL_STATE_TRANSITION

    def __init__(self, before: SessionState, after: SessionState) -> None:
        super().__init__(f"Illegal transition: {before.value} -> {after.value}")
        self.before = before
        self.after = after


class StateMachine:
    """Thread-safe runtime holder for the five-state session lifecycle."""

    def __init__(self, initial: SessionState = SessionState.IDLE) -> None:
        self._state = initial
        self._lock = threading.RLock()

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    def transition(self, after: SessionState) -> SessionState:
        with self._lock:
            if not validate_transition(self._state, after):
                raise IllegalStateTransition(self._state, after)
            self._state = after
            return self._state
