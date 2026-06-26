"""Session entity and single-session registry."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from robolineage_contracts.core import RolloutMode
from robolineage_contracts.session import SessionState

from robolineage_session.rollout_id import generate_rollout_id
from robolineage_session.state_machine import StateMachine


@dataclass
class Session:
    session_id: str
    rollout_id: str
    rollout_dir: Path
    runtime_dir: Path
    task_id: str
    mode: RolloutMode
    operator_id: str
    policy_version: str | None
    started_at: str
    state_machine: StateMachine
    events_path: Path

    @property
    def state(self) -> SessionState:
        return self.state_machine.state


class SessionRegistry:
    """MVP registry: exactly one active session at a time."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: Session | None = None

    def current(self) -> Session | None:
        with self._lock:
            return self._current

    def require_current(self) -> Session:
        session = self.current()
        if session is None:
            raise RuntimeError("no active session")
        return session

    def create(
        self,
        *,
        data_root: Path,
        runtime_root: Path,
        task_id: str,
        mode: RolloutMode,
        operator_id: str,
        policy_version: str | None,
        started_at: str,
    ) -> Session:
        with self._lock:
            if self._current is not None:
                raise RuntimeError("session already active")
            rollout_id = generate_rollout_id()
            session_id = str(uuid.uuid4())
            rollout_dir = data_root / rollout_id
            runtime_dir = runtime_root / session_id
            session = Session(
                session_id=session_id,
                rollout_id=rollout_id,
                rollout_dir=rollout_dir,
                runtime_dir=runtime_dir,
                task_id=task_id,
                mode=mode,
                operator_id=operator_id,
                policy_version=policy_version,
                started_at=started_at,
                state_machine=StateMachine(),
                events_path=rollout_dir / "events.jsonl",
            )
            self._current = session
            return session

    def clear(self) -> None:
        with self._lock:
            self._current = None

    def set_current(self, session: Session | None) -> None:
        with self._lock:
            self._current = session


DEFAULT_REGISTRY = SessionRegistry()
