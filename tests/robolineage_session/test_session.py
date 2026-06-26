from pathlib import Path

import pytest

from robolineage_contracts.core import RolloutMode
from robolineage_contracts.session import SessionState
from robolineage_session.session import SessionRegistry


def test_registry_creates_single_active_session(tmp_path: Path):
    registry = SessionRegistry()

    session = registry.create(
        data_root=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        task_id="task_1",
        mode=RolloutMode.A,
        operator_id="op",
        policy_version=None,
        started_at="2026-04-25T00:00:00.000Z",
    )

    assert registry.current() == session
    assert session.state == SessionState.IDLE
    assert session.rollout_dir == tmp_path / "data" / session.rollout_id
    assert session.runtime_dir == tmp_path / "runtime" / session.session_id


def test_registry_rejects_second_active_session(tmp_path: Path):
    registry = SessionRegistry()
    kwargs = dict(
        data_root=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        task_id="task_1",
        mode=RolloutMode.A,
        operator_id="op",
        policy_version=None,
        started_at="2026-04-25T00:00:00.000Z",
    )

    registry.create(**kwargs)

    with pytest.raises(RuntimeError, match="already active"):
        registry.create(**kwargs)


def test_registry_clear_removes_active_session(tmp_path: Path):
    registry = SessionRegistry()
    registry.create(
        data_root=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        task_id="task_1",
        mode=RolloutMode.A,
        operator_id="op",
        policy_version=None,
        started_at="2026-04-25T00:00:00.000Z",
    )

    registry.clear()

    assert registry.current() is None
