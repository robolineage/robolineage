from pathlib import Path

from robolineage_contracts.agents import SnapshotAssessment
from robolineage_contracts.core import RolloutMode
from robolineage_contracts.session import FeedbackEventName, SessionState
from robolineage_session.dispatcher import FeedbackDispatcher
from robolineage_session.events import EventLogger
from robolineage_session.session import SessionRegistry


def _session(tmp_path: Path):
    registry = SessionRegistry()
    session = registry.create(
        data_root=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        task_id="task_1",
        mode=RolloutMode.B1,
        operator_id="op",
        policy_version="1.0.0",
        started_at="2026-04-25T00:00:00.000Z",
    )
    session.rollout_dir.mkdir(parents=True)
    session.state_machine.transition(SessionState.COLLECTING)
    return session


def _snapshot(**overrides) -> SnapshotAssessment:
    base = dict(
        timestamp=1.0,
        frame_id=1,
        progress="advancing",
        risk_level="low",
        phase="approach",
        imminent_failure=False,
        confidence=0.9,
        needs_review=False,
        raw_response="ok",
    )
    base.update(overrides)
    return SnapshotAssessment(**base)


def test_low_risk_only_emits_assessment_updated(tmp_path: Path):
    session = _session(tmp_path)
    emitted = []
    dispatcher = FeedbackDispatcher(
        session=session,
        logger=EventLogger(session.events_path),
        broadcast=emitted.append,
    )

    dispatcher.on_snapshot(_snapshot())

    assert [e.event for e in emitted] == [FeedbackEventName.ASSESSMENT_UPDATED.value]


def test_high_risk_emits_risk_alert(tmp_path: Path):
    session = _session(tmp_path)
    emitted = []
    dispatcher = FeedbackDispatcher(
        session=session,
        logger=EventLogger(session.events_path),
        broadcast=emitted.append,
    )

    dispatcher.on_snapshot(_snapshot(risk_level="high"))

    assert [e.event for e in emitted] == [
        FeedbackEventName.ASSESSMENT_UPDATED.value,
        FeedbackEventName.RISK_ALERT.value,
    ]


def test_vlm_failure_emits_failure_and_pauses_session(tmp_path: Path):
    session = _session(tmp_path)
    emitted = []
    dispatcher = FeedbackDispatcher(
        session=session,
        logger=EventLogger(session.events_path),
        broadcast=emitted.append,
    )

    dispatcher.on_snapshot(_snapshot(vlm_meta={"error": "timeout"}))

    assert emitted[-1].event == FeedbackEventName.VLM_FAILURE.value
    assert session.state == SessionState.PAUSED
