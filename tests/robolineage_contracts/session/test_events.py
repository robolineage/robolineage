"""H5 — Control / Feedback enums + ErrorCode + EventEnvelope construction."""
import json
from pathlib import Path

import pytest

from robolineage_contracts.session import (
    ControlEventName,
    ErrorCode,
    EventEnvelope,
    EventSource,
    FeedbackEventName,
)


FIXTURES = Path(__file__).resolve().parents[2] / "_shared_fixtures"


# ── Enum coverage (locked to the session event contract) ──────────────

def test_control_events_6_values():
    assert {e.value for e in ControlEventName} == {
        "START_COLLECTING",
        "PAUSE_COLLECTING",
        "RESUME_COLLECTING",
        "STOP_COLLECTING",
        "SUBMIT_ROLLOUT",
        "DISCARD_ROLLOUT",
    }


def test_feedback_events_6_values():
    assert {e.value for e in FeedbackEventName} == {
        "SESSION_OPENED",
        "SESSION_CLOSED",
        "FRAME_DROPPED",
        "ASSESSMENT_UPDATED",
        "RISK_ALERT",
        "VLM_FAILURE",
    }


def test_event_source_4_values():
    assert {s.value for s in EventSource} == {"ux", "ar", "policy", "vsa"}


def test_error_code_4_values():
    assert {e.value for e in ErrorCode} == {
        "E_SESSION_OPEN_FAILED",
        "E_SESSION_CLOSE_FAILED",
        "E_ILLEGAL_STATE_TRANSITION",
        "E_VLM_TIMEOUT",
    }


# ── Envelope construction ───────────────────────────────────────────────

def _env(**overrides):
    base = dict(
        event=ControlEventName.START_COLLECTING.value,
        event_id="11111111-2222-3333-4444-555555555555",
        timestamp="2026-04-25T10:00:00.123Z",
        rollout_id=None,
        source=EventSource.UX,
        payload={"task_id": "task_98", "mode": "B1", "operator_id": "op01"},
    )
    base.update(overrides)
    return EventEnvelope(**base)


def test_envelope_minimal_construction():
    e = _env()
    assert e.event == "START_COLLECTING"
    assert e.source == EventSource.UX


def test_envelope_is_frozen():
    e = _env()
    with pytest.raises(Exception):  # FrozenInstanceError
        e.event_id = "x"  # type: ignore[misc]


def test_envelope_default_payload_is_empty_dict():
    e = EventEnvelope(
        event="X", event_id="id", timestamp="t", rollout_id=None, source=EventSource.AR,
    )
    assert e.payload == {}


def test_envelope_rejects_empty_event():
    with pytest.raises(ValueError, match="event must"):
        _env(event="")


def test_envelope_rejects_empty_event_id():
    with pytest.raises(ValueError, match="event_id"):
        _env(event_id="")


def test_envelope_rejects_empty_timestamp():
    with pytest.raises(ValueError, match="timestamp"):
        _env(timestamp="")


def test_envelope_rollout_id_can_be_none():
    """For events that fire before SESSION_OPENED — rollout_id is None."""
    e = _env(rollout_id=None)
    assert e.rollout_id is None


# ── Fixture round-trip ───────────────────────────────────────────────────

def test_events_jsonl_fixture_roundtrip():
    """Read tests/_shared_fixtures/mini_rollout/events.jsonl and verify each
    line constructs a valid EventEnvelope."""
    path = FIXTURES / "mini_rollout" / "events.jsonl"
    if not path.exists():
        pytest.skip(f"events.jsonl fixture not present: {path}")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 5, "events.jsonl should have at least 5 envelopes"
    for i, line in enumerate(lines):
        d = json.loads(line)
        env = EventEnvelope(
            event=d["event"],
            event_id=d["event_id"],
            timestamp=d["timestamp"],
            rollout_id=d.get("rollout_id"),
            source=EventSource(d["source"]),
            payload=d.get("payload", {}),
        )
        assert env.event_id, f"line {i} missing event_id"
