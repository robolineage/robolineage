"""H2 — SnapshotAssessment construction + invariants."""
import pytest

from robolineage_contracts.agents import (
    SnapshotAssessment,
    SnapshotTrigger,
)


def _make(**overrides):
    base = dict(
        timestamp=1.5,
        frame_id=45,
        progress="advancing",
        risk_level="low",
        phase="grasp",
        imminent_failure=False,
        confidence=0.9,
        needs_review=False,
        raw_response="...",
    )
    base.update(overrides)
    return SnapshotAssessment(**base)


def test_minimal_construction():
    s = _make()
    assert s.frame_id == 45
    assert s.trigger is None
    assert s.frame_index_range is None
    assert s.vlm_meta is None


def test_full_realtime_construction():
    s = _make(
        trigger=SnapshotTrigger.GRIPPER_CLOSE,
        frame_index_range=(15, 45),
        vlm_meta={"model": "gpt-4o", "latency_ms": 1300, "prompt_version": "v1"},
    )
    assert s.trigger == SnapshotTrigger.GRIPPER_CLOSE
    assert s.frame_index_range == (15, 45)
    assert s.vlm_meta["model"] == "gpt-4o"


def test_is_frozen():
    s = _make()
    with pytest.raises(Exception):
        s.confidence = 0.5  # type: ignore[misc]


# ── progress / risk enum coverage ────────────────────────────────────────

@pytest.mark.parametrize("val", ["advancing", "stalled", "regressing", "unknown"])
def test_progress_accepted_values(val):
    _make(progress=val)


@pytest.mark.parametrize("val", ["advancing_well", "ADVANCING", "going", ""])
def test_progress_rejected_values(val):
    with pytest.raises(ValueError, match="progress"):
        _make(progress=val)


@pytest.mark.parametrize("val", ["low", "medium", "high", "unknown"])
def test_risk_accepted_values(val):
    _make(risk_level=val)


@pytest.mark.parametrize("val", ["LOW", "danger", ""])
def test_risk_rejected_values(val):
    with pytest.raises(ValueError, match="risk_level"):
        _make(risk_level=val)


# ── numeric / index invariants ───────────────────────────────────────────

@pytest.mark.parametrize("c", [-0.01, 1.01, 1.5, -1])
def test_confidence_out_of_range_rejected(c):
    with pytest.raises(ValueError, match="confidence"):
        _make(confidence=c)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_confidence_inclusive_bounds_accepted(c):
    _make(confidence=c)


def test_negative_frame_id_rejected():
    with pytest.raises(ValueError, match="frame_id"):
        _make(frame_id=-1)


def test_negative_timestamp_rejected():
    with pytest.raises(ValueError, match="timestamp"):
        _make(timestamp=-0.001)


def test_frame_index_range_invalid_order():
    with pytest.raises(ValueError, match="frame_index_range"):
        _make(frame_index_range=(10, 5))


def test_frame_index_range_negative():
    with pytest.raises(ValueError, match="frame_index_range"):
        _make(frame_index_range=(-1, 10))


def test_frame_index_range_equal_lo_hi_allowed():
    """Single-frame window is legal (e.g. heartbeat at exactly one anchor)."""
    s = _make(frame_index_range=(7, 7))
    assert s.frame_index_range == (7, 7)


# ── trigger enum ─────────────────────────────────────────────────────────

def test_trigger_enum_complete():
    assert {t.value for t in SnapshotTrigger} == {
        "sequence_start",
        "gripper_close",
        "gripper_open",
        "gripper_burst",
        "contact_transition",
        "still_start",
        "motion_resume",
        "periodic_sample",
        "heartbeat",
        "final_observation",
    }


def test_trigger_can_be_none_offline_mode():
    """Offline / batch-mode VSA records may omit trigger."""
    s = _make(trigger=None)
    assert s.trigger is None
