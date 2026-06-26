"""H1 — RolloutRecord / RolloutMode / RolloutOutcome."""
import pytest

from robolineage_contracts.core import RolloutRecord, RolloutMode, RolloutOutcome


def test_rollout_mode_values():
    assert {m.value for m in RolloutMode} == {"A", "B1", "B2", "C1"}


def test_rollout_outcome_values():
    assert {o.value for o in RolloutOutcome} == {
        "success", "failure", "interrupted", "unknown",
    }


def _make(**overrides):
    base = dict(
        rollout_id="027b72ff-aaaa-bbbb-cccc-000000000001",
        task_id="task_98",
        mode=RolloutMode.B2,
        policy_version="1.2.0",
        operator_id="op-001",
        started_at="2026-04-25T10:00:00Z",
        ended_at="2026-04-25T10:05:00Z",
        outcome=RolloutOutcome.SUCCESS,
        intervention_count=0,
        storage_path="data/rollouts/027b72ff-aaaa-bbbb-cccc-000000000001",
    )
    base.update(overrides)
    return RolloutRecord(**base)


def test_rollout_record_constructs_minimal():
    r = _make()
    assert r.task_id == "task_98"
    assert r.mode == RolloutMode.B2


def test_rollout_record_is_frozen():
    r = _make()
    with pytest.raises(Exception):  # FrozenInstanceError
        r.task_id = "task_99"  # type: ignore[misc]


@pytest.mark.parametrize("mode", [RolloutMode.B1, RolloutMode.B2])
def test_rollout_record_policy_version_required_for_b_modes(mode):
    with pytest.raises(ValueError, match="policy_version"):
        _make(mode=mode, policy_version=None)


@pytest.mark.parametrize("mode", [RolloutMode.A, RolloutMode.C1])
def test_rollout_record_mode_a_c1_allow_null_policy_version(mode):
    r = _make(mode=mode, policy_version=None)
    assert r.policy_version is None


def test_rollout_record_rejects_negative_intervention_count():
    with pytest.raises(ValueError, match="intervention_count"):
        _make(intervention_count=-1)
