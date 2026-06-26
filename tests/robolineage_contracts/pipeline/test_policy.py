"""H4 — PolicyMeta SemVer + source-dataset binding invariants."""
import json
from pathlib import Path

import pytest

from robolineage_contracts.pipeline import PolicyMeta


FIXTURES = Path(__file__).resolve().parents[2] / "_shared_fixtures"


def _meta(**overrides):
    base = dict(
        version_id="1.0.0",
        trained_on_dataset="v1",
        architecture="diffusion_policy",
        training_steps=50000,
        created_at="2026-04-25T13:00:00Z",
        eval_success_rate=None,
        deployed=False,
        deployment_gating_result="pending",
    )
    base.update(overrides)
    return PolicyMeta(**base)


# ── Construction + invariants ────────────────────────────────────────────

def test_minimal_construction():
    m = _meta()
    assert m.version_id == "1.0.0"
    assert m.deployed is False


def test_is_frozen():
    m = _meta()
    with pytest.raises(Exception):  # FrozenInstanceError
        m.deployed = True  # type: ignore[misc]


@pytest.mark.parametrize("bad_version", ["1.0", "v1.0.0", "1.0.0a1", "1", "1.0.0.0", ""])
def test_rejects_non_semver_version_id(bad_version):
    with pytest.raises(ValueError, match="version_id"):
        _meta(version_id=bad_version)


def test_rejects_eval_success_rate_out_of_range():
    with pytest.raises(ValueError, match="eval_success_rate"):
        _meta(eval_success_rate=1.5)


def test_eval_success_rate_none_allowed():
    m = _meta(eval_success_rate=None)
    assert m.eval_success_rate is None


def test_rejects_negative_training_steps():
    with pytest.raises(ValueError, match="training_steps"):
        _meta(training_steps=-1)


@pytest.mark.parametrize("bad", ["passed", "failed", "PENDING", ""])
def test_rejects_invalid_gating_result(bad):
    with pytest.raises(ValueError, match="deployment_gating_result"):
        _meta(deployment_gating_result=bad)


# ── Critical invariant: deployed implies pass ────────────────────────────

def test_deployed_requires_gating_pass():
    """The most important PolicyMeta invariant: you can't deploy a policy that
    didn't pass the gate (per docs/artifact_contracts.md)."""
    with pytest.raises(ValueError, match="deployment_gating_result"):
        _meta(deployed=True, deployment_gating_result="pending")
    with pytest.raises(ValueError, match="deployment_gating_result"):
        _meta(deployed=True, deployment_gating_result="fail")


def test_deployed_with_gating_pass_is_valid():
    m = _meta(deployed=True, deployment_gating_result="pass")
    assert m.deployed is True


def test_passed_but_not_deployed_is_valid():
    """The reverse direction is fine — you can pass the gate but not yet deploy."""
    m = _meta(deployed=False, deployment_gating_result="pass")
    assert m.deployed is False


# ── Fixture round-trip ───────────────────────────────────────────────────

def test_fixture_policy_meta_loads():
    raw = json.loads((FIXTURES / "policy_meta.json").read_text())
    m = PolicyMeta(**raw)
    assert m.version_id == "1.0.0"
    assert m.trained_on_dataset == "v1"
    assert m.deployed is False
    assert m.deployment_gating_result == "pending"
