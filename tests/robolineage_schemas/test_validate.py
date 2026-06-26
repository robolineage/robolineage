"""H1+H2 — robolineage_schemas.validate against the shared fixture pack.

Asserts the fixtures in tests/_shared_fixtures/ pass schema validation with
zero ERRORs. If a future PR breaks one of these, either the fixture is wrong
or the schema is wrong — fix one, then bump CONTRACTS_VERSION accordingly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robolineage_schemas import validate, load_schema


FIXTURES = Path(__file__).resolve().parents[1] / "_shared_fixtures"


# ── load_schema ──────────────────────────────────────────────────────────

def test_load_schema_metadata():
    s = load_schema("metadata")
    assert s["title"] == "RoboLineage Sample Metadata"


def test_load_schema_snapshot():
    s = load_schema("snapshot")
    assert s["title"] == "RoboLineage SnapshotAssessment"


def test_load_schema_dataset_lock():
    s = load_schema("dataset_lock")
    assert s["title"] == "RoboLineage DatasetLock"


def test_load_schema_policy_meta():
    s = load_schema("policy_meta")
    assert s["title"] == "RoboLineage PolicyMeta"


def test_load_lifecycle_artifact_schemas():
    names = [
        "annotation_final",
        "failure_analysis",
        "dataset_admission",
        "policy_evaluation",
        "deployment_decision",
        "collection_recommendation",
        "next_collection_brief",
        "training_status",
        "training_result",
    ]
    for name in names:
        assert load_schema(name)["type"] == "object"


def test_fixture_dataset_lock_validates_clean():
    raw = json.loads((FIXTURES / "dataset.lock").read_text())
    issues = validate(raw, "dataset_lock")
    errs = [i for i in issues if i.severity == "error"]
    assert errs == [], f"shared fixture dataset.lock has schema errors: {errs}"


def test_fixture_policy_meta_validates_clean():
    raw = json.loads((FIXTURES / "policy_meta.json").read_text())
    issues = validate(raw, "policy_meta")
    errs = [i for i in issues if i.severity == "error"]
    assert errs == [], f"shared fixture policy_meta.json has schema errors: {errs}"


def test_policy_meta_accepts_framework_provenance_fields():
    raw = json.loads((FIXTURES / "policy_meta.json").read_text())
    raw.update(
        {
            "framework_name": "host_repo",
            "framework_type": "generic",
            "adapter_version": "RoboLineage.framework_adapter.v1",
            "checkpoint_path": "checkpoints/1.0.0/policy.ckpt",
            "training_result_path": "training_runs/run_1/framework/training_result.json",
            "eval_result_path": "training_runs/run_1/framework/eval/result.json",
            "ROBOLINEAGE_context_path": "training_runs/run_1/policy.ROBOLINEAGE_context.json",
        }
    )
    issues = validate(raw, "policy_meta")
    errs = [i for i in issues if i.severity == "error"]
    assert errs == [], f"policy_meta framework provenance fields failed schema: {errs}"


def test_dataset_lock_rejects_invalid_sha():
    bad = {
        "version_id": "v1",
        "created_at": "2026-04-25T12:00:00Z",
        "base_version_id": None,
        "included_rollout_ids": [],
        "total_samples": 0,
        "manifest_sha256": "not-hex",
        "changelog": "",
    }
    issues = validate(bad, "dataset_lock")
    assert any(i.code.startswith("schema:") for i in issues)


def test_policy_meta_rejects_deployed_with_pending_gate():
    """The schema doesn't enforce the deployed↔pass invariant (that's
    enforced by the dataclass). But schema does check the enum values."""
    bad = {
        "version_id": "1.0.0",
        "trained_on_dataset": "v1",
        "architecture": "x",
        "training_steps": 0,
        "created_at": "2026-04-25T13:00:00Z",
        "eval_success_rate": None,
        "deployed": False,
        "deployment_gating_result": "INVALID",
    }
    issues = validate(bad, "policy_meta")
    assert any(i.code.startswith("schema:") for i in issues)


def test_load_schema_unknown_raises():
    with pytest.raises(FileNotFoundError, match="Schema not found"):
        load_schema("nope_nonexistent")


# ── metadata schema ──────────────────────────────────────────────────────

def test_fixture_metadata_validates_clean():
    raw = json.loads((FIXTURES / "mini_rollout" / "metadata.json").read_text())
    issues = validate(raw, "metadata")
    errs = [i for i in issues if i.severity == "error"]
    assert errs == [], f"shared fixture metadata has schema errors: {errs}"


def test_metadata_missing_required_field_caught():
    raw = json.loads((FIXTURES / "mini_rollout" / "metadata.json").read_text())
    del raw["alignment"]
    issues = validate(raw, "metadata")
    assert any(i.code.startswith("schema:") for i in issues)


def test_metadata_with_l1_validates():
    raw = json.loads((FIXTURES / "mini_rollout" / "metadata.json").read_text())
    raw["annotation"]["l1"] = {
        "schema_version": "1.0",
        "annotator": "agent:l1_prelabeler@0.3",
        "annotated_at": "2026-04-25T11:00:00Z",
        "phases": ["approach", "grasp", "transfer", "place"],
        "goal": "place cube into container",
        "success_criterion": {
            "type": "visual",
            "description": "cube fully inside container",
        },
        "phase_segments": [
            {"phase": "approach", "start_frame": 0, "end_frame": 9, "start_ts": 0.0, "end_ts": 0.3},
            {"phase": "grasp", "start_frame": 10, "end_frame": 17, "start_ts": 0.333, "end_ts": 0.566},
            {"phase": "transfer", "start_frame": 18, "end_frame": 24, "start_ts": 0.6, "end_ts": 0.8},
            {"phase": "place", "start_frame": 25, "end_frame": 29, "start_ts": 0.833, "end_ts": 0.966},
        ],
    }
    issues = validate(raw, "metadata")
    errs = [i for i in issues if i.severity == "error"]
    assert errs == [], f"metadata + l1 has schema errors: {errs}"


# ── snapshot schema ──────────────────────────────────────────────────────

def test_all_snapshot_jsonl_lines_validate():
    path = FIXTURES / "snapshots.jsonl"
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 10, "fixture should contain exactly 10 snapshot records"
    for i, line in enumerate(lines):
        s = json.loads(line)
        issues = validate(s, "snapshot")
        errs = [x for x in issues if x.severity == "error"]
        assert errs == [], f"snapshots.jsonl line {i} invalid: {errs}"


def test_snapshot_with_invalid_progress_caught():
    bad = {
        "timestamp": 0.0,
        "frame_id": 0,
        "progress": "going_well",
        "risk_level": "low",
        "phase": "x",
        "imminent_failure": False,
        "confidence": 0.5,
        "needs_review": False,
        "raw_response": "x",
    }
    issues = validate(bad, "snapshot")
    assert any(i.code.startswith("schema:") for i in issues)


def test_snapshot_confidence_out_of_range_caught():
    bad = {
        "timestamp": 0.0,
        "frame_id": 0,
        "progress": "advancing",
        "risk_level": "low",
        "phase": "x",
        "imminent_failure": False,
        "confidence": 1.5,
        "needs_review": False,
        "raw_response": "x",
    }
    issues = validate(bad, "snapshot")
    assert any(i.code.startswith("schema:") for i in issues)


# ── path / structure issues ──────────────────────────────────────────────

def test_validation_issue_path_is_human_readable():
    bad = {"exportId": "x", "project": {"id": "not_int", "name": "y"}}
    issues = validate(bad, "metadata")
    paths = [i.message.split(":", 1)[0] for i in issues]
    # at least one issue should mention either project/id or a top-level required key
    assert any("project" in p or p == "<root>" for p in paths)
