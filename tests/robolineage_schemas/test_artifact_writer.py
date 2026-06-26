from __future__ import annotations

import json

import pytest

from robolineage_schemas.artifacts import ArtifactValidationError, write_validated_json_atomic


def test_write_validated_json_atomic_rejects_invalid_payload(tmp_path):
    target = tmp_path / "dataset_admission.json"
    payload = {
        "schema_version": "post_review.admission.v1",
        "decision": "accepted",
    }

    with pytest.raises(ArtifactValidationError, match="dataset_admission") as exc:
        write_validated_json_atomic(target, payload, "dataset_admission")

    assert not target.exists()
    assert any(issue.code.startswith("schema:") for issue in exc.value.issues)


def test_write_validated_json_atomic_writes_valid_payload(tmp_path):
    target = tmp_path / "dataset_admission.json"
    payload = {
        "schema_version": "post_review.admission.v1",
        "agent_version": "test@1",
        "rollout_id": "rollout_a",
        "decision": "accepted",
        "accepted_for_training": True,
        "label_quality": "clean",
        "review_reason": None,
        "admission_class": "accepted_with_labels",
        "reasons": ["complete_success"],
        "data_use": ["train"],
        "recommended_split": "train",
        "requires_review": False,
        "created_at": "2026-05-16T00:00:00Z",
    }

    path = write_validated_json_atomic(target, payload, "dataset_admission")

    assert path == target
    assert json.loads(target.read_text(encoding="utf-8")) == payload
