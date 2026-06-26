"""H4 — TrainManifestEntry / ReviewManifestEntry / RejectManifestEntry."""
import pytest

from robolineage_contracts.pipeline import (
    RejectManifestEntry,
    ReviewManifestEntry,
    TrainManifestEntry,
)


def test_train_entry_minimal():
    e = TrainManifestEntry(
        export_id="027b72ff", rollout_id="027b72ff",
        sample_dir="data/rollouts/027b72ff",
        review_score="A", confidence=0.9,
    )
    assert e.l1_phases is None
    assert e.reasons == ()


def test_train_entry_with_l1_phases_tuple():
    e = TrainManifestEntry(
        export_id="x", rollout_id="x", sample_dir="d", review_score="S",
        confidence=1.0,
        l1_phases=("approach", "grasp", "transfer", "place"),
    )
    assert e.l1_phases == ("approach", "grasp", "transfer", "place")


def test_train_entry_rejects_confidence_out_of_range():
    with pytest.raises(ValueError, match="confidence"):
        TrainManifestEntry(
            export_id="x", rollout_id="x", sample_dir="d", review_score="A",
            confidence=1.1,
        )


def test_review_entry_required_issues_tuple():
    e = ReviewManifestEntry(
        export_id="x", rollout_id="x", sample_dir="d",
        review_score="B",
        issues=("low_confidence",),
        confidence=0.6,
    )
    assert e.issues == ("low_confidence",)


def test_review_entry_rejects_confidence_out_of_range():
    with pytest.raises(ValueError, match="confidence"):
        ReviewManifestEntry(
            export_id="x", rollout_id="x", sample_dir="d", review_score="B",
            issues=(), confidence=-0.1,
        )


def test_reject_entry_minimal():
    e = RejectManifestEntry(
        export_id="x", rollout_id="x", sample_dir="d",
        reason="missing raw manifest",
        issues=("missing_file",),
    )
    assert e.reason == "missing raw manifest"


def test_all_entries_are_frozen():
    entries = [
        TrainManifestEntry(export_id="x", rollout_id="x", sample_dir="d", review_score="A", confidence=0.9),
        ReviewManifestEntry(export_id="x", rollout_id="x", sample_dir="d", review_score="B", issues=(), confidence=0.5),
        RejectManifestEntry(export_id="x", rollout_id="x", sample_dir="d", reason="r", issues=()),
    ]
    for e in entries:
        with pytest.raises(Exception):  # FrozenInstanceError
            e.export_id = "y"  # type: ignore[misc]
