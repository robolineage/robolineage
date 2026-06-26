"""MetadataModel construction + validate_metadata_transition.

Covers legacy metadata write-stage ownership:
  legacy_data_source_collect / legacy_data_source_close
      → legacy data source may write everything except annotation.l1
  legacy_l1_writeback
      → legacy L1 may only write annotation.l1 (+ l1_updated_at)
  readonly                      → F/G/D must not change anything
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from robolineage_contracts.core import (
    L1Annotation,
    MetadataModel,
    PhaseSegment,
    SuccessCriterion,
    validate_metadata_transition,
)


# ── shared fixture (in-test factory; do NOT replace with file fixture so
#    these tests stay independent of tests/_shared_fixtures/) ─────────────

def _base_metadata() -> dict:
    return {
        "exportId": "027b72ff-aaaa-bbbb-cccc-000000000001",
        "project": {"id": 1, "name": "demo"},
        "task": {"id": 98, "name": "grasp_cube"},
        "annotation": {
            "annotationId": "ann-1",
            "description": {"zh-CN": "pick up the cube"},
            "review": {"score": "A", "reviewedAt": "2026-04-25T10:00:00Z"},
            "l1": None,
        },
        "dataPackage": {"id": "pkg-1", "name": "demo_pkg"},
        "exportedAt": "2026-04-25T10:00:00Z",
        "exportedBy": {"userId": 1, "username": "op01"},
        "alignment": {
            "referenceCam": "camera_h",
            "totalFrames": 30,
            "timeAligned": True,
            "method": "frame-based",
            "note": "",
            "frames_schema_version": "1.0",
        },
    }


def _full_l1() -> dict:
    return {
        "schema_version": "1.0",
        "annotator": "agent:l1_prelabeler@0.3",
        "annotated_at": "2026-04-25T10:10:00Z",
        "phases": ["grasp"],
        "goal": "grab cube",
        "success_criterion": {
            "type": "visual",
            "description": "cube held above table",
        },
        "phase_segments": [{
            "phase": "grasp",
            "start_frame": 0,
            "end_frame": 29,
            "start_ts": 0.0,
            "end_ts": 1.0,
        }],
    }


# ── MetadataModel parsing ────────────────────────────────────────────────

def test_metadata_model_parses_base():
    m = MetadataModel.model_validate(_base_metadata())
    assert m.annotation.review.score == "A"
    assert m.annotation.l1 is None
    assert m.alignment.referenceCam == "camera_h"


def test_metadata_model_parses_with_l1():
    raw = _base_metadata()
    raw["annotation"]["l1"] = _full_l1()
    m = MetadataModel.model_validate(raw)
    assert m.annotation.l1 is not None
    assert m.annotation.l1.phases == ["grasp"]
    assert isinstance(m.annotation.l1.phase_segments[0], PhaseSegment)
    assert isinstance(m.annotation.l1.success_criterion, SuccessCriterion)


def test_metadata_model_rejects_invalid_review_score():
    raw = _base_metadata()
    raw["annotation"]["review"]["score"] = "Z"
    with pytest.raises(Exception):  # pydantic.ValidationError
        MetadataModel.model_validate(raw)


def test_metadata_model_rejects_invalid_camera():
    raw = _base_metadata()
    raw["alignment"]["referenceCam"] = "camera_back"
    with pytest.raises(Exception):
        MetadataModel.model_validate(raw)


def test_metadata_model_round_trip_preserves_unknown_top_level_keys():
    """extra='allow' at top so future schema bumps don't drop unknown fields."""
    raw = _base_metadata()
    raw["unknownTopLevelField"] = "preserve me"
    m = MetadataModel.model_validate(raw)
    dumped = m.model_dump()
    assert dumped.get("unknownTopLevelField") == "preserve me"


def test_l1_annotation_rejects_empty_phases():
    bad = _full_l1()
    bad["phases"] = []
    with pytest.raises(Exception):
        L1Annotation.model_validate(bad)


def test_l1_annotation_rejects_empty_phase_segments():
    bad = _full_l1()
    bad["phase_segments"] = []
    with pytest.raises(Exception):
        L1Annotation.model_validate(bad)


# ── validate_metadata_transition: legacy data-source stages ──────────────

def test_transition_legacy_data_source_first_write_allowed():
    """before=None means first write under legacy_data_source_collect."""
    issues = validate_metadata_transition(
        before=None, after=_base_metadata(), stage="legacy_data_source_collect",
    )
    assert [i for i in issues if i.severity == "error"] == []


def test_transition_legacy_data_source_close_allowed():
    """legacy_data_source_close may rewrite alignment/outcome fields."""
    before = _base_metadata()
    after = deepcopy(before)
    after["alignment"]["totalFrames"] = 31  # late-arriving frame counted
    issues = validate_metadata_transition(before, after, stage="legacy_data_source_close")
    assert [i for i in issues if i.severity == "error"] == []


def test_transition_legacy_data_source_collect_rejects_l1_write():
    before = _base_metadata()
    after = deepcopy(before)
    after["annotation"]["l1"] = _full_l1()
    issues = validate_metadata_transition(before, after, stage="legacy_data_source_collect")
    errs = [i for i in issues if i.severity == "error"]
    assert errs, "data-source must not be allowed to write annotation.l1"
    assert errs[0].code == "illegal_l1_write_before_closed"


def test_transition_legacy_data_source_close_rejects_l1_write_too():
    before = _base_metadata()
    after = deepcopy(before)
    after["annotation"]["l1"] = _full_l1()
    issues = validate_metadata_transition(before, after, stage="legacy_data_source_close")
    assert any(i.code == "illegal_l1_write_before_closed" for i in issues)


def test_transition_legacy_data_source_rejects_l1_updated_at_write():
    """l1_updated_at is also L1 writer territory."""
    before = _base_metadata()
    after = deepcopy(before)
    after["annotation"]["l1_updated_at"] = "2026-04-25T11:00:00Z"
    issues = validate_metadata_transition(before, after, stage="legacy_data_source_collect")
    assert any(i.code == "illegal_l1_write_before_closed" for i in issues)


# ── validate_metadata_transition: legacy L1 writeback ───────────────────

def test_transition_legacy_l1_writeback_can_write_l1():
    before = _base_metadata()
    after = deepcopy(before)
    after["annotation"]["l1"] = _full_l1()
    after["annotation"]["l1_updated_at"] = "2026-04-25T11:00:00Z"
    issues = validate_metadata_transition(before, after, stage="legacy_l1_writeback")
    assert [i for i in issues if i.severity == "error"] == []


def test_transition_legacy_l1_writeback_rejects_non_l1_field_change():
    before = _base_metadata()
    after = deepcopy(before)
    after["alignment"]["totalFrames"] = 999
    issues = validate_metadata_transition(before, after, stage="legacy_l1_writeback")
    errs = [i for i in issues if i.severity == "error"]
    assert errs
    assert errs[0].code == "illegal_non_l1_field_change"


def test_transition_legacy_l1_writeback_rejects_changing_review_score():
    before = _base_metadata()
    after = deepcopy(before)
    after["annotation"]["review"]["score"] = "S"
    issues = validate_metadata_transition(before, after, stage="legacy_l1_writeback")
    assert any(i.code == "illegal_non_l1_field_change" for i in issues)


def test_transition_legacy_l1_writeback_no_change_is_fine():
    """Idempotent re-write of identical metadata: zero changes → zero issues."""
    before = _base_metadata()
    after = deepcopy(before)
    issues = validate_metadata_transition(before, after, stage="legacy_l1_writeback")
    assert issues == []


# ── validate_metadata_transition: readonly ───────────────────────────────

def test_transition_readonly_rejects_any_change():
    before = _base_metadata()
    after = deepcopy(before)
    after["alignment"]["note"] = "tampered"
    issues = validate_metadata_transition(before, after, stage="readonly")
    errs = [i for i in issues if i.severity == "error"]
    assert errs
    assert errs[0].code == "illegal_readonly_write"


def test_transition_readonly_allows_no_change():
    before = _base_metadata()
    after = deepcopy(before)
    issues = validate_metadata_transition(before, after, stage="readonly")
    assert issues == []
