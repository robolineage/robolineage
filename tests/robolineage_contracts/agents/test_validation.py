"""H2 — ValidationIssue snake_case code + severity enum."""
import pytest

from robolineage_contracts.agents import ValidationIssue


def test_validation_issue_basic():
    i = ValidationIssue(severity="error", code="missing_file", message="raw_manifest.json not found")
    assert i.severity == "error"
    assert i.code == "missing_file"
    assert i.message == "raw_manifest.json not found"


def test_validation_issue_warning_severity():
    ValidationIssue(severity="warning", code="left_pose_time_non_monotonic", message="...")


@pytest.mark.parametrize("bad_severity", ["FATAL", "info", "ERROR", "Error", ""])
def test_validation_issue_rejects_other_severities(bad_severity):
    with pytest.raises(ValueError, match="severity"):
        ValidationIssue(severity=bad_severity, code="x", message="y")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_code",
    ["MissingFile", "missing-file", "missing.file", "Missing_File",
     "1missing", "_missing", "missing__file_double_underscore_ok",
     "schema:Type"],  # sub-id must be lowercase too
)
def test_validation_issue_rejects_non_snake_case_code(bad_code):
    if bad_code == "missing__file_double_underscore_ok":
        # this one is actually OK; double-underscore is allowed by [a-z0-9_]+
        ValidationIssue(severity="error", code=bad_code, message="x")
        return
    with pytest.raises(ValueError, match="snake_case"):
        ValidationIssue(severity="error", code=bad_code, message="x")


@pytest.mark.parametrize("good_code", [
    "missing_file",
    "pose_width_mismatch",
    "left_pose_time_non_monotonic",
    "schema:required",
    "schema:enum",
    "schema:additional-properties",
])
def test_validation_issue_accepts_canonical_codes(good_code):
    ValidationIssue(severity="error", code=good_code, message="x")


def test_validation_issue_is_frozen():
    i = ValidationIssue(severity="error", code="x", message="y")
    with pytest.raises(Exception):  # FrozenInstanceError
        i.severity = "warning"  # type: ignore[misc]
