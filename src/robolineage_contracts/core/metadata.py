"""MetadataModel — Pydantic v2 mirror of doc/schemas/metadata.schema.json.

Source of truth: docs/artifact_contracts.md + docs/artifact_contracts.md.

Why Pydantic and not a frozen dataclass? metadata.json is the most-touched
contract in the project. The current production path keeps L1-style phase
annotation in post-review artifacts; `annotation.l1` remains here for legacy
metadata compatibility. Pydantic gives us:
  - JSON ↔ model round-trip with field validation in one step
  - Field-level constraints (regex, ge, min_length) without manual __post_init__
  - Stable error messages for invalid payloads

The same module also exposes `validate_metadata_transition(before, after,
stage)` — the runtime enforcement of metadata write-stage ownership rules.
This is the runtime counterpart to the doc-level rule and is retained for
legacy metadata writers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from robolineage_contracts.agents.validation import ValidationIssue


# ───────────────────────────────────────────────────────────────────────────
# Field-level type aliases
# ───────────────────────────────────────────────────────────────────────────

ReviewScore = Literal["S", "A", "B", "C", "D"]
"""Five-grade human review (see docs/artifact_contracts.md). S/A → train,
B → review, C/D → reject by default in training readiness logic."""


# ───────────────────────────────────────────────────────────────────────────
# Sub-models
# ───────────────────────────────────────────────────────────────────────────

class Project(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    name: str


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    name: str
    steps: Optional[str] = None  # legacy free-form description; phases now come from task_config/post-review


class Review(BaseModel):
    model_config = ConfigDict(extra="allow")
    score: ReviewScore
    comment: Optional[str] = None
    errorType: Optional[str] = None
    reviewedAt: str  # ISO8601


class SuccessCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["visual", "kinematic", "hybrid"]
    description: str
    visual_hints: Optional[List[str]] = None
    kinematic_hints: Optional[List[str]] = None


class PhaseSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_ts: float = Field(ge=0)
    end_ts: float = Field(ge=0)


class FrameTag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: int = Field(ge=0)
    tag: str
    note: Optional[str] = None


class Subtask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    phase_ids: List[str]


class L1Annotation(BaseModel):
    """Legacy metadata L1 lightweight annotation node.

    Three layers of granularity:
      - sample-level: phases, goal, success_criterion, object_roles, subtasks
      - segment-level: phase_segments (must cover the whole rollout)
      - frame-level (optional): frame_tags

    Current production L1 drafts live in annotation.final.json.l1_annotation.
    This node is preserved so legacy metadata files continue to validate.
    """
    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    annotator: str  # "user:<name>" | "agent:<name>@<v>" | combined with "+"
    annotated_at: str  # ISO8601
    phases: List[str] = Field(min_length=1)
    subtasks: Optional[List[Subtask]] = None
    goal: str = Field(min_length=1)
    success_criterion: SuccessCriterion
    object_roles: Optional[Dict[str, str]] = None
    phase_segments: List[PhaseSegment] = Field(min_length=1)
    frame_tags: Optional[List[FrameTag]] = None


class Annotation(BaseModel):
    model_config = ConfigDict(extra="allow")
    annotationId: str
    description: Dict[str, str]  # ISO language code → localised text
    review: Review
    l1: Optional[L1Annotation] = None
    # `l1_updated_at` is added by the L1 writer when l1 is filled.
    l1_updated_at: Optional[str] = None


class DataPackage(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str


class ExportedBy(BaseModel):
    model_config = ConfigDict(extra="allow")
    userId: int
    username: str


class Alignment(BaseModel):
    model_config = ConfigDict(extra="allow")
    referenceCam: Literal["camera_h", "camera_l", "camera_r"]
    totalFrames: int = Field(ge=0)
    timeAligned: bool
    method: Literal["frame-based", "timestamp-based"]
    note: str = ""
    frames_schema_version: str = Field(pattern=r"^\d+\.\d+$")


# ───────────────────────────────────────────────────────────────────────────
# Top-level model
# ───────────────────────────────────────────────────────────────────────────

class MetadataModel(BaseModel):
    """The whole metadata.json document.

    `extra="allow"` at the top level so unknown fields are preserved on
    round-trip (forward compat for fields added by future schema bumps).
    """
    model_config = ConfigDict(extra="allow")
    exportId: str  # UUID; must equal containing directory name
    project: Project
    task: TaskSpec
    annotation: Annotation
    dataPackage: DataPackage
    exportedAt: str
    exportedBy: ExportedBy
    alignment: Alignment


# ───────────────────────────────────────────────────────────────────────────
# Transition validator — legacy metadata write permissions
# ───────────────────────────────────────────────────────────────────────────
#
# Stage table:
#   "legacy_data_source_collect"  legacy data-source initial write or updates;
#                                 everything except annotation.l1 is writable
#   "legacy_data_source_close"    legacy data-source atomic close; same surface
#   "legacy_l1_writeback"         legacy L1 metadata writeback; may write only
#                                 annotation.l1 and annotation.l1_updated_at
#   "readonly"                    consumers must not change anything

TransitionStage = Literal[
    "legacy_data_source_collect",
    "legacy_data_source_close",
    "legacy_l1_writeback",
    "readonly",
]


def _diff_paths(
    before: dict | None,
    after: dict,
    *,
    prefix: str = "",
) -> list[str]:
    """Return dotted-path identifiers for every key whose value differs.

    Recurses into nested dicts. Lists are compared as opaque values (any
    change to a list — element added/removed/edited — emits the parent path
    once). This is intentional: L1's `phase_segments` and `frame_tags`
    are rewrite-the-whole-list semantics for legacy L1 annotations.
    """
    changed: list[str] = []
    if before is None:
        for k in after.keys():
            changed.append(f"{prefix}{k}" if not prefix else f"{prefix}.{k}")
        return changed

    all_keys = set(before.keys()) | set(after.keys())
    for k in all_keys:
        path = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        b, a = before.get(k), after.get(k)
        if isinstance(b, dict) and isinstance(a, dict):
            changed.extend(_diff_paths(b, a, prefix=path))
        elif b != a:
            changed.append(path)
    return changed


def validate_metadata_transition(
    before: dict | None,
    after: dict,
    *,
    stage: TransitionStage,
) -> List[ValidationIssue]:
    """Validate that a metadata.json write is legal for the given stage.

    `before` is None when metadata does not yet exist on disk (first write).
    Returns a list of ValidationIssue; empty list means the transition is
    OK. The caller is expected to refuse the write if any issue has
    severity == "error".

    Current production annotation lives in post-review artifacts. These stages
    are retained for legacy metadata writers and compatibility import tools.
    """
    issues: list[ValidationIssue] = []

    if stage == "readonly":
        changed = _diff_paths(before, after)
        if changed:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="illegal_readonly_write",
                    message=(
                        f"Stage={stage!r} forbids any change; "
                        f"changed paths: {changed}"
                    ),
                )
            )
        return issues

    changed = _diff_paths(before, after)

    if stage in ("legacy_data_source_collect", "legacy_data_source_close"):
        # data-source must not touch annotation.l1 or annotation.l1_updated_at
        l1_changed = [
            p for p in changed
            if p.startswith("annotation.l1") or p == "annotation.l1_updated_at"
        ]
        if l1_changed:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="illegal_l1_write_before_closed",
                    message=(
                        f"data-source may not write annotation.l1 (stage={stage!r}); "
                        f"changed paths: {l1_changed}"
                    ),
                )
            )
        return issues

    if stage == "legacy_l1_writeback":
        # Legacy L1 writeback may only touch annotation.l1 + annotation.l1_updated_at
        non_l1 = [
            p for p in changed
            if not p.startswith("annotation.l1") and p != "annotation.l1_updated_at"
        ]
        if non_l1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="illegal_non_l1_field_change",
                    message=(
                        "legacy L1 writeback may only write annotation.l1 + l1_updated_at; "
                        f"also changed: {non_l1}"
                    ),
                )
            )
        return issues

    # Should be unreachable given the Literal type; defensive.
    issues.append(
        ValidationIssue(
            severity="error",
            code="unknown_transition_stage",
            message=f"Unknown stage={stage!r}",
        )
    )
    return issues
