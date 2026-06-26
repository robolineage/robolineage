"""Core data structures shared across collection, post-review, dataset,
training and evaluation lifecycles.

Source of truth: docs/artifact_contracts.md.

In v0.1.0 (H1) this exposes:

    Rollout layer
      RolloutRecord, RolloutMode {A,B1,B2,C1}, RolloutOutcome

    Metadata layer (Pydantic v2 — mirrors doc/schemas/metadata.schema.json)
      MetadataModel, L1Annotation, Annotation, Review, SuccessCriterion,
      PhaseSegment, FrameTag, Subtask, Alignment, ReviewScore,
      validate_metadata_transition()
"""
from robolineage_contracts.core.rollout import RolloutRecord, RolloutMode, RolloutOutcome
from robolineage_contracts.core.metadata import (
    MetadataModel,
    L1Annotation,
    Annotation,
    Review,
    SuccessCriterion,
    PhaseSegment,
    FrameTag,
    Subtask,
    Alignment,
    ReviewScore,
    Project,
    TaskSpec,
    DataPackage,
    ExportedBy,
    validate_metadata_transition,
    TransitionStage,
)

__all__ = [
    # rollout
    "RolloutRecord", "RolloutMode", "RolloutOutcome",
    # metadata
    "MetadataModel", "L1Annotation", "Annotation", "Review",
    "SuccessCriterion", "PhaseSegment", "FrameTag", "Subtask", "Alignment",
    "ReviewScore", "Project", "TaskSpec", "DataPackage", "ExportedBy",
    "validate_metadata_transition", "TransitionStage",
]
