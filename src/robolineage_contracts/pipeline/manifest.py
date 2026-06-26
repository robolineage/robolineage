"""Train / review / reject manifest entry types.

Producer: current TrainingLifecycleRunner or compatibility manifest tooling.
Consumer: current `robolineage_train` lifecycle and external framework adapters;
human review tooling can still consume review/reject manifests.

Source of truth: doc/agent contract.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class TrainManifestEntry:
    """One row of train_manifest.jsonl."""
    export_id: str            # = metadata.exportId (UUID, matches dir name)
    rollout_id: str           # = metadata.exportId; kept separate for forward-compat
    sample_dir: str           # path relative to project root, e.g. data/rollouts/<uuid>
    review_score: str         # "S" | "A" | "B" | "C" | "D"
    confidence: float         # readiness/admission confidence
    l1_phases: Optional[Tuple[str, ...]] = None  # from post-review l1_annotation when present
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0]; got {self.confidence}")


@dataclass(frozen=True)
class ReviewManifestEntry:
    """One row of review_manifest.jsonl. Needs human follow-up before promotion."""
    export_id: str
    rollout_id: str
    sample_dir: str
    review_score: str
    issues: Tuple[str, ...]   # textual issues + warning codes from validation
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0]; got {self.confidence}")


@dataclass(frozen=True)
class RejectManifestEntry:
    """One row of reject_manifest.jsonl. Excluded from any dataset version."""
    export_id: str
    rollout_id: str
    sample_dir: str
    reason: str               # the deciding reason (e.g. "missing raw manifest")
    issues: Tuple[str, ...]   # full issue list for traceability
