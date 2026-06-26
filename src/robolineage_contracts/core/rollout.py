"""RolloutRecord contract for one collection or evaluation rollout.

Source of truth: docs/artifact_contracts.md §1.

A `RolloutRecord` is produced by the data source layer at
session-open time and accompanies every downstream artifact (snapshots,
P_rollout segments, dataset.lock entries, policy meta). The `rollout_id`
is the immutable correlation key for the full lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RolloutMode(str, Enum):
    """C_rollout collection mode (see docs/session_collection_design)."""
    A = "A"     # Generalisation collection — operator drives, no policy
    B1 = "B1"   # Trajectory-referenced correction — policy infers, no robot drive
    B2 = "B2"   # Deployment monitoring + intervention — policy drives, human takes over
    C1 = "C1"   # Basic demonstration (pre-policy bootstrap)


class RolloutOutcome(str, Enum):
    """Final disposition of a rollout, recorded at SUBMIT_ROLLOUT time."""
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RolloutRecord:
    """One RoboLineage rollout — covers a single START..SUBMIT session.

    Field semantics match docs/artifact_contracts.md §1 verbatim. Frozen so the
    record can be safely shared across threads / Plans without defensive
    copying.
    """
    rollout_id: str            # UUID (matches the on-disk directory name)
    task_id: str               # e.g. "task_98"
    mode: RolloutMode
    policy_version: str | None  # null ONLY for mode A or C1
    operator_id: str
    started_at: str            # ISO8601
    ended_at: str              # ISO8601
    outcome: RolloutOutcome
    intervention_count: int    # human-takeover count; only meaningful for B2
    storage_path: str          # path to rollout dir, relative to project root

    def __post_init__(self) -> None:
        # B1/B2 require a policy_version because the rollout is conditioned
        # on a specific policy snapshot. A and C1 are policy-free by design.
        if self.mode in (RolloutMode.B1, RolloutMode.B2) and self.policy_version is None:
            raise ValueError(
                f"policy_version is required for mode={self.mode.value}; got None"
            )
        if self.intervention_count < 0:
            raise ValueError(
                f"intervention_count must be ≥0; got {self.intervention_count}"
            )
