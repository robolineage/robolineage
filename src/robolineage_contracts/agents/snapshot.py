"""SnapshotAssessment — Visual Snapshot Agent output contract.

Source of truth: docs/artifact_contracts.md §5 + docs/visual_snapshot_agent_design §4.

`VisualSnapshotAgent.process_window()` returns this type. Current consumers
are the session UI feedback path, PostRolloutReview and PolicyEvaluation.
Historical consumers still share the same contract for compatibility.

The four required core fields (`progress / risk_level / phase /
imminent_failure`) match the §5 contract verbatim. The three optional
realtime-only fields (`trigger / frame_index_range / vlm_meta`) carry
streaming-pipeline metadata that batch / offline runs may leave unset —
they default to None so older offline records still round-trip cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


# ── Type aliases (re-used by other contracts that consume snapshots) ─────

Progress = Literal["advancing", "stalled", "regressing", "unknown"]
RiskLevel = Literal["low", "medium", "high", "unknown"]


# ── Trigger enum (event-driven schedule from VSA realtime design §4) ────────

class SnapshotTrigger(str, Enum):
    """Why this snapshot was emitted (which event the VSA pipeline anchored on)."""
    SEQUENCE_START = "sequence_start"  # first action arrived → initial assessment
    GRIPPER_CLOSE = "gripper_close"    # gripper crossed close threshold
    GRIPPER_OPEN = "gripper_open"      # gripper crossed open threshold
    GRIPPER_BURST = "gripper_burst"    # close/open chatter coalesced into one observation
    CONTACT_TRANSITION = "contact_transition"  # contact-related edge burst
    STILL_START = "still_start"        # motion energy stayed below threshold
    MOTION_RESUME = "motion_resume"    # static segment ended
    PERIODIC_SAMPLE = "periodic_sample"  # fixed-interval VSA observation
    HEARTBEAT = "heartbeat"            # slow fallback / health observation
    FINAL_OBSERVATION = "final_observation"  # rollout stop -> terminal visual evidence


_VALID_PROGRESS = {"advancing", "stalled", "regressing", "unknown"}
_VALID_RISK = {"low", "medium", "high", "unknown"}


# ── Main contract ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SnapshotAssessment:
    """Output of one VSA inference call.

    Required fields (always present, both in offline and realtime modes):

      timestamp: float            seconds from rollout start
      frame_id: int               anchor frame in the source recording
      progress: Progress          task progression state
      risk_level: RiskLevel       current risk
      phase: str                  current phase (from task_config.phases or "unknown")
      imminent_failure: bool      VSA prediction of imminent failure (UX uses this for
                                  early-warning overlay)
      confidence: float           [0.0, 1.0]; <0.3 typically also flips needs_review
      needs_review: bool          flag for human review (low-confidence or anomalous)
      raw_response: str           raw VLM text, preserved for debugging

    Optional realtime-only enrichment (None in batch / offline records):

      trigger: SnapshotTrigger | None
                                  which event caused the snapshot
      frame_index_range: (int, int) | None
                                  source frame window used for this assessment
      vlm_meta: dict | None       provider-specific metadata (model, latency_ms,
                                  prompt_version, optional error, etc.). VSA writes:
                                    {"model": str,
                                     "latency_ms": int,
                                     "prompt_version": str,
                                     ["error"]: str,           # optional explicit failure
                                     ["tokens"]: int,          # if backend reports
                                    }
                                  Session dispatcher inspects vlm_meta.error when
                                  present; current realtime pipeline usually records
                                  backend failure as a low-confidence needs_review
                                  snapshot instead of setting error.
    """
    timestamp: float
    frame_id: int
    progress: Progress
    risk_level: RiskLevel
    phase: str
    imminent_failure: bool
    confidence: float
    needs_review: bool
    raw_response: str
    trigger: Optional[SnapshotTrigger] = None
    frame_index_range: Optional[tuple[int, int]] = None
    vlm_meta: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.progress not in _VALID_PROGRESS:
            raise ValueError(
                f"progress must be one of {sorted(_VALID_PROGRESS)}; "
                f"got {self.progress!r}"
            )
        if self.risk_level not in _VALID_RISK:
            raise ValueError(
                f"risk_level must be one of {sorted(_VALID_RISK)}; "
                f"got {self.risk_level!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0]; got {self.confidence}"
            )
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be ≥0; got {self.frame_id}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be ≥0; got {self.timestamp}")
        if self.frame_index_range is not None:
            lo, hi = self.frame_index_range
            if lo < 0 or hi < lo:
                raise ValueError(
                    f"frame_index_range must satisfy 0≤lo≤hi; got {(lo, hi)}"
                )
