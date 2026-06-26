"""PolicyMeta + CheckpointVersion — policy artifact metadata.

Source of truth: docs/artifact_contracts.md §2 (constraint 5) +
docs/artifact_contracts.md §14.

Producer: current T_update / training-framework lifecycle
(`src/robolineage_train/policy_meta.py`).
Consumer: policy evaluation, deployment governance, deployment systems and
check tooling.

**Critical invariant**: `trained_on_dataset` is bound to a single
`DatasetLock.version_id` at training time and **must never be edited**
afterwards. The current `robolineage_train.PolicyMetaWriter` enforces this by:
  1. Reading the DatasetLock fed into the trainer
  2. Constructing PolicyMeta with `trained_on_dataset = lock.version_id`
  3. Writing the file atomically + `chmod 0o444`
  4. Providing a `verify_integrity()` helper that re-reads and compares

If anyone manually edits the field on disk, `verify_integrity` returns a
`policy_source_dataset_mismatch` ValidationIssue and deployment is refused.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


GatingResult = Literal["pass", "fail", "pending"]


@dataclass(frozen=True)
class PolicyMeta:
    """The metadata file written next to each trained policy checkpoint.

    Stored at: checkpoints/<version_id>/policy.meta.json
    """
    version_id: str                       # SemVer "1.2.0"
    trained_on_dataset: str               # = DatasetLock.version_id, immutable
    architecture: str                     # "diffusion_policy" / "act_v2" / ...
    training_steps: int
    created_at: str                       # ISO8601
    eval_success_rate: Optional[float]    # None until evaluation completes
    deployed: bool
    deployment_gating_result: GatingResult
    framework_name: Optional[str] = None
    framework_type: Optional[str] = None
    adapter_version: Optional[str] = None
    checkpoint_path: Optional[str] = None
    training_result_path: Optional[str] = None
    eval_result_path: Optional[str] = None
    ROBOLINEAGE_context_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not _SEMVER_RE.match(self.version_id):
            raise ValueError(
                f"version_id must be SemVer (X.Y.Z); got {self.version_id!r}"
            )
        if self.eval_success_rate is not None and not (0.0 <= self.eval_success_rate <= 1.0):
            raise ValueError(
                f"eval_success_rate must be in [0.0, 1.0] or None; "
                f"got {self.eval_success_rate}"
            )
        if self.training_steps < 0:
            raise ValueError(f"training_steps must be ≥0; got {self.training_steps}")
        if self.deployment_gating_result not in ("pass", "fail", "pending"):
            raise ValueError(
                f"deployment_gating_result must be 'pass'|'fail'|'pending'; "
                f"got {self.deployment_gating_result!r}"
            )
        # A deployed policy must have passed the gate. The reverse isn't required
        # (you can pass the gate but not yet deploy).
        if self.deployed and self.deployment_gating_result != "pass":
            raise ValueError(
                "deployed=True requires deployment_gating_result='pass'; "
                f"got result={self.deployment_gating_result!r}"
            )


@dataclass(frozen=True)
class CheckpointVersion:
    """High-level checkpoint version (companion to PolicyMeta).

    Used by deployment / evaluation tooling that wants a quick summary
    without parsing the full PolicyMeta file.
    """
    version_id: str
    trained_on: str                        # = DatasetLock.version_id
    created_at: str
    architecture: str
    training_steps: int
    eval_success_rate: Optional[float]
    deployed: bool
    deployment_gating_result: GatingResult

    def __post_init__(self) -> None:
        if not _SEMVER_RE.match(self.version_id):
            raise ValueError(
                f"version_id must be SemVer; got {self.version_id!r}"
            )
        if self.training_steps < 0:
            raise ValueError(f"training_steps must be ≥0; got {self.training_steps}")
