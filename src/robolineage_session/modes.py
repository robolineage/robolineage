"""Collection mode behavior for the session service."""
from __future__ import annotations

from dataclasses import dataclass

from robolineage_contracts.core import RolloutMode


@dataclass(frozen=True)
class ModeBehavior:
    policy_inference: bool
    drive_robot: bool


_BEHAVIORS: dict[RolloutMode, ModeBehavior] = {
    RolloutMode.A: ModeBehavior(policy_inference=False, drive_robot=False),
    RolloutMode.B1: ModeBehavior(policy_inference=True, drive_robot=False),
    RolloutMode.B2: ModeBehavior(policy_inference=True, drive_robot=True),
    RolloutMode.C1: ModeBehavior(policy_inference=False, drive_robot=False),
}


def behavior_for(mode: RolloutMode) -> ModeBehavior:
    return _BEHAVIORS[mode]


def trajectory_accepted(mode: RolloutMode) -> bool:
    return behavior_for(mode).policy_inference
