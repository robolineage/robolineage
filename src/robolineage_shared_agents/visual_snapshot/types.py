from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from robolineage_contracts.agents import (
    Progress,
    RiskLevel,
    SnapshotAssessment,
    SnapshotTrigger,
)

VALID_PROGRESS: frozenset[str] = frozenset({"advancing", "stalled", "regressing", "unknown"})
VALID_RISK_LEVEL: frozenset[str] = frozenset({"low", "medium", "high", "unknown"})

GripperState = Literal["open", "closed"]
GripperEdge = Literal["closing_edge", "opening_edge", "none"]


@dataclass
class TaskConfig:
    task_description: str
    phases: list[str]
    phase_definitions: dict[str, str] = field(default_factory=dict)
    success_signals: list[str] = field(default_factory=list)
    failure_signals: list[str] = field(default_factory=list)
    phase_transition_hint: dict[str, list[str]] = field(default_factory=dict)
    # Internal implementation note.
    # Internal implementation note.
    # Internal implementation note.
    phase_action_hints: dict[str, dict[str, object]] = field(default_factory=dict)
    # Internal implementation note.
    phase_visual_hints: dict[str, str] = field(default_factory=dict)


@dataclass
class PhasePriorResult:
    phase_scores: dict[str, float]
    top_phase: str
    top_margin: float
    prior_reason: str


@dataclass
class RolloutMemoryContext:
    """Compact per-rollout context injected into online VSA prompts.

    This is intentionally small: realtime VSA should remember enough recent
    history to reason like a human operator, without turning each prompt into a
    full post-rollout review.
    """

    last_confirmed_phase: Optional[str] = None
    recent_final_phases: list[str] = field(default_factory=list)
    recent_visual_phases: list[str] = field(default_factory=list)
    recent_events: list[str] = field(default_factory=list)
    phase_first_seen_frames: dict[str, int] = field(default_factory=dict)
    phase_confidence: dict[str, float] = field(default_factory=dict)
    summary: str = ""


@dataclass
class MemoryEntry:
    iteration_id: str
    policy_version: str
    timestamp: str
    failure_distribution: dict[str, int]
    phase_failure_rates: dict[str, float]
    recommended_mode: str
    operator_override: bool
    final_mode: str
    outcome_note: str = ""


@dataclass
class TaskMemory:
    task_id: str
    entries: list[MemoryEntry] = field(default_factory=list)


@dataclass
class VisualObservationWindow:
    rollout_id: str
    frame_ids: list[int]
    timestamps: list[float]
    camera_name: str
    color_frames: list[Optional[np.ndarray]]
    depth_frames: list[Optional[np.ndarray]]
    end_frame_id: int
    end_timestamp: float


@dataclass
class FrameActionRecord:
    episode: str
    frame_index: int
    timestamp_sec: float
    mp4_file: str
    hdf5_file: str
    eef_xyz: tuple[float, float, float]
    eef_rxyz: tuple[float, float, float]
    gripper: float


@dataclass
class ActionDerivedSignal:
    frame_index: int
    timestamp_sec: float
    gripper_state: GripperState
    gripper_edge: GripperEdge
    translation_speed: float
    rotation_speed: float
    motion_energy: float
    motion_energy_avg: float
    is_still: bool


@dataclass
class ActionEvent:
    event_type: str
    anchor_frame: int
    timestamp_sec: float
    confidence: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class ActionGuidedWindow:
    rollout_id: str
    frame_ids: list[int]
    timestamps: list[float]
    camera_name: str
    color_frames: list[Optional[np.ndarray]]
    depth_frames: list[Optional[np.ndarray]]
    end_frame_id: int
    end_timestamp: float
    anchor_frame_id: int
    event_type: str
    keyframe_ids: list[int]
    action_summary: dict[str, object] = field(default_factory=dict)
    event_details: dict[str, object] = field(default_factory=dict)
    source_video_name: str = ""
    keyframe_image_paths: list[str] = field(default_factory=list)
    keyframe_window_id: str = ""
