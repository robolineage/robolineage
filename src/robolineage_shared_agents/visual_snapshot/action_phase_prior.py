from __future__ import annotations

import logging
import math
from typing import Optional

from .types import ActionGuidedWindow, PhasePriorResult, TaskConfig, VisualObservationWindow

logger = logging.getLogger(__name__)

_STILL_THRESHOLD = 3e-4
_SLOW_THRESHOLD = 1e-3
_Z_UP_THRESHOLD = 0.01
_Z_DOWN_THRESHOLD = -0.005
_XY_MOVE_THRESHOLD = 0.01


class ActionPhasePriorScorer:
    """
    Implementation note.

    Implementation note.
    Implementation note.

    Implementation note.
    Implementation note.

    Implementation note.
    Implementation note.

    Implementation note.
        Implementation note.
        Implementation note.
        Implementation note.
            Implementation note.
            Implementation note.
            Implementation note.
            Implementation note.
            Implementation note.
    """

    def score(
        self,
        window: ActionGuidedWindow | VisualObservationWindow,
        task_config: TaskConfig,
        last_phase: Optional[str] = None,
    ) -> PhasePriorResult:
        phases = task_config.phases
        if not phases:
            return PhasePriorResult(
                phase_scores={},
                top_phase="",
                top_margin=0.0,
                prior_reason="no_phases_defined",
            )

        if not isinstance(window, ActionGuidedWindow):
            uniform = 1.0 / len(phases)
            return PhasePriorResult(
                phase_scores={p: round(uniform, 4) for p in phases},
                top_phase=phases[0],
                top_margin=0.0,
                prior_reason="no_hints_uniform",
            )

        # Internal implementation note.
        event_type: str = window.action_summary.get("event_type", "")
        if event_type == "sequence_start" and last_phase is None:
            scores = {p: 0.0 for p in phases}
            scores[phases[0]] = 1.0
            return PhasePriorResult(
                phase_scores=scores,
                top_phase=phases[0],
                top_margin=1.0,
                prior_reason="sequence_start_force_first_phase",
            )

        hints = task_config.phase_action_hints
        if not hints:
            uniform = 1.0 / len(phases)
            return PhasePriorResult(
                phase_scores={p: round(uniform, 4) for p in phases},
                top_phase=phases[0],
                top_margin=0.0,
                prior_reason="no_hints_uniform",
            )

        features = self._extract_features(window)
        scores = self._score_phases(features, phases, hints, last_phase)

        sorted_phases = sorted(scores, key=scores.__getitem__, reverse=True)
        top_phase = sorted_phases[0]
        second_score = scores[sorted_phases[1]] if len(sorted_phases) > 1 else 0.0
        top_margin = round(scores[top_phase] - second_score, 4)
        reason = self._build_reason(features, top_phase, hints)

        logger.debug(
            "Phase prior: top=%s margin=%.3f scores=%s reason=%s",
            top_phase,
            top_margin,
            {p: round(v, 3) for p, v in scores.items()},
            reason,
        )

        return PhasePriorResult(
            phase_scores={p: round(v, 4) for p, v in scores.items()},
            top_phase=top_phase,
            top_margin=top_margin,
            prior_reason=reason,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(window: ActionGuidedWindow) -> dict:
        summary = window.action_summary
        pos_delta = summary.get("position_delta", (0.0, 0.0, 0.0))
        dx = float(pos_delta[0])
        dy = float(pos_delta[1])
        dz = float(pos_delta[2])
        return {
            "event_type": summary.get("event_type", ""),
            "gripper_state_before": summary.get("gripper_state_before", "open"),
            "gripper_state_after": summary.get("gripper_state_after", "open"),
            "mean_motion_energy": float(summary.get("mean_motion_energy", 0.0)),
            "still_fraction": float(summary.get("still_fraction", 0.0)),
            "delta_eef_z": dz,
            "delta_eef_xy": math.sqrt(dx ** 2 + dy ** 2),
            "delta_eef_xyz_norm": math.sqrt(dx ** 2 + dy ** 2 + dz ** 2),
        }

    @staticmethod
    def _score_phases(
        features: dict,
        phases: list[str],
        hints: dict[str, dict],
        last_phase: Optional[str] = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        event_type = features["event_type"]
        gripper_after = features["gripper_state_after"]
        still_fraction = features["still_fraction"]
        delta_z = features["delta_eef_z"]
        delta_xy = features["delta_eef_xy"]
        mean_motion = features["mean_motion_energy"]

        for phase in phases:
            hint = hints.get(phase, {})
            score = 0.0

            # Internal implementation note.
            hint_events = hint.get("event_type", [])
            if isinstance(hint_events, list) and event_type in hint_events:
                score += 2.0

            # Internal implementation note.
            hint_gripper = hint.get("gripper_state")
            if hint_gripper:
                if gripper_after == hint_gripper:
                    score += 1.0
                else:
                    score -= 0.5

            # Internal implementation note.
            motion_pattern = hint.get("motion_pattern", "")
            if motion_pattern == "z_up":
                if delta_z > _Z_UP_THRESHOLD:
                    score += 1.5
                elif delta_z < _Z_DOWN_THRESHOLD:
                    score -= 0.5
            elif motion_pattern == "xy_move":
                if delta_xy > _XY_MOVE_THRESHOLD and abs(delta_z) < delta_xy:
                    score += 1.5
            elif motion_pattern == "moving_to_object":
                if mean_motion > _SLOW_THRESHOLD:
                    score += 1.0
            elif motion_pattern == "contact_or_capture":
                if mean_motion < _SLOW_THRESHOLD:
                    score += 0.5
            elif motion_pattern == "release_or_settle":
                if still_fraction > 0.5 or event_type in ("still_start", "gripper_open"):
                    score += 1.0

            scores[phase] = max(0.0, score)

        # Internal implementation note.
        if event_type == "heartbeat" and last_phase is not None and last_phase in scores:
            scores[last_phase] += 0.5

        # Internal implementation note.
        if last_phase is None and len(phases) > 1:
            tail_start_idx = len(phases) // 2
            for i, phase in enumerate(phases):
                if i >= tail_start_idx:
                    scores[phase] = max(0.0, scores[phase] - 1.0)

        total = sum(scores.values())
        if total > 0:
            return {p: v / total for p, v in scores.items()}

        uniform = 1.0 / len(phases)
        return {p: uniform for p in phases}

    @staticmethod
    def _build_reason(features: dict, top_phase: str, hints: dict) -> str:
        hint = hints.get(top_phase, {})
        parts = [f"top={top_phase}"]

        if features["event_type"] in hint.get("event_type", []):
            parts.append(f"event={features['event_type']}")
        if hint.get("gripper_state") == features["gripper_state_after"]:
            parts.append(f"gripper={features['gripper_state_after']}")
        motion = hint.get("motion_pattern", "")
        if motion:
            parts.append(f"motion={motion}")

        return " ".join(parts)
