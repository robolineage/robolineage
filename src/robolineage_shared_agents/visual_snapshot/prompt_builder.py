from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .types import (
    ActionGuidedWindow,
    PhasePriorResult,
    RolloutMemoryContext,
    TaskConfig,
    TaskMemory,
    VisualObservationWindow,
)


_LOG = logging.getLogger(__name__)


class PromptBuilder:
    SYSTEM_PROMPT = (
        "You are an embodied visual assessment agent for robot task execution monitoring. "
        "Analyze the provided keyframes as an ordered video snippet around a target timepoint. "
        "Use visual evidence as the primary evidence for task phase, object state, contact, placement, and failure risk. "
        "Use measured robot action signals only as auxiliary temporal evidence."
    )

    def build(
        self,
        window: VisualObservationWindow | ActionGuidedWindow,
        task_config: TaskConfig,
        task_memory_context: Optional[TaskMemory] = None,
        prior: Optional[PhasePriorResult] = None,
        rollout_context: Optional[RolloutMemoryContext] = None,
    ) -> tuple[str, list[np.ndarray]]:
        _ = prior  # action-derived phase scores are logged internally, not injected as a trusted prior.
        phases_str = ", ".join(f'"{phase}"' for phase in task_config.phases)

        phase_defs_section = ""
        if task_config.phase_definitions:
            lines = [f'  "{key}": {value}' for key, value in task_config.phase_definitions.items()]
            phase_defs_section = "\nPhase descriptions:\n" + "\n".join(lines)

        phase_visual_section = ""
        if task_config.phase_visual_hints:
            lines = [f'  "{key}": {value}' for key, value in task_config.phase_visual_hints.items()]
            phase_visual_section = "\nPhase visual signatures (what the scene looks like from the camera):\n" + "\n".join(lines)

        failure_section = ""
        if task_config.failure_signals:
            failure_section = "\nKnown failure signals: " + "; ".join(task_config.failure_signals)

        memory_section = ""
        if task_memory_context and task_memory_context.entries:
            recent_entries = task_memory_context.entries[-3:]
            pattern_lines: list[str] = []
            for entry in recent_entries:
                top = sorted(entry.failure_distribution.items(), key=lambda item: -item[1])[:2]
                if top:
                    pattern_lines.append(", ".join(f"{name}({count}x)" for name, count in top))
            if pattern_lines:
                memory_section = (
                    "\nHistorical failure context (background only, do not override current evidence):\n  "
                    + "; ".join(pattern_lines)
                )

        rollout_section = ""
        if rollout_context and rollout_context.summary:
            phase_first_seen = ", ".join(
                f"{phase}@{frame}"
                for phase, frame in rollout_context.phase_first_seen_frames.items()
            )
            confidence_lines = ", ".join(
                f"{phase}={confidence:.2f}"
                for phase, confidence in rollout_context.phase_confidence.items()
            )
            rollout_section = (
                "\nOnline rollout memory from previous assessments in this same episode "
                "(temporal context only):\n"
                f"- {rollout_context.summary}\n"
            )
            if phase_first_seen:
                rollout_section += f"- phase_first_seen_frames: {phase_first_seen}\n"
            if confidence_lines:
                rollout_section += f"- strongest_phase_confidence_so_far: {confidence_lines}\n"
            rollout_section += (
                "Use this history to keep the phase sequence coherent. "
                "Do not regress to an earlier phase unless the current images clearly show the task being undone, retried, or corrected."
            )

        timestamps = ", ".join(f"{timestamp:.2f}s" for timestamp in window.timestamps)
        camera_info = getattr(window, "camera_name", "camera_h") or "camera_h"

        action_section = ""
        if isinstance(window, ActionGuidedWindow):
            summary_lines = [f"- {key}: {value}" for key, value in window.action_summary.items()]
            detail_lines = [f"- {key}: {value}" for key, value in window.event_details.items()]
            action_section = (
                f"\nMeasured robot action evidence (auxiliary, not a phase prior):\n"
                f"- event_type: {window.event_type}\n"
                f"- anchor_frame: {window.anchor_frame_id}\n"
                f"- target_timestamp: {window.end_timestamp:.2f}s\n"
                f"- keyframe_ids: {window.keyframe_ids}\n"
            )
            if summary_lines:
                action_section += "\nAction summary:\n" + "\n".join(summary_lines)
            if detail_lines:
                action_section += "\nEvent details:\n" + "\n".join(detail_lines)
            action_section += (
                "\nInterpret action metadata as measured temporal signals: gripper open/close, "
                "EEF translation/rotation, stillness, and event timing. "
                "These signals can help disambiguate contact, lifting, transfer, release, retries, or stalls, "
                "but they must not override clear visual evidence. "
                "The phase label must describe the state at the anchor_frame / target_timestamp."
            )

        evidence_rules = """
Embodied evidence checklist:
- Identify the task-relevant object(s), end effector/tool, and target location/container/surface.
- Determine whether the end effector is approaching, contacting, grasping/holding, lifting, transferring, placing, releasing, retracting, or retrying.
- Prefer object state and spatial relation over raw gripper state when they conflict.
- Use EEF motion and gripper changes to refine timing of phase boundaries, not to blindly choose a phase.
- Use rollout history to maintain coherent temporal progress across this episode.
- If a phase was skipped because the sampled windows are sparse, choose the visually supported current phase rather than delaying the label.
"""

        prompt = f"""{self.SYSTEM_PROMPT}

Task: {task_config.task_description}
Defined phases (you MUST choose exactly one): [{phases_str}]{phase_defs_section}{phase_visual_section}{failure_section}{memory_section}{rollout_section}

You are observing keyframes from camera view(s): [{camera_info}] at timestamps: [{timestamps}].{action_section}
{evidence_rules}

Assess the current robot task execution. Output ONLY a valid JSON object with these exact fields:

{{
  "phase": <one of [{phases_str}]>,
  "progress": <one of ["advancing", "stalled", "regressing", "unknown"]>,
  "risk_level": <one of ["low", "medium", "high", "unknown"]>,
  "imminent_failure": <boolean>,
  "needs_review": <boolean>,
  "confidence": <float 0.0 to 1.0>
}}

Field rules:
- "phase": MUST be exactly one value from the provided list. Do not invent or rename phases.
- The chosen "phase" must describe the robot task at the anchor_frame / target_timestamp, which is the decision point for this window.
- Read the keyframes in temporal order as a short video segment. Use earlier frames as context, but do not let an earlier phase dominate the answer if the transition to a later phase has already started by the target_timestamp.
- If the first half of the window looks like an earlier phase and the later frames show a clear transition into the next phase, choose the later phase.
- Prefer a temporally reasonable phase boundary: once the object/robot behavior is already consistent with the next phase near the target_timestamp, output the next phase rather than extending the previous phase too long.
- "progress": advancing=task moving forward, stalled=no meaningful progress, regressing=moving backward or undoing progress, unknown=cannot determine.
- "risk_level": low=proceeding safely, medium=some concern, high=likely to fail soon, unknown=cannot determine.
- "imminent_failure": true only when the current visual/action evidence suggests failure is likely very soon.
- "needs_review": true when the evidence is ambiguous, low confidence, or risk is high enough that a human should inspect the snapshot.
- "confidence": overall confidence in this assessment.
- Output ONLY the JSON object. No preamble, no explanation, no markdown fences."""

        images = [frame for frame in window.color_frames if frame is not None]
        if not images:
            images = self._load_images_from_paths(
                getattr(window, "keyframe_image_paths", [])
            )
        max_images = 6
        if len(images) > max_images:
            step = len(images) / max_images
            images = [images[int(i * step)] for i in range(max_images)]

        return prompt, images

    @staticmethod
    def _load_images_from_paths(paths: list[str]) -> list[np.ndarray]:
        if not paths:
            return []
        try:
            import cv2
        except Exception:
            _LOG.exception("Failed to import cv2 for persisted keyframe loading")
            return []

        images: list[np.ndarray] = []
        for raw_path in paths:
            path = Path(raw_path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                _LOG.warning("Persisted VSA keyframe is unreadable: %s", path)
                continue
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return images
