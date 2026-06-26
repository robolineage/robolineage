from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ADVANCING_EVENTS = frozenset({"gripper_close", "gripper_open", "motion_resume"})
_STALLED_EVENTS = frozenset({"still_start"})
_PERIODIC_STILL_THRESHOLD = 0.6
_PERIODIC_MOTION_THRESHOLD = 8e-4


class ProgressDeriver:
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
    """

    def derive(
        self,
        event_type: str,
        fused_phase: str,
        last_phase: Optional[str],
        action_summary: dict,
        vlm_progress: str,
    ) -> tuple[str, str]:
        """
        Returns:
            (progress, derive_reason)
        """
        # Internal implementation note.
        if last_phase is not None and fused_phase != last_phase:
            return "advancing", f"phase_changed:{last_phase}->{fused_phase}"

        # Internal implementation note.
        if event_type in _ADVANCING_EVENTS:
            return "advancing", f"strong_event:{event_type}"

        # Internal implementation note.
        if event_type == "sequence_start" and last_phase is None:
            return "advancing", "sequence_start_initial"

        # Internal implementation note.
        if event_type in _STALLED_EVENTS:
            return "stalled", f"still_event:{event_type}"

        # Internal implementation note.
        if event_type in {"heartbeat", "periodic_sample"}:
            still_fraction = float(action_summary.get("still_fraction", 0.0))
            mean_motion = float(action_summary.get("mean_motion_energy", 0.0))
            if still_fraction > _PERIODIC_STILL_THRESHOLD:
                return "stalled", f"{event_type}_still:fraction={still_fraction:.2f}"
            if mean_motion > _PERIODIC_MOTION_THRESHOLD:
                return "advancing", f"{event_type}_moving:motion={mean_motion:.5f}"
            # Internal implementation note.
            return vlm_progress, f"{event_type}_fallback_vlm:fraction={still_fraction:.2f}"

        # Internal implementation note.
        return vlm_progress, f"fallback_vlm:{event_type}"
