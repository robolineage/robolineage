"""
Implementation note.

Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.
"""

from __future__ import annotations

import logging
from collections import deque

from .types import TaskConfig

logger = logging.getLogger(__name__)

# Internal implementation note.
_RISK_ORDER: dict[str, int] = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
_RISK_BY_VAL: dict[int, str] = {v: k for k, v in _RISK_ORDER.items()}


class TemporalStabilizer:
    """
    Implementation note.

    Args:
        Implementation note.
        Implementation note.
        Implementation note.
    """

    # Internal implementation note.
    _PHASE_RESET_EVENTS: frozenset[str] = frozenset({
        "gripper_close",
        "gripper_open",
        "motion_resume",
        "still_start",
    })
    _DIRECT_PHASE_CONFIDENCE = 0.60
    _BOUNDARY_EVENT_CONFIDENCE = 0.50
    _RISK_PHASE_CONFIDENCE = 0.45
    _REPEATED_PHASE_CONFIDENCE = 0.50

    def __init__(
        self,
        phase_window: int = 3,
        progress_window: int = 3,
        risk_down_steps: int = 3,
    ):
        self.phase_window = phase_window
        self.progress_window = progress_window
        self.risk_down_steps = risk_down_steps

        self._phase_history: deque[str] = deque(maxlen=phase_window)
        self._progress_history: deque[str] = deque(maxlen=progress_window)
        self._risk_history: deque[str] = deque(maxlen=risk_down_steps + 2)
        self._stable_risk: str = "unknown"
        self._phase_candidate: str | None = None
        self._phase_candidate_count: int = 0

    def stabilize(
        self,
        raw: dict,
        task_config: TaskConfig,
        event_type: str | None = None,
    ) -> dict:
        """
        Implementation note.

        Args:
            Implementation note.
            Implementation note.
            Implementation note.

        Returns:
            Implementation note.
        """
        # Internal implementation note.
        if event_type in self._PHASE_RESET_EVENTS:
            self._phase_history.clear()
            logger.debug("Phase history cleared on strong event: %s", event_type)

        result = dict(raw)

        # Internal implementation note.
        raw_phase = raw.get("phase", "")
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        if raw_phase == self._phase_candidate:
            self._phase_candidate_count += 1
        else:
            self._phase_candidate = raw_phase
            self._phase_candidate_count = 1

        if raw_phase in task_config.phases:
            self._phase_history.append(raw_phase)
        else:
            logger.warning(
                "VLM returned phase %r which is not in valid phases %s — discarding.",
                raw_phase,
                task_config.phases,
            )
        direct_threshold = self._direct_phase_threshold(raw, event_type)
        repeated_phase = (
            self._phase_candidate_count >= 2
            and confidence >= self._REPEATED_PHASE_CONFIDENCE
        )
        if (
            raw_phase in task_config.phases
            and (confidence >= direct_threshold or repeated_phase)
        ):
            # Internal implementation note.
            # Internal implementation note.
            stable_phase = raw_phase
            self._phase_history.clear()
            self._phase_history.append(raw_phase)
            self._phase_candidate = None
            self._phase_candidate_count = 0
        else:
            stable_phase = self._majority_vote(
                list(self._phase_history),
                default=task_config.phases[0] if task_config.phases else raw_phase,
                valid_set=set(task_config.phases),
            )
        result["phase"] = stable_phase

        # Internal implementation note.
        raw_progress = raw.get("progress", "unknown")
        self._progress_history.append(raw_progress)
        result["progress"] = self._majority_vote(list(self._progress_history), default=raw_progress)

        # Internal implementation note.
        raw_risk = raw.get("risk_level", "unknown")
        self._risk_history.append(raw_risk)
        result["risk_level"] = self._stabilize_risk(raw_risk)

        return result

    def reset(self) -> None:
        """Implementation note."""
        self._phase_history.clear()
        self._progress_history.clear()
        self._risk_history.clear()
        self._stable_risk = "unknown"
        self._phase_candidate = None
        self._phase_candidate_count = 0

    # ------------------------------------------------------------------
    # Internal implementation note.
    # ------------------------------------------------------------------

    @staticmethod
    def _direct_phase_threshold(raw: dict, event_type: str | None) -> float:
        threshold = TemporalStabilizer._DIRECT_PHASE_CONFIDENCE
        if event_type in TemporalStabilizer._PHASE_RESET_EVENTS:
            threshold = min(threshold, TemporalStabilizer._BOUNDARY_EVENT_CONFIDENCE)
        if raw.get("risk_level") == "high" or bool(raw.get("imminent_failure")):
            threshold = min(threshold, TemporalStabilizer._RISK_PHASE_CONFIDENCE)
        return threshold

    @staticmethod
    def _majority_vote(
        history: list[str],
        default: str,
        valid_set: set[str] | None = None,
    ) -> str:
        if not history:
            return default
        counts: dict[str, int] = {}
        for v in history:
            if valid_set is None or v in valid_set:
                counts[v] = counts.get(v, 0) + 1
        if not counts:
            return default
        return max(counts, key=counts.__getitem__)

    def _stabilize_risk(self, new_risk: str) -> str:
        new_val = _RISK_ORDER.get(new_risk, 0)
        current_val = _RISK_ORDER.get(self._stable_risk, 0)

        # Internal implementation note.
        if new_val > current_val:
            self._stable_risk = new_risk
            return self._stable_risk

        # Internal implementation note.
        if new_val < current_val:
            recent = list(self._risk_history)[-(self.risk_down_steps):]
            if (
                len(recent) >= self.risk_down_steps
                and all(_RISK_ORDER.get(r, 0) <= new_val for r in recent)
            ):
                self._stable_risk = new_risk
                logger.debug(f"Risk level decreased to {new_risk} after {self.risk_down_steps} confirmations.")

        return self._stable_risk
