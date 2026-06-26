"""
Implementation note.

Implementation note.
    Implementation note.
    Implementation note.
"""

from __future__ import annotations

import logging

from .exceptions import ContractViolationError
from .types import (
    VALID_PROGRESS,
    VALID_RISK_LEVEL,
    SnapshotAssessment,
    SnapshotTrigger,
    TaskConfig,
)

logger = logging.getLogger(__name__)


class ContractValidator:
    """Implementation note."""

    def validate_and_build(
        self,
        parsed: dict,
        raw_response: str,
        frame_id: int,
        timestamp: float,
        task_config: TaskConfig,
        event_type: str | None = None,
        frame_index_range: tuple[int, int] | None = None,
        vlm_meta: dict | None = None,
    ) -> SnapshotAssessment:
        """
        Args:
            Implementation note.
            Implementation note.
            frame_id     : observation_window.end_frame_id
            timestamp    : observation_window.end_timestamp
            Implementation note.
            Implementation note.
            Implementation note.
            Implementation note.

        Returns:
            Implementation note.

        Raises:
            Implementation note.
        """
        # Internal implementation note.
        if frame_id is None or timestamp is None:
            raise ContractViolationError("Missing frame_id or timestamp — cannot build SnapshotAssessment.")
        if not raw_response:
            raise ContractViolationError("raw_response must not be empty.")

        # --- progress ---
        progress = str(parsed.get("progress", "unknown"))
        if progress not in VALID_PROGRESS:
            logger.warning(f"Invalid progress value '{progress}', correcting to 'unknown'.")
            progress = "unknown"

        # --- risk_level ---
        risk_level = str(parsed.get("risk_level", "unknown"))
        if risk_level not in VALID_RISK_LEVEL:
            logger.warning(f"Invalid risk_level value '{risk_level}', correcting to 'unknown'.")
            risk_level = "unknown"

        # --- phase ---
        phase = str(parsed.get("phase", ""))
        if phase not in task_config.phases:
            fallback = task_config.phases[0] if task_config.phases else "unknown"
            logger.warning(
                f"Phase '{phase}' not in task phases {task_config.phases}, "
                f"correcting to '{fallback}'."
            )
            phase = fallback

        # --- confidence ---
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        if not (0.0 <= confidence <= 1.0):
            confidence = max(0.0, min(1.0, confidence))

        imminent_failure = self._coerce_bool(parsed.get("imminent_failure"), default=False)
        needs_review = self._coerce_optional_bool(parsed.get("needs_review"))
        if needs_review is None:
            needs_review = confidence < 0.3 or risk_level == "high"

        trigger = self._coerce_trigger(event_type)

        return SnapshotAssessment(
            timestamp=timestamp,
            frame_id=frame_id,
            progress=progress,        # type: ignore[arg-type]
            risk_level=risk_level,    # type: ignore[arg-type]
            phase=phase,
            imminent_failure=imminent_failure,
            confidence=confidence,
            needs_review=needs_review,
            raw_response=raw_response,
            trigger=trigger,
            frame_index_range=frame_index_range,
            vlm_meta=vlm_meta,
        )

    @staticmethod
    def _coerce_bool(value: object, default: bool) -> bool:
        parsed = ContractValidator._coerce_optional_bool(value)
        return default if parsed is None else parsed

    @staticmethod
    def _coerce_optional_bool(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        return None

    @staticmethod
    def _coerce_trigger(event_type: str | None) -> SnapshotTrigger | None:
        if not event_type:
            return None
        try:
            return SnapshotTrigger(event_type)
        except ValueError:
            return None
