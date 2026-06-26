from __future__ import annotations

import logging

from .types import PhasePriorResult, TaskConfig

logger = logging.getLogger(__name__)

_DEFAULT_STRONG_PRIOR_MARGIN = 0.35   # compatibility name; now means action-assist margin
_DEFAULT_HIGH_VLM_CONFIDENCE = 0.80
_DEFAULT_PRIOR_STICKY_FRAMES = 2      # kept for config compatibility
_DEFAULT_LOW_VLM_CONFIDENCE = 0.35


class PhaseFusion:
    """
    Implementation note.

    Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.
    """

    def __init__(
        self,
        strong_prior_margin: float = _DEFAULT_STRONG_PRIOR_MARGIN,
        high_vlm_confidence: float = _DEFAULT_HIGH_VLM_CONFIDENCE,
        prior_sticky_frames: int = _DEFAULT_PRIOR_STICKY_FRAMES,
        low_vlm_confidence: float = _DEFAULT_LOW_VLM_CONFIDENCE,
    ):
        self.action_assist_margin = strong_prior_margin
        self.high_vlm_confidence = high_vlm_confidence
        self.prior_sticky_frames = max(1, prior_sticky_frames)
        self.low_vlm_confidence = low_vlm_confidence
        self._sticky_candidate: str | None = None
        self._sticky_count: int = 0

    def reset(self) -> None:
        """Reset sticky state — call when agent resets between rollouts."""
        self._sticky_candidate = None
        self._sticky_count = 0

    def fuse(
        self,
        prior: PhasePriorResult,
        vlm_phase: str,
        vlm_confidence: float,
        task_config: TaskConfig,
    ) -> tuple[str, str]:
        """
        Returns:
            (final_phase, fusion_reason)
        """
        phases = set(task_config.phases)
        has_action_cue = (
            prior.top_margin > 0.0
            and prior.prior_reason not in ("no_hints_uniform", "no_phases_defined")
            and prior.top_phase in phases
        )

        # Internal implementation note.
        if vlm_phase not in phases:
            if has_action_cue:
                return prior.top_phase, f"invalid_vlm_phase:action_fallback:{prior.top_phase}"
            return vlm_phase, "invalid_vlm_phase:no_action_fallback"

        # Internal implementation note.
        if not has_action_cue:
            self._clear_sticky()
            return vlm_phase, "no_action_cue:visual_used"

        # Internal implementation note.
        if prior.top_phase == vlm_phase:
            self._clear_sticky()
            return vlm_phase, f"action_confirms_visual:{vlm_phase}"

        # Internal implementation note.
        if vlm_confidence >= self.low_vlm_confidence:
            self._clear_sticky()
            logger.debug(
                "Fusion: visual overrides action cue  visual=%s conf=%.2f action=%s margin=%.3f",
                vlm_phase, vlm_confidence, prior.top_phase, prior.top_margin,
            )
            return (
                vlm_phase,
                f"visual_primary:{vlm_phase}(conf={vlm_confidence:.2f})>action:{prior.top_phase}",
            )

        # Internal implementation note.
        if prior.top_margin >= self.action_assist_margin:
            self._clear_sticky()
            return (
                prior.top_phase,
                f"low_conf_visual_action_assist:{prior.top_phase}"
                f"(margin={prior.top_margin:.3f})>visual:{vlm_phase}(conf={vlm_confidence:.2f})",
            )

        self._clear_sticky()
        return vlm_phase, f"weak_action_cue:visual_used:{vlm_phase}(action={prior.top_phase})"

    def _clear_sticky(self) -> None:
        self._sticky_candidate = None
        self._sticky_count = 0
