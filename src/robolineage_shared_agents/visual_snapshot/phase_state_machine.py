from __future__ import annotations

import logging

from .types import TaskConfig

logger = logging.getLogger(__name__)


class PhaseStateMachine:
    """
    Implementation note.

    Implementation note.
    Implementation note.
    Implementation note.

    Implementation note.
    """

    def __init__(
        self,
        allow_regression: bool = False,
        forward_jump_confidence: float = 0.60,
        event_forward_jump_confidence: float = 0.50,
        risk_forward_jump_confidence: float = 0.45,
        forward_jump_repeat: int = 2,
        terminal_forward_jump_confidence: float = 0.80,
        terminal_forward_jump_repeat: int = 2,
        correction_confidence: float = 0.82,
        correction_repeat: int = 2,
        retry_after_terminal_confidence: float = 0.70,
    ):
        self._current_phase: str | None = None
        self.allow_regression = allow_regression
        self.forward_jump_confidence = forward_jump_confidence
        self.event_forward_jump_confidence = event_forward_jump_confidence
        self.risk_forward_jump_confidence = risk_forward_jump_confidence
        self.forward_jump_repeat = max(1, forward_jump_repeat)
        self.terminal_forward_jump_confidence = terminal_forward_jump_confidence
        self.terminal_forward_jump_repeat = max(1, terminal_forward_jump_repeat)
        self.correction_confidence = correction_confidence
        self.correction_repeat = max(1, correction_repeat)
        self.retry_after_terminal_confidence = retry_after_terminal_confidence
        self.last_reason: str = "not_applied"
        self._forward_candidate: str | None = None
        self._forward_count: int = 0
        self._regression_candidate: str | None = None
        self._regression_count: int = 0

    def reset(self) -> None:
        self._current_phase = None
        self.last_reason = "reset"
        self._forward_candidate = None
        self._forward_count = 0
        self._regression_candidate = None
        self._regression_count = 0

    def apply(
        self,
        proposed_phase: str,
        task_config: TaskConfig,
        event_type: str | None = None,
        confidence: float | None = None,
        risk_level: str | None = None,
        imminent_failure: bool | None = None,
        progress: str | None = None,
    ) -> str:
        phases = task_config.phases
        if not phases:
            self.last_reason = "no_phases"
            return proposed_phase

        if self._current_phase is None:
            self._current_phase = proposed_phase if proposed_phase in phases else phases[0]
            self.last_reason = f"initialize:{self._current_phase}"
            return self._current_phase

        if proposed_phase not in phases:
            self.last_reason = f"invalid_proposed:{proposed_phase}:keep:{self._current_phase}"
            return self._current_phase

        current = self._current_phase
        if proposed_phase == current:
            self._clear_forward_candidate()
            self._clear_regression_candidate()
            self.last_reason = f"keep:{current}"
            return current

        current_index = phases.index(current)
        target_index = phases.index(proposed_phase)
        conf = 0.5 if confidence is None else float(confidence)

        # Internal implementation note.
        if target_index > current_index:
            self._clear_regression_candidate()
            if proposed_phase == self._forward_candidate:
                self._forward_count += 1
            else:
                self._forward_candidate = proposed_phase
                self._forward_count = 1
            allowed_forward = task_config.phase_transition_hint.get(current)
            distance = target_index - current_index
            boundary_event = event_type in {"gripper_close", "gripper_open", "motion_resume", "still_start"}
            risky_evidence = risk_level == "high" or bool(imminent_failure)
            terminal_index = len(phases) - 1
            early_terminal_skip = target_index == terminal_index and current_index < terminal_index - 1
            if early_terminal_skip and risky_evidence:
                self._current_phase = current
                self.last_reason = (
                    f"terminal_failure_hold:{proposed_phase}->keep:{current}"
                    f"(distance={distance}, conf={conf:.2f}, risk={risk_level}, "
                    f"imminent_failure={bool(imminent_failure)})"
                )
            elif allowed_forward and proposed_phase in allowed_forward:
                self._current_phase = proposed_phase
                self.last_reason = f"transition_hint:{current}->{proposed_phase}"
            elif target_index == current_index + 1:
                self._current_phase = proposed_phase
                self.last_reason = f"single_step:{current}->{proposed_phase}"
            elif early_terminal_skip:
                if (
                    self._forward_count >= self.terminal_forward_jump_repeat
                    and conf >= self.terminal_forward_jump_confidence
                ):
                    self._current_phase = proposed_phase
                    self.last_reason = (
                        f"repeated_terminal_jump:{current}->{proposed_phase}"
                        f"(conf={conf:.2f}, repeat={self._forward_count})"
                    )
                else:
                    self._current_phase = current
                    self.last_reason = (
                        f"terminal_skip_pending:{proposed_phase}->keep:{current}"
                        f"(distance={distance}, conf={conf:.2f}, "
                        f"repeat={self._forward_count}/{self.terminal_forward_jump_repeat})"
                    )
            elif conf >= self.forward_jump_confidence:
                self._current_phase = proposed_phase
                self.last_reason = (
                    f"visual_forward_jump:{current}->{proposed_phase}"
                    f"(conf={conf:.2f})"
                )
            elif risky_evidence and conf >= self.risk_forward_jump_confidence:
                self._current_phase = proposed_phase
                self.last_reason = (
                    f"risk_forward_jump:{current}->{proposed_phase}"
                    f"(conf={conf:.2f}, risk={risk_level}, imminent_failure={bool(imminent_failure)})"
                )
            elif boundary_event and conf >= self.event_forward_jump_confidence:
                self._current_phase = proposed_phase
                self.last_reason = (
                    f"event_forward_jump:{current}->{proposed_phase}"
                    f"(event={event_type}, conf={conf:.2f})"
                )
            elif self._forward_count >= self.forward_jump_repeat and conf >= self.event_forward_jump_confidence:
                self._current_phase = proposed_phase
                self.last_reason = (
                    f"repeated_forward_jump:{current}->{proposed_phase}"
                    f"(conf={conf:.2f}, repeat={self._forward_count})"
                )
            else:
                self._current_phase = current
                self.last_reason = (
                    f"forward_jump_pending:{proposed_phase}->keep:{current}"
                    f"(distance={distance}, conf={conf:.2f}, repeat={self._forward_count}/{self.forward_jump_repeat})"
                )
            if self._current_phase == proposed_phase:
                self._clear_forward_candidate()
            logger.debug(
                "Phase advance: %s -> %s (proposed=%s confidence=%.2f reason=%s)",
                current, self._current_phase, proposed_phase, conf, self.last_reason,
            )
            return self._current_phase

        # Internal implementation note.
        self._clear_forward_candidate()
        if self.allow_regression:
            # Regression-enabled mode: heartbeat/sequence_start keep the current phase.
            if event_type in {"sequence_start", "heartbeat"}:
                self.last_reason = f"regression_suppressed:{event_type}"
                return current
            self._current_phase = phases[max(current_index - 1, 0)]
            self.last_reason = f"regression_allowed:{current}->{self._current_phase}"
            logger.debug("Phase regression allowed: %s -> %s", current, self._current_phase)
        else:
            if proposed_phase == self._regression_candidate:
                self._regression_count += 1
            else:
                self._regression_candidate = proposed_phase
                self._regression_count = 1

            terminal_index = len(phases) - 1
            if current_index == terminal_index:
                retry_evidence = self._terminal_retry_evidence(
                    event_type=event_type,
                    risk_level=risk_level,
                    imminent_failure=imminent_failure,
                    progress=progress,
                    confidence=conf,
                )
                if conf >= self.retry_after_terminal_confidence and retry_evidence:
                    self._current_phase = proposed_phase
                    self.last_reason = (
                        f"retry_after_terminal:{current}->{proposed_phase}"
                        f"(event={event_type}, conf={conf:.2f}, "
                        f"evidence={','.join(retry_evidence)})"
                    )
                    self._clear_regression_candidate()
                    logger.debug("Phase retry after terminal allowed: %s", self.last_reason)
                    return self._current_phase
                self.last_reason = (
                    f"regression_pending:{proposed_phase}->keep:{current}"
                    f"(conf={conf:.2f}, repeat={self._regression_count}/{self.correction_repeat}, "
                    f"terminal_retry_evidence={','.join(retry_evidence) or 'none'})"
                )
                logger.debug("Terminal phase regression suppressed: %s", self.last_reason)
                return current

            repeated_correction = (
                self._regression_count >= self.correction_repeat
                and conf >= self.forward_jump_confidence
            )
            if conf >= self.correction_confidence or repeated_correction:
                self._current_phase = proposed_phase
                self.last_reason = (
                    f"visual_correction:{current}->{proposed_phase}"
                    f"(conf={conf:.2f}, repeat={self._regression_count})"
                )
                self._clear_regression_candidate()
                logger.debug("Phase correction allowed: %s", self.last_reason)
            else:
                self.last_reason = (
                    f"regression_pending:{proposed_phase}->keep:{current}"
                    f"(conf={conf:.2f}, repeat={self._regression_count}/{self.correction_repeat})"
                )
                logger.debug("Phase regression suppressed: %s", self.last_reason)

        return self._current_phase

    def _terminal_retry_evidence(
        self,
        *,
        event_type: str | None,
        risk_level: str | None,
        imminent_failure: bool | None,
        progress: str | None,
        confidence: float,
    ) -> list[str]:
        evidence: list[str] = []
        if risk_level == "high":
            evidence.append("high_risk")
        if bool(imminent_failure):
            evidence.append("imminent_failure")
        if progress == "regressing":
            evidence.append("progress_regressing")
        if event_type in {"gripper_burst", "contact_transition"}:
            evidence.append(f"composite_event:{event_type}")
        if (
            self._regression_count >= self.correction_repeat
            and confidence >= self.retry_after_terminal_confidence
        ):
            evidence.append("repeated_visual_regression")
        return evidence

    def _clear_forward_candidate(self) -> None:
        self._forward_candidate = None
        self._forward_count = 0

    def _clear_regression_candidate(self) -> None:
        self._regression_candidate = None
        self._regression_count = 0
