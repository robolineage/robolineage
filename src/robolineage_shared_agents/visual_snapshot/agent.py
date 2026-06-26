"""
Implementation note.

Implementation note.
    VisualObservationWindow
        -> PromptBuilder
        -> VLMRunner
        -> ResponseParser
        -> TemporalStabilizer
        -> ContractValidator
        -> SnapshotWriter
        -> SnapshotAssessment

Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.

Implementation note.
    Implementation note.
    Implementation note.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from .action_phase_prior import ActionPhasePriorScorer
from .contract_validator import ContractValidator
from .progress_deriver import ProgressDeriver
from .exceptions import (
    CollectionPausedError,
    ContractViolationError,
    RolloutClosedError,
    VLMInferenceError,
)
from .phase_fusion import PhaseFusion
from .phase_state_machine import PhaseStateMachine
from .prompt_builder import PromptBuilder
from .rollout_memory import PhaseDecisionRecord, RolloutMemory
from .response_parser import ResponseParser
from .snapshot_writer import SnapshotWriter
from .temporal_stabilizer import TemporalStabilizer
from .types import (
    ActionGuidedWindow,
    RolloutMemoryContext,
    SnapshotAssessment,
    TaskConfig,
    TaskMemory,
    VisualObservationWindow,
)
from .vlm_runner import BaseVLMRunner

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PreparedInference:
    """Implementation note."""
    window: VisualObservationWindow | ActionGuidedWindow
    prompt: str
    images: list
    prior: object                    # action-derived auxiliary evidence
    rollout_context: RolloutMemoryContext
    n_images: int
    # Internal implementation note.
    fallback_parsed: dict | None
    fallback_raw_response: str | None
    step_timings: dict[str, float]  # Internal implementation note.


class VisualSnapshotAgent:
    """
    Implementation note.

    Implementation note.
        agent = VisualSnapshotAgent(
            task_config=TaskConfig(task_description="...", phases=["grasp", "place"]),
            vlm_runner=OpenAIVLMRunner(),
            rollout_dir=Path("data/rollouts/abc123"),
        )
        assessment = agent.process_window(window)

    Implementation note.
        # Internal implementation note.
        agent.is_paused  # True
        # Internal implementation note.
        agent.resume()
        assessment = agent.process_window(next_window)

    Args:
        Implementation note.
        Implementation note.
        Implementation note.
        Implementation note.
        Implementation note.
        Implementation note.
        Implementation note.
        Implementation note.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        vlm_runner: BaseVLMRunner,
        rollout_dir: Path,
        output_jsonl: Path | None = None,
        task_memory_context: Optional[TaskMemory] = None,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        on_pause: Optional[Callable[[str], None]] = None,
        strong_prior_margin: float = 0.35,
        prior_sticky_frames: int = 2,
    ):
        self.task_config = task_config
        self.vlm_runner = vlm_runner
        self.rollout_dir = Path(rollout_dir)
        self.output_jsonl = Path(output_jsonl) if output_jsonl else self.rollout_dir / "snapshots.jsonl"
        self.task_memory_context = task_memory_context
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_pause = on_pause or (lambda msg: logger.error("[PAUSE ALERT] %s", msg))

        self._prompt_builder = PromptBuilder()
        self._prior_scorer = ActionPhasePriorScorer()
        self._phase_fusion = PhaseFusion(
            strong_prior_margin=strong_prior_margin,
            prior_sticky_frames=prior_sticky_frames,
        )
        self._progress_deriver = ProgressDeriver()
        self._response_parser = ResponseParser()
        self._stabilizer = TemporalStabilizer()
        self._phase_state_machine = PhaseStateMachine()
        self._rollout_memory = RolloutMemory()
        self._validator = ContractValidator()
        self._writer = SnapshotWriter(self.output_jsonl)

        self._paused = False
        self._last_phase: Optional[str] = None

        # Internal implementation note.
        self._debug_log_path = self.rollout_dir / "logs" / "tiaoshi.log"
        self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal implementation note.
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """Implementation note."""
        return self._paused

    # ------------------------------------------------------------------
    # Internal implementation note.
    # ------------------------------------------------------------------

    def process_window(
        self,
        window: VisualObservationWindow | ActionGuidedWindow,
    ) -> SnapshotAssessment:
        """
        Implementation note.

        Args:
            Implementation note.

        Returns:
            Implementation note.

        Raises:
            Implementation note.
            Implementation note.
        """
        if self._paused:
            raise CollectionPausedError(
                "VSA is paused. Call agent.resume() after manual recovery."
            )

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                attempt_start = time.perf_counter()
                assessment, step_timings, n_images = self._run_once(window)
                write_start = time.perf_counter()
                self._writer.write(assessment)
write_elapsed = time.perf_counter() - write_start
                attempt_elapsed = time.perf_counter() - attempt_start
                logger.info(
                    "VSA timing: %s attempt=%d/%d images=%d prior=%.3fs prompt=%.3fs vlm=%.3fs parse=%.3fs fusion=%.3fs stabilize=%.3fs validate=%.3fs write=%.3fs total=%.3fs",
                    self._describe_window(window),
                    attempt + 1,
                    self.max_retries + 1,
                    n_images,
                    step_timings["prior"],
                    step_timings["prompt"],
                    step_timings["vlm"],
                    step_timings["parse"],
                    step_timings["fusion"],
                    step_timings["stabilize"],
                    step_timings["validate"],
                    write_elapsed,
                    attempt_elapsed,
                )
                return assessment

            except RolloutClosedError:
                # Internal implementation note.
                raise

            except (VLMInferenceError, ContractViolationError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "VSA attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, self.max_retries + 1, exc, self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
                else:
                    self._trigger_pause(str(exc))
                    raise CollectionPausedError(
                        f"VSA paused collection after {self.max_retries + 1} failed attempt(s): {exc}"
                    ) from exc

        # Internal implementation note.
        raise CollectionPausedError("Unexpected: retries exhausted.") from last_error

    def resume(self) -> None:
        """
        Implementation note.
        Implementation note.
        """
        self._paused = False
        self._last_phase = None
        self._phase_fusion.reset()
        self._stabilizer.reset()
        self._phase_state_machine.reset()
        self._rollout_memory.reset()
        logger.info("VSA resumed. Temporal stabilizer cache cleared.")

    def close(self) -> None:
        """Close the underlying SnapshotWriter."""
        self._writer.close()

    def write_assessment(self, assessment: SnapshotAssessment) -> None:
        """Append one prepared SnapshotAssessment to the output JSONL."""
        self._writer.write(assessment)
# ------------------------------------------------------------------
    # Internal implementation note.
    # ------------------------------------------------------------------

    def prepare(
        self,
        window: VisualObservationWindow | ActionGuidedWindow,
    ) -> PreparedInference:
        """
        Implementation note.

        Implementation note.
        Implementation note.
        """
        step_timings: dict[str, float] = {}

        # Internal implementation note.
        step_start = time.perf_counter()
        prior = self._prior_scorer.score(window, self.task_config, self._last_phase)
        step_timings["prior"] = time.perf_counter() - step_start

        rollout_context = self._rollout_memory.context()

        # Internal implementation note.
        step_start = time.perf_counter()
        prompt, images = self._prompt_builder.build(
            window,
            self.task_config,
            self.task_memory_context,
            prior,
            rollout_context,
        )
        n_images = len(images)
        step_timings["prompt"] = time.perf_counter() - step_start

        # Internal implementation note.
        if n_images == 0:
            fallback_phase = (
                self._last_phase
                or (self.task_config.phases[0] if self.task_config.phases else "unknown")
            )
            fallback_parsed: dict | None = {
                "phase": fallback_phase,
                "progress": "unknown",
                "risk_level": "unknown",
                "confidence": 0.2,
                "imminent_failure": False,
                "needs_review": True,
            }
            fallback_raw: str | None = "skipped:no_images"
            step_timings["vlm"] = 0.0
            step_timings["parse"] = 0.0
        else:
            fallback_parsed = None
            fallback_raw = None

        return PreparedInference(
            window=window,
            prompt=prompt,
            images=images,
            prior=prior,
            rollout_context=rollout_context,
            n_images=n_images,
            fallback_parsed=fallback_parsed,
            fallback_raw_response=fallback_raw,
            step_timings=step_timings,
        )

    def apply(
        self,
        prepared: PreparedInference,
        raw_response: str,
    ) -> SnapshotAssessment:
        """
        Implementation note.

        Implementation note.
        """
        window = prepared.window
        prior = prepared.prior
        step_timings = dict(prepared.step_timings)  # Internal implementation note.

        # Internal implementation note.
        if prepared.fallback_parsed is not None:
            parsed = dict(prepared.fallback_parsed)
        else:
            step_start = time.perf_counter()
            parsed = self._response_parser.parse(raw_response, self.task_config)
            step_timings["parse"] = time.perf_counter() - step_start
        vlm_parsed = dict(parsed)

        # Internal implementation note.
        step_start = time.perf_counter()
        event_type: Optional[str] = getattr(window, "event_type", None)
        vlm_phase = str(parsed.get("phase", ""))
        try:
            vlm_confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            vlm_confidence = 0.5
        if raw_response.startswith("error:"):
            # A backend timeout is not visual evidence. Keep the online phase
            # conservative instead of letting action hints advance the state.
            fused_phase = (
                self._last_phase
                or (self.task_config.phases[0] if self.task_config.phases else vlm_phase)
            )
            parsed["confidence"] = min(vlm_confidence, 0.1)
            parsed["needs_review"] = True
            fusion_reason = f"vlm_error_hold_phase:{fused_phase}"
        else:
            fused_phase, fusion_reason = self._phase_fusion.fuse(
                prior, vlm_phase, vlm_confidence, self.task_config
            )

            # Internal implementation note.
            if event_type == "sequence_start" and self._last_phase is None and self.task_config.phases:
                fused_phase = self.task_config.phases[0]
                fusion_reason = "sequence_start_force_first_phase"

        parsed["phase"] = fused_phase
        step_timings["fusion"] = time.perf_counter() - step_start

        # Internal implementation note.
        action_summary = getattr(window, "action_summary", {})
        vlm_progress = str(parsed.get("progress", "unknown"))
        derived_progress, progress_reason = self._progress_deriver.derive(
            event_type=event_type or "",
            fused_phase=fused_phase,
            last_phase=self._last_phase,
            action_summary=action_summary,
            vlm_progress=vlm_progress,
        )
        parsed["progress"] = derived_progress
        fused_parsed = dict(parsed)

        logger.info(
            "VSA phase decision: action_top=%s action_margin=%.3f vlm_phase=%s vlm_conf=%.2f "
            "fused_phase=%s fusion_reason=%s | progress: vlm=%s derived=%s progress_reason=%s",
            prior.top_phase,
            prior.top_margin,
            vlm_phase,
            vlm_confidence,
            fused_phase,
            fusion_reason,
            vlm_progress,
            derived_progress,
            progress_reason,
        )

        # Internal implementation note.
        step_start = time.perf_counter()
        stabilized = self._stabilizer.stabilize(parsed, self.task_config, event_type)
        try:
            state_confidence = float(stabilized.get("confidence", vlm_confidence))
        except (TypeError, ValueError):
            state_confidence = vlm_confidence
        stabilized["phase"] = self._phase_state_machine.apply(
            stabilized.get("phase", ""),
            self.task_config,
            event_type,
            confidence=state_confidence,
            risk_level=str(stabilized.get("risk_level", "unknown")),
            imminent_failure=bool(stabilized.get("imminent_failure", False)),
            progress=str(stabilized.get("progress", "unknown")),
        )
        state_reason = self._phase_state_machine.last_reason
        if state_reason.startswith(
            (
                "forward_jump_pending",
                "regression_pending",
                "terminal_failure_hold",
                "terminal_skip_pending",
            )
        ):
            stabilized["needs_review"] = True
        step_timings["stabilize"] = time.perf_counter() - step_start
        final_parsed = dict(stabilized)

        self._log_debug_inference(
            window,
            prepared.prompt,
            prepared.images,
            raw_response,
            final_parsed,
            prior,
            progress_reason,
            vlm_parsed=vlm_parsed,
            fused_parsed=fused_parsed,
            fusion_reason=fusion_reason,
            state_machine_reason=state_reason,
            rollout_context=prepared.rollout_context,
        )

        # Internal implementation note.
        step_start = time.perf_counter()
        assessment = self._validator.validate_and_build(
            stabilized,
            raw_response=raw_response,
            frame_id=window.end_frame_id,
            timestamp=window.end_timestamp,
            task_config=self.task_config,
            event_type=event_type,
            frame_index_range=self._frame_index_range(window),
            vlm_meta={
                "model": getattr(self.vlm_runner, "model_name", type(self.vlm_runner).__name__),
                "latency_ms": int(round(step_timings.get("vlm", 0.0) * 1000)),
                "prompt_version": "inline-v1",
            },
        )
        step_timings["validate"] = time.perf_counter() - step_start

        self._last_phase = assessment.phase
        self._rollout_memory.add(
            PhaseDecisionRecord(
                frame_id=assessment.frame_id,
                timestamp=assessment.timestamp,
                event_type=event_type or "",
                visual_phase=vlm_phase,
                final_phase=assessment.phase,
                confidence=assessment.confidence,
                risk_level=assessment.risk_level,
                progress=assessment.progress,
                action_hint_phase=prior.top_phase,
                fusion_reason=fusion_reason,
                state_reason=state_reason,
            )
        )
        prepared.step_timings.update(step_timings)
        return assessment

    def _run_once(
        self,
        window: VisualObservationWindow | ActionGuidedWindow,
    ) -> tuple[SnapshotAssessment, dict[str, float], int]:
        """Implementation note."""
        prepared = self.prepare(window)
        step_timings = dict(prepared.step_timings)

        if prepared.fallback_parsed is not None:
            raw_response = "skipped:no_images"
        else:
            step_start = time.perf_counter()
            raw_response = self.vlm_runner.run(prepared.prompt, prepared.images)
            step_timings["vlm"] = time.perf_counter() - step_start
            prepared.step_timings["vlm"] = step_timings["vlm"]

        assessment = self.apply(prepared, raw_response)
        step_timings.update(prepared.step_timings)
        return assessment, step_timings, prepared.n_images

    # ------------------------------------------------------------------
    # Internal implementation note.
    # ------------------------------------------------------------------

    def _trigger_pause(self, reason: str) -> None:
        self._paused = True
        alert_msg = (
            f"[VSA CRITICAL] Visual Snapshot Agent has paused the collection pipeline.\n"
            f"Reason: {reason}\n"
            f"Required action: Investigate VLM service, then call agent.resume() to continue.\n"
            f"Do NOT write placeholder SnapshotAssessments; call resume() on a fresh window."
        )
        logger.error(alert_msg)
        self.on_pause(alert_msg)

    @staticmethod
    def _describe_window(window: VisualObservationWindow | ActionGuidedWindow) -> str:
        event_type = getattr(window, "event_type", "window")
        anchor_frame = getattr(window, "anchor_frame_id", window.end_frame_id)
        image_count = len([frame for frame in window.color_frames if frame is not None])
        if image_count == 0:
            image_count = len(getattr(window, "keyframe_image_paths", []) or [])
        return (
            f"event_type={event_type} anchor_frame={anchor_frame} "
            f"end_frame={window.end_frame_id} images={image_count}"
        )

    @staticmethod
    def _frame_index_range(window: VisualObservationWindow | ActionGuidedWindow) -> tuple[int, int] | None:
        summary = getattr(window, "action_summary", None)
        if isinstance(summary, dict):
            frame_range = summary.get("frame_range")
            if (
                isinstance(frame_range, (list, tuple))
                and len(frame_range) == 2
            ):
                try:
                    return int(frame_range[0]), int(frame_range[1])
                except (TypeError, ValueError):
                    pass

        if window.frame_ids:
            return min(window.frame_ids), max(window.frame_ids)
        return None

    def _log_debug_inference(
        self,
        window: VisualObservationWindow | ActionGuidedWindow,
        prompt: str,
        images: list,
        raw_response: str,
        parsed: dict,
        prior,
        progress_reason: str = "",
        *,
        vlm_parsed: dict | None = None,
        fused_parsed: dict | None = None,
        fusion_reason: str = "",
        state_machine_reason: str = "",
        rollout_context: RolloutMemoryContext | None = None,
    ) -> None:
        """Implementation note."""
        try:
            import json
            from datetime import datetime
            import cv2

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            event_type = getattr(window, "event_type", "window")
            anchor_frame = getattr(window, "anchor_frame_id", window.end_frame_id)

            image_paths = []
            persisted_paths = getattr(window, "keyframe_image_paths", []) or []
            if persisted_paths:
                for raw_path in persisted_paths:
                    img_path = Path(raw_path)
                    try:
                        image_paths.append(str(img_path.relative_to(self.rollout_dir)))
                    except ValueError:
                        image_paths.append(str(img_path))
            else:
                # Internal implementation note.
                image_dir = self._debug_log_path.parent / "tiaoshi_images"
                image_dir.mkdir(parents=True, exist_ok=True)
                for idx, img in enumerate(images):
                    if img is not None:
                        img_path = image_dir / f"{anchor_frame:06d}_{idx}.jpg"
                        # Internal implementation note.
                        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        image_paths.append(str(img_path.relative_to(self.rollout_dir)))

            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "anchor_frame": anchor_frame,
                "end_frame": window.end_frame_id,
                "n_images": len(images),
                "final_parsed": parsed,
                "image_paths": image_paths,
                "prior": {
                    "top_phase": prior.top_phase,
                    "top_margin": prior.top_margin,
                    "phase_scores": prior.phase_scores,
                    "prior_reason": prior.prior_reason,
                },
                "action_evidence": {
                    "action_only_phase_hint": prior.top_phase,
                    "action_hint_margin": prior.top_margin,
                    "action_phase_scores": prior.phase_scores,
                    "reason": prior.prior_reason,
                    "role": "auxiliary_only_visual_evidence_is_primary",
                },
                "rollout_context": (
                    dataclasses.asdict(rollout_context)
                    if rollout_context is not None
                    else None
                ),
                "prompt": prompt,
                "raw_response": raw_response,
                "vlm_parsed": vlm_parsed,
                "fused_parsed": fused_parsed,
                "parsed": parsed,
                "fusion_reason": fusion_reason,
                "state_machine_reason": state_machine_reason,
                "progress_reason": progress_reason,
            }

            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write debug log: %s", e)
