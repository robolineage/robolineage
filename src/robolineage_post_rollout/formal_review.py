from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robolineage_shared_agents.visual_snapshot.exceptions import VLMInferenceError
from robolineage_shared_agents.visual_snapshot.vlm_runner import BaseVLMRunner
from robolineage_contracts.agents import SnapshotAssessment
from robolineage_schemas.artifacts import write_validated_json_atomic

_LOG = logging.getLogger(__name__)

AGENT_VERSION = "post_rollout_review@0.2"
EVIDENCE_SCHEMA_VERSION = "post_review.evidence.v1"
ANNOTATION_SCHEMA_VERSION = "post_review.annotation.v1"
FAILURE_SCHEMA_VERSION = "post_review.failure.v1"
ADMISSION_SCHEMA_VERSION = "post_review.admission.v1"
SUMMARY_SCHEMA_VERSION = "post_review.summary.v1"
L1_SCHEMA_VERSION = "1.0"
_MAX_ISOLATED_JITTER_CONFIDENCE = 0.65
DEFAULT_MAX_REVIEW_IMAGES = 12

_VLM_ADMISSION_FIELDS = {
    "accepted_for_training",
    "admission_class",
    "data_use",
    "dataset_decision",
    "decision",
    "recommended_split",
    "requires_review",
}


@dataclass(frozen=True)
class PostRolloutReviewResult:
    rollout_dir: Path
    rollout_id: str
    status: str
    artifacts: dict[str, str]
    used_vlm: bool
    vlm_error: str | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    rollout_dir: Path
    rollout_id: str
    task_config: dict[str, Any]
    snapshots: list[SnapshotAssessment]
    snapshot_records: list[dict[str, Any]]
    image_index: dict[int, list[Path]]
    vsa_windows: list[dict[str, Any]]
    key_frames: list[dict[str, Any]]
    visual_disagreements: list[dict[str, Any]]
    sources: dict[str, Any]

    @property
    def phases(self) -> list[str]:
        return [str(item) for item in self.task_config.get("phases") or [] if str(item)]

    @property
    def terminal_phase(self) -> str | None:
        return self.phases[-1] if self.phases else None

    def to_index(self) -> dict[str, Any]:
        timestamps = [item.timestamp for item in self.snapshots]
        frame_ids = [item.frame_id for item in self.snapshots]
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "agent_version": AGENT_VERSION,
            "rollout_id": self.rollout_id,
            "task": {
                "description": self.task_config.get("task_description", ""),
                "phases": self.phases,
                "terminal_phase": self.terminal_phase,
                "failure_signals": list(self.task_config.get("failure_signals") or []),
                "phase_visual_hints": self.task_config.get("phase_visual_hints") or {},
                "phase_action_hints": self.task_config.get("phase_action_hints") or {},
            },
            "sources": self.sources,
            "snapshot_count": len(self.snapshots),
            "image_frame_count": len(self.image_index),
            "image_count": sum(len(paths) for paths in self.image_index.values()),
            "vsa_window_count": len(self.vsa_windows),
            "final_observation_frames": [
                int(item["anchor_frame_id"])
                for item in self.vsa_windows
                if item.get("event_type") == "final_observation"
                and item.get("anchor_frame_id") is not None
            ],
            "frame_span": {
                "start_frame": min(frame_ids) if frame_ids else None,
                "end_frame": max(frame_ids) if frame_ids else None,
                "start_timestamp": min(timestamps) if timestamps else None,
                "end_timestamp": max(timestamps) if timestamps else None,
            },
            "online_phase_sequence": [record["phase"] for record in self.snapshot_records],
            "risk_summary": dict(Counter(record["risk_level"] for record in self.snapshot_records)),
            "progress_summary": dict(Counter(record["progress"] for record in self.snapshot_records)),
            "key_frames": self.key_frames,
            "visual_disagreements": self.visual_disagreements,
            "vsa_windows": self.vsa_windows,
        }


class EvidenceBuilder:
    """Build the offline evidence package for one closed rollout."""

    def build(self, rollout_dir: str | Path) -> EvidenceBundle:
        rollout_path = Path(rollout_dir)
        snapshots, snapshot_source = _load_snapshots(rollout_path)
        task_config = _load_task_config(rollout_path)
        vsa_windows = _load_vsa_window_records(rollout_path)
        image_index = _build_image_index(rollout_path, vsa_windows)
        snapshot_records = [_snapshot_record(item, task_config) for item in snapshots]
        visual_disagreements = _visual_disagreements(snapshot_records, task_config)
        key_frames = _key_frames(snapshot_records, visual_disagreements, image_index, vsa_windows)
        return EvidenceBundle(
            rollout_dir=rollout_path,
            rollout_id=rollout_path.name,
            task_config=task_config,
            snapshots=snapshots,
            snapshot_records=snapshot_records,
            image_index=image_index,
            vsa_windows=vsa_windows,
            key_frames=key_frames,
            visual_disagreements=visual_disagreements,
            sources={
                "snapshots": str(snapshot_source) if snapshot_source is not None else None,
                "task_config": str(rollout_path / "task_config.yaml")
                if (rollout_path / "task_config.yaml").exists()
                else None,
                "tiaoshi_log": _log_summary(rollout_path / "logs" / "tiaoshi.log"),
                "tiaoshi_images_dir": str(rollout_path / "logs" / "tiaoshi_images")
                if (rollout_path / "logs" / "tiaoshi_images").exists()
                else None,
                "vsa_windows_dir": str(rollout_path / "vsa_windows")
                if (rollout_path / "vsa_windows").exists()
                else None,
                "vsa_windows_manifest": str(rollout_path / "vsa_windows" / "manifest.jsonl")
                if (rollout_path / "vsa_windows" / "manifest.jsonl").exists()
                else None,
            },
        )


class AnnotationAgent:
    """Finalize phase timeline and rollout outcome from full episode evidence."""

    def __init__(
        self,
        *,
        vlm_runner: BaseVLMRunner | None = None,
        use_vlm: bool = True,
        max_review_images: int = DEFAULT_MAX_REVIEW_IMAGES,
    ) -> None:
        self.vlm_runner = vlm_runner
        self.use_vlm = use_vlm
        self.max_review_images = max(0, max_review_images)

    def annotate(self, evidence: EvidenceBundle) -> dict[str, Any]:
        points = _initial_phase_points(evidence)
        timeline = _phase_timeline_from_points(points, terminal_phase=evidence.terminal_phase)
        retry_events = _retry_events(timeline, evidence.phases)
        outcome = _determine_outcome(evidence, timeline, retry_events)

        vlm_review: dict[str, Any] | None = None
        vlm_error: str | None = None
        if self.use_vlm and self.vlm_runner is not None and evidence.snapshots:
            review_packets = _review_packets_and_images(
                evidence,
                timeline,
                outcome,
                retry_events,
                self.max_review_images,
            )
            if review_packets:
                packet_reviews: list[dict[str, Any]] = []
                for packet in review_packets:
                    try:
                        raw = self.vlm_runner.run(packet["prompt"], packet["images"])
                        packet_review = _sanitize_vlm_review(
                            _parse_json_object(raw) or {"raw_response": raw}
                        )
                        packet_review.setdefault("packet_id", packet["packet_id"])
                        packet_review.setdefault("packet_index", packet["packet_index"])
                        packet_review.setdefault("purpose", packet["purpose"])
                        packet_review.setdefault("image_frames", packet["image_frames"])
                        packet_review.setdefault("status", "reviewed")
                        packet_reviews.append(packet_review)
                    except VLMInferenceError as exc:
                        packet_reviews.append(_failed_packet_review(packet, exc))

                successful_packet_reviews = [
                    item for item in packet_reviews if item.get("status") != "failed"
                ]
                failed_packet_reviews = [
                    item for item in packet_reviews if item.get("status") == "failed"
                ]
                if successful_packet_reviews:
                    vlm_review = _aggregate_packet_reviews(successful_packet_reviews, evidence, outcome)
                    if failed_packet_reviews:
                        vlm_review["status"] = "packet_partially_aggregated"
                    vlm_review["successful_packet_count"] = len(successful_packet_reviews)
                    vlm_review["failed_packet_count"] = len(failed_packet_reviews)
                    if failed_packet_reviews:
                        vlm_review["packet_failures"] = [
                            _packet_failure_metadata(item) for item in failed_packet_reviews
                        ]
                    vlm_review["image_frames"] = review_packets[0]["image_frames"]
                    vlm_review["terminal_image_coverage"] = _review_image_coverage(
                        evidence,
                        review_packets[0]["image_frames"],
                    )
                    vlm_review["review_packets"] = [
                        _packet_metadata(packet) for packet in review_packets
                    ]
                    vlm_review["packet_reviews"] = packet_reviews
                    if not _vlm_failure_without_terminal_image_coverage(vlm_review, outcome):
                        points = _apply_vlm_phase_corrections(points, vlm_review, evidence.phases)
                        points = _apply_vlm_final_phase(points, vlm_review, evidence.phases)
                        timeline = _phase_timeline_from_points(
                            points,
                            terminal_phase=evidence.terminal_phase,
                        )
                        retry_events = _retry_events(timeline, evidence.phases)
                        outcome = _determine_outcome(evidence, timeline, retry_events)
                    outcome = _outcome_with_vlm(outcome, vlm_review, evidence)
                elif failed_packet_reviews:
                    vlm_error = "; ".join(
                        f"{item.get('packet_id')}: {item.get('error')}"
                        for item in failed_packet_reviews
                    )
                    vlm_review = {
                        "status": "skipped",
                        "reason": "all_packet_reviews_failed",
                        "image_frames": review_packets[0]["image_frames"],
                        "terminal_image_coverage": _review_image_coverage(
                            evidence,
                            review_packets[0]["image_frames"],
                        ),
                        "review_packets": [
                            _packet_metadata(packet) for packet in review_packets
                        ],
                        "packet_reviews": packet_reviews,
                        "successful_packet_count": 0,
                        "failed_packet_count": len(failed_packet_reviews),
                        "packet_failures": [
                            _packet_failure_metadata(item) for item in failed_packet_reviews
                        ],
                    }
                    _LOG.info(
                        "post-rollout VLM packet annotation skipped/failed for %s: %s",
                        evidence.rollout_id,
                        vlm_error,
                    )
            else:
                prompt, image_frames, images = _review_prompt_and_images(
                    evidence,
                    timeline,
                    outcome,
                    retry_events,
                    self.max_review_images,
                )
                if not images:
                    vlm_review = {
                        "status": "skipped",
                        "reason": "insufficient_visual_evidence",
                        "image_frames": [],
                    }
                else:
                    try:
                        raw = self.vlm_runner.run(prompt, images)
                        vlm_review = _sanitize_vlm_review(_parse_json_object(raw) or {"raw_response": raw})
                        vlm_review.setdefault("image_frames", image_frames)
                        vlm_review["terminal_image_coverage"] = _review_image_coverage(
                            evidence,
                            image_frames,
                        )
                        if not _vlm_failure_without_terminal_image_coverage(vlm_review, outcome):
                            points = _apply_vlm_phase_corrections(points, vlm_review, evidence.phases)
                            points = _apply_vlm_final_phase(points, vlm_review, evidence.phases)
                            timeline = _phase_timeline_from_points(
                                points,
                                terminal_phase=evidence.terminal_phase,
                            )
                            retry_events = _retry_events(timeline, evidence.phases)
                            outcome = _determine_outcome(evidence, timeline, retry_events)
                        outcome = _outcome_with_vlm(outcome, vlm_review, evidence)
                    except VLMInferenceError as exc:
                        vlm_error = str(exc)
                        _LOG.info(
                            "post-rollout VLM annotation skipped/failed for %s: %s",
                            evidence.rollout_id,
                            exc,
                        )

        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "agent_version": AGENT_VERSION,
            "rollout_id": evidence.rollout_id,
            "created_at": _now_iso(),
            "task": {
                "description": evidence.task_config.get("task_description", ""),
                "phases": evidence.phases,
                "terminal_phase": evidence.terminal_phase,
            },
            "outcome": {
                **outcome,
                "used_vlm": vlm_review is not None and vlm_review.get("status") != "skipped",
                "vlm_error": vlm_error,
            },
            "phase_timeline": timeline,
            "retry_events": retry_events,
            "frame_tags": _frame_tags_from_evidence(evidence, timeline),
            "l1_annotation": _l1_annotation(evidence, timeline),
            "vlm_review": vlm_review,
            "review_packets": (vlm_review or {}).get("review_packets") if isinstance(vlm_review, dict) else None,
            "vlm_packet_reviews": (vlm_review or {}).get("packet_reviews") if isinstance(vlm_review, dict) else None,
        }


class FailureAnalysisAgent:
    """Attribute failure, retry and anomaly events after final annotation."""

    def analyze(self, evidence: EvidenceBundle, annotation: dict[str, Any]) -> dict[str, Any]:
        timeline = list(annotation.get("phase_timeline") or [])
        outcome = dict(annotation.get("outcome") or {})
        failure_events: list[dict[str, Any]] = []

        for segment in timeline:
            reasons = _failure_reasons(segment)
            if not reasons:
                continue
            failure_events.append(_failure_event(segment, reasons, evidence, outcome))

        retry_events = list(annotation.get("retry_events") or [])
        vlm_review = annotation.get("vlm_review") if isinstance(annotation.get("vlm_review"), dict) else None
        if vlm_review is not None:
            failure_events = _merge_vlm_failure_events(failure_events, vlm_review, evidence)

        final_success = bool(outcome.get("final_success"))
        recovered_count = sum(1 for event in failure_events if event.get("recovered"))
        final_failure_events = [
            event for event in failure_events if not event.get("recovered") and not final_success
        ]
        if not failure_events and not retry_events:
            status = "no_failure_candidates"
        elif final_success:
            status = "recovered_failure_found" if failure_events or retry_events else "no_failure_candidates"
        else:
            status = "final_failure"

        return {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "agent_version": AGENT_VERSION,
            "rollout_id": evidence.rollout_id,
            "status": status,
            "outcome": "success" if final_success else "failure",
            "candidate_count": len(failure_events),
            "candidate_segments": [_candidate_segment(event) for event in failure_events],
            "failure_events": failure_events,
            "retry_events": retry_events,
            "recovered_count": recovered_count,
            "final_failure_type": (
                final_failure_events[-1]["failure_type"] if final_failure_events else None
            ),
            "high_risk_frames": [
                record["frame_id"]
                for record in evidence.snapshot_records
                if record["risk_level"] == "high" or record["imminent_failure"]
            ],
            "failure_signals": list(evidence.task_config.get("failure_signals") or []),
        }


class DataGovernanceAgent:
    """Decide dataset admission from final annotation and failure analysis."""

    def decide(
        self,
        evidence: EvidenceBundle,
        annotation: dict[str, Any],
        failures: dict[str, Any],
    ) -> dict[str, Any]:
        outcome = dict(annotation.get("outcome") or {})
        final_success = bool(outcome.get("final_success"))
        snapshot_count = len(evidence.snapshots)
        failure_count = int(failures.get("candidate_count") or 0)
        recovered_count = int(failures.get("recovered_count") or 0)
        needs_review_count = sum(1 for item in evidence.snapshots if item.needs_review)
        outcome_uncertain = str(outcome.get("success_status") or "") == "uncertain"
        reasons: list[str] = []
        data_use: list[str] = []
        admission_class = "unknown"
        decision = "needs_review"
        accepted_for_training = False
        label_quality = "clean"
        review_reason: str | None = None

        if snapshot_count == 0:
            decision = "rejected"
            admission_class = "empty_rollout"
            accepted_for_training = False
            label_quality = "uncertain"
            review_reason = "No VSA snapshots were available for this rollout."
            reasons.append("no_snapshots")
        elif final_success and outcome_uncertain:
            decision = "needs_review"
            admission_class = "successful_but_ambiguous"
            accepted_for_training = True
            data_use.extend(["success_trajectory", "review_queue"])
            if failure_count > 0:
                data_use.append("failure_analysis")
            if recovered_count > 0:
                data_use.append("recovery_training")
            label_quality = "uncertain"
            review_reason = "Terminal success evidence exists, but final labels or risk require review."
            reasons.append("terminal_success_with_uncertain_outcome")
        elif final_success and recovered_count > 0:
            decision = "accepted"
            admission_class = "accepted_with_labels"
            accepted_for_training = True
            data_use.extend(["success_trajectory", "recovery_training", "failure_taxonomy"])
            reasons.append("final_success_with_recovered_failure")
        elif final_success and failure_count == 0:
            decision = "accepted"
            admission_class = "clean_success"
            accepted_for_training = True
            data_use.append("success_trajectory")
            reasons.append("final_success_without_failure_candidates")
        elif final_success:
            decision = "needs_review"
            admission_class = "successful_but_ambiguous"
            accepted_for_training = True
            data_use.extend(["success_trajectory", "review_queue"])
            label_quality = "uncertain"
            review_reason = "Success was likely, but failure candidates or labels need review."
            reasons.append("success_with_unresolved_failure_candidates")
        elif failure_count > 0:
            decision = "needs_review"
            admission_class = "failure_pool_candidate"
            data_use.extend(["failure_analysis", "future_recovery_training"])
            label_quality = "uncertain"
            review_reason = "Failure evidence is useful for training or taxonomy, but labels need review."
            reasons.append("final_failure_with_structured_failure_evidence")
        else:
            decision = "needs_review" if needs_review_count > 0 else "retry_recommended"
            admission_class = "insufficient_completion_evidence"
            data_use.extend(["review_queue"])
            label_quality = "uncertain"
            review_reason = "Terminal success was not confirmed; keep the trajectory available while labels are reviewed."
            reasons.append("terminal_success_not_confirmed")

        if (
            decision == "accepted"
            and needs_review_count > max(1, snapshot_count // 2)
            and outcome.get("used_vlm") is not True
        ):
            decision = "needs_review"
            admission_class = "accepted_candidate_needs_review"
            label_quality = "uncertain"
            review_reason = "Many VSA snapshots were low-confidence or marked for review."
            reasons.append("many_low_confidence_or_review_snapshots")

        vlm_review = annotation.get("vlm_review")
        if isinstance(vlm_review, dict):
            vlm_label_quality = str(vlm_review.get("label_quality") or "").strip()
            if vlm_label_quality in {"clean", "needs_correction", "uncertain"}:
                label_quality = vlm_label_quality
                if vlm_label_quality != "clean" and review_reason is None:
                    review_reason = str(vlm_review.get("reasoning") or "VLM marked labels for review.")
            if vlm_review.get("training_usability") is False:
                reasons.append("vlm_training_usability_note")

        return {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "agent_version": AGENT_VERSION,
            "rollout_id": evidence.rollout_id,
            "decision": decision,
            "accepted_for_training": accepted_for_training,
            "label_quality": label_quality,
            "review_reason": review_reason,
            "admission_class": admission_class,
            "reasons": _dedupe_preserve_order_str(reasons),
            "data_use": _dedupe_preserve_order_str(data_use),
            "recommended_split": "train" if accepted_for_training else None,
            "requires_review": decision == "needs_review",
            "task_description": evidence.task_config.get("task_description", ""),
            "created_at": _now_iso(),
        }


class PostRolloutReviewAgent:
    """Offline review entrypoint for one closed rollout directory."""

    def __init__(
        self,
        *,
        vlm_runner: BaseVLMRunner | None = None,
        use_vlm: bool = True,
        max_review_images: int = DEFAULT_MAX_REVIEW_IMAGES,
    ) -> None:
        self.evidence_builder = EvidenceBuilder()
        self.annotation_agent = AnnotationAgent(
            vlm_runner=vlm_runner,
            use_vlm=use_vlm,
            max_review_images=max_review_images,
        )
        self.failure_agent = FailureAnalysisAgent()
        self.governance_agent = DataGovernanceAgent()

    def run(self, rollout_dir: str | Path) -> PostRolloutReviewResult:
        rollout_path = Path(rollout_dir)
        rollout_id = rollout_path.name
        artifacts: dict[str, str] = {}
        _write_json_atomic(
            rollout_path / "post_review_status.json",
            {
                "status": "running",
                "rollout_id": rollout_id,
                "started_at": _now_iso(),
                "agent_version": AGENT_VERSION,
            },
        )

        evidence = self.evidence_builder.build(rollout_path)
        annotation = self.annotation_agent.annotate(evidence)
        failures = self.failure_agent.analyze(evidence, annotation)
        admission = self.governance_agent.decide(evidence, annotation, failures)
        summary = _rollout_summary(evidence, annotation, failures, admission)
        timeline = list(annotation.get("phase_timeline") or [])

        artifacts["evidence_index"] = str(
            _write_json_atomic(rollout_path / "evidence_index.json", evidence.to_index())
        )
        artifacts["annotation"] = str(
            write_validated_json_atomic(rollout_path / "annotation.final.json", annotation, "annotation_final")
        )
        review_packets = list(annotation.get("review_packets") or [])
        if review_packets:
            artifacts["review_packets"] = str(
                _write_json_atomic(rollout_path / "review_packets.json", review_packets)
            )
        packet_reviews = list(annotation.get("vlm_packet_reviews") or [])
        if packet_reviews:
            artifacts["vlm_packet_reviews"] = str(
                _write_jsonl_atomic(rollout_path / "vlm_packet_reviews.jsonl", packet_reviews)
            )
        artifacts["rollout_summary"] = str(
            _write_json_atomic(rollout_path / "rollout_summary.json", summary)
        )
        artifacts["failure_analysis"] = str(
            write_validated_json_atomic(rollout_path / "failure_analysis.json", failures, "failure_analysis")
        )
        artifacts["dataset_admission"] = str(
            write_validated_json_atomic(rollout_path / "dataset_admission.json", admission, "dataset_admission")
        )
        artifacts["dataset_decision"] = str(
            write_validated_json_atomic(rollout_path / "dataset_decision.json", admission, "dataset_admission")
        )
        artifacts["phase_timeline"] = str(
            _write_jsonl_atomic(rollout_path / "phase_timeline.final.jsonl", timeline)
        )
        artifacts["review_report"] = str(
            _write_text_atomic(
                rollout_path / "review_report.md",
                _render_report(summary, timeline, failures, admission),
            )
        )

        outcome = annotation.get("outcome") if isinstance(annotation.get("outcome"), dict) else {}
        status_payload = {
            "status": "completed",
            "rollout_id": rollout_id,
            "completed_at": _now_iso(),
            "artifacts": artifacts,
            "used_vlm": bool(outcome.get("used_vlm")),
            "vlm_error": outcome.get("vlm_error"),
            "agent_version": AGENT_VERSION,
        }
        artifacts["status"] = str(
            _write_json_atomic(rollout_path / "post_review_status.json", status_payload)
        )
        return PostRolloutReviewResult(
            rollout_dir=rollout_path,
            rollout_id=rollout_id,
            status="completed",
            artifacts=artifacts,
            used_vlm=bool(outcome.get("used_vlm")),
            vlm_error=outcome.get("vlm_error"),
        )


def _load_snapshots(rollout_dir: Path) -> tuple[list[SnapshotAssessment], Path | None]:
    candidates = [rollout_dir / "snapshots.jsonl", rollout_dir / "snapshot_annotations.jsonl"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return [], None

    snapshots: list[SnapshotAssessment] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if raw.get("frame_index_range") is not None:
                raw["frame_index_range"] = tuple(raw["frame_index_range"])
            snapshots.append(SnapshotAssessment(**raw))
        except Exception as exc:
            _LOG.warning("skipping bad snapshot line %s in %s: %s", line_no, path, exc)
    return sorted(snapshots, key=lambda item: (item.timestamp, item.frame_id)), path


def _load_task_config(rollout_dir: Path) -> dict[str, Any]:
    path = rollout_dir / "task_config.yaml"
    if not path.exists():
        return {"task_description": "", "phases": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {"task_description": "", "phases": []}
    data.setdefault("task_description", data.get("goal", ""))
    data.setdefault("phases", [])
    return data


def _load_vsa_window_records(rollout_dir: Path) -> list[dict[str, Any]]:
    root = rollout_dir / "vsa_windows"
    manifest = root / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    if not manifest.exists():
        return records
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            _LOG.warning("skipping bad VSA window manifest line %s in %s: %s", line_no, manifest, exc)
            continue
        if not isinstance(raw, dict):
            continue
        image_paths: list[str] = []
        for item in raw.get("image_paths") or []:
            path = Path(str(item))
            if not path.is_absolute():
                path = rollout_dir / path
            if path.exists():
                image_paths.append(str(path))
        records.append(
            {
                "window_id": str(raw.get("window_id") or ""),
                "event_type": str(raw.get("event_type") or "unknown"),
                "anchor_frame_id": _int_or_none(raw.get("anchor_frame_id")),
                "end_frame_id": _int_or_none(raw.get("end_frame_id")),
                "keyframe_ids": _int_list(raw.get("keyframe_ids")),
                "image_paths": image_paths,
                "image_format": raw.get("image_format"),
                "camera_name": raw.get("camera_name"),
            }
        )
    return records


def _build_image_index(rollout_dir: Path, vsa_windows: list[dict[str, Any]]) -> dict[int, list[Path]]:
    indexed: dict[int, list[Path]] = {}
    image_dir = rollout_dir / "logs" / "tiaoshi_images"
    if image_dir.exists():
        for path in sorted(image_dir.glob("*.jpg")):
            match = re.match(r"^(\d+)_\d+\.jpg$", path.name)
            if match is None:
                continue
            indexed.setdefault(int(match.group(1)), []).append(path)

    for record in vsa_windows:
        keyframe_ids = [int(item) for item in record.get("keyframe_ids") or []]
        paths = [Path(str(item)) for item in record.get("image_paths") or []]
        for idx, path in enumerate(paths):
            frame_id = (
                keyframe_ids[idx]
                if idx < len(keyframe_ids)
                else _frame_id_from_vsa_keyframe_path(path)
            )
            if frame_id is None:
                frame_id = _int_or_none(record.get("anchor_frame_id"))
            if frame_id is None:
                continue
            indexed.setdefault(frame_id, []).append(path)

    vsa_root = rollout_dir / "vsa_windows"
    if vsa_root.exists():
        for path in sorted(vsa_root.rglob("kf_*_frame_*.png")):
            frame_id = _frame_id_from_vsa_keyframe_path(path)
            if frame_id is None:
                continue
            existing = indexed.setdefault(frame_id, [])
            if path not in existing:
                existing.append(path)
    return indexed


def _frame_id_from_vsa_keyframe_path(path: Path) -> int | None:
    match = re.search(r"_frame_(\d+)\.png$", path.name)
    return int(match.group(1)) if match is not None else None


def _snapshot_record(snapshot: SnapshotAssessment, task_config: dict[str, Any]) -> dict[str, Any]:
    record = asdict(snapshot)
    trigger = record.get("trigger")
    if trigger is not None:
        record["trigger"] = getattr(trigger, "value", str(trigger))
    if record.get("frame_index_range") is not None:
        record["frame_index_range"] = list(record["frame_index_range"])
    raw_obj = _parse_json_object(snapshot.raw_response)
    if raw_obj is not None:
        record["raw_vlm_json"] = raw_obj
        phases = [str(item) for item in task_config.get("phases") or []]
        raw_phase = str(raw_obj.get("phase") or "")
        if raw_phase in phases:
            record["raw_visual_phase"] = raw_phase
        raw_progress = str(raw_obj.get("progress") or "")
        if raw_progress:
            record["raw_visual_progress"] = raw_progress
        raw_risk = str(raw_obj.get("risk_level") or "")
        if raw_risk:
            record["raw_visual_risk_level"] = raw_risk
        record["raw_visual_confidence"] = _float_or_none(raw_obj.get("confidence"))
    return record


def _visual_disagreements(
    snapshot_records: list[dict[str, Any]],
    task_config: dict[str, Any],
) -> list[dict[str, Any]]:
    phases = [str(item) for item in task_config.get("phases") or []]
    disagreements: list[dict[str, Any]] = []
    for record in snapshot_records:
        raw_phase = record.get("raw_visual_phase")
        if raw_phase not in phases or raw_phase == record.get("phase"):
            continue
        disagreements.append(
            {
                "frame_id": record["frame_id"],
                "online_phase": record["phase"],
                "raw_visual_phase": raw_phase,
                "online_confidence": record["confidence"],
                "raw_visual_confidence": record.get("raw_visual_confidence"),
                "risk_level": record["risk_level"],
                "needs_review": record["needs_review"],
                "trigger": record.get("trigger"),
            }
        )
    return disagreements


def _key_frames(
    snapshot_records: list[dict[str, Any]],
    visual_disagreements: list[dict[str, Any]],
    image_index: dict[int, list[Path]],
    vsa_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons: dict[int, set[str]] = {}
    event_types: dict[int, set[str]] = {}
    if snapshot_records:
        reasons.setdefault(snapshot_records[0]["frame_id"], set()).add("start")
        reasons.setdefault(snapshot_records[-1]["frame_id"], set()).add("end")

    last_phase: str | None = None
    for record in snapshot_records:
        frame_id = int(record["frame_id"])
        phase = str(record["phase"])
        if phase != last_phase:
            reasons.setdefault(frame_id, set()).add("phase_boundary")
            last_phase = phase
        if record.get("risk_level") == "high" or record.get("imminent_failure"):
            reasons.setdefault(frame_id, set()).add("failure_candidate")
        if record.get("needs_review"):
            reasons.setdefault(frame_id, set()).add("needs_review")

    for item in visual_disagreements:
        reasons.setdefault(int(item["frame_id"]), set()).add("visual_disagreement")

    for item in vsa_windows:
        event_type = str(item.get("event_type") or "unknown")
        frame_ids = _int_list(item.get("keyframe_ids"))
        anchor = _int_or_none(item.get("anchor_frame_id"))
        if not frame_ids and anchor is not None:
            frame_ids = [anchor]
        for frame_id in frame_ids:
            reasons.setdefault(frame_id, set()).add("vsa_window")
            event_types.setdefault(frame_id, set()).add(event_type)
            if event_type == "final_observation":
                reasons[frame_id].add("final_observation")

    rows: list[dict[str, Any]] = []
    for frame_id in sorted(reasons):
        events = sorted(event_types.get(frame_id) or [])
        row = {
            "frame_id": frame_id,
            "reasons": sorted(reasons[frame_id]),
            "images": [str(path) for path in image_index.get(frame_id, [])],
        }
        if events:
            row["event_types"] = events
            row["event_type"] = events[0]
        rows.append(row)
    return rows


def _initial_phase_points(evidence: EvidenceBundle) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    phases = evidence.phases
    terminal_phase = evidence.terminal_phase
    for record in evidence.snapshot_records:
        phase = str(record["phase"])
        raw_phase = record.get("raw_visual_phase")
        correction_reason: str | None = None
        source = "online_vsa"
        raw_conf = record.get("raw_visual_confidence")
        online_conf = float(record.get("confidence") or 0.0)

        if isinstance(raw_phase, str) and raw_phase in phases and raw_phase != phase:
            phase_index = _phase_index(phases, phase)
            raw_index = _phase_index(phases, raw_phase)
            terminal_lock_suspect = (
                phase == terminal_phase
                and raw_index is not None
                and phase_index is not None
                and raw_index < phase_index
            )
            strong_raw_signal = raw_conf is not None and raw_conf >= max(0.55, online_conf + 0.05)
            uncertain_online = bool(record.get("needs_review")) or record.get("risk_level") == "high"
            if terminal_lock_suspect or strong_raw_signal or uncertain_online:
                phase = raw_phase
                source = "raw_vlm_visual"
                correction_reason = (
                    "terminal_phase_retry_evidence" if terminal_lock_suspect else "raw_visual_phase_override"
                )

        points.append(
            {
                "frame_id": record["frame_id"],
                "timestamp": record["timestamp"],
                "phase": phase,
                "online_phase": record["phase"],
                "raw_visual_phase": raw_phase,
                "progress": record["progress"],
                "risk_level": record["risk_level"],
                "imminent_failure": record["imminent_failure"],
                "confidence": record["confidence"],
                "needs_review": record["needs_review"],
                "trigger": record.get("trigger") or "unknown",
                "source": source,
                "corrected_by_review": source != "online_vsa",
                "correction_reason": correction_reason,
            }
        )
    return points


def _phase_timeline_from_points(
    points: list[dict[str, Any]],
    *,
    terminal_phase: str | None = None,
) -> list[dict[str, Any]]:
    if not points:
        return []
    points = _suppress_isolated_phase_jitter(points)
    groups: list[list[dict[str, Any]]] = [[points[0]]]
    for point in points[1:]:
        if point["phase"] == groups[-1][-1]["phase"]:
            groups[-1].append(point)
        else:
            groups.append([point])
    timeline = [_timeline_record(group, idx) for idx, group in enumerate(groups)]
    return _cleanup_zero_duration_segments(timeline, terminal_phase=terminal_phase)


def _suppress_isolated_phase_jitter(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) < 3:
        return points
    out = [dict(point) for point in points]
    for index in range(1, len(points) - 1):
        prev_phase = str(points[index - 1]["phase"])
        phase = str(points[index]["phase"])
        next_phase = str(points[index + 1]["phase"])
        if prev_phase != next_phase or phase == prev_phase:
            continue
        if not _is_low_evidence_phase_jitter(points[index]):
            continue
        out[index]["phase"] = prev_phase
        out[index]["source"] = "timeline_filter"
        out[index]["corrected_by_review"] = True
        out[index]["correction_reason"] = "isolated_phase_jitter_suppressed"
    return out


def _is_low_evidence_phase_jitter(point: dict[str, Any]) -> bool:
    if bool(point.get("corrected_by_review")):
        return False
    if point.get("risk_level") == "high" or bool(point.get("imminent_failure")):
        return False
    if bool(point.get("needs_review")) or point.get("progress") == "regressing":
        return False
    if str(point.get("trigger") or "") in {"final_observation", "sequence_start"}:
        return False
    return float(point.get("confidence") or 0.0) <= _MAX_ISOLATED_JITTER_CONFIDENCE


def _timeline_record(group: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    start = group[0]
    end = group[-1]
    risks = [str(item["risk_level"]) for item in group]
    progresses = [str(item["progress"]) for item in group]
    confidence = [float(item["confidence"]) for item in group]
    corrected = any(bool(item.get("corrected_by_review")) for item in group)
    high_frames = [
        int(item["frame_id"])
        for item in group
        if item["risk_level"] == "high" or item.get("imminent_failure")
    ]
    evidence_frames = _dedupe_preserve_order_int(
        [int(start["frame_id"]), *high_frames, int(end["frame_id"])]
    )
    return {
        "segment_index": idx,
        "phase": start["phase"],
        "start_frame": start["frame_id"],
        "end_frame": end["frame_id"],
        "start_timestamp": start["timestamp"],
        "end_timestamp": end["timestamp"],
        "duration_sec": max(0.0, float(end["timestamp"]) - float(start["timestamp"])),
        "snapshot_count": len(group),
        "dominant_progress": _dominant(progresses),
        "max_risk": _worst_risk(risks),
        "needs_review_count": sum(1 for item in group if item["needs_review"]),
        "imminent_failure_count": sum(1 for item in group if item["imminent_failure"]),
        "mean_confidence": round(sum(confidence) / len(confidence), 3),
        "triggers": sorted({str(item.get("trigger") or "unknown") for item in group}),
        "source": "review_corrected" if corrected else "online_vsa",
        "corrected_by_review": corrected,
        "correction_reasons": sorted(
            {str(item["correction_reason"]) for item in group if item.get("correction_reason")}
        ),
        "online_phases": sorted({str(item["online_phase"]) for item in group}),
        "raw_visual_phases": sorted(
            {
                str(item["raw_visual_phase"])
                for item in group
                if item.get("raw_visual_phase") is not None
            }
        ),
        "evidence_frames": evidence_frames,
    }


def _cleanup_zero_duration_segments(
    timeline: list[dict[str, Any]],
    *,
    terminal_phase: str | None,
) -> list[dict[str, Any]]:
    if not timeline:
        return timeline
    cleaned: list[dict[str, Any]] = []
    for index, segment in enumerate(timeline):
        duration = float(segment.get("duration_sec") or 0.0)
        if duration > 0.0:
            cleaned.append(segment)
            continue

        reason = _zero_duration_evidence_reason(segment, terminal_phase=terminal_phase)
        if reason is None and index in {0, len(timeline) - 1}:
            reason = "timeline_boundary"
        if reason is None:
            continue
        retained = dict(segment)
        retained["zero_duration_reason"] = reason
        cleaned.append(retained)

    for index, segment in enumerate(cleaned):
        segment["segment_index"] = index
    return cleaned


def _zero_duration_evidence_reason(
    segment: dict[str, Any],
    *,
    terminal_phase: str | None,
) -> str | None:
    triggers = {str(item) for item in segment.get("triggers") or []}
    if segment.get("max_risk") == "high" or int(segment.get("imminent_failure_count") or 0) > 0:
        return "failure_evidence"
    if int(segment.get("needs_review_count") or 0) > 0:
        return "needs_review"
    if segment.get("dominant_progress") == "regressing":
        return "regressing"
    if "final_observation" in triggers:
        return "final_observation"
    if "sequence_start" in triggers:
        return "sequence_start"
    if terminal_phase is not None and str(segment.get("phase")) == terminal_phase:
        return "terminal_phase"
    return None


def _retry_events(timeline: list[dict[str, Any]], phases: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not phases:
        return events
    max_index_seen = -1
    max_phase_seen: str | None = None
    max_segment_seen: dict[str, Any] | None = None
    for segment in timeline:
        index = _phase_index(phases, str(segment["phase"]))
        if index is None:
            continue
        if index < max_index_seen:
            retry_evidence = _retry_evidence(segment, max_segment_seen)
            if retry_evidence:
                events.append(
                    {
                        "event_type": "retry",
                        "start_frame": segment["start_frame"],
                        "end_frame": segment["end_frame"],
                        "phase": segment["phase"],
                        "from_phase": max_phase_seen,
                        "reason": "phase_sequence_returned_to_earlier_phase",
                        "evidence": retry_evidence,
                        "evidence_frames": segment.get("evidence_frames", []),
                    }
                )
        if index > max_index_seen:
            max_index_seen = index
            max_phase_seen = str(segment["phase"])
            max_segment_seen = segment
    return events


def _retry_evidence(segment: dict[str, Any], prior_max_segment: dict[str, Any] | None) -> list[str]:
    evidence: list[str] = []
    for prefix, item in (("current", segment), ("prior", prior_max_segment or {})):
        if item.get("max_risk") == "high":
            evidence.append(f"{prefix}_high_risk")
        if int(item.get("imminent_failure_count") or 0) > 0:
            evidence.append(f"{prefix}_imminent_failure")
        if item.get("dominant_progress") == "regressing":
            evidence.append(f"{prefix}_regressing")
        if int(item.get("needs_review_count") or 0) > 0:
            evidence.append(f"{prefix}_needs_review")
        correction_reasons = {str(reason) for reason in item.get("correction_reasons") or []}
        if correction_reasons.intersection({"terminal_phase_retry_evidence", "offline_vlm_retry_evidence"}):
            evidence.append(f"{prefix}_review_confirmed_retry")
    return _dedupe_preserve_order_str(evidence)


def _terminal_evidence(
    evidence: EvidenceBundle,
    timeline: list[dict[str, Any]],
) -> dict[str, Any] | None:
    terminal_phase = evidence.terminal_phase
    if terminal_phase is None:
        return None

    final_observation_frames = {
        int(item["anchor_frame_id"])
        for item in evidence.vsa_windows
        if item.get("event_type") == "final_observation"
        and item.get("anchor_frame_id") is not None
    }
    for segment in reversed(timeline):
        if str(segment.get("phase")) != terminal_phase:
            continue
        triggers = {str(item) for item in segment.get("triggers") or []}
        evidence_frames = _int_list(segment.get("evidence_frames"))
        if "final_observation" in triggers or final_observation_frames.intersection(evidence_frames):
            return {
                "source": "final_observation_phase_segment",
                "phase": terminal_phase,
                "frame_id": int(segment.get("end_frame")),
                "segment_index": segment.get("segment_index"),
            }

    for record in reversed(evidence.snapshot_records):
        if (
            str(record.get("trigger") or "") == "final_observation"
            and record.get("raw_visual_phase") == terminal_phase
        ):
            return {
                "source": "final_observation_raw_visual_phase",
                "phase": terminal_phase,
                "frame_id": int(record["frame_id"]),
            }

    for record in reversed(evidence.snapshot_records[-3:]):
        if record.get("raw_visual_phase") == terminal_phase:
            return {
                "source": "trailing_raw_visual_phase",
                "phase": terminal_phase,
                "frame_id": int(record["frame_id"]),
            }

    return None


def _determine_outcome(
    evidence: EvidenceBundle,
    timeline: list[dict[str, Any]],
    retry_events: list[dict[str, Any]],
) -> dict[str, Any]:
    terminal_phase = evidence.terminal_phase
    stabilized_final_phase = str(timeline[-1]["phase"]) if timeline else "unknown"
    final_segment = timeline[-1] if timeline else {}
    final_risk_high = final_segment.get("max_risk") == "high"
    final_regressing = final_segment.get("dominant_progress") == "regressing"
    terminal_evidence = (
        None
        if terminal_phase is not None and stabilized_final_phase == terminal_phase
        else _terminal_evidence(evidence, timeline)
    )
    terminal_reached = terminal_phase is not None and (
        stabilized_final_phase == terminal_phase or terminal_evidence is not None
    )
    final_phase = terminal_phase if terminal_evidence is not None else stabilized_final_phase
    final_success = bool(timeline) and bool(terminal_reached)

    if not timeline:
        status = "unknown"
        confidence = 0.0
        reasoning = "no_snapshots"
    elif final_success and (final_risk_high or final_regressing):
        status = "uncertain"
        confidence = max(0.45, float(final_segment.get("mean_confidence") or 0.0))
        reasoning = (
            "terminal_phase_reached_with_unresolved_final_risk"
            if final_risk_high
            else "terminal_phase_reached_but_final_progress_regressing"
        )
    elif final_success and terminal_evidence is not None:
        status = "uncertain"
        confidence = max(0.45, float(final_segment.get("mean_confidence") or 0.0))
        reasoning = "terminal_phase_supported_by_terminal_evidence"
    elif final_success:
        status = "success"
        confidence = max(0.55, float(final_segment.get("mean_confidence") or 0.0))
        reasoning = "terminal_phase_confirmed_without_final_high_risk"
    else:
        status = "failure"
        confidence = max(0.45, float(final_segment.get("mean_confidence") or 0.0))
        if terminal_phase is None:
            reasoning = "task_has_no_terminal_phase"
        elif not terminal_reached:
            reasoning = "final_phase_not_terminal"
        else:
            reasoning = "terminal_phase_has_unresolved_final_risk"

    return {
        "final_success": final_success,
        "success_status": status,
        "success_confidence": round(min(1.0, confidence), 3),
        "final_phase": final_phase,
        "final_segment_index": final_segment.get("segment_index"),
        "terminal_phase": terminal_phase,
        "terminal_reached": bool(terminal_reached),
        "terminal_evidence": terminal_evidence,
        "retry_count": len(retry_events),
        "reasoning": reasoning,
    }


def _outcome_with_vlm(
    outcome: dict[str, Any],
    vlm_review: dict[str, Any],
    evidence: EvidenceBundle,
) -> dict[str, Any]:
    out = dict(outcome)
    success = _vlm_success_value(vlm_review)
    blocked_failure = _vlm_failure_without_terminal_image_coverage(vlm_review, outcome)
    if isinstance(success, bool):
        if blocked_failure:
            out["final_success"] = bool(out.get("final_success"))
            out["success_status"] = "uncertain"
            out["reasoning"] = "offline_vlm_failure_without_terminal_image_coverage"
        else:
            out["final_success"] = success
            out["success_status"] = "success" if success else "failure"
            out["reasoning"] = str(vlm_review.get("reasoning") or "offline_vlm_final_review")

    final_phase = str(vlm_review.get("final_phase") or "").strip()
    if final_phase in evidence.phases and not blocked_failure:
        out["final_phase"] = final_phase
        out["terminal_reached"] = final_phase == evidence.terminal_phase

    confidence = _float_or_none(vlm_review.get("success_confidence"))
    if blocked_failure:
        current = float(out.get("success_confidence") or 0.0)
        out["success_confidence"] = round(min(0.65, max(0.45, current)), 3)
    elif confidence is not None:
        out["success_confidence"] = round(min(1.0, max(0.0, confidence)), 3)
    else:
        out["success_confidence"] = max(float(out.get("success_confidence") or 0.0), 0.65)
    return out


def _vlm_success_value(vlm_review: dict[str, Any]) -> bool | None:
    success = vlm_review.get("final_success")
    if not isinstance(success, bool):
        success = vlm_review.get("success")
    return success if isinstance(success, bool) else None


def _vlm_failure_without_terminal_image_coverage(
    vlm_review: dict[str, Any],
    deterministic_outcome: dict[str, Any],
) -> bool:
    if _vlm_success_value(vlm_review) is not False:
        return False
    if deterministic_outcome.get("final_success") is not True:
        return False
    return not _vlm_terminal_coverage_sufficient(vlm_review)


def _vlm_terminal_coverage_sufficient(vlm_review: dict[str, Any]) -> bool:
    coverage = vlm_review.get("terminal_image_coverage")
    if not isinstance(coverage, dict):
        return True
    return bool(coverage.get("covers_episode_end"))


def _apply_vlm_phase_corrections(
    points: list[dict[str, Any]],
    vlm_review: dict[str, Any],
    phases: list[str],
) -> list[dict[str, Any]]:
    corrections = vlm_review.get("phase_corrections")
    if not isinstance(corrections, list):
        return points
    out = [dict(point) for point in points]
    for item in corrections:
        if not isinstance(item, dict):
            continue
        phase = str(item.get("phase") or "").strip()
        if phase not in phases:
            continue
        start = _int_or_none(item.get("start_frame"))
        end = _int_or_none(item.get("end_frame"))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        reason = str(item.get("reason") or "offline_vlm_phase_correction")
        for point in out:
            frame_id = int(point["frame_id"])
            if start <= frame_id <= end:
                point["phase"] = phase
                point["source"] = "offline_vlm"
                point["corrected_by_review"] = True
                point["correction_reason"] = reason
    return out


def _apply_vlm_final_phase(
    points: list[dict[str, Any]],
    vlm_review: dict[str, Any],
    phases: list[str],
) -> list[dict[str, Any]]:
    final_phase = str(vlm_review.get("final_phase") or "").strip()
    if not points or final_phase not in phases or points[-1]["phase"] == final_phase:
        return points
    out = [dict(point) for point in points]
    out[-1]["phase"] = final_phase
    out[-1]["source"] = "offline_vlm"
    out[-1]["corrected_by_review"] = True
    out[-1]["correction_reason"] = "offline_vlm_final_phase"
    return out


def _failure_reasons(segment: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if segment["max_risk"] == "high":
        reasons.append("high_risk")
    if segment["imminent_failure_count"] > 0:
        reasons.append("imminent_failure")
    if segment["dominant_progress"] == "regressing":
        reasons.append("regressing")
    if reasons and segment["needs_review_count"] > 0:
        reasons.append("needs_review")
    return reasons


def _failure_event(
    segment: dict[str, Any],
    reasons: list[str],
    evidence: EvidenceBundle,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    recovered = bool(outcome.get("final_success")) and (
        segment.get("segment_index") != outcome.get("final_segment_index")
        or str(segment["phase"]) != str(outcome.get("final_phase"))
    )
    text = _segment_raw_text(evidence, segment)
    failure_type = _classify_failure(segment, text)
    return {
        "event_type": "recovered_failure" if recovered else "failure_candidate",
        "failure_type": failure_type,
        "phase": segment["phase"],
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "severity": _severity(segment),
        "reasons": reasons,
        "recovered": recovered,
        "evidence_frames": segment.get("evidence_frames", []),
        "source": "post_review_failure_analysis",
        "reasoning": _failure_reasoning(failure_type, reasons, recovered),
    }


def _merge_vlm_failure_events(
    deterministic_events: list[dict[str, Any]],
    vlm_review: dict[str, Any],
    evidence: EvidenceBundle,
) -> list[dict[str, Any]]:
    events = list(deterministic_events)
    raw_events = (
        vlm_review.get("retry_or_failure_evidence")
        or vlm_review.get("failure_events")
        or vlm_review.get("failure_modes")
    )
    if not isinstance(raw_events, list):
        return events
    for raw in raw_events:
        if isinstance(raw, str):
            events.append(
                {
                    "event_type": "vlm_failure_mode",
                    "failure_type": raw,
                    "phase": "unknown",
                    "start_frame": None,
                    "end_frame": None,
                    "severity": "medium",
                    "reasons": ["offline_vlm_failure_mode"],
                    "recovered": bool(vlm_review.get("success")),
                    "evidence_frames": [],
                    "source": "offline_vlm",
                    "reasoning": str(vlm_review.get("reasoning") or ""),
                }
            )
            continue
        if not isinstance(raw, dict):
            continue
        phase = str(raw.get("phase") or "unknown")
        if phase not in evidence.phases and phase != "unknown":
            phase = "unknown"
        events.append(
            {
                "event_type": str(raw.get("event_type") or "offline_vlm_failure_event"),
                "failure_type": str(raw.get("failure_type") or raw.get("type") or "unknown"),
                "phase": phase,
                "start_frame": _int_or_none(raw.get("start_frame")),
                "end_frame": _int_or_none(raw.get("end_frame")),
                "severity": str(raw.get("severity") or "medium"),
                "reasons": _string_list(raw.get("reasons")) or ["offline_vlm_failure_event"],
                "recovered": bool(raw.get("recovered", vlm_review.get("success", False))),
                "evidence_frames": _int_list(raw.get("evidence_frames")),
                "source": "offline_vlm",
                "reasoning": str(raw.get("reasoning") or vlm_review.get("reasoning") or ""),
            }
        )
    return _dedupe_failure_events(events)


def _candidate_segment(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": event.get("phase"),
        "start_frame": event.get("start_frame"),
        "end_frame": event.get("end_frame"),
        "reasons": event.get("reasons") or [],
        "severity": event.get("severity", "medium"),
        "failure_type": event.get("failure_type", "unknown"),
        "recovered": bool(event.get("recovered")),
        "evidence_frames": event.get("evidence_frames") or [],
    }


def _rollout_summary(
    evidence: EvidenceBundle,
    annotation: dict[str, Any],
    failures: dict[str, Any],
    admission: dict[str, Any],
) -> dict[str, Any]:
    timeline = list(annotation.get("phase_timeline") or [])
    outcome = dict(annotation.get("outcome") or {})
    high_risk_count = sum(1 for item in evidence.snapshots if item.risk_level == "high")
    needs_review_count = sum(1 for item in evidence.snapshots if item.needs_review)
    timestamps = [item.timestamp for item in evidence.snapshots]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "rollout_id": evidence.rollout_id,
        "task_description": evidence.task_config.get("task_description", ""),
        "snapshot_count": len(evidence.snapshots),
        "phase_count": len(timeline),
        "phases_seen": [segment["phase"] for segment in timeline],
        "final_phase": outcome.get("final_phase", "unknown"),
        "terminal_phase": outcome.get("terminal_phase"),
        "final_success": bool(outcome.get("final_success")),
        "success_likely": bool(outcome.get("final_success")),
        "success_status": outcome.get("success_status", "unknown"),
        "success_confidence": outcome.get("success_confidence", 0.0),
        "success_reasoning": outcome.get("reasoning", ""),
        "used_vlm": bool(outcome.get("used_vlm")),
        "vlm_error": outcome.get("vlm_error"),
        "retry_count": outcome.get("retry_count", 0),
        "recovered_failure_count": failures.get("recovered_count", 0),
        "high_risk_count": high_risk_count,
        "needs_review_count": needs_review_count,
        "failure_candidate_count": failures.get("candidate_count", 0),
        "dataset_decision": admission.get("decision"),
        "admission_class": admission.get("admission_class"),
        "started_at": min(timestamps) if timestamps else None,
        "ended_at": max(timestamps) if timestamps else None,
    }


def _review_prompt_and_images(
    evidence: EvidenceBundle,
    timeline: list[dict[str, Any]],
    outcome: dict[str, Any],
    retry_events: list[dict[str, Any]],
    max_images: int,
) -> tuple[str, list[int], list[np.ndarray]]:
    frame_ids = _select_review_frame_ids(evidence, max_images)
    image_rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    for index, frame_id in enumerate(frame_ids, start=1):
        image_path = _best_image_for_frame(evidence.image_index, frame_id)
        if image_path is None:
            continue
        image = _load_image(image_path)
        if image is None:
            continue
        image_rows.append({"image_index": index, "frame_id": frame_id, "path": str(image_path)})
        images.append(image)

    compact = {
        "task": {
            "description": evidence.task_config.get("task_description", ""),
            "phases": evidence.phases,
            "terminal_phase": evidence.terminal_phase,
            "failure_signals": evidence.task_config.get("failure_signals") or [],
            "phase_visual_hints": evidence.task_config.get("phase_visual_hints") or {},
        },
        "deterministic_outcome": outcome,
        "deterministic_timeline": timeline,
        "retry_events": retry_events,
        "visual_disagreements": evidence.visual_disagreements,
        "key_frames": evidence.key_frames,
        "selected_images": image_rows,
    }
    prompt = (
        "You are RoboLineage's offline post-rollout Annotation and Failure Analysis Agent.\n"
        "Review the complete evidence summary plus the ordered images. The online VSA may be wrong, "
        "especially after retries or after a terminal phase was temporarily reached.\n\n"
        "Return JSON only with these keys:\n"
        "{\n"
        '  "final_success": true|false,\n'
        '  "success_confidence": 0.0-1.0,\n'
        '  "final_phase": "one of the provided phases",\n'
        '  "terminal_evidence": "short visual evidence for the terminal state",\n'
        '  "phase_corrections": [\n'
        '    {"start_frame": int, "end_frame": int, "phase": "phase_name", "reason": "short reason"}\n'
        "  ],\n"
        '  "retry_or_failure_evidence": [\n'
        '    {"failure_type": "grasp_miss|slip|collision|wrong_target|release_failure|timeout|uncertain", '
        '"phase": "phase_name", "start_frame": int, "end_frame": int, '
        '"recovered": true|false, "evidence_frames": [int], "reasoning": "short reason"}\n'
        "  ],\n"
        '  "label_quality": "clean|needs_correction|uncertain",\n'
        '  "training_usability": true|false,\n'
        '  "reasoning": "one concise explanation"\n'
        "}\n\n"
        "Important: retry is not a phase name. If the operator/robot retries, express it as an event "
        "and reuse the original phase names in phase_corrections. Do not decide dataset admission; "
        "training_usability and label_quality are advisory only. Do not wrap the JSON in markdown or "
        "add explanatory prose outside the JSON object. If selected_images is empty, do not "
        "invent slip, grasp miss, terminal failure, or object state; return insufficient_visual_evidence "
        "in reasoning.\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )
    return prompt, [row["frame_id"] for row in image_rows], images


def _review_packets_and_images(
    evidence: EvidenceBundle,
    timeline: list[dict[str, Any]],
    outcome: dict[str, Any],
    retry_events: list[dict[str, Any]],
    max_images: int,
) -> list[dict[str, Any]]:
    if max_images <= 0:
        return []
    candidate_frame_ids = _review_candidate_frame_ids(evidence)
    if len(candidate_frame_ids) <= max_images:
        return []

    packet_frames: list[tuple[str, list[int]]] = []
    selected_terminal_frames: set[int] = set()
    for group_index, group in enumerate(_terminal_review_frame_groups(evidence)):
        purpose = "terminal_focus" if group_index == 0 else "post_terminal_context"
        available = [
            frame_id
            for frame_id in group
            if frame_id in evidence.image_index and frame_id not in selected_terminal_frames
        ]
        if not available:
            continue
        for index in range(0, len(available), max_images):
            chunk = available[index:index + max_images]
            packet_frames.append((purpose, chunk))
            selected_terminal_frames.update(chunk)

    if not packet_frames:
        packet_frames.append(("terminal_focus", _select_review_frame_ids(evidence, max_images)))

    selected_frames = {frame_id for _, frame_ids in packet_frames for frame_id in frame_ids}
    remaining = [
        frame_id
        for frame_id in candidate_frame_ids
        if frame_id not in selected_frames
    ]
    for index in range(0, len(remaining), max_images):
        packet_frames.append(("temporal_context", remaining[index:index + max_images]))

    packets: list[dict[str, Any]] = []
    for packet_index, (purpose, frame_ids) in enumerate(packet_frames):
        rows, images = _image_rows_for_frames(evidence, frame_ids)
        if not images:
            continue
        packet_id = f"packet_{packet_index:03d}_{purpose}"
        packet = {
            "packet_id": packet_id,
            "packet_index": packet_index,
            "purpose": purpose,
            "image_frames": [row["frame_id"] for row in rows],
            "selected_images": rows,
        }
        packet["prompt"] = _packet_review_prompt(
            evidence,
            timeline,
            outcome,
            retry_events,
            packet,
        )
        packet["images"] = images
        packets.append(packet)
    return packets if len(packets) > 1 else []


def _packet_review_prompt(
    evidence: EvidenceBundle,
    timeline: list[dict[str, Any]],
    outcome: dict[str, Any],
    retry_events: list[dict[str, Any]],
    packet: dict[str, Any],
) -> str:
    compact = {
        "task": {
            "description": evidence.task_config.get("task_description", ""),
            "phases": evidence.phases,
            "terminal_phase": evidence.terminal_phase,
            "failure_signals": evidence.task_config.get("failure_signals") or [],
            "phase_visual_hints": evidence.task_config.get("phase_visual_hints") or {},
        },
        "deterministic_outcome": outcome,
        "deterministic_timeline": timeline,
        "retry_events": retry_events,
        "visual_disagreements": evidence.visual_disagreements,
        "key_frames": evidence.key_frames,
        "review_packet": {
            "packet_id": packet["packet_id"],
            "packet_index": packet["packet_index"],
            "purpose": packet["purpose"],
            "selected_images": packet["selected_images"],
        },
    }
    return (
        "You are RoboLineage's offline post-rollout packet reviewer. Review only this packet's ordered "
        "images, and return local visual evidence. Do not make a whole-episode dataset "
        "admission decision.\n\n"
        "Return JSON only with these keys:\n"
        "{\n"
        '  "packet_terminal_state": "success|failure|uncertain|not_visible",\n'
        '  "terminal_intact_at_packet_end": true|false|null,\n'
        '  "post_terminal_status": "intact|broken|uncertain|not_applicable",\n'
        '  "success_confidence": 0.0-1.0,\n'
        '  "final_phase": "one of the provided phases or unknown",\n'
        '  "evidence_frames": [int],\n'
        '  "retry_or_failure_evidence": [\n'
        '    {"failure_type": "grasp_miss|slip|collision|wrong_target|release_failure|timeout|uncertain", '
        '"phase": "phase_name", "start_frame": int, "end_frame": int, '
        '"recovered": true|false, "evidence_frames": [int], "reasoning": "short reason"}\n'
        "  ],\n"
        '  "reasoning": "one concise explanation"\n'
        "}\n\n"
        "Important: terminal_intact_at_packet_end should be true when this packet shows the "
        "task's final object arrangement is achieved and remains intact by the packet's last "
        "image. If this is a post-terminal contact or gripper motion that does not break the "
        "terminal object state, use post_terminal_status=intact. If the terminal state is "
        "destroyed or the object is re-grasped and removed after success, use "
        "post_terminal_status=broken.\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def _aggregate_packet_reviews(
    packet_reviews: list[dict[str, Any]],
    evidence: EvidenceBundle,
    deterministic_outcome: dict[str, Any],
) -> dict[str, Any]:
    success_reviews: list[dict[str, Any]] = []
    broken_reviews: list[dict[str, Any]] = []
    failure_events: list[dict[str, Any]] = []
    reasoning: list[str] = []
    for review in packet_reviews:
        purpose = str(review.get("purpose") or "terminal_focus")
        terminal_state = str(review.get("packet_terminal_state") or "").strip()
        post_terminal_status = str(review.get("post_terminal_status") or "").strip()
        terminal_intact = review.get("terminal_intact_at_packet_end")
        if (
            terminal_state == "success"
            or terminal_intact is True
            or post_terminal_status == "intact"
        ):
            success_reviews.append(review)
        if (
            (terminal_state == "failure" and purpose == "terminal_focus")
            or terminal_intact is False
            or post_terminal_status == "broken"
        ):
            broken_reviews.append(review)
        for item in review.get("retry_or_failure_evidence") or []:
            if isinstance(item, dict):
                failure_events.append(item)
        if review.get("reasoning"):
            reasoning.append(str(review["reasoning"]))

    latest_success = max(success_reviews, key=lambda item: int(item.get("packet_index") or 0), default=None)
    latest_broken = max(broken_reviews, key=lambda item: int(item.get("packet_index") or 0), default=None)
    success_index = int(latest_success.get("packet_index") or 0) if latest_success else -1
    broken_index = int(latest_broken.get("packet_index") or 0) if latest_broken else -1

    confidence_values = [
        value
        for value in (_float_or_none(item.get("success_confidence")) for item in packet_reviews)
        if value is not None
    ]
    confidence = max(confidence_values) if confidence_values else 0.65

    final_success: bool | None
    if latest_broken is not None and broken_index >= success_index:
        final_success = False
        final_phase = str(latest_broken.get("final_phase") or deterministic_outcome.get("final_phase") or "unknown")
    elif latest_success is not None:
        final_success = True
        final_phase = evidence.terminal_phase or str(deterministic_outcome.get("final_phase") or "unknown")
    else:
        final_success = None
        final_phase = str(deterministic_outcome.get("final_phase") or "unknown")

    out = {
        "status": "packet_aggregated",
        "packet_count": len(packet_reviews),
        "success_confidence": round(min(1.0, max(0.0, confidence)), 3),
        "final_phase": final_phase,
        "retry_or_failure_evidence": failure_events,
        "label_quality": "clean" if final_success is True and not failure_events else "uncertain",
        "training_usability": final_success is not False,
        "reasoning": " | ".join(_dedupe_preserve_order_str(reasoning)) or "packet_reviews_aggregated",
    }
    if final_success is not None:
        out["final_success"] = final_success
    return out


def _packet_metadata(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "packet_index": packet["packet_index"],
        "purpose": packet["purpose"],
        "image_frames": packet["image_frames"],
        "selected_images": packet["selected_images"],
    }


def _failed_packet_review(packet: dict[str, Any], exc: VLMInferenceError) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "packet_index": packet["packet_index"],
        "purpose": packet["purpose"],
        "image_frames": packet["image_frames"],
        "status": "failed",
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def _packet_failure_metadata(packet_review: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet_review.get("packet_id"),
        "packet_index": packet_review.get("packet_index"),
        "purpose": packet_review.get("purpose"),
        "image_frames": packet_review.get("image_frames"),
        "error": packet_review.get("error"),
        "error_type": packet_review.get("error_type"),
    }


def _image_rows_for_frames(
    evidence: EvidenceBundle,
    frame_ids: list[int],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    image_rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    for index, frame_id in enumerate(frame_ids, start=1):
        image_path = _best_image_for_frame(evidence.image_index, frame_id)
        if image_path is None:
            continue
        image = _load_image(image_path)
        if image is None:
            continue
        image_rows.append({"image_index": index, "frame_id": frame_id, "path": str(image_path)})
        images.append(image)
    return image_rows, images


def _select_review_frame_ids(evidence: EvidenceBundle, max_images: int) -> list[int]:
    if max_images <= 0:
        return []
    selected: list[int] = []
    for group in _terminal_review_frame_groups(evidence):
        for frame_id in group:
            if len(selected) >= max_images:
                return sorted(selected)
            if frame_id in evidence.image_index and frame_id not in selected:
                selected.append(frame_id)

    scored: dict[int, int] = {}
    for row in evidence.key_frames:
        frame_id = int(row["frame_id"])
        if frame_id not in evidence.image_index or frame_id in selected:
            continue
        score = 1
        reasons = set(row.get("reasons") or [])
        event_types = {str(item) for item in row.get("event_types") or []}
        if row.get("event_type") is not None:
            event_types.add(str(row.get("event_type")))
        if "failure_candidate" in reasons:
            score += 4
        if "visual_disagreement" in reasons:
            score += 3
        if "phase_boundary" in reasons:
            score += 2
        if "end" in reasons:
            score += 3
        if "final_observation" in reasons:
            score += 5
        if "vsa_window" in reasons and "sequence_start" not in event_types:
            score += 1
        scored[frame_id] = max(scored.get(frame_id, 0), score)
    ordered = sorted(scored, key=lambda frame_id: (-scored[frame_id], frame_id))
    for frame_id in ordered:
        if len(selected) >= max_images:
            break
        selected.append(frame_id)
    return sorted(selected)


def _review_candidate_frame_ids(evidence: EvidenceBundle) -> list[int]:
    scored: dict[int, int] = {}
    for row in evidence.key_frames:
        frame_id = int(row["frame_id"])
        if frame_id not in evidence.image_index:
            continue
        score = 1
        reasons = set(row.get("reasons") or [])
        if "failure_candidate" in reasons:
            score += 4
        if "visual_disagreement" in reasons:
            score += 3
        if "phase_boundary" in reasons:
            score += 2
        if "end" in reasons:
            score += 3
        if "final_observation" in reasons:
            score += 5
        if "vsa_window" in reasons:
            score += 1
        scored[frame_id] = max(scored.get(frame_id, 0), score)
    for frame_id in evidence.image_index:
        scored.setdefault(int(frame_id), 0)
    return sorted(scored)


def _review_image_coverage(
    evidence: EvidenceBundle,
    selected_frame_ids: list[int],
) -> dict[str, Any]:
    selected = set(selected_frame_ids)
    required_groups = _terminal_review_frame_groups(evidence)
    missing_groups = [
        group
        for group in required_groups
        if group and not selected.intersection(group)
    ]
    return {
        "covers_episode_end": not missing_groups,
        "required_terminal_frame_groups": required_groups,
        "missing_terminal_frame_groups": missing_groups,
    }


def _terminal_review_frame_groups(evidence: EvidenceBundle) -> list[list[int]]:
    groups: list[list[int]] = []

    def add_group(frame_ids: list[int]) -> None:
        group = _dedupe_preserve_order_int([int(item) for item in frame_ids])
        if not group:
            return
        group_set = set(group)
        if any(group_set.issubset(set(existing)) for existing in groups):
            return
        groups.append(group)

    final_windows = [
        window
        for window in evidence.vsa_windows
        if str(window.get("event_type") or "") == "final_observation"
    ]
    latest_final_frame: int | None = None
    if final_windows:
        latest_final = max(final_windows, key=_vsa_window_sort_frame)
        latest_final_frame = _vsa_window_sort_frame(latest_final)
        add_group(_vsa_window_frame_ids(latest_final))

    if latest_final_frame is not None:
        post_final_windows = [
            window
            for window in evidence.vsa_windows
            if _vsa_window_sort_frame(window) > latest_final_frame
        ]
        if post_final_windows:
            add_group(_vsa_window_frame_ids(max(post_final_windows, key=_vsa_window_sort_frame)))
    elif evidence.vsa_windows:
        add_group(_vsa_window_frame_ids(max(evidence.vsa_windows, key=_vsa_window_sort_frame)))

    episode_end = _episode_end_frame_id(evidence)
    if episode_end is not None:
        add_group([episode_end])
    return groups


def _vsa_window_frame_ids(window: dict[str, Any]) -> list[int]:
    frame_ids = _int_list(window.get("keyframe_ids"))
    anchor = _int_or_none(window.get("anchor_frame_id"))
    if not frame_ids and anchor is not None:
        frame_ids.append(anchor)
    end = _int_or_none(window.get("end_frame_id"))
    if end is not None:
        frame_ids.append(end)
    return _dedupe_preserve_order_int(frame_ids)


def _vsa_window_sort_frame(window: dict[str, Any]) -> int:
    frame_ids = _vsa_window_frame_ids(window)
    return max(frame_ids) if frame_ids else -1


def _episode_end_frame_id(evidence: EvidenceBundle) -> int | None:
    frame_ids: list[int] = []
    frame_ids.extend(int(record["frame_id"]) for record in evidence.snapshot_records)
    for window in evidence.vsa_windows:
        frame_ids.extend(_vsa_window_frame_ids(window))
    frame_ids.extend(int(frame_id) for frame_id in evidence.image_index)
    return max(frame_ids) if frame_ids else None


def _best_image_for_frame(image_index: dict[int, list[Path]], frame_id: int) -> Path | None:
    paths = image_index.get(frame_id) or []
    if not paths:
        return None
    preferred = [path for path in paths if path.name.endswith("_2.jpg")]
    return preferred[0] if preferred else paths[len(paths) // 2]


def _load_image(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image as PILImage

        return np.asarray(PILImage.open(path).convert("RGB"))
    except Exception as exc:
        _LOG.warning("failed to load review image %s: %s", path, exc)
        return None


def _sanitize_vlm_review(vlm_review: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(vlm_review)
    ignored = [field for field in _VLM_ADMISSION_FIELDS if field in sanitized]
    for field in ignored:
        sanitized.pop(field, None)
    if ignored:
        sanitized["ignored_legacy_fields"] = sorted(ignored)
    return sanitized


def _l1_annotation(evidence: EvidenceBundle, timeline: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not evidence.phases or not timeline:
        return None
    return {
        "schema_version": L1_SCHEMA_VERSION,
        "annotator": f"agent:{AGENT_VERSION}",
        "annotated_at": _now_iso(),
        "phases": evidence.phases,
        "goal": evidence.task_config.get("goal")
        or evidence.task_config.get("task_description")
        or "complete rollout task",
        "success_criterion": evidence.task_config.get("success_criterion")
        or {
            "type": "visual",
            "description": f"Task reaches terminal phase {evidence.terminal_phase!r}.",
        },
        "object_roles": evidence.task_config.get("object_roles"),
        "subtasks": evidence.task_config.get("subtasks"),
        "phase_segments": [
            {
                "phase": segment["phase"],
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
                "start_ts": segment["start_timestamp"],
                "end_ts": segment["end_timestamp"],
            }
            for segment in timeline
        ],
        "frame_tags": _frame_tags_from_evidence(evidence, timeline) or None,
    }


def _frame_tags_from_evidence(
    evidence: EvidenceBundle,
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    for record in evidence.snapshot_records:
        frame_id = int(record["frame_id"])
        if record.get("risk_level") == "high":
            tags.append({"frame_id": frame_id, "tag": "risk", "note": "risk_level=high"})
        if record.get("needs_review"):
            tags.append({"frame_id": frame_id, "tag": "review", "note": "online_vsa_needs_review"})
    for segment in timeline:
        if segment.get("corrected_by_review"):
            tags.append(
                {
                    "frame_id": int(segment["start_frame"]),
                    "tag": "phase_correction",
                    "note": ",".join(segment.get("correction_reasons") or []),
                }
            )
    return tags


def _render_report(
    summary: dict[str, Any],
    timeline: list[dict[str, Any]],
    failures: dict[str, Any],
    admission: dict[str, Any],
) -> str:
    rows = [
        "# Post-rollout Review",
        "",
        f"- rollout_id: `{summary['rollout_id']}`",
        f"- task: {summary.get('task_description') or '(unknown)'}",
        f"- final_success: `{summary['final_success']}`",
        f"- success_confidence: `{summary['success_confidence']}`",
        f"- final_phase: `{summary['final_phase']}`",
        f"- retry_count: `{summary['retry_count']}`",
        f"- recovered_failure_count: `{summary['recovered_failure_count']}`",
        f"- dataset_decision: `{admission['decision']}`",
        f"- admission_class: `{admission.get('admission_class')}`",
        "",
        "## Phase Timeline",
        "",
    ]
    for segment in timeline:
        corrected = " corrected" if segment.get("corrected_by_review") else ""
        rows.append(
            "- "
            f"{segment['segment_index']}: `{segment['phase']}` "
            f"frames {segment['start_frame']}..{segment['end_frame']} "
            f"risk={segment['max_risk']} progress={segment['dominant_progress']}"
            f"{corrected}"
        )
    rows.extend(["", "## Failure Events", ""])
    if not failures.get("failure_events"):
        rows.append("- none")
    for event in failures.get("failure_events", []):
        recovered = " recovered" if event.get("recovered") else ""
        rows.append(
            "- "
            f"`{event.get('failure_type')}` phase={event.get('phase')} "
            f"frames {event.get('start_frame')}..{event.get('end_frame')} "
            f"severity={event.get('severity')} reasons={','.join(event.get('reasons') or [])}"
            f"{recovered}"
        )
    rows.extend(["", "## Admission Reasons", ""])
    rows.extend(f"- {reason}" for reason in admission.get("reasons", []))
    data_use = admission.get("data_use") or []
    if data_use:
        rows.extend(["", "## Data Use", ""])
        rows.extend(f"- {item}" for item in data_use)
    rows.append("")
    return "\n".join(rows)


def _segment_raw_text(evidence: EvidenceBundle, segment: dict[str, Any]) -> str:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    texts = [
        str(record.get("raw_response") or "")
        for record in evidence.snapshot_records
        if start <= int(record["frame_id"]) <= end
    ]
    return "\n".join(texts).lower()


def _classify_failure(segment: dict[str, Any], text: str) -> str:
    if "wrong target" in text or "wrong object" in text:
        return "wrong_target"
    if "collision" in text or "collide" in text:
        return "collision"
    if "timeout" in text:
        return "timeout"
    if "slip" in text or "slipped" in text:
        return "slip"
    if "drop" in text or "dropped" in text:
        return "drop"
    if "release" in text and ("premature" in text or "not on" in text or "table" in text):
        return "release_failure"
    if "grasp" in text and ("miss" in text or "failed" in text or "not lifted" in text):
        return "grasp_miss"
    phase = str(segment.get("phase") or "")
    if "grasp" in phase and segment.get("max_risk") == "high":
        return "grasp_failure"
    if ("place" in phase or "release" in phase) and segment.get("max_risk") == "high":
        return "placement_or_release_failure"
    return "uncertain"


def _failure_reasoning(failure_type: str, reasons: list[str], recovered: bool) -> str:
    base = f"classified_as={failure_type}; signals={','.join(reasons)}"
    if recovered:
        return base + "; final annotation indicates the rollout recovered later"
    return base


def _severity(segment: dict[str, Any]) -> str:
    if segment["max_risk"] == "high" or segment["imminent_failure_count"] > 0:
        return "high"
    if segment["needs_review_count"] > 0:
        return "medium"
    return "low"


def _dominant(values: list[str], fallback: str = "unknown") -> str:
    if not values:
        return fallback
    return Counter(values).most_common(1)[0][0]


def _worst_risk(values: list[str]) -> str:
    for risk in ("high", "medium", "low", "unknown"):
        if risk in values:
            return risk
    return "unknown"


def _phase_index(phases: list[str], phase: str) -> int | None:
    try:
        return phases.index(phase)
    except ValueError:
        return None


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 2 else text
        if text.startswith("json"):
            text = "\n".join(text.splitlines()[1:]).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    for candidate in sorted(re.findall(r"\{.*?\}", text, flags=re.DOTALL), key=len, reverse=True):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _log_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {"exists": True, "error": str(exc)}
    return {"exists": True, "line_count": len(lines)}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        maybe = _int_or_none(item)
        if maybe is not None:
            out.append(maybe)
    return out


def _dedupe_preserve_order_int(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_preserve_order_str(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_failure_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for event in events:
        key = (
            event.get("failure_type"),
            event.get("phase"),
            event.get("start_frame"),
            event.get("end_frame"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def _write_json_atomic(path: Path, data: dict[str, Any]) -> Path:
    return _write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> Path:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return _write_text_atomic(path, text)


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
