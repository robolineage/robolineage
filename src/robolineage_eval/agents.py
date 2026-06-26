from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robolineage_shared_agents.json_llm import OpenAICompatibleJSONClient
from robolineage_shared_agents.visual_snapshot.vlm_runner import BaseVLMRunner
from robolineage_post_rollout.formal_review import (
    DEFAULT_MAX_REVIEW_IMAGES,
    AnnotationAgent,
    EvidenceBuilder,
    FailureAnalysisAgent,
)
from robolineage_schemas.artifacts import write_validated_json_atomic


EVAL_SCHEMA_VERSION = "RoboLineage.policy_evaluation.rollout.v1"
SUMMARY_SCHEMA_VERSION = "RoboLineage.policy_evaluation.session.v1"
DECISION_SCHEMA_VERSION = "RoboLineage.deployment_governance.v1"
GOVERNANCE_UNDERSTANDING_SCHEMA_VERSION = "RoboLineage.deployment_governance_understanding.v1"
COLLECTION_SCHEMA_VERSION = "RoboLineage.collection_recommendation.v1"
BRIEF_SCHEMA_VERSION = "RoboLineage.next_collection_brief.v1"
AGENT_VERSION = "policy_eval_governance@0.1"


@dataclass(frozen=True)
class PolicyEvaluationResult:
    rollout_dir: Path
    rollout_id: str
    status: str
    artifacts: dict[str, str]
    used_vlm: bool
    policy_version: str | None = None


class PolicyEvaluationAgent:
    """Evaluation/deployment rollout review.

    This reuses the post-rollout evidence, final annotation and failure
    analysis path, then replaces dataset admission with policy evaluation and
    next-collection recommendations.
    """

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

    def run(
        self,
        rollout_dir: str | Path,
        *,
        policy_version: str | None = None,
        evaluation_session_id: str | None = None,
        evaluation_mode: str = "evaluation",
    ) -> PolicyEvaluationResult:
        rollout_path = Path(rollout_dir)
        rollout_id = rollout_path.name
        artifacts: dict[str, str] = {}
        _write_json_atomic(
            rollout_path / "policy_eval_status.json",
            {
                "status": "running",
                "rollout_id": rollout_id,
                "policy_version": policy_version,
                "evaluation_session_id": evaluation_session_id,
                "started_at": _now_iso(),
                "agent_version": AGENT_VERSION,
            },
        )

        evidence = self.evidence_builder.build(rollout_path)
        annotation = self.annotation_agent.annotate(evidence)
        failures = self.failure_agent.analyze(evidence, annotation)
        rollout_eval = _rollout_evaluation(
            evidence=evidence,
            annotation=annotation,
            failures=failures,
            policy_version=policy_version,
            evaluation_session_id=evaluation_session_id,
            evaluation_mode=evaluation_mode,
        )
        collection = _collection_recommendation_from_rollout(rollout_eval, failures)

        artifacts["evidence_index"] = str(_write_json_atomic(rollout_path / "evidence_index.json", evidence.to_index()))
        artifacts["annotation"] = str(
            write_validated_json_atomic(rollout_path / "annotation.final.json", annotation, "annotation_final")
        )
        artifacts["failure_analysis"] = str(
            write_validated_json_atomic(rollout_path / "failure_analysis.json", failures, "failure_analysis")
        )
        artifacts["policy_evaluation"] = str(
            write_validated_json_atomic(rollout_path / "policy_evaluation.json", rollout_eval, "policy_evaluation")
        )
        artifacts["collection_recommendation"] = str(
            write_validated_json_atomic(
                rollout_path / "collection_recommendation.json",
                collection,
                "collection_recommendation",
            )
        )
        artifacts["eval_review_report"] = str(
            _write_text_atomic(
                rollout_path / "eval_review_report.md",
                _render_eval_report(rollout_eval, failures, collection),
            )
        )

        outcome = annotation.get("outcome") if isinstance(annotation.get("outcome"), dict) else {}
        status_payload = {
            "status": "completed",
            "rollout_id": rollout_id,
            "policy_version": policy_version,
            "evaluation_session_id": evaluation_session_id,
            "completed_at": _now_iso(),
            "artifacts": artifacts,
            "used_vlm": bool(outcome.get("used_vlm")),
            "vlm_error": outcome.get("vlm_error"),
            "agent_version": AGENT_VERSION,
        }
        artifacts["status"] = str(_write_json_atomic(rollout_path / "policy_eval_status.json", status_payload))
        return PolicyEvaluationResult(
            rollout_dir=rollout_path,
            rollout_id=rollout_id,
            status="completed",
            artifacts=artifacts,
            used_vlm=bool(outcome.get("used_vlm")),
            policy_version=policy_version,
        )


class DeploymentGovernanceAgent:
    """Session-level policy evaluation and deployment governance."""

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        enable_llm_understanding: bool = True,
        enable_env_llm: bool = True,
    ) -> None:
        self.enable_llm_understanding = enable_llm_understanding
        self.llm_client = (
            llm_client
            if llm_client is not None
            else OpenAICompatibleJSONClient.from_env(
                prefix="DEPLOYMENT_GOVERNANCE_LLM",
                system_prompt=(
                    "You are RoboLineage Deployment Governance Agent. Read deterministic policy evaluation "
                    "summary, deployment decision, collection recommendation and next collection brief. "
                    "Return JSON only with keys: summary, risk_notes, llm_suggested_decision, "
                    "operator_brief, confidence. The deterministic decision is authoritative."
                ),
                timeout_default=30.0,
            )
            if enable_env_llm
            else None
        )

    def summarize_session(
        self,
        *,
        rollout_dirs: list[Path],
        output_dir: Path,
        session_id: str,
        policy_version: str | None,
        mode: str = "deployment",
        deploy_success_threshold: float = 0.8,
        rollback_success_threshold: float = 0.5,
    ) -> dict[str, Any]:
        evaluations = [
            payload
            for payload in (_read_json(path / "policy_evaluation.json") for path in rollout_dirs)
            if payload
        ]
        summary = aggregate_policy_evaluations(
            evaluations,
            session_id=session_id,
            policy_version=policy_version,
            mode=mode,
        )
        decision = _deployment_decision(
            summary,
            deploy_success_threshold=deploy_success_threshold,
            rollback_success_threshold=rollback_success_threshold,
            mode=mode,
        )
        collection = _session_collection_recommendation(summary, decision)
        brief = _next_collection_brief(summary, decision, collection)
        understanding = self._generate_understanding(summary, decision, collection, brief)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_dir / "policy_eval_summary.json", summary)
        write_validated_json_atomic(output_dir / "deployment_decision.json", decision, "deployment_decision")
        write_validated_json_atomic(
            output_dir / "collection_recommendation.json",
            collection,
            "collection_recommendation",
        )
        write_validated_json_atomic(output_dir / "next_collection_brief.json", brief, "next_collection_brief")
        _write_json_atomic(output_dir / "deployment_governance_understanding.json", understanding)
        _write_text_atomic(output_dir / "deployment_governance_understanding.md", _render_governance_understanding(understanding))
        _write_text_atomic(
            output_dir / "deployment_session_report.md",
            _render_session_report(summary, decision, collection, brief, understanding),
        )
        return {
            "session_id": session_id,
            "output_dir": str(output_dir),
            "policy_eval_summary": summary,
            "deployment_decision": decision,
            "collection_recommendation": collection,
            "next_collection_brief": brief,
            "deployment_governance_understanding": understanding,
        }

    def _generate_understanding(
        self,
        summary: dict[str, Any],
        decision: dict[str, Any],
        collection: dict[str, Any],
        brief: dict[str, Any],
    ) -> dict[str, Any]:
        model = str(getattr(self.llm_client, "model", "") or "")
        base = {
            "summary": _governance_summary(summary, decision),
            "risk_notes": [],
            "llm_suggested_decision": decision.get("decision"),
            "operator_brief": brief.get("operator_brief") or _governance_summary(summary, decision),
            "confidence": None,
        }
        if not self.enable_llm_understanding:
            return _governance_understanding_payload(
                status="skipped",
                model=None,
                deterministic_decision=str(decision.get("decision") or ""),
                body={**base, "reason": "disabled"},
            )
        if self.llm_client is None:
            return _governance_understanding_payload(
                status="not_configured",
                model=None,
                deterministic_decision=str(decision.get("decision") or ""),
                body={**base, "reason": "DEPLOYMENT_GOVERNANCE_LLM_API_KEY is not set"},
            )
        try:
            generated = self.llm_client.generate(
                {
                    "policy_eval_summary": summary,
                    "deterministic_decision": decision,
                    "collection_recommendation": collection,
                    "next_collection_brief": brief,
                    "contract": {
                        "summary": "short governance summary",
                        "risk_notes": "list of deployment risks",
                        "llm_suggested_decision": "diagnostic suggestion only; deterministic decision remains authoritative",
                        "operator_brief": "operator-facing paragraph",
                        "confidence": "0-1 confidence",
                    },
                }
            )
            if not isinstance(generated, dict):
                raise RuntimeError("Deployment governance LLM returned a non-object response")
            return _governance_understanding_payload(
                status="generated",
                model=model or None,
                deterministic_decision=str(decision.get("decision") or ""),
                body=generated,
            )
        except Exception as exc:
            return _governance_understanding_payload(
                status="failed",
                model=model or None,
                deterministic_decision=str(decision.get("decision") or ""),
                body={**base, "error": str(exc)},
            )


def aggregate_policy_evaluations(
    evaluations: list[dict[str, Any]],
    *,
    session_id: str,
    policy_version: str | None,
    mode: str,
) -> dict[str, Any]:
    rollout_count = len(evaluations)
    success_count = sum(1 for item in evaluations if item.get("final_success") is True)
    failure_count = rollout_count - success_count
    failure_types: Counter[str] = Counter()
    phase_failures: Counter[str] = Counter()
    phase_attempts: Counter[str] = Counter()
    recommendations: Counter[str] = Counter()

    for item in evaluations:
        for phase in item.get("phases_seen") or []:
            phase_attempts[str(phase)] += 1
        for failure_type, count in (item.get("failure_type_counts") or {}).items():
            failure_types[str(failure_type)] += int(count)
        for weakness in item.get("phase_weakness") or []:
            phase = str(weakness.get("phase") or "unknown")
            phase_failures[phase] += 1
        recommendation = item.get("recommended_next_action")
        if recommendation:
            recommendations[str(recommendation)] += 1

    phase_weakness = [
        {
            "phase": phase,
            "failure_or_weak_rollouts": count,
            "attempted_rollouts": phase_attempts.get(phase, 0),
            "weakness_rate": round(count / max(1, phase_attempts.get(phase, rollout_count)), 3),
        }
        for phase, count in phase_failures.most_common()
    ]
    success_rate = round(success_count / rollout_count, 3) if rollout_count else None
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "session_id": session_id,
        "mode": mode,
        "policy_version": policy_version,
        "rollout_count": rollout_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_rate,
        "failure_type_counts": dict(failure_types),
        "phase_weakness": phase_weakness,
        "dominant_next_action": recommendations.most_common(1)[0][0] if recommendations else None,
        "rollout_ids": [str(item.get("rollout_id")) for item in evaluations],
        "created_at": _now_iso(),
    }


def _rollout_evaluation(
    *,
    evidence: Any,
    annotation: dict[str, Any],
    failures: dict[str, Any],
    policy_version: str | None,
    evaluation_session_id: str | None,
    evaluation_mode: str,
) -> dict[str, Any]:
    timeline = list(annotation.get("phase_timeline") or [])
    outcome = dict(annotation.get("outcome") or {})
    final_success = bool(outcome.get("final_success"))
    failure_events = list(failures.get("failure_events") or [])
    failure_type_counts = dict(Counter(str(item.get("failure_type") or "unknown") for item in failure_events))
    phase_weakness = _phase_weakness(timeline, failure_events)
    recommended = _recommended_next_action(final_success, failures, phase_weakness)
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "rollout_id": evidence.rollout_id,
        "evaluation_session_id": evaluation_session_id,
        "evaluation_mode": evaluation_mode,
        "policy_version": policy_version,
        "task_description": evidence.task_config.get("task_description", ""),
        "final_success": final_success,
        "success_status": outcome.get("success_status", "unknown"),
        "success_confidence": outcome.get("success_confidence", 0.0),
        "final_phase": outcome.get("final_phase", "unknown"),
        "terminal_phase": outcome.get("terminal_phase"),
        "terminal_reached": bool(outcome.get("terminal_reached")),
        "phases_seen": [segment.get("phase") for segment in timeline],
        "phase_weakness": phase_weakness,
        "failure_type_counts": failure_type_counts,
        "failure_events": failure_events,
        "retry_events": list(annotation.get("retry_events") or []),
        "policy_behavior_summary": _policy_behavior_summary(final_success, outcome, failures, phase_weakness),
        "recommended_next_action": recommended,
        "created_at": _now_iso(),
    }


def _phase_weakness(timeline: list[dict[str, Any]], failure_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failure_phases = Counter(str(event.get("phase") or "unknown") for event in failure_events)
    rows: list[dict[str, Any]] = []
    for segment in timeline:
        reasons: list[str] = []
        if segment.get("max_risk") == "high":
            reasons.append("high_risk")
        if segment.get("dominant_progress") in {"stalled", "regressing"}:
            reasons.append(str(segment.get("dominant_progress")))
        if int(segment.get("needs_review_count") or 0) > 0:
            reasons.append("needs_review")
        phase = str(segment.get("phase") or "unknown")
        if failure_phases.get(phase):
            reasons.append("failure_event")
        if reasons:
            rows.append(
                {
                    "phase": phase,
                    "segment_index": segment.get("segment_index"),
                    "start_frame": segment.get("start_frame"),
                    "end_frame": segment.get("end_frame"),
                    "severity": _severity_from_reasons(reasons),
                    "reasons": _dedupe(reasons),
                }
            )
    return rows


def _recommended_next_action(final_success: bool, failures: dict[str, Any], phase_weakness: list[dict[str, Any]]) -> str:
    if final_success and not phase_weakness:
        return "continue_evaluation"
    if final_success:
        return "collect_recovery_or_edge_cases"
    if failures.get("candidate_count"):
        return "collect_targeted_failure_data"
    return "collect_more_generalization_data"


def _policy_behavior_summary(
    final_success: bool,
    outcome: dict[str, Any],
    failures: dict[str, Any],
    phase_weakness: list[dict[str, Any]],
) -> str:
    if final_success and not phase_weakness:
        return "policy completed the task without strong weakness signals"
    if final_success:
        return "policy eventually completed the task but showed weak or risky phases"
    failure_type = failures.get("final_failure_type") or "unknown"
    return f"policy failed before stable task completion; final_failure_type={failure_type}; reason={outcome.get('reasoning', '')}"


def _collection_recommendation_from_rollout(eval_payload: dict[str, Any], failures: dict[str, Any]) -> dict[str, Any]:
    weak_phases = [str(item.get("phase")) for item in eval_payload.get("phase_weakness") or [] if item.get("phase")]
    failure_modes = list((eval_payload.get("failure_type_counts") or {}).keys())
    action = str(eval_payload.get("recommended_next_action") or "continue_evaluation")
    if action == "continue_evaluation":
        mode = "none"
    elif failures.get("recovered_count"):
        mode = "B1_reference_correction"
    elif eval_payload.get("final_success") is False:
        mode = "B2_deployment_monitoring"
    else:
        mode = "A_generalization"
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "scope": "rollout",
        "rollout_id": eval_payload.get("rollout_id"),
        "policy_version": eval_payload.get("policy_version"),
        "recommended_mode": mode,
        "focus_phases": _dedupe(weak_phases),
        "focus_failure_modes": _dedupe(failure_modes),
        "rationale": action,
        "created_at": _now_iso(),
    }


def _deployment_decision(
    summary: dict[str, Any],
    *,
    deploy_success_threshold: float,
    rollback_success_threshold: float,
    mode: str,
) -> dict[str, Any]:
    success_rate = summary.get("success_rate")
    reasons: list[str] = []
    if success_rate is None:
        decision = "hold"
        gating_result = "pending"
        reasons.append("no_completed_evaluation_rollouts")
    elif mode == "deployment" and success_rate < rollback_success_threshold:
        decision = "rollback_recommended"
        gating_result = "fail"
        reasons.append("deployment_success_rate_below_rollback_threshold")
    elif success_rate >= deploy_success_threshold:
        decision = "deploy_recommended"
        gating_result = "pass"
        reasons.append("success_rate_meets_deploy_threshold")
    elif success_rate >= rollback_success_threshold:
        decision = "collect_more_data"
        gating_result = "fail"
        reasons.append("success_rate_between_rollback_and_deploy_threshold")
    else:
        decision = "hold"
        gating_result = "fail"
        reasons.append("success_rate_below_minimum")

    if summary.get("phase_weakness"):
        reasons.append("phase_level_weakness_detected")
    if summary.get("failure_type_counts"):
        reasons.append("failure_modes_detected")
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "session_id": summary.get("session_id"),
        "policy_version": summary.get("policy_version"),
        "decision": decision,
        "gating_result": gating_result,
        "success_rate": success_rate,
        "deploy_success_threshold": deploy_success_threshold,
        "rollback_success_threshold": rollback_success_threshold,
        "reasons": _dedupe(reasons),
        "future_work": {
            "real_deploy_executor": "not implemented; deploy/rollback commands and policy server switching remain future work"
        },
        "created_at": _now_iso(),
    }


def _session_collection_recommendation(summary: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    weak = summary.get("phase_weakness") or []
    focus_phases = [str(item.get("phase")) for item in weak if item.get("phase")]
    failure_modes = list((summary.get("failure_type_counts") or {}).keys())
    if decision.get("decision") == "deploy_recommended":
        mode = "none"
        rationale = "policy meets deployment threshold"
    elif failure_modes:
        mode = "B2_deployment_monitoring"
        rationale = "deployment/eval failures should be replayed as targeted monitoring rollouts"
    elif focus_phases:
        mode = "A_generalization"
        rationale = "phase weakness without clear failure taxonomy"
    else:
        mode = "A_generalization"
        rationale = "insufficient positive evidence"
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "scope": "session",
        "session_id": summary.get("session_id"),
        "policy_version": summary.get("policy_version"),
        "recommended_mode": mode,
        "focus_phases": _dedupe(focus_phases),
        "focus_failure_modes": _dedupe(failure_modes),
        "rationale": rationale,
        "created_at": _now_iso(),
    }


def _next_collection_brief(
    summary: dict[str, Any],
    decision: dict[str, Any],
    collection: dict[str, Any],
) -> dict[str, Any]:
    focus_phases = list(collection.get("focus_phases") or [])
    failure_modes = list(collection.get("focus_failure_modes") or [])
    mode = str(collection.get("recommended_mode") or "A_generalization")
    decision_value = str(decision.get("decision") or "hold")
    target_count = _brief_target_count(decision_value, focus_phases, failure_modes)
    if mode == "none" or decision_value == "deploy_recommended":
        operator_brief = "No additional collection is recommended for this policy version. Prepare for human deployment review."
        scene_variations: list[str] = []
        success_criteria = "Policy maintains the observed deployment success criteria."
    else:
        phase_text = ", ".join(focus_phases) if focus_phases else "the weakest observed phases"
        failure_text = ", ".join(failure_modes) if failure_modes else "unclear or generalization failures"
        operator_brief = (
            f"Collect {target_count} rollout(s) focused on {phase_text}. "
            f"Prioritize cases that reproduce or correct {failure_text}."
        )
        scene_variations = _scene_variations(failure_modes, focus_phases)
        success_criteria = "A rollout is useful when it clearly shows either stable task completion or the target failure/recovery pattern."
    return {
        "schema_version": BRIEF_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "session_id": summary.get("session_id"),
        "policy_version": summary.get("policy_version"),
        "source_decision": decision_value,
        "recommended_mode": mode,
        "target_count": target_count,
        "focus_phases": focus_phases,
        "focus_failure_modes": failure_modes,
        "scene_variations": scene_variations,
        "operator_brief": operator_brief,
        "success_criteria": success_criteria,
        "created_at": _now_iso(),
    }


def _brief_target_count(decision: str, focus_phases: list[Any], failure_modes: list[Any]) -> int:
    if decision == "deploy_recommended":
        return 0
    if decision == "rollback_recommended":
        return 30
    if failure_modes:
        return 20
    if focus_phases:
        return 15
    return 12


def _scene_variations(failure_modes: list[Any], focus_phases: list[Any]) -> list[str]:
    variations: list[str] = []
    if failure_modes:
        variations.append("recreate the observed failure conditions with small object pose changes")
        variations.append("include at least a few corrected or recovered attempts")
    if focus_phases:
        variations.append("vary object placement near the weak phase boundary")
        variations.append("capture both clean and borderline executions for the focus phases")
    if not variations:
        variations.append("vary object pose, distance and mild occlusion while keeping the same task goal")
    return _dedupe(variations)


def _severity_from_reasons(reasons: list[str]) -> str:
    if "failure_event" in reasons or "high_risk" in reasons:
        return "high"
    if "regressing" in reasons:
        return "medium"
    return "low"


def _render_eval_report(eval_payload: dict[str, Any], failures: dict[str, Any], collection: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Policy Evaluation Rollout",
            "",
            f"- rollout_id: `{eval_payload.get('rollout_id')}`",
            f"- policy_version: `{eval_payload.get('policy_version')}`",
            f"- final_success: `{eval_payload.get('final_success')}`",
            f"- final_phase: `{eval_payload.get('final_phase')}`",
            f"- failure_candidates: `{failures.get('candidate_count', 0)}`",
            f"- recommended_next_action: `{eval_payload.get('recommended_next_action')}`",
            f"- collection_mode: `{collection.get('recommended_mode')}`",
            "",
        ]
    )


def _render_session_report(
    summary: dict[str, Any],
    decision: dict[str, Any],
    collection: dict[str, Any],
    brief: dict[str, Any],
    understanding: dict[str, Any] | None = None,
) -> str:
    return "\n".join(
        [
            "# Deployment Governance Session",
            "",
            f"- session_id: `{summary.get('session_id')}`",
            f"- policy_version: `{summary.get('policy_version')}`",
            f"- rollout_count: `{summary.get('rollout_count')}`",
            f"- success_rate: `{summary.get('success_rate')}`",
            f"- decision: `{decision.get('decision')}`",
            f"- gating_result: `{decision.get('gating_result')}`",
            f"- collection_mode: `{collection.get('recommended_mode')}`",
            f"- next_collection_target_count: `{brief.get('target_count')}`",
            f"- llm_understanding: `{(understanding or {}).get('status')}`",
            f"- operator_brief: {brief.get('operator_brief')}",
            "",
            "## Future Work",
            "",
            "- Real deploy executor: policy server switching, rollback command execution, launch/systemd integration and safety interlocks.",
            "",
        ]
    )


def _governance_understanding_payload(
    *,
    status: str,
    model: str | None,
    deterministic_decision: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": GOVERNANCE_UNDERSTANDING_SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "created_at": _now_iso(),
        "status": status,
        "model": model,
        "deterministic_decision": deterministic_decision,
        "summary": str(body.get("summary") or ""),
        "risk_notes": _string_list(body.get("risk_notes")),
        "llm_suggested_decision": str(body.get("llm_suggested_decision") or deterministic_decision),
        "operator_brief": str(body.get("operator_brief") or body.get("summary") or ""),
        "confidence": _float_or_none(body.get("confidence")),
        "reason": str(body.get("reason") or "") if body.get("reason") else None,
        "error": str(body.get("error") or "") if body.get("error") else None,
    }


def _governance_summary(summary: dict[str, Any], decision: dict[str, Any]) -> str:
    return (
        f"Deployment decision is {decision.get('decision')} with "
        f"success_rate={summary.get('success_rate')}."
    )


def _render_governance_understanding(understanding: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Deployment Governance Understanding",
            "",
            f"- status: `{understanding.get('status')}`",
            f"- model: `{understanding.get('model')}`",
            f"- deterministic_decision: `{understanding.get('deterministic_decision')}`",
            f"- llm_suggested_decision: `{understanding.get('llm_suggested_decision')}`",
            f"- summary: {understanding.get('summary') or ''}",
            f"- operator_brief: {understanding.get('operator_brief') or ''}",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, data: dict[str, Any]) -> Path:
    return _write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
