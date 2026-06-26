from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robolineage_shared_agents.json_llm import OpenAICompatibleJSONClient


DATASET_HEALTH_AGENT_VERSION = "dataset_health_agent@0.1"
DATASET_HEALTH_SCHEMA_VERSION = "RoboLineage.dataset_health.v1"
DATASET_HEALTH_UNDERSTANDING_SCHEMA_VERSION = "RoboLineage.dataset_health_understanding.v1"


@dataclass(frozen=True)
class DatasetHealthResult:
    report: dict[str, Any]
    understanding: dict[str, Any]
    report_path: Path | None = None
    understanding_path: Path | None = None
    report_markdown_path: Path | None = None
    events_path: Path | None = None


class DatasetHealthAgent:
    """Dataset-level health and coverage agent before training."""

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
                prefix="DATASET_HEALTH_LLM",
                system_prompt=(
                    "You are RoboLineage Dataset Health Agent. Read deterministic rollout coverage, "
                    "failure taxonomy and task phase coverage. Return JSON only with keys: "
                    "summary, coverage_notes, risk_notes, recommended_collection, confidence. "
                    "Do not claim that data exists beyond supplied artifacts."
                ),
                timeout_default=30.0,
            )
            if enable_env_llm
            else None
        )

    def analyze(
        self,
        *,
        selected_rollouts: list[Any],
        post_review_artifacts: list[Any] | dict[str, Any] | None = None,
        dataset_history: dict[str, Any] | None = None,
        task_config: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> DatasetHealthResult:
        events = [_event("dataset_health_started", {"selected_rollout_count": len(selected_rollouts)})]
        report = self._deterministic_report(
            selected_rollouts=selected_rollouts,
            post_review_artifacts=post_review_artifacts,
            dataset_history=dataset_history or {},
            task_config=task_config or {},
        )
        events.append(_event("deterministic_report_built", {"status": report.get("status")}))
        understanding = self._generate_understanding(report, task_config or {}, dataset_history or {})
        events.append(_event("llm_understanding_" + str(understanding.get("status") or "unknown"), {}))

        report_with_ref = {
            **report,
            "llm_understanding": _understanding_ref(understanding, None),
        }
        report_path: Path | None = None
        understanding_path: Path | None = None
        markdown_path: Path | None = None
        events_path: Path | None = None
        if output_dir is not None:
            out = Path(output_dir)
            report_path = out / "dataset_health_report.json"
            understanding_path = out / "dataset_health_understanding.json"
            markdown_path = out / "dataset_health_report.md"
            events_path = out / "dataset_health_events.jsonl"
            report_with_ref["llm_understanding"] = _understanding_ref(understanding, understanding_path)
            _write_json_atomic(understanding_path, understanding)
            _write_json_atomic(report_path, report_with_ref)
            _write_text_atomic(markdown_path, _render_dataset_health_report(report_with_ref, understanding))
            events.append(_event("artifacts_written", {"report_path": str(report_path)}))
            _write_jsonl(events_path, events)

        return DatasetHealthResult(
            report=report_with_ref,
            understanding=understanding,
            report_path=report_path,
            understanding_path=understanding_path,
            report_markdown_path=markdown_path,
            events_path=events_path,
        )

    def _deterministic_report(
        self,
        *,
        selected_rollouts: list[Any],
        post_review_artifacts: list[Any] | dict[str, Any] | None,
        dataset_history: dict[str, Any],
        task_config: dict[str, Any],
    ) -> dict[str, Any]:
        artifacts_by_id = _artifacts_by_rollout_id(post_review_artifacts)
        decision_counts: Counter[str] = Counter()
        phase_counts: Counter[str] = Counter()
        failure_type_counts: Counter[str] = Counter()
        success_count = 0
        failure_count = 0
        rollout_ids: list[str] = []

        for raw in selected_rollouts:
            item = _as_dict(raw)
            rollout_id = str(item.get("rollout_id") or item.get("id") or "")
            if rollout_id:
                rollout_ids.append(rollout_id)
            merged = {**artifacts_by_id.get(rollout_id, {}), **item}
            decision = str(merged.get("decision") or "unknown")
            decision_counts[decision] += 1
            final_success = merged.get("final_success")
            if final_success is True:
                success_count += 1
            elif final_success is False:
                failure_count += 1
            for phase in _phases_from_rollout(merged):
                phase_counts[phase] += 1
            for failure_type, count in _failure_types_from_rollout(merged).items():
                failure_type_counts[failure_type] += count

        expected_phases = _expected_phases(task_config)
        missing_phases = [phase for phase in expected_phases if phase not in phase_counts]
        selected_count = len(selected_rollouts)
        if selected_count == 0:
            status = "insufficient_data"
            recommended_action = "collect_more_data"
        elif missing_phases:
            status = "coverage_gap"
            recommended_action = "collect_phase_coverage"
        elif failure_count > success_count and failure_type_counts:
            status = "needs_targeted_data"
            recommended_action = "collect_targeted_failure_data"
        else:
            status = "healthy"
            recommended_action = "proceed_to_training"

        return {
            "schema_version": DATASET_HEALTH_SCHEMA_VERSION,
            "agent_version": DATASET_HEALTH_AGENT_VERSION,
            "created_at": _now_iso(),
            "status": status,
            "selected_rollout_count": selected_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "decision_counts": dict(decision_counts),
            "rollout_ids": rollout_ids,
            "phase_coverage": {
                "expected_phases": expected_phases,
                "observed_phase_counts": dict(phase_counts),
                "missing_phases": missing_phases,
            },
            "failure_type_counts": dict(failure_type_counts),
            "dataset_history": dataset_history,
            "recommended_action": recommended_action,
        }

    def _generate_understanding(
        self,
        report: dict[str, Any],
        task_config: dict[str, Any],
        dataset_history: dict[str, Any],
    ) -> dict[str, Any]:
        model = str(getattr(self.llm_client, "model", "") or "")
        base = {
            "summary": _deterministic_summary(report),
            "coverage_notes": [],
            "risk_notes": [],
            "recommended_collection": {
                "mode": "none" if report.get("recommended_action") == "proceed_to_training" else "A_generalization",
                "focus_phases": report.get("phase_coverage", {}).get("missing_phases") or [],
            },
            "confidence": None,
        }
        if not self.enable_llm_understanding:
            return _understanding_payload(status="skipped", model=None, body={**base, "reason": "disabled"})
        if self.llm_client is None:
            return _understanding_payload(
                status="not_configured",
                model=None,
                body={**base, "reason": "DATASET_HEALTH_LLM_API_KEY is not set"},
            )
        try:
            generated = self.llm_client.generate(
                {
                    "deterministic_report": report,
                    "task_config": task_config,
                    "dataset_history": dataset_history,
                    "contract": {
                        "summary": "short dataset health summary",
                        "coverage_notes": "list of coverage notes",
                        "risk_notes": "list of risk notes",
                        "recommended_collection": "mode/focus_phases/target_count object",
                        "confidence": "0-1 confidence",
                    },
                }
            )
            if not isinstance(generated, dict):
                raise RuntimeError("Dataset health LLM returned a non-object response")
            return _understanding_payload(status="generated", model=model or None, body=generated)
        except Exception as exc:
            return _understanding_payload(status="failed", model=model or None, body={**base, "error": str(exc)})


def _artifacts_by_rollout_id(raw: list[Any] | dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        if all(isinstance(value, dict) for value in raw.values()):
            return {str(key): dict(value) for key, value in raw.items()}
        rollout_id = str(raw.get("rollout_id") or "")
        return {rollout_id: dict(raw)} if rollout_id else {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            payload = _as_dict(item)
            rollout_id = str(payload.get("rollout_id") or "")
            if rollout_id:
                out[rollout_id] = payload
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        return dict(data) if isinstance(data, dict) else {}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _phases_from_rollout(item: dict[str, Any]) -> list[str]:
    phases: list[str] = []
    for key in ("phase_timeline", "phases_seen"):
        raw = item.get(key)
        if not isinstance(raw, list):
            continue
        for entry in raw:
            if isinstance(entry, dict):
                phase = entry.get("phase")
            else:
                phase = entry
            if phase:
                phases.append(str(phase))
    annotation = item.get("annotation") if isinstance(item.get("annotation"), dict) else {}
    if annotation:
        phases.extend(_phases_from_rollout(annotation))
    return _dedupe(phases)


def _failure_types_from_rollout(item: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key in ("failure_type_counts",):
        raw_counts = item.get(key)
        if isinstance(raw_counts, dict):
            for failure_type, count in raw_counts.items():
                try:
                    counts[str(failure_type)] += int(count)
                except (TypeError, ValueError):
                    counts[str(failure_type)] += 1
    failures = item.get("failure_analysis") if isinstance(item.get("failure_analysis"), dict) else item
    events = failures.get("failure_events") if isinstance(failures.get("failure_events"), list) else []
    for event in events:
        if isinstance(event, dict):
            counts[str(event.get("failure_type") or "unknown")] += 1
    return dict(counts)


def _expected_phases(task_config: dict[str, Any]) -> list[str]:
    phases = task_config.get("phases") if isinstance(task_config, dict) else None
    out: list[str] = []
    if isinstance(phases, list):
        for item in phases:
            if isinstance(item, dict):
                value = item.get("name") or item.get("phase") or item.get("id")
            else:
                value = item
            if value:
                out.append(str(value))
    return _dedupe(out)


def _understanding_payload(
    *,
    status: str,
    model: str | None,
    body: dict[str, Any],
) -> dict[str, Any]:
    recommended = body.get("recommended_collection")
    return {
        "schema_version": DATASET_HEALTH_UNDERSTANDING_SCHEMA_VERSION,
        "agent_version": DATASET_HEALTH_AGENT_VERSION,
        "created_at": _now_iso(),
        "status": status,
        "model": model,
        "summary": str(body.get("summary") or ""),
        "coverage_notes": _string_list(body.get("coverage_notes")),
        "risk_notes": _string_list(body.get("risk_notes")),
        "recommended_collection": recommended if isinstance(recommended, dict) else {},
        "confidence": _float_or_none(body.get("confidence")),
        "reason": str(body.get("reason") or "") if body.get("reason") else None,
        "error": str(body.get("error") or "") if body.get("error") else None,
    }


def _understanding_ref(payload: dict[str, Any], path: Path | None) -> dict[str, Any]:
    out = {
        "status": payload.get("status"),
        "model": payload.get("model"),
        "summary": payload.get("summary") or "",
        "reason": payload.get("reason"),
        "error": payload.get("error"),
    }
    if path is not None:
        out["path"] = str(path)
    return out


def _deterministic_summary(report: dict[str, Any]) -> str:
    return (
        f"Dataset health is {report.get('status')} with "
        f"{report.get('selected_rollout_count', 0)} selected rollout(s)."
    )


def _render_dataset_health_report(report: dict[str, Any], understanding: dict[str, Any]) -> str:
    coverage = report.get("phase_coverage") if isinstance(report.get("phase_coverage"), dict) else {}
    return "\n".join(
        [
            "# Dataset Health",
            "",
            f"- status: `{report.get('status')}`",
            f"- selected_rollout_count: `{report.get('selected_rollout_count')}`",
            f"- success_count: `{report.get('success_count')}`",
            f"- missing_phases: `{coverage.get('missing_phases') or []}`",
            f"- recommended_action: `{report.get('recommended_action')}`",
            f"- llm_understanding: `{understanding.get('status')}`",
            f"- summary: {understanding.get('summary') or ''}",
            "",
        ]
    )


def _event(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "RoboLineage.dataset_health_event.v1",
        "agent_version": DATASET_HEALTH_AGENT_VERSION,
        "created_at": _now_iso(),
        "event": event,
        "payload": payload,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    return _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
