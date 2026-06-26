from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robolineage_shared_agents.json_llm import OpenAICompatibleJSONClient


TRAINING_MONITOR_AGENT_VERSION = "training_monitor_agent@0.1"
TRAINING_MONITOR_UNDERSTANDING_SCHEMA_VERSION = "RoboLineage.training_monitor_understanding.v1"
TRAINING_MONITOR_REPORT_SCHEMA_VERSION = "RoboLineage.training_monitor_report.v1"


@dataclass(frozen=True)
class TrainingMonitorReport:
    status: str
    latest_step: int | None = None
    latest_epoch: int | None = None
    latest_loss: float | None = None
    best_loss: float | None = None
    latest_success_rate: float | None = None
    best_success_rate: float | None = None
    checkpoints: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = "continue_monitoring"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingMonitorAgentResult:
    report: dict[str, Any]
    understanding: dict[str, Any]
    report_path: Path | None = None
    understanding_path: Path | None = None
    report_markdown_path: Path | None = None
    events_path: Path | None = None


class TrainingLogMonitor:
    """Framework-agnostic training/eval log interpreter.

    It prefers JSONL metrics when available, uses profile regexes as hints, and
    falls back to common plain-text conventions used by policy training stacks.
    """

    def __init__(
        self,
        custom_patterns: dict[str, str] | None = None,
        monitor_spec: dict[str, Any] | None = None,
    ) -> None:
        self.custom_patterns = custom_patterns or {}
        self.monitor_spec = monitor_spec or {}

    def parse(self, text: str) -> TrainingMonitorReport:
        steps: list[int] = []
        epochs: list[int] = []
        losses: list[float] = []
        success_rates: list[float] = []
        checkpoints: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            json_obj = _json_object(_strip_json_prefix(line, self.monitor_spec))
            if json_obj is not None:
                _consume_json_metrics(json_obj, steps, epochs, losses, success_rates, checkpoints, metrics)
            _consume_custom_patterns(line, self.custom_patterns, metrics)
            _consume_text_metrics(line, steps, epochs, losses, success_rates, checkpoints)
            _consume_error_signals(line, errors, warnings)

        latest_loss = losses[-1] if losses else None
        best_loss = min(losses) if losses else None
        latest_success = success_rates[-1] if success_rates else None
        best_success = max(success_rates) if success_rates else None
        latest_step = steps[-1] if steps else _int_metric(metrics.get("step") or metrics.get("steps"))
        latest_epoch = epochs[-1] if epochs else _int_metric(metrics.get("epoch"))
        if latest_loss is None:
            latest_loss = _float_metric(metrics.get("loss") or metrics.get("train_loss"))
        if best_loss is None and latest_loss is not None:
            best_loss = latest_loss
        if latest_success is None:
            latest_success = _float_metric(
                metrics.get("success_rate") or metrics.get("eval_success") or metrics.get("success")
            )
        if best_success is None and latest_success is not None:
            best_success = latest_success

        if any("nan" in item.lower() or "diverg" in item.lower() for item in errors + warnings):
            status = "unstable"
            action = "inspect_training_instability"
        elif errors:
            status = "failed"
            action = "inspect_failed_training"
        else:
            status = "completed"
            action = "evaluate_checkpoint" if checkpoints else "inspect_missing_checkpoint"
            if not checkpoints:
                warnings.append("no_checkpoint_detected")
            if len(losses) >= 4 and losses[-1] > losses[0] * 1.2:
                warnings.append("loss_trending_up")
                status = "unstable"
                action = "inspect_loss_trend"

        flat_metrics = dict(metrics)
        if latest_step is not None:
            flat_metrics["step"] = latest_step
        if latest_epoch is not None:
            flat_metrics["epoch"] = latest_epoch
        if latest_loss is not None:
            flat_metrics["loss"] = latest_loss
        if latest_success is not None:
            flat_metrics["success_rate"] = latest_success

        return TrainingMonitorReport(
            status=status,
            latest_step=latest_step,
            latest_epoch=latest_epoch,
            latest_loss=latest_loss,
            best_loss=best_loss,
            latest_success_rate=latest_success,
            best_success_rate=best_success,
            checkpoints=tuple(_dedupe(checkpoints)),
            errors=tuple(_dedupe(errors)),
            warnings=tuple(_dedupe(warnings)),
            metrics=flat_metrics,
            recommended_action=action,
        )


class TrainingMonitorAgent:
    """Deterministic training log monitor with optional LLM diagnosis."""

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
                prefix="TRAINING_MONITOR_LLM",
                system_prompt=(
                    "You are RoboLineage Training Monitor Agent. Read deterministic training log metrics "
                    "and a bounded log excerpt. Return JSON only with keys: diagnosis, likely_causes, "
                    "recommended_action, operator_brief, confidence. Do not override deterministic "
                    "status unless explaining it."
                ),
                timeout_default=30.0,
            )
            if enable_env_llm
            else None
        )

    def analyze(
        self,
        text: str,
        *,
        patterns: dict[str, str] | None = None,
        monitor_spec: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        deterministic_report: dict[str, Any] | None = None,
    ) -> TrainingMonitorAgentResult:
        report = (
            dict(deterministic_report)
            if isinstance(deterministic_report, dict)
            else TrainingLogMonitor(patterns or {}, monitor_spec).parse(text).to_dict()
        )
        understanding = self._generate_understanding(
            deterministic_report=report,
            log_excerpt=_log_excerpt(text),
            context=context or {},
        )
        report_with_ref = {
            "schema_version": TRAINING_MONITOR_REPORT_SCHEMA_VERSION,
            "agent_version": TRAINING_MONITOR_AGENT_VERSION,
            "created_at": _now_iso(),
            **report,
            "llm_understanding": _understanding_ref(understanding, None),
        }

        report_path: Path | None = None
        understanding_path: Path | None = None
        markdown_path: Path | None = None
        events_path: Path | None = None
        if output_dir is not None:
            out = Path(output_dir)
            report_path = out / "training_monitor_report.json"
            understanding_path = out / "training_monitor_understanding.json"
            markdown_path = out / "training_monitor_report.md"
            events_path = out / "training_monitor_events.jsonl"
            report_with_ref["llm_understanding"] = _understanding_ref(understanding, understanding_path)
            _write_json_atomic(understanding_path, understanding)
            _write_json_atomic(report_path, report_with_ref)
            _write_text_atomic(markdown_path, _render_monitor_report(report_with_ref, understanding))
            _write_jsonl(
                events_path,
                [
                    _event("monitor_started", {}),
                    _event("deterministic_report_built", {"status": report.get("status")}),
                    _event("llm_understanding_" + str(understanding.get("status") or "unknown"), {}),
                    _event("artifacts_written", {"report_path": str(report_path)}),
                ],
            )

        return TrainingMonitorAgentResult(
            report=report_with_ref,
            understanding=understanding,
            report_path=report_path,
            understanding_path=understanding_path,
            report_markdown_path=markdown_path,
            events_path=events_path,
        )

    def _generate_understanding(
        self,
        *,
        deterministic_report: dict[str, Any],
        log_excerpt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        model = str(getattr(self.llm_client, "model", "") or "")
        base = {
            "diagnosis": _deterministic_diagnosis(deterministic_report),
            "likely_causes": [],
            "recommended_action": deterministic_report.get("recommended_action") or "continue_monitoring",
            "operator_brief": _deterministic_diagnosis(deterministic_report),
            "confidence": None,
        }
        if not self.enable_llm_understanding:
            return _understanding_payload(status="skipped", model=None, body={**base, "reason": "disabled"})
        if self.llm_client is None:
            return _understanding_payload(
                status="not_configured",
                model=None,
                body={**base, "reason": "TRAINING_MONITOR_LLM_API_KEY is not set"},
            )
        try:
            generated = self.llm_client.generate(
                {
                    "deterministic_report": deterministic_report,
                    "log_excerpt": log_excerpt,
                    "context": context,
                    "contract": {
                        "diagnosis": "one sentence explanation",
                        "likely_causes": "list of likely causes",
                        "recommended_action": "one of deterministic/actionable monitor actions",
                        "operator_brief": "operator-facing next step",
                        "confidence": "0-1 confidence",
                    },
                }
            )
            if not isinstance(generated, dict):
                raise RuntimeError("Training monitor LLM returned a non-object response")
            return _understanding_payload(status="generated", model=model or None, body=generated)
        except Exception as exc:
            return _understanding_payload(status="failed", model=model or None, body={**base, "error": str(exc)})


def _json_object(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _strip_json_prefix(line: str, monitor_spec: dict[str, Any]) -> str:
    prefixes = monitor_spec.get("json_line_prefixes")
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    if not isinstance(prefixes, list):
        prefixes = []
    for prefix in prefixes:
        prefix_s = str(prefix)
        if prefix_s and line.startswith(prefix_s):
            return line[len(prefix_s):].strip()
    return line


def _consume_json_metrics(
    obj: dict[str, Any],
    steps: list[int],
    epochs: list[int],
    losses: list[float],
    success_rates: list[float],
    checkpoints: list[str],
    metrics: dict[str, Any],
) -> None:
    for key, value in obj.items():
        key_s = str(key)
        metrics[key_s] = value
        lower = key_s.lower()
        if lower in {"step", "steps", "global_step", "iter", "iteration"}:
            maybe = _int_metric(value)
            if maybe is not None:
                steps.append(maybe)
        elif lower == "epoch":
            maybe = _int_metric(value)
            if maybe is not None:
                epochs.append(maybe)
        elif "loss" in lower:
            maybe = _float_metric(value)
            if maybe is not None:
                losses.append(maybe)
        elif lower in {"success", "success_rate", "eval_success", "eval_success_rate"}:
            maybe = _float_metric(value)
            if maybe is not None:
                success_rates.append(maybe)
        elif "checkpoint" in lower or lower in {"ckpt", "checkpoint_path"}:
            if value:
                checkpoints.append(str(value))


def _consume_custom_patterns(line: str, patterns: dict[str, str], metrics: dict[str, Any]) -> None:
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            metrics[key] = _coerce_scalar(match.group(1) if match.groups() else match.group(0))


def _consume_text_metrics(
    line: str,
    steps: list[int],
    epochs: list[int],
    losses: list[float],
    success_rates: list[float],
    checkpoints: list[str],
) -> None:
    for pattern in (r"\b(?:step|steps|iter|iteration|global_step)\s*[=:]\s*(\d+)",):
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            steps.append(int(match.group(1)))
    match = re.search(r"\bepoch\s*[=:]\s*(\d+)", line, flags=re.IGNORECASE)
    if match:
        epochs.append(int(match.group(1)))
    match = re.search(r"\b(?:train_)?loss\s*[=:]\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", line, flags=re.IGNORECASE)
    if match:
        losses.append(float(match.group(1)))
    match = re.search(
        r"\b(?:success_rate|eval_success|eval_success_rate|success)\s*[=:]\s*([-+]?\d*\.?\d+)",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        success_rates.append(float(match.group(1)))
    match = re.search(
        r"(?:saved checkpoint to|checkpoint saved(?: at| to)?|ckpt(?:_path)?\s*[=:])\s*(.+)$",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        checkpoints.append(match.group(1).strip())


def _consume_error_signals(line: str, errors: list[str], warnings: list[str]) -> None:
    lower = line.lower()
    if any(token in lower for token in ("traceback", "exception", "runtimeerror", "cuda out of memory", "oom", "segmentation fault")):
        errors.append(line)
    if re.search(r"\b(?:nan|inf)\b", lower):
        errors.append(line)
    if any(token in lower for token in ("diverg", "exploding", "gradient overflow")):
        warnings.append(line)


def _int_metric(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_metric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_scalar(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _understanding_payload(
    *,
    status: str,
    model: str | None,
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_MONITOR_UNDERSTANDING_SCHEMA_VERSION,
        "agent_version": TRAINING_MONITOR_AGENT_VERSION,
        "created_at": _now_iso(),
        "status": status,
        "model": model,
        "diagnosis": str(body.get("diagnosis") or ""),
        "likely_causes": _string_list(body.get("likely_causes")),
        "recommended_action": str(body.get("recommended_action") or "continue_monitoring"),
        "operator_brief": str(body.get("operator_brief") or body.get("diagnosis") or ""),
        "confidence": _float_metric(body.get("confidence")),
        "reason": str(body.get("reason") or "") if body.get("reason") else None,
        "error": str(body.get("error") or "") if body.get("error") else None,
    }


def _understanding_ref(payload: dict[str, Any], path: Path | None) -> dict[str, Any]:
    out = {
        "status": payload.get("status"),
        "model": payload.get("model"),
        "summary": payload.get("diagnosis") or "",
        "recommended_action": payload.get("recommended_action"),
        "reason": payload.get("reason"),
        "error": payload.get("error"),
    }
    if path is not None:
        out["path"] = str(path)
    return out


def _deterministic_diagnosis(report: dict[str, Any]) -> str:
    status = report.get("status") or "unknown"
    action = report.get("recommended_action") or "continue_monitoring"
    if status == "unstable":
        return f"Training appears unstable; recommended_action={action}."
    if status == "failed":
        return f"Training failed; recommended_action={action}."
    if status == "completed":
        return f"Training completed; recommended_action={action}."
    return f"Training status is {status}; recommended_action={action}."


def _log_excerpt(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _render_monitor_report(report: dict[str, Any], understanding: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Training Monitor",
            "",
            f"- status: `{report.get('status')}`",
            f"- latest_step: `{report.get('latest_step')}`",
            f"- latest_loss: `{report.get('latest_loss')}`",
            f"- recommended_action: `{report.get('recommended_action')}`",
            f"- llm_understanding: `{understanding.get('status')}`",
            f"- diagnosis: {understanding.get('diagnosis') or ''}",
            "",
        ]
    )


def _event(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "RoboLineage.training_monitor_event.v1",
        "agent_version": TRAINING_MONITOR_AGENT_VERSION,
        "created_at": _now_iso(),
        "event": event,
        "payload": payload,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
