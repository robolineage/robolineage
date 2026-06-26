from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robolineage_shared_agents.llm_routes import DEFAULT_OPENAI_COMPAT_MODEL, resolve_ai_route

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a project dependency
    yaml = None


MASTER_AGENT_VERSION = "master_agent@0.2"
MASTER_STATE_SCHEMA_VERSION = "RoboLineage.master_state.v1"
MASTER_REVIEW_SCHEMA_VERSION = "RoboLineage.master_review.v1"
MASTER_MEMORY_SCHEMA_VERSION = "RoboLineage.master_memory.v1"
MASTER_EVENT_SCHEMA_VERSION = "RoboLineage.master_event.v1"
MASTER_UNDERSTANDING_SCHEMA_VERSION = "RoboLineage.master_understanding.v1"


@dataclass(frozen=True)
class MasterReviewResult:
    task_root: Path
    state_path: Path
    memory_path: Path
    events_path: Path
    review_path: Path
    report_path: Path
    understanding_path: Path
    understanding_report_path: Path
    state: dict[str, Any]
    review: dict[str, Any]
    understanding: dict[str, Any]


class OpenAICompatibleMasterLLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> OpenAICompatibleMasterLLMClient | None:
        route = resolve_ai_route(
            "MASTER_LLM",
            fallback_prefixes=("ROBOLINEAGE_AGENT", "TASK_LLM", "OPENAI"),
            base_url_default="https://api.openai.com/v1",
            timeout_default=30.0,
        )
        api_key = route.api_key
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=route.model or DEFAULT_OPENAI_COMPAT_MODEL,
            base_url=route.base_url or "https://api.openai.com/v1",
            timeout=float(route.timeout_s or 30.0),
        )

    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the RoboLineage Master Agent. Read compact lifecycle artifacts and "
                        "return only JSON with keys: summary, operator_brief, "
                        "risk_interpretation, suggested_next_action, memory_updates. "
                        "Do not invent runtime facts beyond the supplied artifacts."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # nosec B310 - configured API endpoint
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Master LLM HTTP {exc.code}: {detail[:500]}") from exc
        content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        parsed = _parse_json_object(str(content))
        if not parsed:
            raise RuntimeError("Master LLM response did not contain a JSON object")
        return parsed


class MasterAgent:
    """Global lifecycle observer for one RoboLineage task directory.

    It scans lightweight artifacts, writes compact state/review/memory, and
    always writes a Master understanding artifact. The LLM layer adds narrative
    judgment; deterministic state remains the source of runtime truth.
    """

    def __init__(self, *, llm_client: Any | None = None, enable_env_llm: bool = True) -> None:
        self._llm_client = (
            llm_client
            if llm_client is not None
            else OpenAICompatibleMasterLLMClient.from_env()
            if enable_env_llm
            else None
        )

    def review(
        self,
        *,
        task_root: str | Path,
        health_summary: dict[str, Any] | None = None,
        trigger: str = "manual",
    ) -> MasterReviewResult:
        root = Path(task_root)
        master_dir = root / "master"
        events_path = master_dir / "master_events.jsonl"
        memory_path = master_dir / "master_memory.jsonl"
        state_path = master_dir / "master_state.json"
        review_path = master_dir / "master_review.json"
        report_path = master_dir / "master_report.md"
        understanding_path = master_dir / "master_understanding.json"
        understanding_report_path = master_dir / "master_understanding_report.md"

        self._append_event(events_path, "master_started", {"trigger": trigger, "task_root": str(root)})
        artifacts = self._scan_artifacts(root)
        self._append_event(
            events_path,
            "artifacts_scanned",
            {
                "current_stage": artifacts["current_stage"],
                "counts": artifacts["counts"],
            },
        )

        state = self._state(root, artifacts, health_summary or {})
        base_review = self._review_payload(root, state)
        understanding = self._generate_understanding(state, base_review, events_path)
        _write_json_atomic(understanding_path, understanding)
        _write_text_atomic(understanding_report_path, self._render_understanding_report(understanding))
        self._append_event(events_path, "understanding_written", {"path": str(understanding_path)})

        state["llm_understanding"] = self._understanding_ref(understanding, understanding_path)
        _write_json_atomic(state_path, state)
        self._append_event(events_path, "state_written", {"path": str(state_path)})

        memory_entry = self._memory_entry(root, state, understanding)
        _append_jsonl(memory_path, memory_entry)
        self._append_event(events_path, "memory_updated", {"path": str(memory_path)})

        review = self._review_payload(root, state)
        self._apply_understanding_to_review(review, understanding, understanding_path)
        _write_json_atomic(review_path, review)
        _write_text_atomic(report_path, self._render_report(state, review, understanding))
        self._append_event(events_path, "review_written", {"path": str(review_path)})

        return MasterReviewResult(
            task_root=root,
            state_path=state_path,
            memory_path=memory_path,
            events_path=events_path,
            review_path=review_path,
            report_path=report_path,
            understanding_path=understanding_path,
            understanding_report_path=understanding_report_path,
            state=state,
            review=review,
            understanding=understanding,
        )

    def _scan_artifacts(self, task_root: Path) -> dict[str, Any]:
        manifest = _read_json(task_root / "task_manifest.json")
        task_config = _read_task_config(task_root / "task_config.latest.yaml")
        rollouts_root = task_root / "rollouts"
        training_root = task_root / "training_runs"
        deployment_root = task_root / "deployment_sessions"

        onboarding_reports = _collect_json_files(task_root / "robot_onboarding", "robot_onboarding_report.json")
        admissions = _collect_json_files(rollouts_root, "dataset_admission.json")
        training_statuses = _collect_json_files(training_root, "training_status.json")
        deployment_decisions = _collect_json_files(deployment_root, "deployment_decision.json")
        framework_discoveries = _collect_json_files(task_root / "framework_profiles", "framework_discovery.json")

        latest_onboarding = onboarding_reports[-1][1] if onboarding_reports else {}
        latest_deployment = deployment_decisions[-1][1] if deployment_decisions else {}
        latest_training = training_statuses[-1][1] if training_statuses else {}
        latest_admission = admissions[-1][1] if admissions else {}

        if latest_deployment:
            stage = "deployment_governance"
        elif latest_training:
            stage = "training"
        elif latest_admission:
            stage = "post_review"
        elif task_config:
            stage = "task_understanding"
        elif latest_onboarding or manifest.get("robot"):
            stage = "robot_onboarding"
        else:
            stage = "not_started"

        return {
            "manifest": manifest,
            "task_config": task_config,
            "latest_robot_onboarding": latest_onboarding,
            "latest_deployment_decision": latest_deployment,
            "latest_training_status": latest_training,
            "latest_dataset_admission": latest_admission,
            "framework_discovery_count": len(framework_discoveries),
            "counts": {
                "robot_onboarding_reports": len(onboarding_reports),
                "dataset_admissions": len(admissions),
                "training_statuses": len(training_statuses),
                "deployment_decisions": len(deployment_decisions),
                "framework_discoveries": len(framework_discoveries),
            },
            "current_stage": stage,
        }

    def _state(
        self,
        task_root: Path,
        artifacts: dict[str, Any],
        health_summary: dict[str, Any],
    ) -> dict[str, Any]:
        task_description = (
            artifacts["manifest"].get("task_description")
            or artifacts["task_config"].get("task_description")
            or artifacts["task_config"].get("goal")
        )
        risks = self._risks(artifacts, health_summary)
        return {
            "schema_version": MASTER_STATE_SCHEMA_VERSION,
            "agent_version": MASTER_AGENT_VERSION,
            "task_id": task_root.name,
            "task_root": str(task_root),
            "task_description": task_description,
            "robot": artifacts["manifest"].get("robot") or _robot_from_onboarding(artifacts["latest_robot_onboarding"]),
            "current_stage": artifacts["current_stage"],
            "counts": artifacts["counts"],
            "health": health_summary,
            "latest": {
                "robot_onboarding": artifacts["latest_robot_onboarding"],
                "deployment_decision": artifacts["latest_deployment_decision"],
                "training_status": artifacts["latest_training_status"],
                "dataset_admission": artifacts["latest_dataset_admission"],
            },
            "risks": risks,
            "blocking": [item for item in risks if item.get("severity") == "high"],
            "next_action": self._next_action(artifacts, risks),
            "created_at": _now_iso(),
        }

    def _generate_understanding(
        self,
        state: dict[str, Any],
        review: dict[str, Any],
        events_path: Path,
    ) -> dict[str, Any]:
        model = str(getattr(self._llm_client, "model", "") or "")
        if self._llm_client is None:
            self._append_event(events_path, "llm_understanding_not_configured", {})
            return self._understanding_payload(
                status="not_configured",
                model=None,
                body={
                    "summary": self._summary_sentence(state),
                    "operator_brief": self._summary_sentence(state),
                    "risk_interpretation": [],
                    "suggested_next_action": state.get("next_action") or {},
                    "memory_updates": [],
                    "reason": "MASTER_LLM_API_KEY is not set",
                },
            )

        self._append_event(events_path, "llm_understanding_started", {"model": model})
        context = {
            "state": state,
            "review": review,
            "contract": {
                "summary": "short lifecycle summary",
                "operator_brief": "one actionable paragraph for the operator",
                "risk_interpretation": "list of risk objects",
                "suggested_next_action": "action/reason/confidence object",
                "memory_updates": "short reusable facts",
            },
        }
        try:
            generated = self._llm_client.generate(context)
            if isinstance(generated, str):
                generated = _parse_json_object(generated)
            if not isinstance(generated, dict):
                raise RuntimeError("Master LLM client returned a non-object response")
            understanding = self._understanding_payload(
                status="generated",
                model=model or None,
                body=generated,
            )
            self._append_event(
                events_path,
                "llm_understanding_completed",
                {"model": model, "summary": understanding.get("summary")},
            )
            return understanding
        except Exception as exc:
            self._append_event(events_path, "llm_understanding_failed", {"model": model, "error": str(exc)})
            return self._understanding_payload(
                status="failed",
                model=model or None,
                body={
                    "summary": self._summary_sentence(state),
                    "operator_brief": self._summary_sentence(state),
                    "risk_interpretation": [],
                    "suggested_next_action": state.get("next_action") or {},
                    "memory_updates": [],
                    "error": str(exc),
                },
            )

    def _understanding_payload(
        self,
        *,
        status: str,
        model: str | None,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        suggested = body.get("suggested_next_action")
        risks = body.get("risk_interpretation")
        memory_updates = body.get("memory_updates")
        return {
            "schema_version": MASTER_UNDERSTANDING_SCHEMA_VERSION,
            "agent_version": MASTER_AGENT_VERSION,
            "created_at": _now_iso(),
            "status": status,
            "model": model,
            "summary": str(body.get("summary") or ""),
            "operator_brief": str(body.get("operator_brief") or body.get("summary") or ""),
            "risk_interpretation": risks if isinstance(risks, list) else [],
            "suggested_next_action": suggested if isinstance(suggested, dict) else {},
            "memory_updates": memory_updates if isinstance(memory_updates, list) else [],
            "reason": body.get("reason"),
            "error": body.get("error"),
        }

    def _understanding_ref(self, understanding: dict[str, Any], path: Path) -> dict[str, Any]:
        return {
            "status": understanding.get("status"),
            "model": understanding.get("model"),
            "path": str(path),
            "summary": understanding.get("summary"),
        }

    def _apply_understanding_to_review(
        self,
        review: dict[str, Any],
        understanding: dict[str, Any],
        path: Path,
    ) -> None:
        review["llm_understanding"] = self._understanding_ref(understanding, path)
        if understanding.get("status") == "generated":
            operator_brief = str(understanding.get("operator_brief") or "").strip()
            if operator_brief:
                review["summary"] = operator_brief
            suggested = understanding.get("suggested_next_action")
            if isinstance(suggested, dict) and suggested.get("action"):
                review["llm_suggested_next_action"] = suggested
            risks = understanding.get("risk_interpretation")
            if isinstance(risks, list):
                review["llm_risk_interpretation"] = risks

    def _risks(self, artifacts: dict[str, Any], health_summary: dict[str, Any]) -> list[dict[str, Any]]:
        risks: list[dict[str, Any]] = []
        health_status = str(health_summary.get("status") or health_summary.get("state") or "").lower()
        if health_status in {"failed", "error", "degraded"}:
            risks.append({"severity": "high", "code": "health_not_ok", "message": "Health endpoint is not OK."})

        training_status = str(artifacts["latest_training_status"].get("status") or "").lower()
        if training_status in {"failed", "unstable"}:
            risks.append({"severity": "high", "code": "training_not_stable", "message": f"Training status is {training_status}."})

        deployment = artifacts["latest_deployment_decision"]
        if deployment and deployment.get("gating_result") == "fail":
            risks.append({"severity": "medium", "code": "deployment_gate_failed", "message": "Deployment gate did not pass."})
        return risks

    def _next_action(self, artifacts: dict[str, Any], risks: list[dict[str, Any]]) -> dict[str, Any]:
        if any(item.get("code") == "health_not_ok" for item in risks):
            return {"action": "inspect_health", "reason": "health endpoint is not OK"}
        deployment = artifacts["latest_deployment_decision"]
        decision = str(deployment.get("decision") or "") if deployment else ""
        if decision:
            if decision == "deploy_recommended":
                return {"action": "prepare_human_deploy_review", "reason": "deployment governance recommends deploy"}
            if decision == "rollback_recommended":
                return {"action": "prepare_human_rollback_review", "reason": "deployment governance recommends rollback"}
            if decision == "collect_more_data":
                return {"action": "collect_more_data", "reason": "deployment governance requests more data"}
            return {"action": "hold", "reason": f"deployment decision is {decision}"}

        training_status = str(artifacts["latest_training_status"].get("status") or "")
        if training_status:
            if training_status == "completed":
                return {"action": "start_deployment_evaluation", "reason": "training completed"}
            return {"action": "inspect_training", "reason": f"training status is {training_status}"}

        admission = artifacts["latest_dataset_admission"]
        if admission:
            return {"action": "create_training_selection", "reason": "post-review artifacts are available"}
        if artifacts["task_config"]:
            return {"action": "start_collection", "reason": "task config is available"}
        return {"action": "define_task", "reason": "no task config found"}

    def _memory_entry(
        self,
        task_root: Path,
        state: dict[str, Any],
        understanding: dict[str, Any],
    ) -> dict[str, Any]:
        entry = {
            "schema_version": MASTER_MEMORY_SCHEMA_VERSION,
            "agent_version": MASTER_AGENT_VERSION,
            "created_at": state["created_at"],
            "task_id": task_root.name,
            "stage": state["current_stage"],
            "summary": self._summary_sentence(state),
            "next_action": state["next_action"],
            "risk_count": len(state["risks"]),
            "llm_understanding_status": understanding.get("status"),
        }
        memory_updates = understanding.get("memory_updates")
        if isinstance(memory_updates, list) and memory_updates:
            entry["llm_memory_updates"] = [str(item) for item in memory_updates[:5]]
        return entry

    def _review_payload(self, task_root: Path, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": MASTER_REVIEW_SCHEMA_VERSION,
            "agent_version": MASTER_AGENT_VERSION,
            "created_at": state["created_at"],
            "task_id": task_root.name,
            "current_stage": state["current_stage"],
            "summary": self._summary_sentence(state),
            "next_action": state["next_action"],
            "risks": state["risks"],
            "blocking": state["blocking"],
        }

    def _render_report(
        self,
        state: dict[str, Any],
        review: dict[str, Any],
        understanding: dict[str, Any],
    ) -> str:
        risks = state.get("risks") or []
        lines = [
            "# Master Review",
            "",
            f"- task_id: `{state.get('task_id')}`",
            f"- current_stage: `{state.get('current_stage')}`",
            f"- next_action: `{state.get('next_action', {}).get('action')}`",
            f"- risk_count: `{len(risks)}`",
            f"- llm_understanding: `{understanding.get('status')}`",
            "",
            review.get("summary") or "No summary available.",
            "",
        ]
        return "\n".join(lines)

    def _render_understanding_report(self, understanding: dict[str, Any]) -> str:
        lines = [
            "# Master Understanding",
            "",
            f"- status: `{understanding.get('status')}`",
            f"- model: `{understanding.get('model')}`",
            "",
            str(understanding.get("operator_brief") or understanding.get("summary") or "No understanding available."),
            "",
        ]
        return "\n".join(lines)

    def _summary_sentence(self, state: dict[str, Any]) -> str:
        action = state.get("next_action", {}).get("action") or "unknown"
        return f"Task {state.get('task_id')} is at {state.get('current_stage')}; recommended next action is {action}."

    def _append_event(self, path: Path, event: str, payload: dict[str, Any]) -> None:
        _append_jsonl(
            path,
            {
                "schema_version": MASTER_EVENT_SCHEMA_VERSION,
                "agent_version": MASTER_AGENT_VERSION,
                "created_at": _now_iso(),
                "event": event,
                "payload": payload,
            },
        )


def _collect_json_files(root: Path, filename: str) -> list[tuple[Path, dict[str, Any]]]:
    if not root.exists():
        return []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob(filename)):
        payload = _read_json(path)
        if payload:
            rows.append((path, payload))
    return rows


def _robot_from_onboarding(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "robot_id": report.get("robot_id"),
        "display_name": report.get("display_name"),
        "active_camera": report.get("active_camera"),
        "active_robot_state": report.get("active_robot_state"),
        "validation_status": report.get("validation_status"),
        "onboarding_status": report.get("status"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_task_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if yaml is None:
        return {}
    try:
        data = yaml.safe_load(text) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
