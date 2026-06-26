from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from robolineage_shared_agents.json_llm import OpenAICompatibleJSONClient

from .profile import (
    ROBOT_PROFILE_SCHEMA_VERSION,
    RobotProfile,
    load_robot_profile,
    profile_to_vsa_topics,
)

ROBOT_ONBOARDING_AGENT_VERSION = "robot_onboarding_agent@0.1"
ROBOT_ONBOARDING_REPORT_SCHEMA_VERSION = "RoboLineage.robot_onboarding_report.v1"
ROBOT_ONBOARDING_EVENT_SCHEMA_VERSION = "RoboLineage.robot_onboarding_event.v1"
ROBOT_ONBOARDING_UNDERSTANDING_SCHEMA_VERSION = "RoboLineage.robot_onboarding_understanding.v1"


@dataclass
class RobotOnboardingResult:
    status: str
    job_id: str
    robot_id: str
    generated_profile_path: Path
    artifact_profile_path: Path
    report_path: Path
    events_path: Path
    understanding_path: Path
    report: dict[str, Any]
    events: list[dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "job_id": self.job_id,
            "robot_id": self.robot_id,
            "generated_profile_path": str(self.generated_profile_path),
            "artifact_profile_path": str(self.artifact_profile_path),
            "report_path": str(self.report_path),
            "events_path": str(self.events_path),
            "understanding_path": str(self.understanding_path),
            "report": self.report,
            "events": self.events,
        }


class RobotOnboardingAgent:
    """Understand and normalize a pasted robot profile YAML.

    This first version is artifact-first and read-only with respect to runtime
    behavior: it writes a generated profile plus report/trace, then callers can
    use the existing profile activation and validation APIs.
    """

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
                prefix="ROBOT_ONBOARDING_LLM",
                system_prompt=(
                    "You are RoboLineage Robot Onboarding Agent. Read a normalized robot profile, "
                    "operator note and deterministic warnings. Return JSON only with keys: "
                    "profile_summary, binding_explanation, warnings, assumptions, recommended_checks. "
                    "Do not invent ROS topics beyond the supplied profile."
                ),
                timeout_default=30.0,
            )
            if enable_env_llm
            else None
        )

    def run(
        self,
        *,
        profile_yaml: str,
        task_root: str | Path,
        generated_profiles_root: str | Path | None = None,
        robot_note: str | None = None,
        job_id: str | None = None,
    ) -> RobotOnboardingResult:
        if not str(profile_yaml or "").strip():
            raise ValueError("profile_yaml is required")

        task_root = Path(task_root)
        generated_profiles_root = Path(generated_profiles_root or Path.cwd() / "robot_profiles")
        job_id = _safe_id(job_id or f"onboard_{uuid.uuid4().hex[:12]}")
        job_dir = task_root / "robot_onboarding" / job_id
        events_path = job_dir / "events.jsonl"
        report_path = job_dir / "robot_onboarding_report.json"
        understanding_path = job_dir / "robot_onboarding_understanding.json"
        artifact_profile_path = job_dir / "robot_profile.generated.yaml"

        events: list[dict[str, Any]] = []
        _record(events, "onboarding_started", {"job_id": job_id, "robot_note": robot_note})

        loaded = yaml.safe_load(profile_yaml) or {}
        if not isinstance(loaded, dict):
            raise ValueError("profile_yaml must decode to a mapping")
        _record(events, "profile_loaded", {"top_level_keys": sorted(str(key) for key in loaded.keys())})

        normalized, warnings, assumptions = self._normalize_profile(loaded)
        _record(events, "schema_understood", {"schema_version": normalized.get("schema_version")})

        streams = normalized.get("streams") if isinstance(normalized.get("streams"), dict) else {}
        color_images = streams.get("color_images") if isinstance(streams.get("color_images"), dict) else {}
        robot_states = streams.get("robot_states") if isinstance(streams.get("robot_states"), dict) else {}
        _record(events, "streams_identified", {"color_images": list(color_images.keys()), "robot_states": list(robot_states.keys())})

        bindings = normalized.get("ROBOLINEAGE_bindings") if isinstance(normalized.get("ROBOLINEAGE_bindings"), dict) else {}
        _record(events, "bindings_identified", {"vsa": bindings.get("vsa") if isinstance(bindings.get("vsa"), dict) else {}})
        recorder = bindings.get("recorder") if isinstance(bindings.get("recorder"), dict) else {}
        _record(events, "recorder_policy_identified", {"camera_names": recorder.get("camera_names") or []})

        understanding = self._generate_understanding(
            normalized_profile=normalized,
            robot_note=robot_note,
            warnings=warnings,
            assumptions=assumptions,
            events=events,
        )
        _write_json_atomic(understanding_path, understanding)
        _record(events, "understanding_written", {"path": str(understanding_path), "status": understanding.get("status")})

        robot_id = str(normalized.get("robot_id") or "generated_robot")
        generated_profile_path = generated_profiles_root / f"generated_{_safe_id(robot_id)}.yaml"
        profile_text = yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False)
        _write_text_atomic(artifact_profile_path, profile_text)
        _write_text_atomic(generated_profile_path, profile_text)

        profile = load_robot_profile(generated_profile_path)
        report = self._report(
            profile=profile,
            job_id=job_id,
            robot_note=robot_note,
            warnings=warnings,
            assumptions=assumptions,
            generated_profile_path=generated_profile_path,
            artifact_profile_path=artifact_profile_path,
            understanding=understanding,
            understanding_path=understanding_path,
        )
        _write_json_atomic(report_path, report)
        _record(events, "profile_written", {"generated_profile_path": str(generated_profile_path), "artifact_profile_path": str(artifact_profile_path)})
        _record(events, "onboarding_completed", {"robot_id": profile.robot_id, "status": "generated"})
        _write_jsonl(events_path, events)

        return RobotOnboardingResult(
            status="generated",
            job_id=job_id,
            robot_id=profile.robot_id,
            generated_profile_path=generated_profile_path,
            artifact_profile_path=artifact_profile_path,
            report_path=report_path,
            events_path=events_path,
            understanding_path=understanding_path,
            report=report,
            events=events,
        )

    def attach_validation(
        self,
        result: RobotOnboardingResult,
        validation: dict[str, Any],
    ) -> RobotOnboardingResult:
        result.report["validation"] = validation
        result.report["validation_status"] = validation.get("status")
        _write_json_atomic(result.report_path, result.report)
        event = _event("validation_completed", {"status": validation.get("status")})
        result.events.append(event)
        _append_jsonl(result.events_path, event)
        return result

    def _normalize_profile(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
        normalized = copy.deepcopy(payload)
        warnings: list[str] = []
        assumptions: list[str] = []

        schema_version = str(normalized.get("schema_version") or "").strip()
        if not schema_version:
            normalized["schema_version"] = ROBOT_PROFILE_SCHEMA_VERSION
            warnings.append("schema_version was missing; set to RoboLineage.robot_profile.v1")
        elif schema_version != ROBOT_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported robot profile schema: {schema_version!r}")

        robot_id = str(normalized.get("robot_id") or "generated_robot").strip()
        if not robot_id:
            robot_id = "generated_robot"
            warnings.append("robot_id was empty; set to generated_robot")
        normalized["robot_id"] = robot_id
        normalized.setdefault("display_name", robot_id)

        connection = normalized.get("connection")
        if not isinstance(connection, dict):
            connection = {}
            normalized["connection"] = connection
            warnings.append("connection was missing; defaulted to ros2")
        connection.setdefault("type", "ros2")
        connection.setdefault("ros_domain_id", 0)
        connection.setdefault("namespace", "")
        connection.setdefault("spin_threads", 2)

        capabilities = normalized.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}
            normalized["capabilities"] = capabilities
        capabilities.setdefault("read_only", True)
        capabilities.setdefault("policy_drive", False)

        streams = normalized.get("streams")
        if not isinstance(streams, dict):
            raise ValueError("robot profile missing streams mapping")
        color_images = streams.get("color_images")
        robot_states = streams.get("robot_states")
        if not isinstance(color_images, dict) or not color_images:
            raise ValueError("robot profile missing streams.color_images")
        if not isinstance(robot_states, dict) or not robot_states:
            raise ValueError("robot profile missing streams.robot_states")

        active = normalized.get("active_streams")
        if not isinstance(active, dict):
            active = {}
            normalized["active_streams"] = active
        if not active.get("color_image"):
            active["color_image"] = _first_key(color_images)
            assumptions.append(f"active color stream inferred as {active['color_image']}")
        if not active.get("robot_state"):
            active["robot_state"] = _first_key(robot_states)
            assumptions.append(f"active robot state inferred as {active['robot_state']}")

        active_camera = str(active["color_image"])
        active_state = str(active["robot_state"])
        color_spec = color_images.get(active_camera) if isinstance(color_images.get(active_camera), dict) else {}
        state_spec = robot_states.get(active_state) if isinstance(robot_states.get(active_state), dict) else {}

        bindings = normalized.get("ROBOLINEAGE_bindings")
        if not isinstance(bindings, dict):
            bindings = {}
            normalized["ROBOLINEAGE_bindings"] = bindings
        vsa = bindings.get("vsa")
        if not isinstance(vsa, dict):
            vsa = {}
            bindings["vsa"] = vsa
        vsa.setdefault("color_image_stream", active_camera)
        vsa.setdefault("robot_state_stream", active_state)
        camera_stream_id = color_spec.get("canonical_topic") or color_spec.get("stream_id")
        arm_stream_id = state_spec.get("canonical_state_topic") or state_spec.get("state_stream_id")
        if camera_stream_id:
            vsa.setdefault("canonical_camera_topic", camera_stream_id)
        if arm_stream_id:
            vsa.setdefault("canonical_arm_topic", arm_stream_id)

        recorder = bindings.get("recorder")
        if not isinstance(recorder, dict):
            recorder = {}
            bindings["recorder"] = recorder
        recorder.setdefault("default_output_dir", "data/rollouts")
        camera_names = recorder.get("camera_names")
        if not isinstance(camera_names, list) or not camera_names:
            recorder["camera_names"] = [active_camera]
            assumptions.append(f"recorder camera_names inferred as [{active_camera}]")

        health = bindings.get("health")
        if not isinstance(health, dict):
            health = {}
            bindings["health"] = health
        required = health.get("required_streams")
        if not isinstance(required, list) or not required:
            health["required_streams"] = [active_camera, active_state]

        return normalized, warnings, assumptions

    def _report(
        self,
        *,
        profile: RobotProfile,
        job_id: str,
        robot_note: str | None,
        warnings: list[str],
        assumptions: list[str],
        generated_profile_path: Path,
        artifact_profile_path: Path,
        understanding: dict[str, Any],
        understanding_path: Path,
    ) -> dict[str, Any]:
        camera_topic, arm_topic = profile_to_vsa_topics(profile)
        bindings = profile.payload.get("ROBOLINEAGE_bindings") if isinstance(profile.payload.get("ROBOLINEAGE_bindings"), dict) else {}
        recorder = bindings.get("recorder") if isinstance(bindings.get("recorder"), dict) else {}
        camera_names = recorder.get("camera_names") if isinstance(recorder.get("camera_names"), list) else []
        return {
            "schema_version": ROBOT_ONBOARDING_REPORT_SCHEMA_VERSION,
            "agent_version": ROBOT_ONBOARDING_AGENT_VERSION,
            "created_at": _now_iso(),
            "job_id": job_id,
            "status": "generated",
            "robot_id": profile.robot_id,
            "display_name": profile.display_name,
            "robot_note": robot_note,
            "active_camera": profile.active_color_stream_id,
            "active_robot_state": profile.active_robot_state_id,
            "recorder_cameras": [str(item) for item in camera_names],
            "vsa_topics": {"camera": camera_topic, "arm": arm_topic},
            "generated_profile_path": str(generated_profile_path),
            "artifact_profile_path": str(artifact_profile_path),
            "summary": profile.to_summary(active=False),
            "warnings": warnings,
            "assumptions": assumptions,
            "llm_understanding": _understanding_ref(understanding, understanding_path),
        }

    def _generate_understanding(
        self,
        *,
        normalized_profile: dict[str, Any],
        robot_note: str | None,
        warnings: list[str],
        assumptions: list[str],
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        model = str(getattr(self.llm_client, "model", "") or "")
        base = {
            "profile_summary": f"Robot profile {normalized_profile.get('robot_id') or 'generated_robot'} normalized.",
            "binding_explanation": "Deterministic profile normalization selected active streams and RoboLineage bindings.",
            "warnings": list(warnings),
            "assumptions": list(assumptions),
            "recommended_checks": [],
        }
        if not self.enable_llm_understanding:
            _record(events, "llm_understanding_skipped", {"reason": "disabled"})
            return _understanding_payload(status="skipped", model=None, body={**base, "reason": "disabled"})
        if self.llm_client is None:
            _record(events, "llm_understanding_not_configured", {})
            return _understanding_payload(
                status="not_configured",
                model=None,
                body={**base, "reason": "ROBOT_ONBOARDING_LLM_API_KEY is not set"},
            )
        _record(events, "llm_understanding_started", {"model": model})
        context = {
            "normalized_profile": normalized_profile,
            "robot_note": robot_note,
            "deterministic_warnings": warnings,
            "deterministic_assumptions": assumptions,
            "contract": {
                "profile_summary": "short summary of robot streams and capabilities",
                "binding_explanation": "explanation of selected RoboLineage bindings",
                "warnings": "list of operator-facing warnings",
                "assumptions": "list of explicit assumptions",
                "recommended_checks": "list of safe validation checks",
            },
        }
        try:
            generated = self.llm_client.generate(context)
            if not isinstance(generated, dict):
                raise RuntimeError("Robot onboarding LLM returned a non-object response")
            understanding = _understanding_payload(status="generated", model=model or None, body=generated)
            _record(events, "llm_understanding_completed", {"model": model, "summary": understanding.get("profile_summary")})
            return understanding
        except Exception as exc:
            _record(events, "llm_understanding_failed", {"model": model, "error": str(exc)})
            return _understanding_payload(
                status="failed",
                model=model or None,
                body={**base, "error": str(exc)},
            )


def _first_key(value: dict[str, Any]) -> str:
    return str(next(iter(value.keys())))


def _understanding_payload(
    *,
    status: str,
    model: str | None,
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ROBOT_ONBOARDING_UNDERSTANDING_SCHEMA_VERSION,
        "agent_version": ROBOT_ONBOARDING_AGENT_VERSION,
        "created_at": _now_iso(),
        "status": status,
        "model": model,
        "profile_summary": str(body.get("profile_summary") or body.get("summary") or ""),
        "binding_explanation": str(body.get("binding_explanation") or ""),
        "warnings": _string_list(body.get("warnings")),
        "assumptions": _string_list(body.get("assumptions")),
        "recommended_checks": _string_list(body.get("recommended_checks")),
        "reason": str(body.get("reason") or "") if body.get("reason") else None,
        "error": str(body.get("error") or "") if body.get("error") else None,
    }


def _understanding_ref(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "model": payload.get("model"),
        "path": str(path),
        "summary": payload.get("profile_summary") or "",
        "reason": payload.get("reason"),
        "error": payload.get("error"),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._-") or "generated"


def _record(events: list[dict[str, Any]], event: str, payload: dict[str, Any]) -> None:
    events.append(_event(event, payload))


def _event(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ROBOT_ONBOARDING_EVENT_SCHEMA_VERSION,
        "agent_version": ROBOT_ONBOARDING_AGENT_VERSION,
        "created_at": _now_iso(),
        "event": event,
        "payload": payload,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
