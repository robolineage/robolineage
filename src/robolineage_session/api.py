"""FastAPI application for the RoboLineage session service."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from robolineage_ar.server import create_app as create_ar_app
from robolineage_ar.types import CameraParams, RenderConfig
from robolineage_ar.video_source import SyntheticVideoSource, VideoSource
from robolineage_contracts.core import RolloutMode
from robolineage_contracts.session import (
    ALLOWED_TRANSITIONS,
    ControlEventName,
    ErrorCode,
    EventEnvelope,
    EventSource,
    FeedbackEventName,
    SessionState,
)
from robolineage_shared_agents.llm_routes import (
    DEFAULT_OPENAI_COMPAT_MODEL,
    all_ai_route_statuses,
    resolve_ai_route,
)

from robolineage_session.dispatcher import make_envelope, utc_now_iso
from robolineage_session.events import EventBroadcaster, EventLogger, envelope_to_dict
from robolineage_session.modes import trajectory_accepted
from robolineage_session.runtime_archive import archive
from robolineage_session.session import DEFAULT_REGISTRY, Session, SessionRegistry
from robolineage_session.state_machine import IllegalStateTransition


def _json_response(status_code: int, payload: dict) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload)


def _error(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    return _json_response(status_code, {"event": "ERROR", "code": code.value, "message": message})


def _bad_request(exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _control_envelope(body: dict[str, Any], rollout_id: str | None) -> EventEnvelope:
    return EventEnvelope(
        event=str(body.get("event", "")),
        event_id=str(body.get("event_id") or make_envelope(
            event="noop", rollout_id=None, source=EventSource.UX
        ).event_id),
        timestamp=str(body.get("timestamp") or utc_now_iso()),
        rollout_id=rollout_id,
        source=EventSource.UX,
        payload=dict(body.get("payload") or {}),
    )


def _state_payload(session: Session | None) -> dict:
    if session is None:
        return {"state": SessionState.IDLE.value, "rollout_id": None, "mode": None}
    return {
        "state": session.state.value,
        "session_id": session.session_id,
        "rollout_id": session.rollout_id,
        "mode": session.mode.value,
        "task_id": session.task_id,
        "operator_id": session.operator_id,
        "policy_version": session.policy_version,
    }


def _write_metadata(session: Session) -> None:
    session.rollout_dir.mkdir(parents=True, exist_ok=True)
    tmp = session.rollout_dir / "metadata.json.tmp"
    final = session.rollout_dir / "metadata.json"
    payload = {
        "rollout_id": session.rollout_id,
        "session_id": session.session_id,
        "task_id": session.task_id,
        "mode": session.mode.value,
        "policy_version": session.policy_version,
        "operator_id": session.operator_id,
        "started_at": session.started_at,
        "storage_path": str(session.rollout_dir),
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(final)


def _append_and_broadcast(logger: EventLogger, broadcaster: EventBroadcaster, env: EventEnvelope) -> None:
    logger.append(env)
    broadcaster.broadcast(env)


def _frame_to_jpeg_data_url(frame: Any, *, max_side: int = 768, quality: int = 80) -> str | None:
    """Encode a BGR/RGB numpy frame as a compact JPEG data URL for VLM input."""
    if frame is None:
        return None
    try:
        import base64

        import cv2

        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            return None
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if height <= 0 or width <= 0:
            return None

        scale = min(1.0, float(max_side) / float(max(height, width)))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
        )
        if not ok:
            return None
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    except Exception:
        return None


def _task_config_user_content(task_description: str, image_data_url: str | None) -> Any:
    text = (
        f"Task description: {task_description}\n\n"
        "Use the current camera image as scene context when it is provided. "
        "Let visible geometry, handles, object pose, obstacles, gripper-object "
        "relations, and likely contact points influence the phase split, visual "
        "hints, action hints, failure signals, and success criteria."
    )
    if image_data_url is None:
        return (
            text
            + "\n\nNo current camera image was available; infer the task config "
            "from the text only and keep assumptions conservative."
        )
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": image_data_url, "detail": "low"},
        },
    ]


def _call_task_llm(
    system_prompt: str,
    task_description: str,
    frame: Any,
) -> str:
    """Call the task-config LLM using the backend selected by TASK_LLM_BACKEND.

    Returns the raw JSON string from the model.
    Backends: openai (default) | google | anthropic
    Env route: TASK_LLM_* with OPENAI_* fallback for key/url/model.
    """
    route = resolve_ai_route("TASK_LLM", fallback_prefixes=("OPENAI",))
    backend = route.backend
    api_key = route.api_key
    base_url = route.base_url
    model = route.model
    model_was_configured = route.sources.get("model") is not None

    image_data_url = _frame_to_jpeg_data_url(frame) if frame is not None else None

    if backend == "google":
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                f"google-generativeai not installed: {exc}. Run: pip install google-generativeai"
            ) from exc
        genai.configure(api_key=api_key)
        model_name = model if model_was_configured else "gemini-2.0-flash"
        llm = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
        )
        parts: list = [
            task_description
            + "\n\nOutput ONLY a valid JSON object as described in the system prompt."
        ]
        if image_data_url:
            from PIL import Image as PILImage
            import base64 as _b64
            raw = _b64.b64decode(image_data_url.split(",", 1)[1])
            import io as _io
            parts.append(PILImage.open(_io.BytesIO(raw)))
        response = llm.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=1024),
        )
        return (response.text or "").strip()

    if backend == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                f"anthropic not installed: {exc}. Run: pip install anthropic"
            ) from exc
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**kwargs)
        model_name = model if model_was_configured else DEFAULT_OPENAI_COMPAT_MODEL
        content: list = []
        if image_data_url:
            import base64 as _b64
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data_url.split(",", 1)[1],
                },
            })
        content.append({"type": "text", "text": (
            task_description
            + "\n\nOutput ONLY a valid JSON object as described in the system prompt."
        )})
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return (response.content[0].text or "").strip()

    # default: openai-compatible
    import httpx
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        http_client=httpx.Client(trust_env=False),
    )
    model_name = model or DEFAULT_OPENAI_COMPAT_MODEL
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _task_config_user_content(task_description, image_data_url)},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content or "").strip()


def _save_versioned_task_config(
    *,
    compatibility_path: Path,
    cfg: dict[str, Any],
    task_description: str,
    current_frame_available: bool,
) -> dict[str, Any]:
    """Write an immutable task_config version plus latest/compat aliases."""
    import yaml

    root = compatibility_path.parent
    versions_dir = root / "task_configs"
    index_path = versions_dir / "task_config_index.json"
    versions_dir.mkdir(parents=True, exist_ok=True)
    index = _read_task_config_index(index_path)
    version_number = _next_task_config_version(index, versions_dir)
    version_id = f"v{version_number}"
    version_path = versions_dir / f"task_config.{version_id}.yaml"
    latest_path = root / "task_config.latest.yaml"
    created_at = utc_now_iso()

    yaml_text = yaml.dump(cfg, allow_unicode=True, sort_keys=False)
    _write_text_atomic(version_path, yaml_text)
    _write_text_atomic(latest_path, yaml_text)
    _write_text_atomic(compatibility_path, yaml_text)
    llm_route = resolve_ai_route("TASK_LLM", fallback_prefixes=("OPENAI",))

    entry = {
        "version_id": version_id,
        "created_at": created_at,
        "task_description": task_description,
        "phases": list(cfg.get("phases") or []),
        "version_path": str(version_path),
        "latest_path": str(latest_path),
        "compatibility_path": str(compatibility_path),
        "source": {
            "current_frame_available": current_frame_available,
            "llm_backend": llm_route.backend,
            "llm_model": llm_route.model,
            "llm_configured": llm_route.configured,
        },
    }
    entries = [item for item in index.get("entries", []) if item.get("version_id") != version_id]
    entries.append(entry)
    payload = {
        "schema_version": "RoboLineage.task_config_index.v1",
        "latest_version": version_id,
        "latest_path": str(latest_path),
        "compatibility_path": str(compatibility_path),
        "entries": entries,
    }
    _write_json_atomic(index_path, payload)
    return {
        **entry,
        "index_path": str(index_path),
        "task_config_path": str(version_path),
    }


def _read_task_config_index(index_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _next_task_config_version(index: dict[str, Any], versions_dir: Path) -> int:
    numbers: list[int] = []
    for entry in index.get("entries") or []:
        version = str(entry.get("version_id") or "")
        match = re.fullmatch(r"v(\d+)", version)
        if match:
            numbers.append(int(match.group(1)))
    for path in versions_dir.glob("task_config.v*.yaml"):
        match = re.fullmatch(r"task_config\.v(\d+)\.yaml", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def create_app(
    *,
    data_root: Path | str = Path("data/rollouts"),
    runtime_root: Path | str = Path("runtime"),
    registry: SessionRegistry = DEFAULT_REGISTRY,
    broadcaster: EventBroadcaster | None = None,
    video_source: VideoSource | None = None,
    camera: CameraParams | None = None,
    render_config: RenderConfig | None = None,
    on_task_configure: Optional[Callable[[str], Any]] = None,
    on_rollout_start: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_rollout_stop: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_rollout_state: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_task_stop: Optional[Callable[[], None]] = None,
    on_post_review_state: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_post_review_rollouts: Optional[Callable[[int], dict[str, Any] | None]] = None,
    on_post_review_detail: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_training_framework_state: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_training_framework_runs: Optional[Callable[[int], dict[str, Any] | None]] = None,
    on_training_framework_detail: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_training_framework_run_demo: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_training_framework_discover: Optional[Callable[[dict[str, Any]], dict[str, Any] | None]] = None,
    on_training_framework_discovery_job: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_tasks: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_task_create: Optional[Callable[[dict[str, Any]], dict[str, Any] | None]] = None,
    on_task_activate: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_task_detail: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_task_collection_sessions: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_task_collection_session_detail: Optional[Callable[[str, str], dict[str, Any] | None]] = None,
    on_task_deployment_sessions: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_task_deployment_session_detail: Optional[Callable[[str, str], dict[str, Any] | None]] = None,
    on_training_selections: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_training_selection_create: Optional[Callable[[str, dict[str, Any]], dict[str, Any] | None]] = None,
    on_framework_profiles: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_training_data_adapt_start: Optional[Callable[[str, dict[str, Any]], dict[str, Any] | None]] = None,
    on_training_run_start: Optional[Callable[[str, dict[str, Any]], dict[str, Any] | None]] = None,
    on_policies: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_rollout_session_state: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_collection_session_start: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_collection_session_stop: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_deployment_session_start: Optional[Callable[[dict[str, Any]], dict[str, Any] | None]] = None,
    on_deployment_session_stop: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_robots: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_robot_detail: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_robot_activate: Optional[Callable[[str], dict[str, Any] | None]] = None,
    on_robot_validate: Optional[Callable[[str | None], dict[str, Any] | None]] = None,
    on_robot_onboard: Optional[Callable[[dict[str, Any]], dict[str, Any] | None]] = None,
    on_master_status: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_master_review: Optional[Callable[[], dict[str, Any] | None]] = None,
    on_ai_routes_status: Optional[Callable[[], dict[str, Any] | None]] = None,
) -> FastAPI:
    """Build the session FastAPI app.

    The mounted AR sub-app receives overlay trajectories through its
    HTTP-facing trajectory endpoint. ROS2 live streams are consumed directly by
    the runtime components that need them.
    """
    data_root = Path(data_root)
    runtime_root = Path(runtime_root)
    broadcaster = broadcaster or EventBroadcaster()
    ar_app_ref: dict[str, FastAPI] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            ar_app = ar_app_ref.get("app")
            if ar_app is not None:
                ar_app.state.shutdown_overlay_consumers()

    app = FastAPI(title="RoboLineage Session Service", version="0.1.0", lifespan=lifespan)

    def trajectory_gate() -> bool:
        session = registry.current()
        return (
            session is not None
            and session.state == SessionState.COLLECTING
            and trajectory_accepted(session.mode)
        )

    @app.post("/events")
    def post_event(body: dict[str, Any]):
        try:
            event = ControlEventName(str(body.get("event")))
        except ValueError:
            return _error(400, ErrorCode.E_ILLEGAL_STATE_TRANSITION, "unknown control event")

        if event == ControlEventName.START_COLLECTING:
            payload = dict(body.get("payload") or {})
            try:
                mode = RolloutMode(payload["mode"])
                session = registry.create(
                    data_root=data_root,
                    runtime_root=runtime_root,
                    task_id=str(payload["task_id"]),
                    mode=mode,
                    operator_id=str(payload.get("operator_id", "unknown")),
                    policy_version=payload.get("policy_version"),
                    started_at=utc_now_iso(),
                )
                session.runtime_dir.mkdir(parents=True, exist_ok=True)
                _write_metadata(session)
                session.state_machine.transition(SessionState.COLLECTING)
            except Exception as exc:
                registry.clear()
                return _error(500, ErrorCode.E_SESSION_OPEN_FAILED, str(exc))

            logger = EventLogger(session.events_path)
            _append_and_broadcast(logger, broadcaster, _control_envelope(body, session.rollout_id))
            opened = make_envelope(
                event=FeedbackEventName.SESSION_OPENED.value,
                rollout_id=session.rollout_id,
                source=EventSource.AR,
                payload=_state_payload(session),
            )
            _append_and_broadcast(logger, broadcaster, opened)
            return _state_payload(session)

        session = registry.current()
        if session is None:
            return _error(400, ErrorCode.E_ILLEGAL_STATE_TRANSITION, "no active session")

        logger = EventLogger(session.events_path)
        try:
            if event == ControlEventName.PAUSE_COLLECTING:
                session.state_machine.transition(SessionState.PAUSED)
            elif event == ControlEventName.RESUME_COLLECTING:
                session.state_machine.transition(SessionState.COLLECTING)
            elif event == ControlEventName.STOP_COLLECTING:
                session.state_machine.transition(SessionState.REVIEWING)
            elif event == ControlEventName.SUBMIT_ROLLOUT:
                if SessionState.SUBMITTED not in ALLOWED_TRANSITIONS[session.state]:
                    raise IllegalStateTransition(session.state, SessionState.SUBMITTED)
            elif event == ControlEventName.DISCARD_ROLLOUT:
                pass
        except IllegalStateTransition as exc:
            return _error(400, exc.error_code, str(exc))

        _append_and_broadcast(logger, broadcaster, _control_envelope(body, session.rollout_id))

        if event == ControlEventName.SUBMIT_ROLLOUT:
            try:
                archived = archive(session.runtime_dir, session.rollout_dir)
                closed_tmp = session.rollout_dir / ".closed.tmp"
                closed_tmp.write_text(utc_now_iso() + "\n", encoding="utf-8")
                closed_tmp.replace(session.rollout_dir / ".closed")
                session.state_machine.transition(SessionState.SUBMITTED)
                session.state_machine.transition(SessionState.IDLE)
            except Exception as exc:
                return _error(500, ErrorCode.E_SESSION_CLOSE_FAILED, str(exc))
            closed = make_envelope(
                event=FeedbackEventName.SESSION_CLOSED.value,
                rollout_id=session.rollout_id,
                source=EventSource.AR,
                payload={"archived_snapshots_path": str(archived)},
            )
            _append_and_broadcast(logger, broadcaster, closed)
            payload = _state_payload(session)
            logger.close()
            registry.clear()
            return payload

        if event == ControlEventName.DISCARD_ROLLOUT:
            trash_root = session.rollout_dir.parent / ".trash"
            trash_root.mkdir(parents=True, exist_ok=True)
            if session.rollout_dir.exists():
                shutil.move(str(session.rollout_dir), str(trash_root / session.rollout_id))
            logger.close()
            registry.clear()
            return {"state": SessionState.IDLE.value, "rollout_id": None, "discarded": session.rollout_id}

        return _state_payload(session)

    @app.get("/state")
    def get_state():
        return _state_payload(registry.current())

    @app.get("/events/stream")
    def event_stream():
        def plain_stream():
            for env in broadcaster.subscribe():
                yield f"event: {env.event}\ndata: {json.dumps(envelope_to_dict(env), ensure_ascii=False)}\n\n"

        try:
            from sse_starlette.sse import EventSourceResponse

            def sse_stream():
                for env in broadcaster.subscribe():
                    yield {
                        "event": env.event,
                        "data": json.dumps(envelope_to_dict(env), ensure_ascii=False),
                    }

            return EventSourceResponse(sse_stream())
        except ImportError:
            return StreamingResponse(plain_stream(), media_type="text/event-stream")

    session_video_source = video_source or SyntheticVideoSource()
    ar_app = create_ar_app(
        video_source=session_video_source,
        camera=camera or CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0),
        render_config=render_config or RenderConfig(),
        trajectory_gate=trajectory_gate,
    )
    for route in ar_app.router.routes:
        app.router.routes.append(route)
    ar_app_ref["app"] = ar_app

    @app.get("/session-health")
    def session_health():
        return {"status": "ok", **_state_payload(registry.current())}

    @app.get("/ai/routes")
    def ai_routes_status():
        try:
            if on_ai_routes_status is not None:
                return JSONResponse(content=on_ai_routes_status() or all_ai_route_statuses())
            return JSONResponse(content=all_ai_route_statuses())
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/robots")
    def robots():
        if on_robots is None:
            return JSONResponse(content={"profiles": [], "active_robot_id": None})
        try:
            return JSONResponse(content=on_robots() or {"profiles": []})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/robots/onboarding")
    def robot_onboarding(body: dict[str, Any]):
        if on_robot_onboard is None:
            return JSONResponse(status_code=501, content={"error": "robot onboarding not supported"})
        try:
            return JSONResponse(content=on_robot_onboard(body) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/robots/{robot_id}")
    def robot_detail(robot_id: str):
        if on_robot_detail is None:
            return JSONResponse(status_code=501, content={"error": "robot profile detail not supported"})
        try:
            return JSONResponse(content=on_robot_detail(robot_id) or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/robots/{robot_id}/activate")
    def robot_activate(robot_id: str):
        if on_robot_activate is None:
            return JSONResponse(status_code=501, content={"error": "robot profile activation not supported"})
        try:
            return JSONResponse(content=on_robot_activate(robot_id) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/robots/{robot_id}/validate")
    def robot_validate(robot_id: str):
        if on_robot_validate is None:
            return JSONResponse(status_code=501, content={"error": "robot profile validation not supported"})
        try:
            return JSONResponse(content=on_robot_validate(robot_id) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    _task_dir = os.environ.get("ROBOLINEAGE_TASK_DIR")
    _TASK_CONFIG_PATH = (
        Path(_task_dir) / "task_config.yaml"
        if _task_dir
        else Path(__file__).resolve().parents[3] / "configs" / "task_pickplace.yaml"
    )

    def _task_config_compatibility_path() -> Path:
        task_dir = os.environ.get("ROBOLINEAGE_TASK_DIR")
        if task_dir:
            return Path(task_dir) / "task_config.yaml"
        return _TASK_CONFIG_PATH

    _GENERATE_SYSTEM_PROMPT = (
        'You are a robot task analyst. Given a robot arm manipulation task description '
        'and, when provided, a current camera image of the scene, '
        'output a single JSON object with exactly five keys: "phases", "hints", "failure_signals", "phase_action_hints", "phase_visual_hints".\n\n'
        'Rules:\n'
        '- "phases": ordered list of 2-6 short snake_case phase names covering the full task.\n'
        '- Use the image to adapt phases to visible affordances and geometry, such as handles, object pose, obstacles, drawer direction, and likely contact points.\n'
        '- "hints": object mapping each phase name to a short human-readable description (1 sentence).\n'
        '- "failure_signals": list of 2-5 short strings describing observable failure modes.\n'
        '- "phase_action_hints": object mapping each phase name to an action-signal descriptor object.\n'
        '  Each descriptor may include any subset of these keys:\n'
        '    "gripper_state": "open" or "closed" — expected gripper state at the end of this phase.\n'
        '    "event_type": list of strings — motion events that typically trigger this phase.\n'
        '      Allowed values: "sequence_start", "gripper_close", "gripper_open", "still_start", "motion_resume", "heartbeat".\n'
        '    "motion_pattern": one of the following strings:\n'
        '      "z_up"              — arm moves upward (lifting object)\n'
        '      "xy_move"           — arm moves horizontally (transporting)\n'
        '      "moving_to_object"  — arm approaches at low speed\n'
        '      "contact_or_capture"— arm contacts or grasps at very low speed\n'
        '      "release_or_settle" — arm releases or stays still\n'
        '- "phase_visual_hints": object mapping each phase name to a 1-2 sentence visual description.\n'
        '  Describe what the scene looks like from a head-mounted camera during this phase.\n'
        '  Focus on observable cues: object position relative to gripper, contact state, arm pose, object movement.\n'
        '- Output ONLY the JSON object. No preamble, no explanation, no markdown fences.'
    )

    @app.post("/task/configure")
    def task_configure(body: dict[str, Any]):
        task_description = str(body.get("task_description", "")).strip()
        if not task_description:
            return JSONResponse(status_code=400, content={"error": "task_description is required"})
        try:
            current_frame = session_video_source.read()
            raw_json = _call_task_llm(_GENERATE_SYSTEM_PROMPT, task_description, current_frame)
            data = json.loads(raw_json or "{}")
            cfg = {
                "task_description": task_description,
                "phases": data.get("phases", []),
                "phase_definitions": data.get("hints", {}),
                "failure_signals": data.get("failure_signals", []),
                "phase_action_hints": data.get("phase_action_hints", {}),
                "phase_visual_hints": data.get("phase_visual_hints", {}),
            }
            version_meta = _save_versioned_task_config(
                compatibility_path=_task_config_compatibility_path(),
                cfg=cfg,
                task_description=task_description,
                current_frame_available=current_frame is not None,
            )
            if on_task_configure is not None:
                on_task_configure(str(version_meta["task_config_path"]))
            return JSONResponse(content={
                "task_description": task_description,
                "phases": cfg["phases"],
                "phase_definitions": cfg["phase_definitions"],
                "failure_signals": cfg["failure_signals"],
                "phase_action_hints": cfg["phase_action_hints"],
                "phase_visual_hints": cfg["phase_visual_hints"],
                "task_config_version": version_meta["version_id"],
                "task_config_path": version_meta["task_config_path"],
                "task_config_latest_path": version_meta["latest_path"],
                "task_config_index_path": version_meta["index_path"],
                "task_config_created_at": version_meta["created_at"],
            })
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/task/rollout/start")
    def task_rollout_start():
        if on_rollout_start is None:
            return JSONResponse(status_code=501, content={"error": "rollout start not supported"})
        try:
            payload = on_rollout_start() or {}
            return JSONResponse(content={"status": "started", **payload})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/task/rollout/stop")
    def task_rollout_stop():
        callback = on_rollout_stop or on_task_stop
        if callback is None:
            return JSONResponse(status_code=501, content={"error": "rollout stop not supported"})
        try:
            payload = callback() or {}
            return JSONResponse(content={"status": "stopped", **payload})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/task/rollout/state")
    def task_rollout_state():
        if on_rollout_state is None:
            return JSONResponse(content={"active": False})
        try:
            return JSONResponse(content=on_rollout_state() or {"active": False})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/task/session/state")
    def task_rollout_session_state():
        if on_rollout_session_state is None:
            return JSONResponse(content={"active": False, "kind": None, "rollout_count": 0})
        try:
            return JSONResponse(content=on_rollout_session_state() or {"active": False})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/task/session/collection/start")
    def task_collection_session_start():
        if on_collection_session_start is None:
            return JSONResponse(status_code=501, content={"error": "collection session start not supported"})
        try:
            return JSONResponse(content=on_collection_session_start() or {})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/task/session/collection/stop")
    def task_collection_session_stop():
        if on_collection_session_stop is None:
            return JSONResponse(status_code=501, content={"error": "collection session stop not supported"})
        try:
            return JSONResponse(content=on_collection_session_stop() or {})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/task/session/deployment/start")
    def task_deployment_session_start(body: dict[str, Any] | None = None):
        if on_deployment_session_start is None:
            return JSONResponse(status_code=501, content={"error": "deployment session start not supported"})
        try:
            return JSONResponse(content=on_deployment_session_start(body or {}) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/task/session/deployment/stop")
    def task_deployment_session_stop():
        if on_deployment_session_stop is None:
            return JSONResponse(status_code=501, content={"error": "deployment session stop not supported"})
        try:
            return JSONResponse(content=on_deployment_session_stop() or {})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/post-review/status")
    def post_review_status():
        if on_post_review_state is None:
            return JSONResponse(content={"active": False, "queue_size": 0})
        try:
            return JSONResponse(content=on_post_review_state() or {"active": False, "queue_size": 0})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/post-review/rollouts")
    def post_review_rollouts(limit: int = 50):
        if on_post_review_rollouts is None:
            return JSONResponse(content={"rollouts": [], "status": {"active": False}})
        try:
            return JSONResponse(content=on_post_review_rollouts(limit) or {"rollouts": []})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/post-review/rollouts/{rollout_id}")
    def post_review_detail(rollout_id: str):
        if on_post_review_detail is None:
            return JSONResponse(status_code=501, content={"error": "post-review detail not supported"})
        try:
            return JSONResponse(content=on_post_review_detail(rollout_id) or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/training-framework/status")
    def training_framework_status():
        if on_training_framework_state is None:
            return JSONResponse(content={"active": False})
        try:
            return JSONResponse(content=on_training_framework_state() or {"active": False})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/training-framework/runs")
    def training_framework_runs(limit: int = 30):
        if on_training_framework_runs is None:
            return JSONResponse(content={"runs": [], "status": {"active": False}})
        try:
            return JSONResponse(content=on_training_framework_runs(limit) or {"runs": []})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/training-framework/runs/{run_id}")
    def training_framework_detail(run_id: str):
        if on_training_framework_detail is None:
            return JSONResponse(status_code=501, content={"error": "training framework detail not supported"})
        try:
            return JSONResponse(content=on_training_framework_detail(run_id) or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/training-framework/run-demo")
    def training_framework_run_demo():
        if on_training_framework_run_demo is None:
            return JSONResponse(status_code=501, content={"error": "training framework demo not supported"})
        try:
            return JSONResponse(content=on_training_framework_run_demo() or {})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/training-framework/discover")
    def training_framework_discover(body: dict[str, Any]):
        if on_training_framework_discover is None:
            return JSONResponse(status_code=501, content={"error": "training framework discovery not supported"})
        try:
            return JSONResponse(content=on_training_framework_discover(body) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/training-framework/discovery/{job_id}")
    def training_framework_discovery_job(job_id: str):
        if on_training_framework_discovery_job is None:
            return JSONResponse(status_code=501, content={"error": "training framework discovery jobs not supported"})
        try:
            return JSONResponse(content=on_training_framework_discovery_job(job_id) or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks")
    def tasks():
        if on_tasks is None:
            return JSONResponse(content={"tasks": []})
        try:
            return JSONResponse(content=on_tasks() or {"tasks": []})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/tasks")
    def task_create(body: dict[str, Any]):
        if on_task_create is None:
            return JSONResponse(status_code=501, content={"error": "task create not supported"})
        try:
            return JSONResponse(content=on_task_create(body) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}")
    def task_detail(task_id: str):
        if on_task_detail is None:
            return JSONResponse(status_code=501, content={"error": "task detail not supported"})
        try:
            return JSONResponse(content=on_task_detail(task_id) or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/master/status")
    def master_status():
        if on_master_status is None:
            return JSONResponse(content={"state": None, "review": None, "available": False})
        try:
            return JSONResponse(content=on_master_status() or {"state": None, "review": None})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/master/review")
    def master_review():
        if on_master_review is None:
            return JSONResponse(status_code=501, content={"error": "master review not supported"})
        try:
            return JSONResponse(content=on_master_review() or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/tasks/{task_id}/activate")
    def task_activate(task_id: str):
        if on_task_activate is None:
            return JSONResponse(status_code=501, content={"error": "task activation not supported"})
        try:
            return JSONResponse(content=on_task_activate(task_id) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}/collection-sessions")
    def task_collection_sessions(task_id: str):
        if on_task_collection_sessions is None:
            return JSONResponse(content={"sessions": []})
        try:
            return JSONResponse(content=on_task_collection_sessions(task_id) or {"sessions": []})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}/collection-sessions/{session_id}")
    def task_collection_session_detail(task_id: str, session_id: str):
        if on_task_collection_session_detail is None:
            return JSONResponse(status_code=501, content={"error": "collection session detail not supported"})
        try:
            return JSONResponse(content=on_task_collection_session_detail(task_id, session_id) or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}/deployment-sessions")
    def task_deployment_sessions(task_id: str):
        if on_task_deployment_sessions is None:
            return JSONResponse(content={"sessions": []})
        try:
            return JSONResponse(content=on_task_deployment_sessions(task_id) or {"sessions": []})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}/deployment-sessions/{session_id}")
    def task_deployment_session_detail(task_id: str, session_id: str):
        if on_task_deployment_session_detail is None:
            return JSONResponse(status_code=501, content={"error": "deployment session detail not supported"})
        try:
            return JSONResponse(content=on_task_deployment_session_detail(task_id, session_id) or {})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}/training-selections")
    def task_training_selections(task_id: str):
        if on_training_selections is None:
            return JSONResponse(content={"selections": []})
        try:
            return JSONResponse(content=on_training_selections(task_id) or {"selections": []})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/tasks/{task_id}/training-selections")
    def task_training_selection_create(task_id: str, body: dict[str, Any]):
        if on_training_selection_create is None:
            return JSONResponse(status_code=501, content={"error": "training selection create not supported"})
        try:
            return JSONResponse(content=on_training_selection_create(task_id, body) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}/framework-profiles")
    def task_framework_profiles(task_id: str):
        if on_framework_profiles is None:
            return JSONResponse(content={"profiles": []})
        try:
            return JSONResponse(content=on_framework_profiles(task_id) or {"profiles": []})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/tasks/{task_id}/training-runs")
    def task_training_run_start(task_id: str, body: dict[str, Any]):
        if on_training_run_start is None:
            return JSONResponse(status_code=501, content={"error": "training run start not supported"})
        try:
            return JSONResponse(content=on_training_run_start(task_id, body) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/tasks/{task_id}/training-data-adapt")
    def task_training_data_adapt_start(task_id: str, body: dict[str, Any]):
        if on_training_data_adapt_start is None:
            return JSONResponse(status_code=501, content={"error": "training data adapt not supported"})
        try:
            return JSONResponse(content=on_training_data_adapt_start(task_id, body) or {})
        except ValueError as exc:
            return _bad_request(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/tasks/{task_id}/policies")
    def task_policies(task_id: str):
        if on_policies is None:
            return JSONResponse(content={"policies": []})
        try:
            return JSONResponse(content=on_policies(task_id) or {"policies": []})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/task/stop")
    def task_stop():
        callback = on_rollout_stop or on_task_stop
        if callback is None:
            return JSONResponse(status_code=501, content={"error": "stop not supported"})
        try:
            payload = callback() or {}
            return JSONResponse(content={"status": "stopped", **payload})
        except ValueError as exc:
            return _bad_request(exc)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    def _latest_tiaoshi_log() -> Path | None:
        import glob as _glob
        if on_rollout_state is not None:
            try:
                state = on_rollout_state() or {}
            except Exception:
                state = {}
            rollout_dir = state.get("rollout_dir")
            if state.get("active") and rollout_dir:
                active_log = Path(str(rollout_dir)) / "logs" / "tiaoshi.log"
                if active_log.exists() and active_log.stat().st_size > 0:
                    return active_log

        task_dir = os.environ.get("ROBOLINEAGE_TASK_DIR")
        if not task_dir:
            return None
        pattern = str(Path(task_dir) / "rollouts" / "*" / "logs" / "tiaoshi.log")
        candidates = _glob.glob(pattern)
        if not candidates:
            return None
        return Path(max(candidates, key=os.path.getmtime))

    @app.get("/vsa/decisions")
    def vsa_decisions(n: int = 50):
        import json as _json
        log_path = _latest_tiaoshi_log()
        if log_path is None or not log_path.exists():
            return JSONResponse(content=[])
        results: list[dict] = []
        rollout_dir = log_path.parent.parent
        rollout_context = _read_json_file(rollout_dir / "rollout_context.json")
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    parsed = entry.get("parsed") or {}
                    results.append({
                        "rollout_id": rollout_dir.name,
                        "rollout_index": rollout_context.get("rollout_index"),
                        "timestamp": entry.get("timestamp", ""),
                        "phase": parsed.get("phase", ""),
                        "progress": parsed.get("progress", ""),
                        "event_type": entry.get("event_type", ""),
                        "anchor_frame": entry.get("anchor_frame"),
                        "end_frame": entry.get("end_frame"),
                        "n_images": entry.get("n_images", 0),
                        "image_paths": entry.get("image_paths", []),
                        "prior": entry.get("prior", {}),
                        "parsed_full": parsed,
                        "raw_response": entry.get("raw_response", ""),
                    })
        except OSError:
            return JSONResponse(content=[])
        return JSONResponse(content=results[-n:])

    @app.get("/vsa/rollout-image")
    def vsa_rollout_image(path: str):
        import mimetypes
        log_path = _latest_tiaoshi_log()
        if log_path is None:
            raise HTTPException(status_code=404, detail="no active rollout")
        rollout_dir = log_path.parent.parent  # logs/ -> rollout_dir
        img_path = (rollout_dir / path).resolve()
        if not str(img_path).startswith(str(rollout_dir.resolve())):
            raise HTTPException(status_code=403, detail="forbidden")
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="image not found")
        mime = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
        return Response(content=img_path.read_bytes(), media_type=mime)

    return app
