"""UnifiedRuntime — composes source adapters + session + AR + VSA in one process.

Raw data capture records ROS2 topics directly with ``ros2 bag record``. Online
VSA subscribes to ROS2 topics directly as well, so high-rate raw capture no
longer competes with an in-process fanout transport.

Lifecycle:
    runtime = UnifiedRuntime(cfg)
    runtime.start()        # session subscribes → data_source publishes → vsa consumes
    ...
    runtime.stop_all()     # reverse order; logs + continues even if a step raises

``services`` flags from yaml decide which sub-runners are spawned. None of
them is mandatory for a "data source only" smoke test.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from robolineage_data_source.config.schema import Config, PostReviewConfig, ServicesToggle
from robolineage_data_source.orchestrator import Orchestrator
from robolineage_robot import (
    RobotProfile,
    RobotProfileRegistry,
    load_robot_profile,
    profile_to_adapter_config,
    profile_to_vsa_topics,
)

_LOG = logging.getLogger(__name__)
_ONLINE_RING_CAPACITY_MAX = 60
_MAX_DRAINING_VSA_ROLLOUTS = 100


@dataclass
class _OnlineRolloutRun:
    rollout_id: str
    rollout_dir: Path
    output_jsonl: Path
    pipeline: Any
    thread: threading.Thread
    stop_flag: threading.Event
    raw_recorder: Any | None
    started_at: str
    rollout_index: int | None = None
    capture_stopped_at: str | None = None
    analysis_completed_at: str | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class _MasterReviewJob:
    trigger: str
    task_root: Path
    enqueued_at: str
    enqueued_mono: float


class _RuntimeDelegate:
    domain = "runtime"

    def __init__(self, owner: Any) -> None:
        self._owner = owner


class _RobotRuntime(_RuntimeDelegate):
    domain = "robot"

    def profiles(self) -> dict[str, Any]:
        owner = self._owner
        active_id = owner._active_robot_profile.robot_id if owner._active_robot_profile else None
        discovered = owner._robot_registry.list_profiles()
        profiles = [
            profile.to_summary(active=profile.robot_id == active_id)
            for profile in discovered
        ]
        if (
            owner._active_robot_profile is not None
            and all(profile.robot_id != active_id for profile in discovered)
        ):
            profiles.insert(0, owner._active_robot_profile.to_summary(active=True))
        return {
            "active_robot_id": active_id,
            "active_profile_path": (
                str(owner._active_robot_profile.path)
                if owner._active_robot_profile is not None
                else None
            ),
            "last_error": owner._robot_last_error,
            "profiles": profiles,
        }

    def profile_detail(self, robot_id: str) -> dict[str, Any]:
        owner = self._owner
        profile = owner._robot_profile_for_id(robot_id)
        active = (
            owner._active_robot_profile is not None
            and owner._active_robot_profile.robot_id == profile.robot_id
        )
        return {
            "profile": profile.to_summary(active=active),
            "payload": profile.payload,
            "validation": owner._robot_validation_for_profile(profile),
        }

    def activate(self, robot_id: str) -> dict[str, Any]:
        owner = self._owner
        if owner._active_vsa_run is not None and owner._active_vsa_run.thread.is_alive():
            raise RuntimeError("cannot switch robot profile while rollout VSA is active")
        if owner._rollout_group is not None:
            raise RuntimeError("cannot switch robot profile while a collection/deployment session is active")
        if owner._training_thread is not None and owner._training_thread.is_alive():
            raise RuntimeError("cannot switch robot profile while training is active")

        profile = owner._robot_profile_for_id(robot_id)
        had_data_source = owner.orchestrator is not None
        should_start_data_source = owner._started and owner._services.data_source
        old_profile = owner._active_robot_profile
        old_robot_profile_path = owner.config.robot_profile_path
        old_adapter = copy.deepcopy(owner.config.adapter)
        old_vsa = copy.deepcopy(owner.config.vsa)
        old_recorder = copy.deepcopy(owner.config.recorder)
        if had_data_source:
            owner.orchestrator.stop()
            owner.orchestrator = None

        try:
            owner._active_robot_profile = profile
            owner.config.robot_profile_path = str(profile.path)
            owner._apply_robot_profile(profile)

            if should_start_data_source:
                owner._start_data_source()
        except Exception:
            if owner.orchestrator is not None:
                try:
                    owner.orchestrator.stop()
                except Exception:
                    _LOG.exception("[robolineage_app] failed to stop partially-started data_source during profile rollback")
                owner.orchestrator = None
            owner._active_robot_profile = old_profile
            owner.config.robot_profile_path = old_robot_profile_path
            owner.config.adapter = old_adapter
            owner.config.vsa = old_vsa
            owner.config.recorder = old_recorder
            if had_data_source and owner._services.data_source:
                try:
                    owner._start_data_source()
                except Exception:
                    _LOG.exception("[robolineage_app] failed to restart previous data_source after profile activation failure")
            raise

        return {
            "status": "activated",
            "profile": profile.to_summary(active=True),
            "validation": owner._robot_validation_for_profile(profile),
            "data_source_restarted": had_data_source,
            "data_source_active": owner.orchestrator is not None,
        }

    def validate(self, robot_id: str | None = None) -> dict[str, Any]:
        owner = self._owner
        if robot_id:
            profile = owner._robot_profile_for_id(robot_id)
        elif owner._active_robot_profile is not None:
            profile = owner._active_robot_profile
        else:
            return {"status": "no_active_profile", "streams": []}
        return owner._robot_validation_for_profile(profile)

    def onboarding_start(self, body: dict[str, Any]) -> dict[str, Any]:
        from robolineage_robot import RobotOnboardingAgent

        owner = self._owner
        profile_yaml = str(body.get("profile_yaml") or "")
        if not profile_yaml.strip():
            raise ValueError("profile_yaml is required")
        generated_root = (
            owner._robot_registry.roots[0]
            if owner._robot_registry.roots
            else Path.cwd() / "robot_profiles"
        )
        agent = RobotOnboardingAgent()
        result = agent.run(
            profile_yaml=profile_yaml,
            task_root=owner._task_root(),
            generated_profiles_root=generated_root,
            robot_note=str(body.get("robot_note") or "").strip() or None,
            job_id=str(body.get("job_id") or "").strip() or None,
        )
        generated_profile = load_robot_profile(result.generated_profile_path)
        validation = owner._robot_validation_for_profile(generated_profile)
        result = agent.attach_validation(result, validation)
        payload = result.to_payload()
        payload["validation"] = validation
        payload["master_review"] = _master_review_ref(
            owner._enqueue_master_review("robot_onboarding_completed")
        )
        return payload


class _MasterRuntime(_RuntimeDelegate):
    domain = "master"

    def status(self) -> dict[str, Any]:
        owner = self._owner
        task_root = owner._task_root()
        master_dir = task_root / "master"
        state = _read_json(master_dir / "master_state.json")
        review = _read_json(master_dir / "master_review.json")
        understanding = _read_json(master_dir / "master_understanding.json")
        memory = _read_jsonl(master_dir / "master_memory.jsonl")
        events = _read_jsonl(master_dir / "master_events.jsonl")
        report = _read_text(master_dir / "master_report.md")
        understanding_report = _read_text(master_dir / "master_understanding_report.md")
        last_started = _latest_event(events, "master_started")
        last_started_payload = (
            last_started.get("payload")
            if isinstance(last_started.get("payload"), dict)
            else {}
        ) if isinstance(last_started, dict) else {}
        return {
            "available": bool(state or review),
            "task_root": str(task_root),
            "master_dir": str(master_dir),
            "last_review_trigger": last_started_payload.get("trigger"),
            "last_review_at": last_started.get("created_at") if isinstance(last_started, dict) else None,
            "last_error": owner._master_last_error,
            "state": state,
            "review": review,
            "understanding": understanding,
            "memory": memory[-20:],
            "events": events[-50:],
            "report": report or None,
            "understanding_report": understanding_report or None,
            "master_queue": owner._master_queue_status(),
            "health_summary": owner._master_health_summary(),
            "ai_routes": self.ai_routes_status(),
        }

    def ai_routes_status(self) -> dict[str, Any]:
        from robolineage_shared_agents.llm_routes import all_ai_route_statuses

        return all_ai_route_statuses()

    def review(self) -> dict[str, Any]:
        owner = self._owner
        result = owner._run_master_review("manual", raise_errors=True)
        payload = self.status()
        payload["paths"] = {
            "state": str(result.state_path),
            "memory": str(result.memory_path),
            "events": str(result.events_path),
            "review": str(result.review_path),
            "report": str(result.report_path),
            "understanding": str(result.understanding_path),
            "understanding_report": str(result.understanding_report_path),
        }
        return payload


class _RolloutRuntime(_RuntimeDelegate):
    domain = "rollout"

    def session_state(self) -> dict[str, Any]:
        owner = self._owner
        group = owner._rollout_group
        post_status = (
            owner.post_review_status()
            if owner._post_review_worker is not None
            else {"active": False, "queue_size": 0}
        )
        eval_status = (
            owner._eval_review_worker.status()
            if owner._eval_review_worker is not None
            else {"active": False, "queue_size": 0}
        )
        if group is None:
            last = owner._last_rollout_group_result or {}
            idle_status = str(last.get("status") or "")
            status = (
                "finalization_failed"
                if idle_status == "finalization_failed"
                else "completed" if last.get("summary") else "idle"
            )
            return {
                "active": False,
                "status": status,
                "finalizing": False,
                "accepting_rollouts": False,
                "kind": None,
                "session_id": None,
                "policy_version": None,
                "rollout_count": 0,
                "rollout_ids": [],
                "post_review": post_status,
                "eval_review": eval_status,
                "stopped_session": last.get("stopped_session"),
                "summary": last.get("summary"),
                "finalization_error": last.get("finalization_error"),
            }
        status = str(group.get("status") or "active")
        finalizing = status == "finalizing"
        return {
            "active": True,
            "status": status,
            "finalizing": finalizing,
            "accepting_rollouts": not finalizing,
            "kind": group.get("kind"),
            "session_id": group.get("session_id"),
            "policy_version": group.get("policy_version"),
            "started_at": group.get("started_at"),
            "stop_requested_at": group.get("stop_requested_at"),
            "finalization_stage": group.get("finalization_stage"),
            "finalization_error": group.get("finalization_error"),
            "rollout_count": len(group.get("rollout_ids") or []),
            "rollout_ids": list(group.get("rollout_ids") or []),
            "post_review": post_status,
            "eval_review": eval_status,
            "summary": group.get("summary"),
        }

    def start_collection_session(self) -> dict[str, Any]:
        return self._owner._start_rollout_group(kind="collection")

    def stop_collection_session(self) -> dict[str, Any]:
        return self._owner._stop_rollout_group(kind="collection")

    def start_deployment_session(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        owner = self._owner
        payload = payload or {}
        owner._start_eval_review()
        return owner._start_rollout_group(
            kind="deployment",
            policy_version=str(payload["policy_version"]) if payload.get("policy_version") else None,
        )

    def stop_deployment_session(self) -> dict[str, Any]:
        return self._owner._stop_rollout_group(kind="deployment")


class _TrainingRuntime(_RuntimeDelegate):
    domain = "training"

    def status(self) -> dict[str, Any]:
        owner = self._owner
        return {
            "active": owner._training_thread is not None and owner._training_thread.is_alive(),
            "current_run": owner._training_current_run,
            "last_error": owner._training_last_error,
            "root": str(owner._training_runs_root()),
        }


class UnifiedRuntime:
    """Orchestrates RoboLineage sub-runners around direct ROS2 topic IO."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._services: ServicesToggle = config.services or ServicesToggle()
        self._robot_registry = RobotProfileRegistry()
        self._robot_last_error: str | None = None
        self._active_robot_profile: RobotProfile | None = self._load_initial_robot_profile()
        if self._active_robot_profile is not None:
            self._apply_robot_profile(self._active_robot_profile)
        self.orchestrator: Optional[Orchestrator] = None
        self.session_app: Any = None
        self._vsa_thread: Optional[threading.Thread] = None
        self._vsa_pipeline: Any = None
        self._vsa_stop_flag = threading.Event()
        self._vsa_rollout_id: str | None = None
        self._vsa_rollout_dir: Path | None = None
        self._vsa_output_jsonl: Path | None = None
        self._vsa_started_at: str | None = None
        self._vsa_last_error: str | None = None
        self._active_vsa_run: _OnlineRolloutRun | None = None
        self._draining_vsa_runs: dict[str, _OnlineRolloutRun] = {}
        self._task_config_meta: dict[str, Any] | None = None
        from robolineage_shared_agents.visual_snapshot.vlm_priority import VLMOnlinePriorityCoordinator

        self._vlm_coordinator = VLMOnlinePriorityCoordinator()
        self._post_review_worker: Any = None
        self._eval_review_worker: Any = None
        self._rollout_group: dict[str, Any] | None = None
        self._rollout_group_finalizer_thread: threading.Thread | None = None
        self._last_rollout_group_result: dict[str, Any] | None = None
        self._training_thread: threading.Thread | None = None
        self._training_current_run: str | None = None
        self._training_last_error: str | None = None
        self._framework_discovery_jobs: dict[str, dict[str, Any]] = {}
        self._framework_discovery_lock = threading.Lock()
        self._master_lock = threading.Lock()
        self._master_last_error: str | None = None
        self._master_review_queue: deque[_MasterReviewJob] = deque()
        self._master_queue_condition = threading.Condition()
        self._master_worker_thread: threading.Thread | None = None
        self._master_stop_flag = threading.Event()
        self._master_queue_running: _MasterReviewJob | None = None
        self._master_queue_last_enqueued: dict[str, Any] | None = None
        self._master_queue_last_completed: dict[str, Any] | None = None
        self._master_queue_last_debounced: dict[str, Any] | None = None
        try:
            self._master_debounce_window_sec = max(
                0.0,
                float(os.environ.get("ROBOLINEAGE_MASTER_REVIEW_DEBOUNCE_SEC", "2.0")),
            )
        except ValueError:
            self._master_debounce_window_sec = 2.0
        self._started = False
        self._vsa_snapshots: deque = deque(maxlen=50)
        self._vsa_lock = threading.Lock()
        self._memory_debug_lock = threading.Lock()
        self.robot_runtime = _RobotRuntime(self)
        self.master_runtime = _MasterRuntime(self)
        self.rollout_runtime = _RolloutRuntime(self)
        self.training_runtime = _TrainingRuntime(self)

    def latest_snapshots(self, n: int = 20) -> list:
        with self._vsa_lock:
            return list(self._vsa_snapshots)[-n:]

    def robot_profiles(self) -> dict[str, Any]:
        return self.robot_runtime.profiles()

    def robot_profile_detail(self, robot_id: str) -> dict[str, Any]:
        return self.robot_runtime.profile_detail(robot_id)

    def robot_profile_activate(self, robot_id: str) -> dict[str, Any]:
        return self.robot_runtime.activate(robot_id)

    def robot_profile_validate(self, robot_id: str | None = None) -> dict[str, Any]:
        return self.robot_runtime.validate(robot_id)

    def robot_onboarding_start(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.robot_runtime.onboarding_start(body)

    def master_status(self) -> dict[str, Any]:
        return self.master_runtime.status()

    def ai_routes_status(self) -> dict[str, Any]:
        return self.master_runtime.ai_routes_status()

    def master_review(self) -> dict[str, Any]:
        return self.master_runtime.review()

    def _run_master_review(
        self,
        trigger: str,
        *,
        task_root: Path | None = None,
        raise_errors: bool = False,
    ) -> Any:
        from robolineage_shared_agents.master import MasterAgent

        root = Path(task_root or self._task_root())
        try:
            with self._master_lock:
                result = MasterAgent().review(
                    task_root=root,
                    health_summary=self._master_health_summary(),
                    trigger=trigger,
                )
            self._master_last_error = None
            return result
        except Exception as exc:
            self._master_last_error = repr(exc)
            _LOG.exception("[robolineage_app] master review failed (trigger=%s task_root=%s)", trigger, root)
            if raise_errors:
                raise
            return {
                "status": "failed",
                "trigger": trigger,
                "task_root": str(root),
                "error": repr(exc),
            }

    def _ensure_master_worker(self) -> None:
        thread = self._master_worker_thread
        if thread is not None and thread.is_alive():
            return
        with self._master_queue_condition:
            thread = self._master_worker_thread
            if thread is not None and thread.is_alive():
                return
            self._master_stop_flag.clear()
            thread = threading.Thread(
                target=self._master_review_worker_loop,
                name="RoboLineageMasterReviewWorker",
                daemon=True,
            )
            self._master_worker_thread = thread
            thread.start()

    def _enqueue_master_review(
        self,
        trigger: str,
        *,
        task_root: Path | None = None,
    ) -> dict[str, Any]:
        root = Path(task_root or self._task_root())
        self._ensure_master_worker()
        now_mono = time.monotonic()
        now_iso = _now_iso()
        with self._master_queue_condition:
            running = self._master_queue_running
            if (
                running is not None
                and running.trigger == trigger
                and running.task_root.resolve() == root.resolve()
                and now_mono - running.enqueued_mono <= self._master_debounce_window_sec
            ):
                payload = {
                    "status": "debounced",
                    "trigger": trigger,
                    "task_root": str(root),
                    "debounced_against": "running",
                    "debounced_at": now_iso,
                    "pending_count": len(self._master_review_queue),
                    "running": True,
                }
                self._master_queue_last_debounced = payload
                return payload
            for pending in self._master_review_queue:
                if pending.trigger == trigger and pending.task_root.resolve() == root.resolve():
                    payload = {
                        "status": "debounced",
                        "trigger": trigger,
                        "task_root": str(root),
                        "debounced_against": "pending",
                        "debounced_at": now_iso,
                        "pending_count": len(self._master_review_queue),
                        "running": running is not None,
                    }
                    self._master_queue_last_debounced = payload
                    return payload
            job = _MasterReviewJob(
                trigger=trigger,
                task_root=root,
                enqueued_at=now_iso,
                enqueued_mono=now_mono,
            )
            self._master_review_queue.append(job)
            payload = {
                "status": "queued",
                "trigger": trigger,
                "task_root": str(root),
                "enqueued_at": now_iso,
                "pending_count": len(self._master_review_queue),
                "running": running is not None,
            }
            self._master_queue_last_enqueued = payload
            self._master_queue_condition.notify()
            return payload

    def _master_review_worker_loop(self) -> None:
        while True:
            with self._master_queue_condition:
                while not self._master_review_queue and not self._master_stop_flag.is_set():
                    self._master_queue_condition.wait(timeout=1.0)
                if self._master_stop_flag.is_set() and not self._master_review_queue:
                    return
                job = self._master_review_queue.popleft()
                self._master_queue_running = job
            result = self._run_master_review(job.trigger, task_root=job.task_root)
            status = "completed"
            if isinstance(result, dict) and result.get("status"):
                status = str(result["status"])
            with self._master_queue_condition:
                self._master_queue_last_completed = {
                    "status": status,
                    "trigger": job.trigger,
                    "task_root": str(job.task_root),
                    "enqueued_at": job.enqueued_at,
                    "completed_at": _now_iso(),
                }
                self._master_queue_running = None
                self._master_queue_condition.notify_all()

    def _master_queue_status(self) -> dict[str, Any]:
        with self._master_queue_condition:
            thread = self._master_worker_thread
            running = self._master_queue_running
            return {
                "pending_count": len(self._master_review_queue),
                "pending_triggers": [job.trigger for job in self._master_review_queue],
                "running": running is not None,
                "current_trigger": running.trigger if running is not None else None,
                "current_task_root": str(running.task_root) if running is not None else None,
                "last_enqueued": copy.deepcopy(self._master_queue_last_enqueued),
                "last_completed": copy.deepcopy(self._master_queue_last_completed),
                "last_debounced": copy.deepcopy(self._master_queue_last_debounced),
                "debounce_window_sec": self._master_debounce_window_sec,
                "worker_alive": bool(thread is not None and thread.is_alive()),
            }

    def _stop_master_review_worker(self, timeout: float = 5.0) -> None:
        thread = self._master_worker_thread
        if thread is None:
            return
        with self._master_queue_condition:
            self._master_stop_flag.set()
            self._master_queue_condition.notify_all()
        thread.join(timeout=timeout)
        if not thread.is_alive():
            self._master_worker_thread = None

    def _master_health_summary(self) -> dict[str, Any]:
        robot = self.robot_profiles()
        vsa = self.vsa_status()
        status = "ok"
        issues: list[str] = []
        if robot.get("last_error"):
            status = "degraded"
            issues.append("robot_profile_error")
        if vsa.get("last_error"):
            status = "degraded"
            issues.append("vsa_error")
        if self._training_last_error:
            status = "degraded"
            issues.append("training_error")
        return {
            "status": status,
            "issues": issues,
            "services": {
                "data_source": bool(self._services.data_source),
                "session": bool(self._services.session),
                "vsa": bool(self._services.vsa),
                "post_review": bool(self._services.post_review),
                "health_check": bool(self._services.health_check),
            },
            "robot": {
                "active_robot_id": robot.get("active_robot_id"),
                "active_profile_path": robot.get("active_profile_path"),
                "last_error": robot.get("last_error"),
            },
            "vsa": {
                "active": vsa.get("active"),
                "analysis_draining": vsa.get("analysis_draining"),
                "draining_rollout_count": vsa.get("draining_rollout_count"),
                "last_error": vsa.get("last_error"),
            },
            "training": {
                "current_run": self._training_current_run,
                "last_error": self._training_last_error,
            },
            "ai_routes": self.ai_routes_status(),
            "data_flow": {
                "raw_capture": "rosbag2",
                "online_vsa": "ros2_topics",
            },
        }

    def _robot_profile_for_id(self, robot_id: str) -> RobotProfile:
        if (
            self._active_robot_profile is not None
            and self._active_robot_profile.robot_id == robot_id
        ):
            return self._active_robot_profile
        return self._robot_registry.get(robot_id)

    def vsa_status(self) -> dict[str, Any]:
        active_run = self._active_vsa_run
        pipeline = active_run.pipeline if active_run is not None else self._vsa_pipeline
        usage = self._vlm_coordinator.snapshot()
        draining = [
            {
                "rollout_id": run.rollout_id,
                "rollout_dir": str(run.rollout_dir),
                "rollout_index": run.rollout_index,
                "capture_stopped_at": run.capture_stopped_at,
                "analysis_completed_at": run.analysis_completed_at,
                "last_error": run.last_error,
            }
            for run in self._draining_vsa_runs.values()
        ]
        return {
            "active": active_run is not None and active_run.thread.is_alive(),
            "capture_active": active_run is not None and active_run.thread.is_alive(),
            "analysis_draining": bool(draining),
            "draining_rollout_count": len(draining),
            "max_draining_rollouts": _MAX_DRAINING_VSA_ROLLOUTS,
            "draining_rollouts": draining,
            "rollout_id": self._vsa_rollout_id,
            "rollout_dir": str(self._vsa_rollout_dir) if self._vsa_rollout_dir else None,
            "output_jsonl": str(self._vsa_output_jsonl) if self._vsa_output_jsonl else None,
            "task_config_path": (
                self.config.vsa.task_config_path
                if self.config.vsa is not None
                else None
            ),
            "task_config_version": (
                self._task_config_meta.get("version_id")
                if isinstance(self._task_config_meta, dict)
                else None
            ),
            "task_config_created_at": (
                self._task_config_meta.get("created_at")
                if isinstance(self._task_config_meta, dict)
                else None
            ),
            "started_at": self._vsa_started_at,
            "last_error": self._vsa_last_error,
            "vlm_usage": {
                "online_active": usage.online_active,
                "online_rollout_id": usage.online_rollout_id,
                "online_rollout_queue": list(usage.online_rollout_queue),
                "offline_waiting": usage.offline_waiting,
                "offline_inflight": usage.offline_inflight,
            },
            "dropped_arm_before_cam": (
                getattr(pipeline, "dropped_arm_before_cam", 0)
                if pipeline is not None
                else 0
            ),
            "vsa_metrics": (
                pipeline.metrics()
                if pipeline is not None and hasattr(pipeline, "metrics")
                else {}
            ),
            "rollout_session": self.rollout_session_state(),
            "memory_debug_log": str(self._memory_debug_log_path()),
            "robot_profile": (
                self._active_robot_profile.to_summary(active=True)
                if self._active_robot_profile is not None
                else None
            ),
        }

    def rollout_session_state(self) -> dict[str, Any]:
        return self.rollout_runtime.session_state()

    def start_collection_session(self) -> dict[str, Any]:
        return self.rollout_runtime.start_collection_session()

    def stop_collection_session(self) -> dict[str, Any]:
        return self.rollout_runtime.stop_collection_session()

    def start_deployment_session(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.rollout_runtime.start_deployment_session(payload)

    def stop_deployment_session(self) -> dict[str, Any]:
        return self.rollout_runtime.stop_deployment_session()

    def post_review_status(self) -> dict[str, Any]:
        usage = self._vlm_coordinator.snapshot()
        vlm_usage = {
            "online_active": usage.online_active,
            "online_rollout_id": usage.online_rollout_id,
            "offline_waiting": usage.offline_waiting,
            "offline_inflight": usage.offline_inflight,
        }
        if self._post_review_worker is None:
            return {"active": False, "queue_size": 0, "vlm_usage": vlm_usage}
        status = self._post_review_worker.status()
        status["vlm_usage"] = vlm_usage
        return status

    def post_review_rollouts(self, limit: int = 50) -> dict[str, Any]:
        root = self._rollouts_root()
        worker_status = self.post_review_status()
        if not root.exists():
            return {"root": str(root), "status": worker_status, "rollouts": []}

        rollout_dirs = [
            path for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        rollout_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        items = [
            self._post_review_rollout_item(path, worker_status)
            for path in rollout_dirs[:max(1, limit)]
        ]
        return {"root": str(root), "status": worker_status, "rollouts": items}

    def post_review_detail(self, rollout_id: str) -> dict[str, Any]:
        root = self._rollouts_root().resolve()
        rollout_dir = (root / rollout_id).resolve()
        try:
            rollout_dir.relative_to(root)
        except ValueError:
            raise FileNotFoundError(f"rollout not found: {rollout_id}")
        if not rollout_dir.exists():
            raise FileNotFoundError(f"rollout not found: {rollout_id}")
        worker_status = self.post_review_status()
        return {
            "rollout": self._post_review_rollout_item(rollout_dir, worker_status),
            "evidence_index": _read_json(rollout_dir / "evidence_index.json"),
            "annotation": _read_json(rollout_dir / "annotation.final.json"),
            "rollout_summary": _read_json(rollout_dir / "rollout_summary.json"),
            "failure_analysis": _read_json(rollout_dir / "failure_analysis.json"),
            "dataset_admission": _read_json(rollout_dir / "dataset_admission.json"),
            "post_review_status": _read_json(rollout_dir / "post_review_status.json"),
            "phase_timeline": _read_jsonl(rollout_dir / "phase_timeline.final.jsonl"),
            "review_report": _read_text(rollout_dir / "review_report.md"),
        }

    def training_framework_status(self) -> dict[str, Any]:
        return self.training_runtime.status()

    def training_framework_runs(self, limit: int = 30) -> dict[str, Any]:
        root = self._training_runs_root()
        status = self.training_framework_status()
        if not root.exists():
            return {"status": status, "runs": []}
        runs = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
        runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return {
            "status": status,
            "runs": [self._training_run_item(path) for path in runs[:max(1, limit)]],
        }

    def training_framework_detail(self, run_id: str) -> dict[str, Any]:
        root = self._training_runs_root().resolve()
        run_dir = (root / run_id).resolve()
        try:
            run_dir.relative_to(root)
        except ValueError:
            raise FileNotFoundError(f"training run not found: {run_id}")
        if not run_dir.exists():
            raise FileNotFoundError(f"training run not found: {run_id}")
        return {
            "run": self._training_run_item(run_dir),
            "training_result": _read_json(run_dir / "framework" / "training_result.json"),
            "training_status": _read_json(run_dir / "framework" / "training_status.json"),
            "deployment_recommendation": _read_json(run_dir / "deployment_recommendation.json"),
            "policy_context": _read_json(run_dir / "policy.ROBOLINEAGE_context.json"),
            "train_manifest": _read_jsonl(run_dir / "train_manifest.jsonl"),
            "dataset_adapt_status": _read_json(run_dir / "framework" / "dataset_adapt_status.json"),
            "dataset_adapt_result": _read_json(run_dir / "framework" / "dataset_adapt_result.json"),
            "dataset_health_report": _read_json(run_dir / "dataset_health_report.json"),
            "dataset_health_understanding": _read_json(run_dir / "dataset_health_understanding.json"),
            "dataset_health_report_md": _read_text(run_dir / "dataset_health_report.md"),
            "training_monitor_report": _read_json(run_dir / "framework" / "training_monitor_report.json"),
            "training_monitor_understanding": _read_json(run_dir / "framework" / "training_monitor_understanding.json"),
            "training_monitor_report_md": _read_text(run_dir / "framework" / "training_monitor_report.md"),
            "dataset_log": _read_text(run_dir / "framework" / "dataset_command.log"),
            "train_log": _read_text(run_dir / "framework" / "train_command.log"),
            "eval_log": _read_text(run_dir / "framework" / "eval_command.log"),
        }

    def training_framework_run_demo(self) -> dict[str, Any]:
        if self._training_thread is not None and self._training_thread.is_alive():
            return {"status": "already_running", "run_id": self._training_current_run}
        run_id = f"demo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self._training_current_run = run_id
        self._training_last_error = None
        thread = threading.Thread(
            target=self._run_training_demo_worker,
            args=(run_id,),
            name="RoboLineageTrainingFrameworkDemo",
            daemon=True,
        )
        self._training_thread = thread
        thread.start()
        return {"status": "started", "run_id": run_id}

    def training_framework_discover_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = f"discover_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        job = {
            "schema_version": "RoboLineage.training_framework_discovery_job.v1",
            "job_id": job_id,
            "status": "running",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "events": [],
        }
        with self._framework_discovery_lock:
            self._framework_discovery_jobs[job_id] = job
        self._framework_discovery_job_event(job_id, "submitted")
        thread = threading.Thread(
            target=self._run_training_framework_discovery_job,
            args=(job_id, dict(payload)),
            name=f"RoboLineageTrainingFrameworkDiscovery.{job_id}",
            daemon=True,
        )
        thread.start()
        return self.training_framework_discovery_job(job_id)

    def training_framework_discovery_job(self, job_id: str) -> dict[str, Any]:
        with self._framework_discovery_lock:
            job = copy.deepcopy(self._framework_discovery_jobs.get(job_id))
        if job is None:
            raise FileNotFoundError(f"training framework discovery job not found: {job_id}")
        file_events = _read_jsonl_events(Path(str(job.get("events_path")))) if job.get("events_path") else []
        if file_events:
            job["events"] = _dedupe_event_records([*(job.get("events") or []), *file_events])
        return job

    def _run_training_framework_discovery_job(self, job_id: str, payload: dict[str, Any]) -> None:
        try:
            result = self.training_framework_discover(
                payload,
                progress=lambda event, **data: self._framework_discovery_job_event(job_id, event, **data),
            )
            with self._framework_discovery_lock:
                job = self._framework_discovery_jobs[job_id]
                job["status"] = "completed"
                job["result"] = result
                job["updated_at"] = _now_iso()
            self._framework_discovery_job_event(job_id, "completed")
        except Exception as exc:
            _LOG.exception("[robolineage_app] training framework discovery job failed (job_id=%s)", job_id)
            self._framework_discovery_job_event(job_id, "failed", error=str(exc), error_type=type(exc).__name__)
            with self._framework_discovery_lock:
                job = self._framework_discovery_jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
                job["error_type"] = type(exc).__name__
                job["updated_at"] = _now_iso()

    def _framework_discovery_job_event(self, job_id: str, event: str, **payload: Any) -> None:
        record = {"event": event, "created_at": _now_iso(), **payload}
        with self._framework_discovery_lock:
            job = self._framework_discovery_jobs.get(job_id)
            if job is None:
                return
            job.setdefault("events", []).append(record)
            job["updated_at"] = record["created_at"]
            for key in ("output_dir", "events_path", "profile_path", "discovery_path", "integration_manifest_path"):
                if payload.get(key):
                    job[key] = payload[key]

    def training_framework_discover(
        self,
        payload: dict[str, Any],
        *,
        progress: Callable[..., None] | None = None,
    ) -> dict[str, Any]:
        from robolineage_train import CommandIntake, FrameworkDiscoveryAgent

        target_dataset_format = str(
            payload.get("target_dataset_format")
            or payload.get("dataset_format")
            or ""
        ).strip()
        command_context = str(
            payload.get("command_context")
            or payload.get("commands_context")
            or ""
        ).strip()
        parsed_context = _parse_framework_command_context(command_context)
        repo_location = str(payload.get("repo_location") or parsed_context.get("repo_location") or "local").strip().lower()
        if repo_location not in {"local", "remote"}:
            raise ValueError("repo_location must be local or remote")
        repo_root_value = str(payload.get("repo_root") or parsed_context.get("repo_root") or "").strip()
        remote_ssh_host = str(payload.get("remote_ssh_host") or parsed_context.get("remote_ssh_host") or "").strip()
        remote_repo_root = str(payload.get("remote_repo_root") or parsed_context.get("remote_repo_root") or repo_root_value or "").strip()
        if repo_location == "remote" and remote_repo_root:
            remote_repo_root = _normalize_remote_repo_root(remote_repo_root)
        display_repo_root = remote_repo_root if repo_location == "remote" else repo_root_value
        if not display_repo_root:
            raise ValueError("repo root is required")
        name = str(payload.get("name") or parsed_context.get("name") or Path(display_repo_root).name or "training_framework")
        safe_name = name.replace("/", "_")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = self._task_root() / "framework_profiles" / f"{safe_name}_{stamp}"
        if progress is not None:
            progress(
                "repo_context_resolved",
                repo_location=repo_location,
                repo_root=display_repo_root,
                output_dir=str(output_dir),
                events_path=str(output_dir / "framework_discovery_events.jsonl"),
            )
        if repo_location == "remote":
            if not remote_ssh_host:
                raise ValueError("remote_ssh_host is required for remote discovery")
            if progress is not None:
                progress(
                    "remote_mirror_started",
                    remote_ssh_host=remote_ssh_host,
                    remote_repo_root=remote_repo_root,
                    local_snapshot=str(output_dir / "repo_snapshot"),
                )
            repo_root = _mirror_remote_training_repo(
                ssh_host=remote_ssh_host,
                remote_repo_root=remote_repo_root,
                destination=output_dir / "repo_snapshot",
            )
            if progress is not None:
                progress("remote_mirror_completed", local_snapshot=str(repo_root))
            command_context = "\n".join(
                [
                    command_context,
                    f"repo location: remote",
                    f"remote ssh host: {remote_ssh_host}",
                    f"remote repo root: {remote_repo_root}",
                    f"local repo snapshot: {repo_root}",
                ]
            ).strip()
        else:
            if not repo_root_value:
                raise ValueError("repo_root is required for local discovery")
            repo_root = Path(repo_root_value).expanduser()
        env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        conda_env = str(payload.get("conda_env") or parsed_context.get("conda_env") or "").strip()
        dataset_command = payload.get("dataset_command") or parsed_context.get("dataset_command")
        train_command = payload.get("train_command") or parsed_context.get("train_command")
        eval_command = payload.get("eval_command") or parsed_context.get("eval_command")
        if repo_location == "remote":
            dataset_command = _normalize_remote_repo_command(dataset_command, remote_repo_root, repo_root) if dataset_command else None
            train_command = _normalize_remote_repo_command(train_command, remote_repo_root, repo_root) if train_command else None
            eval_command = _normalize_remote_repo_command(eval_command, remote_repo_root, repo_root) if eval_command else None
        dataset_command = _normalize_script_command(str(dataset_command)) if dataset_command else None
        train_command = _normalize_script_command(str(train_command)) if train_command else None
        eval_command = _normalize_script_command(str(eval_command)) if eval_command else None
        if conda_env:
            dataset_command = _maybe_prefix_conda_run(dataset_command, conda_env) if dataset_command else None
            train_command = _maybe_prefix_conda_run(train_command, conda_env) if train_command else None
            eval_command = _maybe_prefix_conda_run(eval_command, conda_env) if eval_command else None
        train_launch_mode = str(
            payload.get("train_launch_mode")
            or parsed_context.get("train_launch_mode")
            or "tmux"
        )
        terminal_hold_open = _context_bool(
            payload.get("terminal_hold_open", parsed_context.get("terminal_hold_open")),
            default=True,
        )
        if progress is not None:
            progress("discovery_agent_started", repo_root=str(repo_root))
        result = FrameworkDiscoveryAgent().discover(
            repo_root=repo_root,
            output_dir=output_dir,
            name=name,
            framework_type=str(
                payload.get("framework_type")
                or parsed_context.get("framework_type")
                or ""
            ) or None,
            commands=CommandIntake(
                dataset_command=dataset_command,
                train_command=train_command,
                eval_command=eval_command,
                env={str(k): str(v) for k, v in env.items()},
            ),
            target_dataset_format=target_dataset_format,
            command_context=command_context,
            fixed_input_dir=str(payload.get("fixed_input_dir") or parsed_context.get("fixed_input_dir") or "") or None,
            checkpoint_glob=str(payload.get("checkpoint_glob") or parsed_context.get("checkpoint_glob") or "") or None,
            train_log=str(payload.get("train_log") or parsed_context.get("train_log") or "") or None,
            eval_result=str(payload.get("eval_result") or parsed_context.get("eval_result") or "") or None,
            train_launch_mode=train_launch_mode,
            terminal_command=payload.get("terminal_command") or parsed_context.get("terminal_command"),
            terminal_hold_open=terminal_hold_open,
            enable_llm_understanding=True,
        )
        result_payload = result.to_dict()
        discovery_payload = _read_json(result.discovery_path) or {}
        for key in (
            "events",
            "events_path",
            "target_contract",
            "adapter_registry",
            "dataset_adapter",
            "monitor",
            "outputs",
            "deep_inspection",
            "llm_understanding",
            "integration_manifest",
            "integration_manifest_path",
        ):
            if key in discovery_payload:
                result_payload[key] = discovery_payload[key]
        if progress is not None:
            progress(
                "discovery_agent_completed",
                profile_path=str(result.profile_path),
                discovery_path=str(result.discovery_path),
                integration_manifest_path=str(result_payload.get("integration_manifest_path") or ""),
            )
        result_payload["master_review"] = _master_review_ref(
            self._enqueue_master_review("framework_discovery_completed")
        )
        return {"status": "generated", **result_payload}

    def task_registry(self) -> dict[str, Any]:
        root = self._tasks_root()
        current_task = self._task_root()
        current_is_task = (current_task / "task_manifest.json").is_file()
        current = current_task.resolve() if current_is_task else None
        tasks = []
        if root.exists():
            for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                if path.is_dir() and not path.name.startswith(".") and (path / "task_manifest.json").is_file():
                    tasks.append(self._task_item(path, active=current is not None and path.resolve() == current))
        return {
            "root": str(root),
            "active_task_id": current_task.name if current_is_task else None,
            "active_task_dir": str(current_task) if current_is_task else None,
            "tasks": tasks,
        }

    def task_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        display_name = str(
            payload.get("display_name")
            or payload.get("task_description")
            or payload.get("task_id")
            or "task"
        ).strip()
        slug = _safe_slug(str(payload.get("task_id") or display_name))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        task_dir = self._tasks_root() / f"{slug}_{stamp}"
        task_dir.mkdir(parents=True, exist_ok=False)
        for child in ("rollouts", "logs", "collection_sessions", "training_selections", "framework_profiles", "training_runs", "deployment_sessions", "datasets"):
            (task_dir / child).mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "RoboLineage.task_manifest.v1",
            "task_id": task_dir.name,
            "display_name": display_name,
            "task_description": str(payload.get("task_description") or ""),
            "task_dir": str(task_dir),
            "robot": (
                self._active_robot_profile.robot_id
                if self._active_robot_profile is not None
                else None
            ),
            "created_at": _now_iso(),
        }
        _write_json_atomic(task_dir / "task_manifest.json", manifest)
        return self.task_activate(task_dir.name)

    def task_activate(self, task_id: str) -> dict[str, Any]:
        if self._active_vsa_run is not None and self._active_vsa_run.thread.is_alive():
            raise RuntimeError("cannot activate a different task while rollout is active")
        if self._rollout_group is not None:
            raise RuntimeError("cannot activate a different task while a session is active")
        if self._training_thread is not None and self._training_thread.is_alive():
            raise RuntimeError("cannot activate a different task while training is active")
        task_dir = self._task_dir_for_id(task_id)
        (task_dir / "rollouts").mkdir(parents=True, exist_ok=True)
        if self.config.recorder is not None:
            self.config.recorder.output_dir = str(task_dir / "rollouts")
        os.environ["ROBOLINEAGE_TASK_DIR"] = str(task_dir)
        latest = task_dir / "task_config.latest.yaml"
        compat = task_dir / "task_config.yaml"
        if self.config.vsa is not None:
            if latest.exists():
                self.config.vsa.task_config_path = str(latest)
                self._task_config_meta = _task_config_metadata_for_path(latest)
            elif compat.exists():
                self.config.vsa.task_config_path = str(compat)
                self._task_config_meta = _task_config_metadata_for_path(compat)
            else:
                self.config.vsa.task_config_path = None
                self._task_config_meta = None
        return {"status": "activated", "task": self._task_item(task_dir, active=True)}

    def task_detail(self, task_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        return {
            "task": self._task_item(task_dir, active=task_dir.resolve() == self._task_root().resolve()),
            "collection_sessions": self.task_collection_sessions(task_id)["sessions"],
            "deployment_sessions": self.task_deployment_sessions(task_id)["sessions"],
            "training_selections": self.training_selections(task_id)["selections"],
            "framework_profiles": self.framework_profiles(task_id)["profiles"],
            "training_runs": self.training_framework_runs_for_task(task_id)["runs"],
            "policies": self.policies(task_id)["policies"],
        }

    def task_collection_sessions(self, task_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        root = task_dir / "collection_sessions"
        sessions = []
        if root.exists():
            for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                if path.is_dir() and not path.name.startswith("."):
                    sessions.append(self._collection_session_item(task_dir, path))
        return {"task_id": task_id, "root": str(root), "sessions": sessions}

    def task_collection_session_detail(self, task_id: str, session_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        session_dir = _resolve_child(task_dir / "collection_sessions", session_id)
        summary = _read_json(session_dir / "collection_session_summary.json") or {}
        rollout_ids = [str(item) for item in summary.get("rollout_ids") or []]
        worker_status = self.post_review_status() if task_dir.resolve() == self._task_root().resolve() else {}
        return {
            "session": self._collection_session_item(task_dir, session_dir),
            "summary": summary,
            "rollouts": [
                self._post_review_rollout_item(task_dir / "rollouts" / rollout_id, worker_status)
                for rollout_id in rollout_ids
                if (task_dir / "rollouts" / rollout_id).exists()
            ],
        }

    def task_deployment_sessions(self, task_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        root = task_dir / "deployment_sessions"
        sessions = []
        if root.exists():
            for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                if path.is_dir() and not path.name.startswith("."):
                    sessions.append(self._deployment_session_item(task_dir, path))
        return {"task_id": task_id, "root": str(root), "sessions": sessions}

    def task_deployment_session_detail(self, task_id: str, session_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        session_dir = _resolve_child(task_dir / "deployment_sessions", session_id)
        summary = _read_json(session_dir / "policy_eval_summary.json") or {}
        rollout_ids = [str(item) for item in summary.get("rollout_ids") or []]
        return {
            "session": self._deployment_session_item(task_dir, session_dir),
            "summary": summary,
            "deployment_decision": _read_json(session_dir / "deployment_decision.json") or {},
            "collection_recommendation": _read_json(session_dir / "collection_recommendation.json") or {},
            "next_collection_brief": _read_json(session_dir / "next_collection_brief.json") or {},
            "deployment_governance_understanding": _read_json(session_dir / "deployment_governance_understanding.json") or {},
            "deployment_governance_understanding_report": _read_text(session_dir / "deployment_governance_understanding.md"),
            "deployment_session_report": _read_text(session_dir / "deployment_session_report.md"),
            "rollouts": [
                self._policy_eval_rollout_item(task_dir / "rollouts" / rollout_id)
                for rollout_id in rollout_ids
                if (task_dir / "rollouts" / rollout_id).exists()
            ],
        }

    def training_selections(self, task_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        root = task_dir / "training_selections"
        selections = []
        if root.exists():
            for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
                selections.append(self._selection_item(path))
        return {"task_id": task_id, "root": str(root), "selections": selections}

    def training_selection_create(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        include_decisions = [
            str(item)
            for item in (payload.get("include_decisions") or ["accepted"])
            if str(item)
        ]
        session_ids = [
            str(item)
            for item in (payload.get("source_collection_session_ids") or payload.get("collection_session_ids") or [])
            if str(item)
        ]
        explicit_rollout_ids = [
            str(item)
            for item in (payload.get("rollout_ids") or [])
            if str(item)
        ]
        rollout_ids: list[str] = []
        for session_id in session_ids:
            summary = _read_json(
                task_dir / "collection_sessions" / session_id / "collection_session_summary.json"
            ) or {}
            rollout_ids.extend(str(item) for item in summary.get("rollout_ids") or [])
        rollout_ids.extend(explicit_rollout_ids)
        if not rollout_ids:
            rollouts_root = task_dir / "rollouts"
            rollout_ids = [
                path.name for path in sorted(rollouts_root.iterdir())
                if path.is_dir() and not path.name.startswith(".")
            ] if rollouts_root.exists() else []
        rollout_ids = _dedupe_str(rollout_ids)
        selected: list[str] = []
        rejected: list[dict[str, Any]] = []
        for rollout_id in rollout_ids:
            rollout_dir = task_dir / "rollouts" / rollout_id
            admission = _read_json(rollout_dir / "dataset_admission.json") or {}
            decision = str(admission.get("decision") or "")
            accepted_for_training = admission.get("accepted_for_training")
            trainable = (
                bool(accepted_for_training)
                if isinstance(accepted_for_training, bool)
                else decision in include_decisions
            )
            raw_status = _raw_artifacts_status(rollout_dir)
            allow_missing_raw = bool(payload.get("allow_missing_raw_artifacts"))
            if trainable and (raw_status["present"] or allow_missing_raw):
                selected.append(rollout_id)
            else:
                if trainable and not raw_status["present"]:
                    rejected.append({
                        "rollout_id": rollout_id,
                        "decision": decision or "missing",
                        "reason": "raw_artifacts_missing",
                        "raw_artifacts": raw_status,
                    })
                else:
                    rejected.append(
                        {
                            "rollout_id": rollout_id,
                            "decision": decision or "missing",
                            "reason": "not_accepted_for_training"
                            if accepted_for_training is False
                            else "decision_excluded",
                        }
                    )
        selection_id = f"selection_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        selection = {
            "schema_version": "RoboLineage.training_selection.v1",
            "selection_id": selection_id,
            "task_id": task_dir.name,
            "task_dir": str(task_dir),
            "source_collection_session_ids": session_ids,
            "include_decisions": include_decisions,
            "rollout_ids": selected,
            "excluded_rollouts": rejected,
            "selected_rollout_count": len(selected),
            "created_by": str(payload.get("created_by") or "operator"),
            "created_at": _now_iso(),
            "note": str(payload.get("note") or ""),
        }
        root = task_dir / "training_selections"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{selection_id}.json"
        _write_json_atomic(path, selection)
        return {"status": "created", "selection": {**selection, "selection_path": str(path)}}

    def framework_profiles(self, task_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        root = task_dir / "framework_profiles"
        profiles = []
        if root.exists():
            for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                if path.is_dir() and (path / "framework_profile.generated.yaml").exists():
                    profiles.append(self._framework_profile_item(path))
        return {"task_id": task_id, "root": str(root), "profiles": profiles}

    def training_framework_runs_for_task(self, task_id: str, limit: int = 50) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        root = task_dir / "training_runs"
        if not root.exists():
            return {"task_id": task_id, "runs": []}
        runs = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
        runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return {"task_id": task_id, "runs": [self._training_run_item(path) for path in runs[:max(1, limit)]]}

    def training_data_adapt_start(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._training_thread is not None and self._training_thread.is_alive():
            return {"status": "already_running", "run_id": self._training_current_run}
        task_dir = self._task_dir_for_id(task_id)
        policy_version = str(payload.get("policy_version") or "").strip()
        if not policy_version:
            raise ValueError("policy_version is required before adapting data")
        profile_path = self._resolve_profile_path(task_dir, payload)
        selection_path = self._resolve_selection_path(task_dir, payload)
        selection = _read_json(selection_path) or {}
        rollout_ids = tuple(str(item) for item in selection.get("rollout_ids") or [])
        include_decisions = tuple(str(item) for item in selection.get("include_decisions") or ["accepted"])
        if not rollout_ids:
            raise ValueError("training selection has no rollout_ids")
        request_id = f"adapt_request_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        run_id = f"train_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        workspace = task_dir / "training_runs" / run_id
        _write_json_atomic(
            workspace / "training_run_config.json",
            {
                "schema_version": "RoboLineage.training_run_config.v1",
                "request_id": request_id,
                "run_id": run_id,
                "task_id": task_dir.name,
                "task_dir": str(task_dir),
                "phase": "dataset_adapt_queued",
                "selection_id": selection.get("selection_id") or selection_path.stem,
                "selection_path": str(selection_path),
                "framework_profile_path": str(profile_path),
                "policy_version": policy_version,
                "architecture": str(payload.get("architecture") or "policy"),
                "deploy_success_threshold": float(payload.get("deploy_success_threshold") or 0.7),
                "include_decisions": list(include_decisions),
                "rollout_ids": list(rollout_ids),
                "created_at": _now_iso(),
            },
        )
        self._training_current_run = run_id
        self._training_last_error = None
        thread = threading.Thread(
            target=self._run_dataset_adapt_worker,
            args=(
                task_dir,
                request_id,
                run_id,
                profile_path,
                selection_path,
                rollout_ids,
                include_decisions,
                policy_version,
                str(payload.get("architecture") or "policy"),
                float(payload.get("deploy_success_threshold") or 0.7),
            ),
            name="RoboLineageTrainingDataAdapt",
            daemon=True,
        )
        self._training_thread = thread
        thread.start()
        return {
            "status": "started",
            "request_id": request_id,
            "run_id": run_id,
            "task_id": task_dir.name,
            "profile_path": str(profile_path),
            "selection_path": str(selection_path),
            "selected_rollout_count": len(rollout_ids),
            "policy_version": policy_version,
        }

    def training_run_start(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._training_thread is not None and self._training_thread.is_alive():
            return {"status": "already_running", "run_id": self._training_current_run}
        task_dir = self._task_dir_for_id(task_id)
        requested_run_id = str(payload.get("run_id") or "").strip() or None
        requested_run_config = self._training_run_config_for_request(task_dir, requested_run_id)
        policy_version = str(payload.get("policy_version") or requested_run_config.get("policy_version") or "").strip()
        if not policy_version:
            raise ValueError("policy_version is required")
        profile_path = (
            self._resolve_profile_path(task_dir, payload)
            if payload.get("framework_profile_id") or payload.get("profile_id") or payload.get("framework_profile_path")
            else self._path_from_training_run_config(requested_run_config, "framework_profile_path", "framework profile")
        )
        selection_path = (
            self._resolve_selection_path(task_dir, payload)
            if payload.get("selection_id") or payload.get("selection_path")
            else self._path_from_training_run_config(requested_run_config, "selection_path", "training selection")
        )
        selection = _read_json(selection_path) or {}
        rollout_ids = tuple(str(item) for item in selection.get("rollout_ids") or [])
        include_decisions = tuple(str(item) for item in selection.get("include_decisions") or ["accepted"])
        if not rollout_ids:
            raise ValueError("training selection has no rollout_ids")
        request_id = f"train_request_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        run_dir = self._find_adapted_training_run(
            task_dir=task_dir,
            selection_path=selection_path,
            profile_path=profile_path,
            policy_version=policy_version,
            requested_run_id=requested_run_id,
        )
        run_id = run_dir.name
        execution_override = _training_execution_override(payload)
        if execution_override:
            existing_config = _read_json(run_dir / "training_run_config.json") or {}
            existing_config.update(
                {
                    "phase": "training_queued",
                    "training_execution": execution_override,
                    "updated_at": _now_iso(),
                }
            )
            _write_json_atomic(run_dir / "training_run_config.json", existing_config)
        self._training_current_run = run_id
        self._training_last_error = None
        thread = threading.Thread(
            target=self._run_training_selection_worker,
            args=(
                task_dir,
                request_id,
                run_id,
                profile_path,
                selection_path,
                rollout_ids,
                include_decisions,
                policy_version,
                str(payload.get("architecture") or "policy"),
                float(payload.get("deploy_success_threshold") or 0.7),
            ),
            name="RoboLineageTrainingFrameworkRun",
            daemon=True,
        )
        self._training_thread = thread
        thread.start()
        return {
            "status": "started",
            "request_id": request_id,
            "run_id": run_id,
            "task_id": task_dir.name,
            "profile_path": str(profile_path),
            "selection_path": str(selection_path),
            "selected_rollout_count": len(rollout_ids),
            "policy_version": policy_version,
        }

    def policies(self, task_id: str) -> dict[str, Any]:
        task_dir = self._task_dir_for_id(task_id)
        root = task_dir / "training_runs"
        policies = []
        if root.exists():
            for meta_path in sorted(root.glob("*/framework/checkpoints/*/policy.meta.json")):
                policies.append(self._policy_item(meta_path))
        policies.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {"task_id": task_id, "policies": policies}

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            raise RuntimeError("UnifiedRuntime already started")

        self._ensure_master_worker()

        if self._services.session:
            self._start_session()

        if self._services.data_source and self._data_source_ready_to_start():
            self._start_data_source()
        elif self._services.data_source:
            _LOG.info("[robolineage_app] data_source deferred until a robot profile is activated")

        if self._services.post_review:
            self._start_post_review()

        if self._services.vsa:
            self._start_vsa()

        self._started = True
        _LOG.info(
            "[robolineage_app] unified runtime started "
            "(data_source=%s session=%s vsa=%s)",
            self._services.data_source,
            self._services.session,
            self._services.vsa,
        )
        self._write_memory_debug("runtime_started")

    def stop_all(self) -> None:
        """Reverse-order shutdown; each step logged + continues on failure."""
        if not self._started:
            self._stop_master_review_worker()
            return

        if self._rollout_group is not None:
            kind = str(self._rollout_group.get("kind") or "collection")
            try:
                if self._rollout_group.get("status") != "finalizing":
                    self._stop_rollout_group(kind=kind)
                self._wait_for_rollout_group_finalizer(timeout=None)
            except Exception:
                _LOG.exception("[robolineage_app] rollout session finalization during shutdown failed")
        else:
            self.stop_vsa()
            self._wait_for_vsa_drain(timeout=60.0)

        if self._post_review_worker is not None:
            try:
                self._post_review_worker.stop()
            except Exception:
                _LOG.exception("[robolineage_app] post-review worker shutdown failed")
            self._post_review_worker = None

        if self._eval_review_worker is not None:
            try:
                self._eval_review_worker.stop()
            except Exception:
                _LOG.exception("[robolineage_app] evaluation review worker shutdown failed")
            self._eval_review_worker = None

        # 2. Session AR overlay consumers (their daemon threads).
        if self.session_app is not None:
            try:
                if hasattr(self.session_app.state, "shutdown_overlay_consumers"):
                    self.session_app.state.shutdown_overlay_consumers()
            except Exception:
                _LOG.exception("[robolineage_app] AR overlay consumer shutdown failed")

        # 3. Orchestrator: adapters + recorders.
        if self.orchestrator is not None:
            try:
                self.orchestrator.stop()
            except Exception:
                _LOG.exception("[robolineage_app] orchestrator stop failed")

        self._stop_master_review_worker()
        self._started = False
        _LOG.info("[robolineage_app] unified runtime stopped cleanly")
        self._write_memory_debug("runtime_stopped")

    def stop_vsa(self) -> dict[str, Any]:
        run = self._active_vsa_run
        if run is None:
            return self.vsa_status()
        run.stop_flag.set()
        self._stop_raw_capture(run)
        run.capture_stopped_at = run.capture_stopped_at or _now_iso()
        self._draining_vsa_runs[run.rollout_id] = run
        self._active_vsa_run = None
        self._vsa_thread = None
        self._vsa_pipeline = None
        self._vsa_stop_flag = threading.Event()
        _LOG.info(
            "[robolineage_app] VSA rollout capture stopped; analysis draining in background (rollout_id=%s)",
            run.rollout_id,
        )
        self._write_memory_debug("rollout_capture_stopped", run=run)
        return self.vsa_status()

    def configure_vsa_task(self, task_config_path: str) -> dict[str, Any]:
        if self.config.vsa is None:
            raise RuntimeError("VSA is not configured")
        if self._active_vsa_run is not None and self._active_vsa_run.thread.is_alive():
            raise RuntimeError("cannot change task config while rollout is active")
        self.config.vsa.task_config_path = task_config_path
        self._task_config_meta = _task_config_metadata_for_path(Path(task_config_path))
        _LOG.info("[robolineage_app] VSA task configured (path=%s)", task_config_path)
        status = self.vsa_status()
        status["master_review"] = _master_review_ref(
            self._enqueue_master_review("task_config_updated")
        )
        return status

    def start_vsa_rollout(self) -> dict[str, Any]:
        return self._start_vsa(require_task_config=True)

    def activate_vsa(self, task_config_path: str) -> None:
        """Backward-compatible one-shot path: configure task then start VSA."""
        if self.config.vsa is None:
            raise RuntimeError("VSA is not configured")
        self.configure_vsa_task(task_config_path)
        self.start_vsa_rollout()

    # ── private starters ────────────────────────────────────────────────

    def _load_initial_robot_profile(self) -> RobotProfile | None:
        explicit_path = os.environ.get("ROBOLINEAGE_ROBOT_PROFILE_PATH") or self.config.robot_profile_path
        if explicit_path:
            try:
                return load_robot_profile(explicit_path)
            except Exception as exc:
                self._robot_last_error = str(exc)
                raise

        for profile in self._robot_registry.list_profiles():
            if profile.robot_id == self.config.rollout.task_id:
                return profile
        return None

    def _apply_robot_profile(self, profile: RobotProfile) -> None:
        self.config.adapter = profile_to_adapter_config(profile)
        camera_topic, arm_topic = profile_to_vsa_topics(profile)
        if self.config.vsa is not None:
            if camera_topic:
                self.config.vsa.camera_topic = camera_topic
            if arm_topic:
                self.config.vsa.arm_topic = arm_topic
        if self.config.recorder is not None:
            bindings = profile.payload.get("ROBOLINEAGE_bindings")
            recorder = (
                bindings.get("recorder")
                if isinstance(bindings, dict) and isinstance(bindings.get("recorder"), dict)
                else {}
            )
            default_output = recorder.get("default_output_dir")
            if default_output and self.config.recorder.output_dir in {"", "data/rollouts"}:
                self.config.recorder.output_dir = str(default_output)
            camera_names = recorder.get("camera_names")
            if isinstance(camera_names, list) and camera_names:
                self.config.recorder.camera_names = tuple(str(item) for item in camera_names if str(item).strip())

    def _robot_validation_for_profile(self, profile: RobotProfile) -> dict[str, Any]:
        streams: list[dict[str, Any]] = []
        color = profile.color_stream()
        state = profile.robot_state_stream()
        camera_status = (
            self._robot_camera_status(profile, color)
            if isinstance(color, dict)
            else None
        )
        arm_sample_status = (
            self._robot_arm_status(profile, state)
            if isinstance(state, dict)
            else None
        )
        arm_vector = (
            self._robot_arm_vector(profile, state)
            if isinstance(state, dict)
            else None
        )
        if isinstance(color, dict):
            streams.append(
                self._robot_stream_status(
                    stream_id=profile.active_color_stream_id or "color_image",
                    role="color_image",
                    ros_topic=str(color.get("topic") or ""),
                    required=bool(color.get("required", True)),
                    sample_status=camera_status,
                )
            )
        state_status: dict[str, Any] | None = None
        if isinstance(state, dict):
            state_status = self._robot_stream_status(
                stream_id=profile.active_robot_state_id or "robot_state",
                role="robot_state",
                ros_topic=str(state.get("topic") or ""),
                required=bool(state.get("required", True)),
                sample_status=arm_sample_status,
            )
            streams.append(
                state_status
            )
        required = [item for item in streams if item["required"]]
        present = [item for item in required if item["present"]]
        status = "ok" if required and len(present) == len(required) else "waiting_for_streams"
        return {
            "status": status,
            "robot_id": profile.robot_id,
            "checked_at": _now_iso(),
            "canonical_signals": self._robot_canonical_signals(
                profile,
                state_status,
                camera_status=camera_status,
                arm_vector=arm_vector,
            ),
            "streams": streams,
        }

    def _robot_canonical_signals(
        self,
        profile: RobotProfile,
        state_status: dict[str, Any] | None,
        *,
        camera_status: dict[str, Any] | None = None,
        arm_vector: Any | None = None,
    ) -> dict[str, Any]:
        color = profile.color_stream() or {}
        state = profile.robot_state_stream() or {}
        camera_topic, arm_topic = profile_to_vsa_topics(profile)
        eef_xyz: list[float] | None = None
        eef_rxyz: list[float] | None = None
        gripper_value: float | None = None
        if arm_vector is not None:
            try:
                eef_xyz, eef_rxyz, gripper_value = _extract_ROBOLINEAGE_pose_vector(arm_vector)
            except Exception:
                eef_xyz = None
                eef_rxyz = None
                gripper_value = None
        close_rule = _profile_gripper_close_rule(state)
        gripper_state = (
            _apply_gripper_close_rule(gripper_value, close_rule)
            if gripper_value is not None and close_rule is not None
            else None
        )
        return {
            "primary_image": {
                "present": (
                    camera_status is not None
                    if self.orchestrator is not None
                    else bool(camera_topic)
                ),
                "topic": camera_topic,
                "shape": _status_shape(camera_status),
                "age_sec": _status_age_sec(camera_status),
            },
            "active_eef_pose": {
                "present": eef_xyz is not None,
                "topic": arm_topic,
                "xyz": eef_xyz,
                "rxyz": eef_rxyz,
                "age_sec": state_status.get("age_sec") if state_status else None,
            },
            "gripper": {
                "present": gripper_value is not None,
                "topic": arm_topic,
                "value": gripper_value,
                "state": gripper_state,
                "source": profile.to_summary(active=True).get("gripper_source"),
                "close_rule": close_rule,
            },
            "vsa_binding": {
                "camera_topic": camera_topic,
                "state_topic": arm_topic,
            },
        }

    def _robot_stream_status(
        self,
        *,
        stream_id: str,
        role: str,
        ros_topic: str,
        required: bool,
        sample_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "stream_id": stream_id,
            "role": role,
            "ros_topic": ros_topic,
            "required": required,
            "present": (
                sample_status is not None
                if self.orchestrator is not None
                else bool(ros_topic)
            ),
            "age_sec": _status_age_sec(sample_status),
            "payload_type": sample_status.get("payload_type") if sample_status else None,
            "payload_shape": _status_shape(sample_status),
            "sample_meta": _json_safe(sample_status) if sample_status else None,
        }

    def _robot_camera_status(
        self,
        profile: RobotProfile,
        color: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.orchestrator is None or not isinstance(color, dict):
            return None
        getter = getattr(self.orchestrator, "camera_status", None)
        if not callable(getter):
            return None
        stream_id = str(color.get("stream_id") or profile.active_color_stream_id or "")
        topic = str(color.get("topic") or "") or None
        return getter(stream_id=stream_id or None, topic=topic)

    def _robot_arm_status(
        self,
        profile: RobotProfile,
        state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.orchestrator is None or not isinstance(state, dict):
            return None
        getter = getattr(self.orchestrator, "arm_status", None)
        if not callable(getter):
            return None
        stream_id = str(state.get("state_stream_id") or profile.active_robot_state_id or "")
        topic = str(state.get("topic") or "") or None
        return getter(stream_id=stream_id or None, topic=topic)

    def _robot_arm_vector(
        self,
        profile: RobotProfile,
        state: dict[str, Any] | None,
    ) -> Any | None:
        if self.orchestrator is None or not isinstance(state, dict):
            return None
        getter = getattr(self.orchestrator, "latest_arm_vector", None)
        if not callable(getter):
            return None
        stream_id = str(state.get("state_stream_id") or profile.active_robot_state_id or "")
        topic = str(state.get("topic") or "") or None
        return getter(stream_id=stream_id or None, topic=topic)

    def _start_data_source(self) -> None:
        from robolineage_data_source.orchestrator import default_adapter_factory

        self.orchestrator = Orchestrator(
            self.config,
            adapter_factory=default_adapter_factory,
            recorder_mode="none",
        )
        self.orchestrator.start()
        _LOG.info("[robolineage_app] data_source started (rollout_id=%s)", self.orchestrator.rollout_id)

    def _data_source_ready_to_start(self) -> bool:
        return (
            self.config.adapter is not None
            or bool(self.config.cameras)
            or bool(self.config.robots)
            or bool(self.config.imu)
        )

    def _start_session(self) -> None:
        # Lazy import: keeps `import robolineage_app` lightweight on macOS dev (no
        # FastAPI/uvicorn pulled until launcher actually runs).
        from robolineage_ar.types import CameraParams, RenderConfig
        from robolineage_session.api import create_app as create_session_app

        self.session_app = create_session_app(
            video_source=self._session_video_source(),
            camera=CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0),
            render_config=RenderConfig(),
            on_task_configure=self.configure_vsa_task,
            on_rollout_start=self.start_vsa_rollout,
            on_rollout_stop=self.stop_vsa,
            on_rollout_state=self.vsa_status,
            on_task_stop=self.stop_vsa,
            on_post_review_state=self.post_review_status,
            on_post_review_rollouts=self.post_review_rollouts,
            on_post_review_detail=self.post_review_detail,
            on_training_framework_state=self.training_framework_status,
            on_training_framework_runs=self.training_framework_runs,
            on_training_framework_detail=self.training_framework_detail,
            on_training_framework_run_demo=self.training_framework_run_demo,
            on_training_framework_discover=self.training_framework_discover_start,
            on_training_framework_discovery_job=self.training_framework_discovery_job,
            on_tasks=self.task_registry,
            on_task_create=self.task_create,
            on_task_activate=self.task_activate,
            on_task_detail=self.task_detail,
            on_task_collection_sessions=self.task_collection_sessions,
            on_task_collection_session_detail=self.task_collection_session_detail,
            on_task_deployment_sessions=self.task_deployment_sessions,
            on_task_deployment_session_detail=self.task_deployment_session_detail,
            on_training_selections=self.training_selections,
            on_training_selection_create=self.training_selection_create,
            on_framework_profiles=self.framework_profiles,
            on_training_data_adapt_start=self.training_data_adapt_start,
            on_training_run_start=self.training_run_start,
            on_policies=self.policies,
            on_rollout_session_state=self.rollout_session_state,
            on_collection_session_start=self.start_collection_session,
            on_collection_session_stop=self.stop_collection_session,
            on_deployment_session_start=self.start_deployment_session,
            on_deployment_session_stop=self.stop_deployment_session,
            on_master_status=self.master_status,
            on_master_review=self.master_review,
            on_ai_routes_status=self.ai_routes_status,
            on_robots=self.robot_profiles,
            on_robot_detail=self.robot_profile_detail,
            on_robot_activate=self.robot_profile_activate,
            on_robot_validate=self.robot_profile_validate,
            on_robot_onboard=self.robot_onboarding_start,
        )
        _LOG.info("[robolineage_app] session app constructed")

    def _session_video_source(self) -> Any:
        from robolineage_ar.video_source import LatestFrameVideoSource, SyntheticVideoSource

        fallback = SyntheticVideoSource()

        def _latest_profile_frame() -> Any | None:
            if self.orchestrator is None or self._active_robot_profile is None:
                return None
            color = self._active_robot_profile.color_stream() or {}
            stream_id = (
                str(color.get("stream_id"))
                if color.get("stream_id")
                else self._active_robot_profile.active_color_stream_id
            )
            topic = str(color.get("topic") or "") or None
            getter = getattr(self.orchestrator, "latest_camera_frame", None)
            if not callable(getter):
                return None
            return getter(stream_id=stream_id, topic=topic)

        return LatestFrameVideoSource(_latest_profile_frame, fallback=fallback)

    def _start_vsa(self, *, require_task_config: bool = False) -> dict[str, Any]:
        if self._active_vsa_run is not None and self._active_vsa_run.thread.is_alive():
            raise RuntimeError("VSA rollout is already active")
        if (
            self._rollout_group is not None
            and self._rollout_group.get("status") == "finalizing"
        ):
            raise RuntimeError("cannot start rollout while session is finalizing")
        if len(self._draining_vsa_runs) >= _MAX_DRAINING_VSA_ROLLOUTS:
            raise RuntimeError(
                "too many VSA rollouts are still draining "
                f"({len(self._draining_vsa_runs)}/{_MAX_DRAINING_VSA_ROLLOUTS}); "
                "wait for analysis to finish before starting another rollout"
            )
        if self.config.vsa is None or self.config.vsa.task_config_path is None:
            if require_task_config:
                raise RuntimeError("VSA task_config_path is missing; configure a task first")
            _LOG.warning(
                "[robolineage_app] services.vsa=true but config.vsa.task_config_path is missing; "
                "skipping VSA realtime startup"
            )
            return self.vsa_status()

        # Lazy imports — VSA pulls openai/yaml/etc.
        import yaml as _yaml

        from robolineage_shared_agents.visual_snapshot import TaskConfig
        from robolineage_shared_agents.visual_snapshot.realtime import (
            StreamingRuntimePipeline,
            run_ros_topic_stream,
        )
        from robolineage_shared_agents.visual_snapshot.vlm_priority import OnlineVLMRunner
        from robolineage_shared_agents.visual_snapshot.vlm_runner import make_vlm_runner_from_env

        vsa_cfg = self.config.vsa
        tuning = self.config.tuning
        vlm_cfg = self.config.vlm
        arm_spec = self._vsa_arm_spec(vsa_cfg.arm_topic)
        ros_domain_id = self.config.adapter.ros_domain_id if self.config.adapter is not None else 0

        with open(vsa_cfg.task_config_path, "r", encoding="utf-8") as f:
            task_dict = _yaml.safe_load(f)
        task_config = TaskConfig(**task_dict)
        self._task_config_meta = _task_config_metadata_for_path(Path(vsa_cfg.task_config_path))

        rollout_dir = self._next_vsa_rollout_dir()
        output_jsonl = (
            Path(vsa_cfg.output_jsonl_path)
            if vsa_cfg.output_jsonl_path
            else rollout_dir / "snapshots.jsonl"
        )
        self._copy_task_config(vsa_cfg.task_config_path, rollout_dir)
        self._write_task_config_binding(rollout_dir)
        self._write_rollout_context(rollout_dir)
        stop_flag = threading.Event()
        raw_recorder = self._start_raw_recorder(rollout_dir)
        self._vlm_coordinator.enter_online(rollout_dir.name)

        try:
            base_vlm_runner = make_vlm_runner_from_env(
                "VSA_VLM",
                valid_phases=task_config.phases,
                timeout_default=vlm_cfg.timeout if vlm_cfg else 20.0,
                max_output_tokens_default=vlm_cfg.max_output_tokens if vlm_cfg else 256,
            )
            vlm_runner = OnlineVLMRunner(
                base_vlm_runner,
                self._vlm_coordinator,
                rollout_id=rollout_dir.name,
            )
        except Exception:
            self._stop_raw_capture_obj(raw_recorder)
            self._finalize_raw_recorder(
                raw_recorder,
                outcome_value="interrupted",
                note="VSA VLM runner initialization failed",
            )
            self._vlm_coordinator.exit_online(rollout_dir.name)
            raise

        configured_ring_capacity = tuning.ring_capacity if tuning else 120
        ring_capacity = min(configured_ring_capacity, _ONLINE_RING_CAPACITY_MAX)
        if ring_capacity != configured_ring_capacity:
            _LOG.warning(
                "[robolineage_app] clamped online VSA ring_capacity from %s to %s to bound frame memory",
                configured_ring_capacity,
                ring_capacity,
            )
        still_min_frames = tuning.still_min_frames if tuning else 25
        heartbeat_interval = tuning.heartbeat_interval if tuning else 5.0
        periodic_interval_sec = tuning.periodic_interval_sec if tuning else 2.0
        merge_window_sec = tuning.merge_window_sec if tuning else 1.0
        final_settle_sec = tuning.final_settle_sec if tuning else 1.0
        max_vlm_windows_per_rollout = tuning.max_vlm_windows_per_rollout if tuning else None
        context_frames = tuning.context_frames if tuning else 15
        max_keyframes = tuning.max_keyframes if tuning else 3
        idle_timeout = tuning.idle_timeout if tuning else 10.0
        gripper_close_threshold = tuning.gripper_close_threshold if tuning else -1.0
        still_threshold = tuning.still_threshold if tuning else 3e-4
        rotation_weight = tuning.rotation_weight if tuning else 0.2
        smoothing_window = tuning.smoothing_window if tuning else 10
        motion_resume_threshold = tuning.motion_resume_threshold if tuning else 8e-4
        min_same_event_interval = tuning.min_same_event_interval if tuning else 3.0
        vlm_workers = tuning.vlm_workers if tuning else 1
        strong_prior_margin = tuning.strong_prior_margin if tuning else 0.35
        prior_sticky_frames = tuning.prior_sticky_frames if tuning else 2

        def _on_snapshot(s: Any) -> None:
            with self._vsa_lock:
                self._vsa_snapshots.append(s)

        try:
            pipeline = StreamingRuntimePipeline(
                task_config=task_config,
                vlm_runner=vlm_runner,
                rollout_dir=rollout_dir,
                output_jsonl=output_jsonl,
                context_frames=context_frames,
                max_keyframes=max_keyframes,
                ring_capacity=ring_capacity,
                still_min_frames=still_min_frames,
                heartbeat_interval=heartbeat_interval,
                periodic_interval_sec=periodic_interval_sec,
                merge_window_sec=merge_window_sec,
                final_settle_sec=final_settle_sec,
                max_vlm_windows_per_rollout=max_vlm_windows_per_rollout,
                gripper_close_threshold=gripper_close_threshold,
                still_threshold=still_threshold,
                rotation_weight=rotation_weight,
                smoothing_window=smoothing_window,
                motion_resume_threshold=motion_resume_threshold,
                min_same_event_interval=min_same_event_interval,
                vlm_workers=vlm_workers,
                strong_prior_margin=strong_prior_margin,
                prior_sticky_frames=prior_sticky_frames,
                on_snapshot=_on_snapshot,
            )
        except Exception:
            self._stop_raw_capture_obj(raw_recorder)
            self._finalize_raw_recorder(
                raw_recorder,
                outcome_value="interrupted",
                note="VSA pipeline initialization failed",
            )
            self._vlm_coordinator.exit_online(rollout_dir.name)
            raise
        started_at = datetime.now(timezone.utc).isoformat()
        self._vsa_pipeline = pipeline
        self._vsa_rollout_id = rollout_dir.name
        self._vsa_rollout_dir = rollout_dir
        self._vsa_output_jsonl = output_jsonl
        self._vsa_started_at = started_at
        self._vsa_last_error = None
        with self._vsa_lock:
            self._vsa_snapshots.clear()

        max_events = vsa_cfg.max_events  # None = run forever
        self._vsa_stop_flag = stop_flag
        run_holder: dict[str, _OnlineRolloutRun] = {}

        def _vsa_loop() -> None:
            run = run_holder["run"]
            try:
                run_ros_topic_stream(
                    camera_topic=vsa_cfg.camera_topic,
                    arm_topic=vsa_cfg.arm_topic,
                    arm_spec=arm_spec,
                    ros_domain_id=ros_domain_id,
                    pipeline=pipeline,
                    max_events=max_events,
                    idle_timeout=idle_timeout,
                    stop_event=stop_flag,
                )
            except Exception as exc:
                run.last_error = repr(exc)
                self._vsa_last_error = repr(exc)
                _LOG.exception("[robolineage_app] VSA realtime loop crashed")
            finally:
                self._complete_vsa_run(run)

        thread = threading.Thread(
            target=_vsa_loop, name="robolineage_app.vsa", daemon=True
        )
        rollout_index = self._rollout_index_for_dir(rollout_dir)
        run = _OnlineRolloutRun(
            rollout_id=rollout_dir.name,
            rollout_dir=rollout_dir,
            output_jsonl=output_jsonl,
            pipeline=pipeline,
            thread=thread,
            stop_flag=stop_flag,
            raw_recorder=raw_recorder,
            started_at=started_at,
            rollout_index=rollout_index,
        )
        run_holder["run"] = run
        self._active_vsa_run = run
        self._vsa_thread = thread
        thread.start()
        _LOG.info(
            "[robolineage_app] VSA realtime thread started (rollout_id=%s cam=%s arm=%s)",
            self._vsa_rollout_id,
            vsa_cfg.camera_topic,
            vsa_cfg.arm_topic,
        )
        self._write_memory_debug(
            "rollout_started",
            run=run,
            extra={
                "camera_topic": vsa_cfg.camera_topic,
                "arm_topic": vsa_cfg.arm_topic,
                "ring_capacity": ring_capacity,
                "configured_ring_capacity": configured_ring_capacity,
                "merge_window_sec": merge_window_sec,
                "final_settle_sec": final_settle_sec,
                "max_vlm_windows_per_rollout": max_vlm_windows_per_rollout,
            },
        )
        return self.vsa_status()

    def _start_raw_recorder(self, rollout_dir: Path) -> Any | None:
        if self.config.recorder is None or self.config.adapter is None:
            return None
        from robolineage_data_source.orchestrator import _rosbag_record_topics, create_rosbag_raw_recorder

        recorder = create_rosbag_raw_recorder(
            rollout_dir=rollout_dir,
            topics=_rosbag_record_topics(self.config),
            ros_domain_id=self.config.adapter.ros_domain_id,
        )
        recorder.start()
        return recorder

    def _vsa_arm_spec(self, arm_topic: str) -> Any:
        if self.config.adapter is None:
            raise RuntimeError("ROS2 adapter config is required for realtime VSA")
        for spec in self.config.adapter.arms.values():
            if spec.slave_status == arm_topic:
                return spec
        raise RuntimeError(f"VSA arm topic is not declared in adapter arms: {arm_topic}")

    def _stop_raw_capture(self, run: _OnlineRolloutRun) -> None:
        self._stop_raw_capture_obj(run.raw_recorder)

    @staticmethod
    def _stop_raw_capture_obj(recorder: Any | None) -> None:
        if recorder is None:
            return
        try:
            recorder.stop_capture()
        except Exception:
            _LOG.exception("[robolineage_app] raw recorder stop_capture failed")

    @staticmethod
    def _finalize_raw_recorder(
        recorder: Any | None,
        *,
        outcome_value: str,
        note: str,
    ) -> None:
        if recorder is None:
            return
        from robolineage_contracts.core import RolloutOutcome

        try:
            recorder.finalize(outcome=RolloutOutcome(outcome_value), note=note)
        except Exception:
            _LOG.exception("[robolineage_app] raw recorder finalize failed")

    def _complete_vsa_run(self, run: _OnlineRolloutRun) -> None:
        try:
            self._stop_raw_capture(run)
            run.capture_stopped_at = run.capture_stopped_at or _now_iso()
            outcome = "interrupted" if run.last_error else "success"
            note = (
                f"VSA realtime error: {run.last_error}"
                if run.last_error
                else "closed after VSA online drain"
            )
            self._finalize_raw_recorder(
                run.raw_recorder,
                outcome_value=outcome,
                note=note,
            )
            run.analysis_completed_at = _now_iso()
            context = _read_json(run.rollout_dir / "rollout_context.json") or {}
            if context.get("kind") == "deployment":
                self._enqueue_eval_review(
                    run.rollout_dir,
                    policy_version=context.get("policy_version"),
                    evaluation_session_id=context.get("session_id"),
                    evaluation_mode="deployment",
                )
            else:
                self._enqueue_post_review(run.rollout_dir)
        finally:
            self._vlm_coordinator.exit_online(run.rollout_id)
            self._write_memory_debug("rollout_analysis_completed", run=run)
            self._draining_vsa_runs.pop(run.rollout_id, None)
            if self._active_vsa_run is run:
                self._active_vsa_run = None
                self._vsa_thread = None
                self._vsa_pipeline = None
                self._vsa_stop_flag = threading.Event()

    def _wait_for_vsa_drain(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            runs = list(self._draining_vsa_runs.values())
            if not runs:
                return True
            for run in runs:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    return False
                run.thread.join(timeout=remaining if remaining is not None else 0.5)
            if deadline is not None and time.monotonic() >= deadline:
                return not self._draining_vsa_runs

    def _raw_video_profile(self) -> tuple[int, int, float]:
        if self.config.cameras:
            cam = self.config.cameras.get(
                "camera_h",
                next(iter(self.config.cameras.values())),
            )
            return cam.resolution[0], cam.resolution[1], float(cam.fps)
        return 1280, 720, 30.0

    @staticmethod
    def _rollout_index_for_dir(rollout_dir: Path) -> int | None:
        context = _read_json(rollout_dir / "rollout_context.json") or {}
        value = context.get("rollout_index")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _start_post_review(self) -> None:
        if self._post_review_worker is not None:
            return

        from robolineage_shared_agents.visual_snapshot.vlm_priority import OfflineVLMRunner
        from robolineage_shared_agents.visual_snapshot.vlm_runner import make_vlm_runner_from_env
        from robolineage_post_rollout import PostRolloutReviewAgent, PostRolloutReviewWorker

        post_cfg = self.config.post_review or PostReviewConfig()
        vlm_cfg = self.config.vlm
        worker: PostRolloutReviewWorker

        def _agent_factory() -> PostRolloutReviewAgent:
            runner = None
            if post_cfg.use_vlm:
                base_runner = make_vlm_runner_from_env(
                    "POST_REVIEW_VLM",
                    timeout_default=max(60.0, vlm_cfg.timeout if vlm_cfg else 20.0),
                    max_output_tokens_default=max(
                        4096,
                        vlm_cfg.max_output_tokens if vlm_cfg else 256,
                    ),
                    min_timeout_s=60.0,
                    min_output_tokens=4096,
                )
                runner = OfflineVLMRunner(
                    base_runner,
                    self._vlm_coordinator,
                    stop_event=worker.stop_event,
                    quiet_period_sec=post_cfg.idle_delay_sec,
                )
            return PostRolloutReviewAgent(
                vlm_runner=runner,
                use_vlm=post_cfg.use_vlm,
                max_review_images=post_cfg.max_review_images,
            )

        worker = PostRolloutReviewWorker(
            agent_factory=_agent_factory,
            idle_delay_sec=post_cfg.idle_delay_sec,
        )
        worker.start()
        self._post_review_worker = worker
        _LOG.info(
            "[robolineage_app] post-review worker started (use_vlm=%s idle_delay=%.1fs)",
            post_cfg.use_vlm,
            post_cfg.idle_delay_sec,
        )

    def _start_eval_review(self) -> None:
        if self._eval_review_worker is not None:
            return

        from robolineage_shared_agents.visual_snapshot.vlm_priority import OfflineVLMRunner
        from robolineage_shared_agents.visual_snapshot.vlm_runner import make_vlm_runner_from_env
        from robolineage_eval import EvaluationReviewWorker, PolicyEvaluationAgent

        post_cfg = self.config.post_review or PostReviewConfig()
        vlm_cfg = self.config.vlm
        worker: EvaluationReviewWorker

        def _agent_factory() -> PolicyEvaluationAgent:
            runner = None
            if post_cfg.use_vlm:
                base_runner = make_vlm_runner_from_env(
                    "POLICY_EVAL_VLM",
                    timeout_default=vlm_cfg.timeout if vlm_cfg else 20.0,
                    max_output_tokens_default=max(
                        512,
                        vlm_cfg.max_output_tokens if vlm_cfg else 256,
                    ),
                )
                runner = OfflineVLMRunner(
                    base_runner,
                    self._vlm_coordinator,
                    stop_event=worker.stop_event,
                    quiet_period_sec=post_cfg.idle_delay_sec,
                )
            return PolicyEvaluationAgent(
                vlm_runner=runner,
                use_vlm=post_cfg.use_vlm,
                max_review_images=post_cfg.max_review_images,
            )

        worker = EvaluationReviewWorker(
            agent_factory=_agent_factory,
            idle_delay_sec=post_cfg.idle_delay_sec,
        )
        worker.start()
        self._eval_review_worker = worker
        _LOG.info(
            "[robolineage_app] evaluation review worker started (use_vlm=%s idle_delay=%.1fs)",
            post_cfg.use_vlm,
            post_cfg.idle_delay_sec,
        )

    def _enqueue_post_review(self, rollout_dir: Path) -> None:
        if self._post_review_worker is None:
            return
        if not rollout_dir.exists():
            return
        self._post_review_worker.enqueue(rollout_dir)

    def _enqueue_eval_review(
        self,
        rollout_dir: Path,
        *,
        policy_version: str | None,
        evaluation_session_id: str | None,
        evaluation_mode: str,
    ) -> None:
        self._start_eval_review()
        if self._eval_review_worker is None or not rollout_dir.exists():
            return
        self._eval_review_worker.enqueue(
            rollout_dir,
            policy_version=policy_version,
            evaluation_session_id=evaluation_session_id,
            evaluation_mode=evaluation_mode,
        )

    def _rollouts_root(self) -> Path:
        if self.config.recorder is not None:
            return Path(self.config.recorder.output_dir)
        return Path(os.environ.get("ROBOLINEAGE_TASK_DIR", ".")) / "rollouts"

    def _task_root(self) -> Path:
        return self._rollouts_root().parent

    def _memory_debug_log_path(self) -> Path:
        return self._task_root() / "logs" / "memory_debug.jsonl"

    def _write_memory_debug(
        self,
        event: str,
        *,
        run: _OnlineRolloutRun | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            payload = self._memory_debug_payload(event, run=run, extra=extra)
            path = self._memory_debug_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)
            with self._memory_debug_lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            _LOG.exception("[robolineage_app] failed to write memory debug log")

    def _memory_debug_payload(
        self,
        event: str,
        *,
        run: _OnlineRolloutRun | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        active_run = self._active_vsa_run
        return {
            "schema_version": "RoboLineage.memory_debug.v1",
            "time": _now_iso(),
            "event": event,
            "extra": dict(extra or {}),
            "process": {
                "rss_mb": _current_rss_mb(),
                "thread_count": threading.active_count(),
            },
            "data_flow": {
                "raw_capture": "rosbag2",
                "online_vsa": "ros2_topics",
            },
            "active_rollout": _rollout_memory_stats(active_run),
            "draining_rollouts": {
                rollout_id: _rollout_memory_stats(draining_run)
                for rollout_id, draining_run in self._draining_vsa_runs.items()
            },
            "event_rollout": _rollout_memory_stats(run),
        }

    def _training_runs_root(self) -> Path:
        return self._task_root() / "training_runs"

    def _datasets_root(self) -> Path:
        return self._task_root() / "datasets"

    def _tasks_root(self) -> Path:
        return Path(os.environ.get("ROBOLINEAGE_TASKS_ROOT") or self._task_root().parent)

    def _task_dir_for_id(self, task_id: str) -> Path:
        return _resolve_child(self._tasks_root(), task_id)

    def _task_item(self, task_dir: Path, *, active: bool) -> dict[str, Any]:
        manifest = _read_json(task_dir / "task_manifest.json") or {}
        config_index = _read_json(task_dir / "task_configs" / "task_config_index.json") or {}
        latest = task_dir / "task_config.latest.yaml"
        rollouts_root = task_dir / "rollouts"
        collection_root = task_dir / "collection_sessions"
        training_root = task_dir / "training_runs"
        rollout_count = len([
            path for path in rollouts_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]) if rollouts_root.exists() else 0
        display_name = (
            manifest.get("display_name")
            or manifest.get("task_description")
            or _read_task_description(latest)
            or task_dir.name
        )
        task_description = manifest.get("task_description") or _read_task_description(latest)
        return {
            "task_id": task_dir.name,
            "display_name": display_name,
            "task_description": task_description,
            "task_dir": str(task_dir),
            "active": active,
            "robot": manifest.get("robot"),
            "created_at": manifest.get("created_at") or _mtime_iso(task_dir),
            "latest_task_config_version": config_index.get("latest_version"),
            "latest_task_config_path": str(latest) if latest.exists() else None,
            "latest_task_config": _read_task_config(latest),
            "rollout_count": rollout_count,
            "collection_session_count": _dir_child_count(collection_root),
            "deployment_session_count": _dir_child_count(task_dir / "deployment_sessions"),
            "training_run_count": _dir_child_count(training_root),
        }

    def _collection_session_item(self, task_dir: Path, session_dir: Path) -> dict[str, Any]:
        summary = _read_json(session_dir / "collection_session_summary.json") or {}
        return {
            "session_id": session_dir.name,
            "session_dir": str(session_dir),
            "started_at": summary.get("started_at"),
            "completed_at": summary.get("completed_at"),
            "rollout_count": int(summary.get("rollout_count") or len(summary.get("rollout_ids") or [])),
            "success_count": int(summary.get("success_count") or 0),
            "dataset_decision_counts": summary.get("dataset_decision_counts") or {},
            "rollout_ids": list(summary.get("rollout_ids") or []),
        }

    def _deployment_session_item(self, task_dir: Path, session_dir: Path) -> dict[str, Any]:
        summary = _read_json(session_dir / "policy_eval_summary.json") or {}
        decision = _read_json(session_dir / "deployment_decision.json") or {}
        collection = _read_json(session_dir / "collection_recommendation.json") or {}
        brief = _read_json(session_dir / "next_collection_brief.json") or {}
        understanding = _read_json(session_dir / "deployment_governance_understanding.json") or {}
        rollout_ids = list(summary.get("rollout_ids") or [])
        return {
            "session_id": session_dir.name,
            "session_dir": str(session_dir),
            "policy_version": summary.get("policy_version") or decision.get("policy_version"),
            "mode": summary.get("mode") or "deployment",
            "rollout_count": int(summary.get("rollout_count") or len(rollout_ids)),
            "success_count": int(summary.get("success_count") or 0),
            "failure_count": int(summary.get("failure_count") or 0),
            "success_rate": summary.get("success_rate"),
            "decision": decision.get("decision"),
            "gating_result": decision.get("gating_result"),
            "recommended_mode": collection.get("recommended_mode"),
            "operator_brief": brief.get("operator_brief"),
            "governance_understanding_status": understanding.get("status"),
            "created_at": summary.get("created_at") or decision.get("created_at") or _mtime_iso(session_dir),
            "rollout_ids": rollout_ids,
        }

    def _policy_eval_rollout_item(self, rollout_dir: Path) -> dict[str, Any]:
        evaluation = _read_json(rollout_dir / "policy_evaluation.json") or {}
        status = _read_json(rollout_dir / "policy_eval_status.json") or {}
        failures = _read_json(rollout_dir / "failure_analysis.json") or {}
        return {
            "rollout_id": rollout_dir.name,
            "rollout_dir": str(rollout_dir),
            "status": status.get("status") or ("evaluated" if evaluation else "pending"),
            "policy_version": evaluation.get("policy_version") or status.get("policy_version"),
            "final_success": evaluation.get("final_success"),
            "success_status": evaluation.get("success_status"),
            "success_confidence": evaluation.get("success_confidence"),
            "final_phase": evaluation.get("final_phase"),
            "recommended_next_action": evaluation.get("recommended_next_action"),
            "policy_behavior_summary": evaluation.get("policy_behavior_summary"),
            "failure_type_counts": evaluation.get("failure_type_counts") or {},
            "phase_weakness": evaluation.get("phase_weakness") or [],
            "failure_analysis": failures,
            "policy_evaluation": evaluation,
            "eval_review_report": _read_text(rollout_dir / "eval_review_report.md"),
            "updated_at": status.get("completed_at") or evaluation.get("created_at") or _mtime_iso(rollout_dir),
        }

    def _selection_item(self, path: Path) -> dict[str, Any]:
        payload = _read_json(path) or {}
        return {
            "selection_id": payload.get("selection_id") or path.stem,
            "selection_path": str(path),
            "task_id": payload.get("task_id"),
            "source_collection_session_ids": payload.get("source_collection_session_ids") or [],
            "include_decisions": payload.get("include_decisions") or [],
            "selected_rollout_count": payload.get("selected_rollout_count") or len(payload.get("rollout_ids") or []),
            "rollout_ids": payload.get("rollout_ids") or [],
            "created_at": payload.get("created_at"),
            "note": payload.get("note"),
        }

    def _framework_profile_item(self, profile_dir: Path) -> dict[str, Any]:
        discovery = _read_json(profile_dir / "framework_discovery.json") or {}
        understanding = discovery.get("llm_understanding") if isinstance(discovery.get("llm_understanding"), dict) else {}
        manifest = discovery.get("integration_manifest") if isinstance(discovery.get("integration_manifest"), dict) else {}
        return {
            "profile_id": profile_dir.name,
            "profile_dir": str(profile_dir),
            "profile_path": str(profile_dir / "framework_profile.generated.yaml"),
            "name": discovery.get("name") or profile_dir.name,
            "framework_type": discovery.get("framework_type"),
            "repo_root": discovery.get("repo_root"),
            "created_at": discovery.get("created_at") or _mtime_iso(profile_dir),
            "events": discovery.get("events") if isinstance(discovery.get("events"), list) else [],
            "events_path": discovery.get("events_path"),
            "target_contract": discovery.get("target_contract") if isinstance(discovery.get("target_contract"), dict) else {},
            "adapter_registry": discovery.get("adapter_registry") if isinstance(discovery.get("adapter_registry"), dict) else {},
            "dataset_adapter": discovery.get("dataset_adapter") if isinstance(discovery.get("dataset_adapter"), dict) else {},
            "integration_manifest": manifest,
            "integration_manifest_path": discovery.get("integration_manifest_path"),
            "report_path": str(profile_dir / "framework_discovery_report.md")
            if (profile_dir / "framework_discovery_report.md").exists()
            else None,
            "understanding_status": understanding.get("status"),
            "understanding_report_path": understanding.get("understanding_report_path"),
        }

    def _policy_item(self, meta_path: Path) -> dict[str, Any]:
        meta = _read_json(meta_path) or {}
        run_dir = meta_path.parents[3] if len(meta_path.parents) >= 4 else meta_path.parent
        recommendation = _read_json(run_dir / "deployment_recommendation.json") or {}
        return {
            "policy_version": meta.get("version_id") or meta_path.parent.name,
            "policy_meta_path": str(meta_path),
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "trained_on_dataset": meta.get("trained_on_dataset"),
            "architecture": meta.get("architecture"),
            "checkpoint_path": meta.get("checkpoint_path"),
            "eval_success_rate": meta.get("eval_success_rate"),
            "deploy_decision": recommendation.get("decision"),
            "created_at": meta.get("created_at") or _mtime_iso(meta_path),
        }

    def _resolve_profile_path(self, task_dir: Path, payload: dict[str, Any]) -> Path:
        if payload.get("framework_profile_path"):
            raise ValueError("framework_profile_path is not accepted; use framework_profile_id")
        profile_id = str(payload.get("framework_profile_id") or payload.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("framework_profile_id is required")
        path = _resolve_child(task_dir / "framework_profiles", profile_id) / "framework_profile.generated.yaml"
        if not path.exists():
            raise FileNotFoundError(f"framework profile not found: {path}")
        return path

    def _resolve_selection_path(self, task_dir: Path, payload: dict[str, Any]) -> Path:
        if payload.get("selection_path"):
            raise ValueError("selection_path is not accepted; use selection_id")
        selection_id = str(payload.get("selection_id") or "").strip()
        if not selection_id:
            raise ValueError("selection_id is required")
        filename = selection_id if selection_id.endswith(".json") else f"{selection_id}.json"
        path = _resolve_child(task_dir / "training_selections", filename)
        if not path.exists():
            raise FileNotFoundError(f"training selection not found: {path}")
        return path

    def _training_run_config_for_request(self, task_dir: Path, run_id: str | None) -> dict[str, Any]:
        if not run_id:
            return {}
        run_dir = _resolve_child(task_dir / "training_runs", run_id)
        config = _read_json(run_dir / "training_run_config.json")
        if not config:
            raise FileNotFoundError(f"training run config not found: {run_dir / 'training_run_config.json'}")
        return config

    def _path_from_training_run_config(self, config: dict[str, Any], key: str, label: str) -> Path:
        value = str(config.get(key) or "").strip()
        if not value:
            raise ValueError(f"{label} is required; choose a dataset run or provide the explicit id")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    def _find_adapted_training_run(
        self,
        *,
        task_dir: Path,
        selection_path: Path,
        profile_path: Path,
        policy_version: str,
        requested_run_id: str | None = None,
    ) -> Path:
        root = task_dir / "training_runs"
        if requested_run_id:
            candidates = [_resolve_child(root, requested_run_id)]
        else:
            candidates = [
                path for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
                if path.is_dir() and not path.name.startswith(".")
            ] if root.exists() else []
        latest_status: str | None = None
        latest_reason: str | None = None
        selection_resolved = str(selection_path.resolve())
        profile_resolved = str(profile_path.resolve())
        for run_dir in candidates:
            config = _read_json(run_dir / "training_run_config.json") or {}
            if not requested_run_id and str(config.get("policy_version") or "") != policy_version:
                continue
            adapt_status = _read_json(run_dir / "framework" / "dataset_adapt_status.json") or {}
            latest_status = str(adapt_status.get("status") or "missing")
            if not _dataset_adapt_status_allows_training(adapt_status):
                latest_reason = f"adapt status is {latest_status}"
                continue
            try:
                config_selection = str(Path(str(config.get("selection_path") or "")).resolve())
                config_profile = str(Path(str(config.get("framework_profile_path") or "")).resolve())
            except OSError:
                continue
            if config_selection != selection_resolved:
                latest_reason = "selection does not match selected dataset run"
                continue
            if config_profile != profile_resolved and not (
                requested_run_id
                and self._profile_dataset_contract_compatible(profile_path, config, adapt_status)
            ):
                latest_reason = "profile dataset contract does not match selected dataset run"
                continue
            return run_dir
        suffix = f" Current adapt status: {latest_status}." if latest_status else ""
        if latest_reason:
            suffix += f" Reason: {latest_reason}."
        raise ValueError("Adapt Data must complete before Start Training." + suffix)

    def _profile_dataset_contract_compatible(
        self,
        profile_path: Path,
        adapted_run_config: dict[str, Any],
        adapt_status: dict[str, Any],
    ) -> bool:
        current = _profile_dataset_signature(profile_path)
        original_profile = Path(str(adapted_run_config.get("framework_profile_path") or ""))
        original = _profile_dataset_signature(original_profile) if original_profile.exists() else {}
        status_signature = _adapt_status_dataset_signature(adapt_status)
        original = {**original, **{key: value for key, value in status_signature.items() if value}}
        return _dataset_signatures_compatible(original, current)

    def _start_rollout_group(self, *, kind: str, policy_version: str | None = None) -> dict[str, Any]:
        if self._rollout_group is not None:
            raise RuntimeError(f"rollout session already active: {self._rollout_group.get('kind')}")
        if (
            self._rollout_group_finalizer_thread is not None
            and self._rollout_group_finalizer_thread.is_alive()
        ):
            raise RuntimeError("rollout session finalization is still running")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{kind}_{stamp}_{uuid.uuid4().hex[:6]}"
        self._last_rollout_group_result = None
        self._rollout_group = {
            "kind": kind,
            "status": "active",
            "session_id": session_id,
            "policy_version": policy_version,
            "started_at": _now_iso(),
            "rollout_ids": [],
        }
        return self.rollout_session_state()

    def _stop_rollout_group(self, *, kind: str) -> dict[str, Any]:
        group = self._rollout_group
        if group is None:
            return {"status": "not_active", **self.rollout_session_state()}
        if group.get("kind") != kind:
            raise RuntimeError(f"active rollout session is {group.get('kind')}, not {kind}")
        if group.get("status") == "finalizing":
            return {"status": "finalizing", **self.rollout_session_state()}
        if self._active_vsa_run is not None and self._active_vsa_run.thread.is_alive():
            self.stop_vsa()
        group["status"] = "finalizing"
        group["stop_requested_at"] = _now_iso()
        group["finalization_stage"] = "waiting_vsa"
        group["finalization_error"] = None
        thread = threading.Thread(
            target=self._finalize_rollout_group,
            args=(group,),
            name=f"RolloutSessionFinalizer.{group.get('session_id')}",
            daemon=True,
        )
        self._rollout_group_finalizer_thread = thread
        thread.start()
        return {"status": "finalizing", **self.rollout_session_state()}

    def _finalize_rollout_group(self, group: dict[str, Any]) -> None:
        kind = str(group.get("kind") or "collection")
        try:
            _LOG.info(
                "[robolineage_app] rollout session finalizing (kind=%s session_id=%s rollouts=%d)",
                kind,
                group.get("session_id"),
                len(group.get("rollout_ids") or []),
            )
            group["finalization_stage"] = "waiting_vsa"
            self._wait_for_vsa_drain(timeout=None)

            if kind == "deployment":
                group["finalization_stage"] = "waiting_eval_review"
                self._enqueue_missing_eval_reviews(group)
                if self._eval_review_worker is not None:
                    self._eval_review_worker.wait_idle(timeout=None)
                group["finalization_stage"] = "writing_summary"
                summary = self._write_deployment_session_summary(group)
            else:
                group["finalization_stage"] = "waiting_post_review"
                self._enqueue_missing_post_reviews(group)
                if self._post_review_worker is not None:
                    self._post_review_worker.wait_idle(timeout=None)
                group["finalization_stage"] = "writing_summary"
                summary = self._write_collection_session_summary(group)

            group["summary"] = summary
            group["status"] = "completed"
            group["finalization_stage"] = (
                "ready_for_training"
                if kind == "collection"
                else "completed"
            )
            master_review = self._enqueue_master_review(
                "post_review_completed" if kind == "collection" else "deployment_governance_completed"
            )
            group["master_review"] = _master_review_ref(master_review)
            result = {
                "status": "stopped",
                "stopped_session": copy.deepcopy(group),
                "summary": summary,
                "master_review": _master_review_ref(master_review),
            }
            self._last_rollout_group_result = result
            _LOG.info(
                "[robolineage_app] rollout session finalized (kind=%s session_id=%s)",
                kind,
                group.get("session_id"),
            )
        except Exception as exc:
            group["status"] = "finalization_failed"
            group["finalization_stage"] = "failed"
            group["finalization_error"] = repr(exc)
            self._last_rollout_group_result = {
                "status": "finalization_failed",
                "stopped_session": copy.deepcopy(group),
                "finalization_error": repr(exc),
            }
            _LOG.exception("[robolineage_app] rollout session finalization failed")
        finally:
            if self._rollout_group is group:
                self._rollout_group = None

    def _wait_for_rollout_group_finalizer(self, timeout: float | None = None) -> bool:
        thread = self._rollout_group_finalizer_thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _enqueue_missing_post_reviews(self, group: dict[str, Any]) -> None:
        if self._post_review_worker is None:
            return
        for rollout_id in group.get("rollout_ids") or []:
            rollout_dir = self._rollouts_root() / str(rollout_id)
            if _post_review_complete(rollout_dir):
                continue
            self._enqueue_post_review(rollout_dir)

    def _enqueue_missing_eval_reviews(self, group: dict[str, Any]) -> None:
        if self._eval_review_worker is None:
            return
        for rollout_id in group.get("rollout_ids") or []:
            rollout_dir = self._rollouts_root() / str(rollout_id)
            if _eval_review_complete(rollout_dir):
                continue
            self._enqueue_eval_review(
                rollout_dir,
                policy_version=group.get("policy_version"),
                evaluation_session_id=group.get("session_id"),
                evaluation_mode="deployment",
            )

    def _write_rollout_context(self, rollout_dir: Path) -> None:
        group = self._rollout_group
        if group is None:
            rollout_index = None
            payload = {
                "kind": "collection",
                "session_id": None,
                "policy_version": None,
                "started_at": _now_iso(),
            }
        else:
            rollout_index = len(group.get("rollout_ids") or []) + 1
            payload = {
                "kind": group.get("kind"),
                "session_id": group.get("session_id"),
                "policy_version": group.get("policy_version"),
                "started_at": group.get("started_at"),
                "rollout_index": rollout_index,
            }
            group.setdefault("rollout_ids", []).append(rollout_dir.name)
        payload["rollout_id"] = rollout_dir.name
        if rollout_index is not None:
            payload["rollout_index"] = rollout_index
        payload["task_config"] = self._task_config_context_payload()
        _write_json_atomic(rollout_dir / "rollout_context.json", payload)

    def _task_config_context_payload(self) -> dict[str, Any]:
        meta = self._task_config_meta if isinstance(self._task_config_meta, dict) else {}
        path = (
            self.config.vsa.task_config_path
            if self.config.vsa is not None
            else None
        )
        return {
            "version_id": meta.get("version_id"),
            "created_at": meta.get("created_at"),
            "version_path": meta.get("version_path") or path,
            "latest_path": meta.get("latest_path"),
            "index_path": meta.get("index_path"),
            "task_description": meta.get("task_description"),
            "phases": meta.get("phases") or [],
        }

    def _write_collection_session_summary(self, group: dict[str, Any]) -> dict[str, Any]:
        rollout_dirs = [self._rollouts_root() / rollout_id for rollout_id in group.get("rollout_ids") or []]
        decisions: dict[str, int] = {}
        successes = 0
        for rollout_dir in rollout_dirs:
            admission = _read_json(rollout_dir / "dataset_admission.json") or {}
            decision = str(admission.get("decision") or "pending")
            decisions[decision] = decisions.get(decision, 0) + 1
            summary = _read_json(rollout_dir / "rollout_summary.json") or {}
            if summary.get("final_success") is True:
                successes += 1
        payload = {
            "schema_version": "RoboLineage.collection_session.v1",
            "session_id": group.get("session_id"),
            "kind": "collection",
            "started_at": group.get("started_at"),
            "completed_at": _now_iso(),
            "rollout_ids": list(group.get("rollout_ids") or []),
            "rollout_count": len(rollout_dirs),
            "success_count": successes,
            "dataset_decision_counts": decisions,
            "training_ready": True,
            "next_stage": "training_selection",
        }
        output_dir = self._task_root() / "collection_sessions" / str(group.get("session_id"))
        _write_json_atomic(output_dir / "collection_session_summary.json", payload)
        return {"output_dir": str(output_dir), **payload}

    def _write_deployment_session_summary(self, group: dict[str, Any]) -> dict[str, Any]:
        from robolineage_eval import DeploymentGovernanceAgent

        rollout_dirs = [self._rollouts_root() / rollout_id for rollout_id in group.get("rollout_ids") or []]
        output_dir = self._task_root() / "deployment_sessions" / str(group.get("session_id"))
        return DeploymentGovernanceAgent().summarize_session(
            rollout_dirs=rollout_dirs,
            output_dir=output_dir,
            session_id=str(group.get("session_id")),
            policy_version=group.get("policy_version"),
            mode="deployment",
        )

    def _post_review_rollout_item(
        self,
        rollout_dir: Path,
        worker_status: dict[str, Any],
    ) -> dict[str, Any]:
        rollout_id = rollout_dir.name
        summary = _read_json(rollout_dir / "rollout_summary.json") or {}
        admission = _read_json(rollout_dir / "dataset_admission.json") or {}
        failure = _read_json(rollout_dir / "failure_analysis.json") or {}
        review_status = _read_json(rollout_dir / "post_review_status.json") or {}
        queued = set(worker_status.get("queued_rollouts") or [])
        current = worker_status.get("current_rollout")
        vlm_usage = worker_status.get("vlm_usage") or {}

        if current == rollout_id and vlm_usage.get("online_active"):
            status = "paused_online"
        elif current == rollout_id:
            status = "running"
        elif rollout_id in queued:
            status = "queued"
        elif review_status.get("status"):
            status = str(review_status["status"])
        elif (rollout_dir / "snapshots.jsonl").exists():
            status = "not_reviewed"
        else:
            status = "recording_or_empty"

        return {
            "rollout_id": rollout_id,
            "rollout_dir": str(rollout_dir),
            "status": status,
            "success_likely": summary.get("success_likely", summary.get("final_success")),
            "dataset_decision": admission.get("decision"),
            "accepted_for_training": admission.get("accepted_for_training"),
            "requires_review": admission.get("requires_review"),
            "admission_class": admission.get("admission_class"),
            "label_quality": admission.get("label_quality"),
            "raw_artifacts": _raw_artifacts_status(rollout_dir),
            "final_phase": summary.get("final_phase"),
            "snapshot_count": summary.get("snapshot_count"),
            "needs_review_count": summary.get("needs_review_count"),
            "failure_candidate_count": failure.get("candidate_count"),
            "updated_at": review_status.get("completed_at") or review_status.get("started_at"),
        }

    def _training_run_item(self, run_dir: Path) -> dict[str, Any]:
        recommendation = _read_json(run_dir / "deployment_recommendation.json") or {}
        context = _read_json(run_dir / "policy.ROBOLINEAGE_context.json") or {}
        status = _read_json(run_dir / "framework" / "training_status.json") or {}
        result = _read_json(run_dir / "framework" / "training_result.json") or {}
        adapt_status = _read_json(run_dir / "framework" / "dataset_adapt_status.json") or {}
        dataset_health = _read_json(run_dir / "dataset_health_report.json") or {}
        dataset_health_understanding = _read_json(run_dir / "dataset_health_understanding.json") or {}
        monitor_understanding = _read_json(run_dir / "framework" / "training_monitor_understanding.json") or {}
        run_config = _read_json(run_dir / "training_run_config.json") or {}
        dataset = context.get("dataset") if isinstance(context.get("dataset"), dict) else {}
        framework = context.get("framework") if isinstance(context.get("framework"), dict) else {}
        metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
        status_value = status.get("status")
        if not status_value:
            if self._training_current_run == run_dir.name:
                status_value = "adapting" if (
                    adapt_status.get("status") in {"pending", "running"}
                    or str(run_config.get("phase") or "").startswith("dataset_adapt")
                ) else "running"
            elif adapt_status.get("status") in {"completed", "skipped"}:
                status_value = "adapted" if _dataset_adapt_status_allows_training(adapt_status) else "blocked"
            elif adapt_status.get("status"):
                status_value = adapt_status.get("status")
            else:
                status_value = "unknown"
        adapt_status_value = adapt_status.get("status")
        if adapt_status_value == "skipped" and not _dataset_adapt_status_allows_training(adapt_status):
            adapt_status_value = "blocked"
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": status_value,
            "framework": framework.get("name") or result.get("framework"),
            "dataset_version": dataset.get("version_id") or result.get("dataset_version") or run_config.get("dataset_version"),
            "policy_version": result.get("policy_version") or run_config.get("policy_version"),
            "selected_rollout_count": result.get("selected_rollout_count") or adapt_status.get("selected_rollout_count") or len(run_config.get("rollout_ids") or []),
            "selection_id": run_config.get("selection_id"),
            "selection_path": run_config.get("selection_path"),
            "framework_profile_path": run_config.get("framework_profile_path"),
            "checkpoint_path": result.get("checkpoint_path"),
            "dataset_adapt_status": adapt_status_value,
            "dataset_health_status": dataset_health.get("status"),
            "dataset_health_understanding_status": dataset_health_understanding.get("status"),
            "training_monitor_understanding_status": monitor_understanding.get("status"),
            "target_dataset_kind": adapt_status.get("target_dataset_kind"),
            "tmux_session": status.get("tmux_session") or result.get("tmux_session"),
            "launch_mode": metrics.get("launch_mode"),
            "remote_host": metrics.get("remote_host"),
            "remote_train_log": metrics.get("remote_train_log"),
            "remote_dataset_dir": metrics.get("remote_dataset_dir"),
            "remote_checkpoint_dir": metrics.get("remote_checkpoint_dir"),
            "eval_success_rate": recommendation.get("success_rate"),
            "deploy_decision": recommendation.get("decision"),
            "updated_at": recommendation.get("created_at") or status.get("updated_at") or adapt_status.get("updated_at"),
        }

    def _run_dataset_adapt_worker(
        self,
        task_dir: Path,
        request_id: str,
        run_id: str,
        profile_path: Path,
        selection_path: Path,
        rollout_ids: tuple[str, ...],
        include_decisions: tuple[str, ...],
        policy_version: str,
        architecture: str,
        deploy_success_threshold: float,
    ) -> None:
        try:
            from robolineage_train import TrainingLifecycleRunner, load_framework_profile

            profile = load_framework_profile(profile_path)
            selection = _read_json(selection_path) or {}
            workspace = task_dir / "training_runs" / run_id
            run_config = {
                "schema_version": "RoboLineage.training_run_config.v1",
                "request_id": request_id,
                "run_id": run_id,
                "task_id": task_dir.name,
                "task_dir": str(task_dir),
                "phase": "dataset_adapt",
                "selection_id": selection.get("selection_id") or selection_path.stem,
                "selection_path": str(selection_path),
                "framework_profile_path": str(profile_path),
                "policy_version": policy_version,
                "architecture": architecture,
                "deploy_success_threshold": deploy_success_threshold,
                "include_decisions": list(include_decisions),
                "rollout_ids": list(rollout_ids),
                "created_at": _now_iso(),
            }
            _write_json_atomic(workspace / "training_run_config.json", run_config)
            result = TrainingLifecycleRunner(
                profile=profile,
                rollouts_root=task_dir / "rollouts",
                datasets_root=task_dir / "datasets",
                workspace_root=task_dir / "training_runs",
                include_decisions=include_decisions,
                include_rollout_ids=rollout_ids,
                deploy_success_threshold=deploy_success_threshold,
            ).adapt_data(policy_version=policy_version, run_id=run_id)
            run_config.update(
                {
                    "phase": "dataset_adapt_completed",
                    "dataset_version": result.dataset_version,
                    "dataset_lock_path": str(result.dataset_lock_path),
                    "dataset_adapt_status_path": str(result.dataset_adapt_status_path),
                    "dataset_adapt_result_path": str(result.dataset_adapt_result_path),
                    "updated_at": _now_iso(),
                }
            )
            _write_json_atomic(result.workspace_dir / "training_run_config.json", run_config)
            self._training_current_run = result.run_id
            self._enqueue_master_review("dataset_adapt_completed", task_root=task_dir)
        except Exception as exc:
            self._training_last_error = repr(exc)
            if self._training_current_run == run_id:
                self._training_current_run = None
            _LOG.exception("[robolineage_app] training data adapt failed")
        finally:
            self._training_thread = None

    def _run_training_selection_worker(
        self,
        task_dir: Path,
        request_id: str,
        run_id: str,
        profile_path: Path,
        selection_path: Path,
        rollout_ids: tuple[str, ...],
        include_decisions: tuple[str, ...],
        policy_version: str,
        architecture: str,
        deploy_success_threshold: float,
    ) -> None:
        try:
            from robolineage_train import TrainingLifecycleRunner, load_framework_profile

            profile = load_framework_profile(profile_path)
            selection = _read_json(selection_path) or {}
            workspace = task_dir / "training_runs" / run_id
            existing_config = _read_json(workspace / "training_run_config.json") or {}
            profile = _profile_with_training_execution(profile, existing_config.get("training_execution"))
            dataset_version = str(existing_config.get("dataset_version") or "").strip()
            dataset_lock_path = Path(str(existing_config.get("dataset_lock_path") or ""))
            if not dataset_version or not dataset_lock_path.exists():
                raise ValueError("adapt data must complete before training starts")
            run_config = {
                **existing_config,
                "schema_version": "RoboLineage.training_run_config.v1",
                "request_id": request_id,
                "run_id": run_id,
                "task_id": task_dir.name,
                "task_dir": str(task_dir),
                "phase": "training",
                "selection_id": selection.get("selection_id") or selection_path.stem,
                "selection_path": str(selection_path),
                "adapted_framework_profile_path": existing_config.get("adapted_framework_profile_path")
                or existing_config.get("framework_profile_path"),
                "framework_profile_path": str(profile_path),
                "policy_version": policy_version,
                "architecture": architecture,
                "include_decisions": list(include_decisions),
                "rollout_ids": list(rollout_ids),
                "created_at": _now_iso(),
            }
            _write_json_atomic(workspace / "training_run_config.json", run_config)
            result = TrainingLifecycleRunner(
                profile=profile,
                rollouts_root=task_dir / "rollouts",
                datasets_root=task_dir / "datasets",
                workspace_root=task_dir / "training_runs",
                include_decisions=include_decisions,
                include_rollout_ids=rollout_ids,
                deploy_success_threshold=deploy_success_threshold,
            ).train_adapted(
                policy_version=policy_version,
                architecture=architecture,
                run_id=run_id,
                dataset_version=dataset_version,
                dataset_lock_path=dataset_lock_path,
            )
            run_config["phase"] = "training_completed"
            run_config["updated_at"] = _now_iso()
            _write_json_atomic(result.workspace_dir / "training_run_config.json", run_config)
            self._training_current_run = result.run_id
            self._enqueue_master_review("training_completed", task_root=task_dir)
        except Exception as exc:
            self._training_last_error = repr(exc)
            if self._training_current_run == run_id:
                self._training_current_run = None
            _LOG.exception("[robolineage_app] training framework run failed")
        finally:
            self._training_thread = None

    def _run_training_demo_worker(self, run_id: str) -> None:
        try:
            from robolineage_train import CommandIntake, FrameworkDiscoveryAgent, TrainingLifecycleRunner, load_framework_profile

            root = self._training_runs_root()
            demo_root = root / ".demo_framework"
            self._write_demo_framework(demo_root)
            discovery = FrameworkDiscoveryAgent().discover(
                repo_root=demo_root,
                output_dir=demo_root / ".ROBOLINEAGE_discovery",
                name="demo_generic_training_framework",
                framework_type="generic_policy",
                commands=CommandIntake(
                    dataset_command=f"{sys.executable} scripts/build_dataset.py {{selected_rollouts_file}} {{dataset_output}}",
                    train_command=f"{sys.executable} scripts/train.py {{dataset_output}} {{checkpoint_dir}}",
                    eval_command=f"{sys.executable} scripts/eval.py {{checkpoint_path}} {{eval_output}}",
                ),
            )
            profile = load_framework_profile(discovery.profile_path)
            result = TrainingLifecycleRunner(
                profile=profile,
                rollouts_root=self._rollouts_root(),
                datasets_root=self._datasets_root(),
                workspace_root=root,
                include_decisions=("accepted", "needs_review"),
                deploy_success_threshold=0.7,
            ).run(policy_version="1.0.0", architecture="generic_policy")
            self._training_current_run = result.run_id
            self._enqueue_master_review("training_completed")
        except Exception as exc:
            self._training_last_error = repr(exc)
            if self._training_current_run == run_id:
                self._training_current_run = None
            _LOG.exception("[robolineage_app] training framework demo failed")
        finally:
            self._training_thread = None

    @staticmethod
    def _write_demo_framework(root: Path) -> None:
        scripts = root / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "build_dataset.py").write_text(
            "\n".join([
                "import json, sys",
                "from pathlib import Path",
                "selected=json.loads(Path(sys.argv[1]).read_text())",
                "out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "rollouts=[item['rollout_id'] for item in selected.get('selected_rollouts', [])]",
                "(out/'dataset.json').write_text(json.dumps({'rollouts': rollouts}, ensure_ascii=False, indent=2))",
                "print('dataset_count=' + str(len(rollouts)))",
            ]),
            encoding="utf-8",
        )
        (scripts / "train.py").write_text(
            "\n".join([
                "import json, sys",
                "from pathlib import Path",
                "dataset=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "ckpt=out/'policy.ckpt'; ckpt.write_text('demo policy')",
                "rows=[",
                "  {'step': 1, 'loss': 1.25},",
                "  {'step': 50, 'loss': 0.42},",
                "  {'step': 100, 'loss': 0.18, 'checkpoint': str(ckpt)},",
                "]",
                "(out/'training.log').write_text('\\n'.join(json.dumps(r) for r in rows))",
                "for row in rows: print(json.dumps(row))",
            ]),
            encoding="utf-8",
        )
        (scripts / "eval.py").write_text(
            "\n".join([
                "import json, sys",
                "from pathlib import Path",
                "ckpt=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "payload={'success_rate': 0.84, 'checkpoint': str(ckpt), 'episodes': 20}",
                "(out/'result.json').write_text(json.dumps(payload, indent=2))",
                "print(json.dumps(payload))",
            ]),
            encoding="utf-8",
        )

    def _next_vsa_rollout_dir(self) -> Path:
        if self.config.recorder is not None:
            root = Path(self.config.recorder.output_dir)
        else:
            root = Path(os.environ.get("ROBOLINEAGE_TASK_DIR", ".")) / "rollouts"
        root.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            rollout_id = uuid.uuid4().hex[:12]
            candidate = root / rollout_id
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
        raise RuntimeError(f"failed to allocate VSA rollout dir under {root}")

    @staticmethod
    def _copy_task_config(task_config_path: str, rollout_dir: Path) -> None:
        try:
            source = Path(task_config_path)
            if source.exists():
                shutil.copyfile(source, rollout_dir / "task_config.yaml")
        except Exception:
            _LOG.exception("[robolineage_app] failed to copy task_config.yaml into rollout dir")

    def _write_task_config_binding(self, rollout_dir: Path) -> None:
        payload = {
            "schema_version": "RoboLineage.task_config_binding.v1",
            "rollout_id": rollout_dir.name,
            "task_config": self._task_config_context_payload(),
            "bound_at": _now_iso(),
        }
        _write_json_atomic(rollout_dir / "task_config_binding.json", payload)


def _dataset_adapt_status_allows_training(status: dict[str, Any]) -> bool:
    status_value = str(status.get("status") or "")
    if status_value == "completed":
        return True
    if status_value != "skipped":
        return False
    adapter_id = str(status.get("adapter_id") or "")
    strategy = str(status.get("adapter_strategy") or "")
    target_kind = str(status.get("target_dataset_kind") or "")
    if target_kind == "unknown_custom":
        return False
    return adapter_id in {"direct_selected_rollouts_file", "no_conversion_required"} and strategy in {
        "direct_manifest",
        "no_conversion_required",
    }


def _profile_dataset_signature(profile_path: Path) -> dict[str, Any]:
    try:
        from robolineage_train import load_framework_profile

        profile = load_framework_profile(profile_path)
    except Exception:
        return {}
    adapter = dict(profile.dataset_adapter or {})
    candidate = adapter.get("adapter_candidate") if isinstance(adapter.get("adapter_candidate"), dict) else {}
    target_contract = adapter.get("target_contract") if isinstance(adapter.get("target_contract"), dict) else {}
    return {
        "target_dataset_kind": str(
            target_contract.get("dataset_kind")
            or adapter.get("target_dataset_kind")
            or candidate.get("target_dataset_kind")
            or ""
        ),
        "adapter_id": str(adapter.get("adapter_id") or candidate.get("adapter_id") or ""),
        "adapter_strategy": str(adapter.get("strategy") or candidate.get("strategy") or ""),
        "fields": _contract_fields_signature(target_contract),
        "camera_names": tuple(sorted(str(item) for item in (target_contract.get("camera_names") or []) if str(item))),
    }


def _adapt_status_dataset_signature(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_dataset_kind": str(status.get("target_dataset_kind") or ""),
        "adapter_id": str(status.get("adapter_id") or ""),
        "adapter_strategy": str(status.get("adapter_strategy") or ""),
    }


def _contract_fields_signature(target_contract: dict[str, Any]) -> tuple[tuple[str, str, bool, str, str], ...]:
    fields = target_contract.get("fields")
    if not isinstance(fields, list):
        return ()
    rows: list[tuple[str, str, bool, str, str]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        rows.append(
            (
                str(field.get("path") or ""),
                str(field.get("role") or ""),
                bool(field.get("required", True)),
                str(field.get("shape") or ""),
                str(field.get("dtype") or ""),
            )
        )
    return tuple(sorted(rows))


def _dataset_signatures_compatible(original: dict[str, Any], current: dict[str, Any]) -> bool:
    original_kind = str(original.get("target_dataset_kind") or "")
    current_kind = str(current.get("target_dataset_kind") or "")
    if not original_kind or not current_kind or "unknown_custom" in {original_kind, current_kind}:
        return False
    if original_kind != current_kind:
        return False
    for key in ("adapter_id", "adapter_strategy"):
        original_value = str(original.get(key) or "")
        current_value = str(current.get(key) or "")
        if original_value and current_value and original_value != current_value:
            return False
    for key in ("fields", "camera_names"):
        original_value = original.get(key) or ()
        current_value = current.get(key) or ()
        if original_value and current_value and original_value != current_value:
            return False
    return True


def _training_execution_override(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("train_launch_mode") or "").strip()
    remote: dict[str, Any] = {}
    for payload_key, remote_key in (
        ("remote_host", "host"),
        ("remote_repo_root", "repo_root"),
        ("remote_dataset_dir", "dataset_dir"),
        ("remote_checkpoint_dir", "checkpoint_dir"),
        ("remote_work_dir", "work_dir"),
        ("remote_train_log", "train_log"),
        ("remote_ssh_args", "ssh_args"),
        ("remote_rsync_args", "rsync_args"),
    ):
        value = payload.get(payload_key)
        if value is not None and str(value).strip():
            text_value = str(value).strip()
            if remote_key == "repo_root":
                text_value = _normalize_remote_repo_root(text_value)
            remote[remote_key] = text_value
    if "remote_sync_checkpoints" in payload:
        remote["sync_checkpoints"] = bool(payload.get("remote_sync_checkpoints"))
    if not mode and remote:
        mode = "remote_tmux"
    out: dict[str, Any] = {}
    if mode:
        out["train_launch_mode"] = mode
    if remote:
        out["remote"] = remote
    return out


def _profile_with_training_execution(profile: Any, raw: Any) -> Any:
    if not isinstance(raw, dict) or not raw:
        return profile
    from robolineage_train import FrameworkRemoteExecution

    execution = profile.execution
    remote_raw = raw.get("remote") if isinstance(raw.get("remote"), dict) else {}
    remote = execution.remote
    if remote_raw:
        remote = replace(
            remote,
            host=_optional_nonempty(remote_raw.get("host"), remote.host),
            repo_root=_optional_nonempty(remote_raw.get("repo_root"), remote.repo_root),
            dataset_dir=_optional_nonempty(remote_raw.get("dataset_dir"), remote.dataset_dir),
            checkpoint_dir=_optional_nonempty(remote_raw.get("checkpoint_dir"), remote.checkpoint_dir),
            work_dir=_optional_nonempty(remote_raw.get("work_dir"), remote.work_dir),
            train_log=_optional_nonempty(remote_raw.get("train_log"), remote.train_log),
            ssh_args=_split_shellish(remote_raw.get("ssh_args")) or remote.ssh_args,
            rsync_args=_split_shellish(remote_raw.get("rsync_args")) or remote.rsync_args,
            sync_checkpoints=bool(remote_raw.get("sync_checkpoints", remote.sync_checkpoints)),
        )
    elif raw.get("train_launch_mode") == "remote_tmux" and not isinstance(remote, FrameworkRemoteExecution):
        remote = FrameworkRemoteExecution()
    execution = replace(
        execution,
        train_launch_mode=str(raw.get("train_launch_mode") or execution.train_launch_mode),
        remote=remote,
    )
    return replace(profile, execution=execution)


def _optional_nonempty(value: Any, default: str | None) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or default


def _split_shellish(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(shlex.split(value)) if value.strip() else ()
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _dedupe_event_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for event in events:
        key = json.dumps(
            {
                "event": event.get("event"),
                "created_at": event.get("created_at"),
                "path": event.get("path"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


_CONTEXT_LABELS = {
    "repo": "repo_root",
    "repo root": "repo_root",
    "repo_root": "repo_root",
    "repo location": "repo_location",
    "repo_location": "repo_location",
    "remote ssh": "remote_ssh_host",
    "remote ssh host": "remote_ssh_host",
    "ssh": "remote_ssh_host",
    "ssh host": "remote_ssh_host",
    "remote repo": "remote_repo_root",
    "remote repo root": "remote_repo_root",
    "remote repository": "remote_repo_root",
    "remote repository root": "remote_repo_root",
    "repository": "repo_root",
    "repository root": "repo_root",
    "repository_root": "repo_root",
    "training repo": "repo_root",
    "training repo root": "repo_root",
    "name": "name",
    "profile name": "name",
    "framework type": "framework_type",
    "framework_type": "framework_type",
    "dataset command": "dataset_command",
    "dataset_command": "dataset_command",
    "convert command": "dataset_command",
    "conversion command": "dataset_command",
    "adapter command": "dataset_command",
    "data command": "dataset_command",
    "train command": "train_command",
    "training command": "train_command",
    "train_command": "train_command",
    "run command": "train_command",
    "training script": "train_command",
    "eval command": "eval_command",
    "evaluation command": "eval_command",
    "eval_command": "eval_command",
    "conda env": "conda_env",
    "conda environment": "conda_env",
    "environment": "conda_env",
    "env": "conda_env",
    "launch mode": "train_launch_mode",
    "train launch": "train_launch_mode",
    "train_launch_mode": "train_launch_mode",
    "terminal command": "terminal_command",
    "terminal_command": "terminal_command",
    "terminal hold open": "terminal_hold_open",
    "terminal_hold_open": "terminal_hold_open",
    "fixed input dir": "fixed_input_dir",
    "framework input dir": "fixed_input_dir",
    "checkpoint glob": "checkpoint_glob",
    "checkpoint output": "checkpoint_glob",
    "train log": "train_log",
    "training log": "train_log",
    "eval result": "eval_result",
    "evaluation result": "eval_result",
}


def _parse_framework_command_context(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key and current_lines:
            value = _clean_context_value(" ".join(line for line in current_lines if line.strip()))
            if value and current_key not in parsed:
                parsed[current_key] = value
        current_key = None
        current_lines = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped in {"```", "~~~"}:
            continue
        label, value = _split_context_label(stripped)
        key = _CONTEXT_LABELS.get(label) if label else None
        if key:
            flush()
            if value:
                parsed[key] = _clean_context_value(value)
            else:
                current_key = key
                current_lines = []
            continue
        if current_key is not None:
            current_lines.append(stripped)
            continue
        if "train_command" not in parsed and _looks_like_shell_command(stripped):
            parsed["train_command"] = _clean_context_value(stripped)
    flush()
    _repair_framework_repo_root(parsed, text)
    return parsed


def _normalize_remote_repo_root(value: str) -> str:
    root = str(value or "").strip()
    if not root:
        return root
    if root.startswith("home/"):
        root = "/" + root
    duplicate_suggestion = _duplicate_tail_path_suggestion(root)
    if duplicate_suggestion is not None:
        raise ValueError(
            "remote_repo_root appears duplicated; "
            f"got {root!r}, did you mean {duplicate_suggestion!r}?"
        )
    if root.startswith("~/") or root.startswith("$HOME/"):
        return root
    if not root.startswith("/"):
        raise ValueError(
            "remote_repo_root must be an absolute remote path such as "
            f"'/home/user/code/repo'; got {root!r}"
        )
    return root


def _duplicate_tail_path_suggestion(root: str) -> str | None:
    parts = [part for part in Path(root).parts if part not in {"/", ""}]
    max_width = min(4, len(parts) // 2)
    for width in range(2, max_width + 1):
        if parts[-width:] == parts[-2 * width : -width]:
            prefix = "/" if root.startswith("/") else ""
            return prefix + "/".join(parts[:-width])
    return None


def _mirror_remote_training_repo(*, ssh_host: str, remote_repo_root: str, destination: Path) -> Path:
    rsync = shutil.which("rsync")
    if not rsync:
        raise FileNotFoundError("rsync command not found; install rsync for remote discovery")
    destination.mkdir(parents=True, exist_ok=True)
    remote_repo_root = _normalize_remote_repo_root(remote_repo_root)
    source = f"{ssh_host}:{remote_repo_root.rstrip('/')}/"
    exclude_patterns = (
        ".git/",
        "__pycache__/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".venv/",
        "node_modules/",
        "wandb/",
        "runs/",
        "logs/",
        "log/",
        "checkpoints/",
        "checkpoint*/",
        "outputs/",
        "output/",
        "data/",
        "raw_data/",
        "datasets_origin/",
        "datasets_generated/",
        "weights/",
        "weights*/",
        "act/logs/",
        "act/datasets_origin/",
        "act/weights*/",
        "*.ckpt",
        "*.pt",
        "*.pth",
        "*.safetensors",
        "*.hdf5",
        "*.h5",
        "*.mp4",
        "*.avi",
        "*.mkv",
        "*.bag",
        "*.db3",
        "*.npy",
        "*.npz",
    )
    cmd = [rsync, "-az", "--delete", "--prune-empty-dirs"]
    for pattern in exclude_patterns:
        cmd.extend(["--exclude", pattern])
    cmd.extend([source, str(destination).rstrip("/") + "/"])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        output_tail = "\n".join((proc.stdout or "").splitlines()[-30:])
        raise RuntimeError(
            "remote repo mirror failed: "
            f"rsync exited with {proc.returncode}; source={source!r}; "
            f"output={output_tail or '(no rsync output)'}"
        )
    _write_json_atomic(
        destination.parent / "remote_repo_snapshot.json",
        {
            "schema_version": "RoboLineage.remote_training_repo_snapshot.v1",
            "ssh_host": ssh_host,
            "remote_repo_root": remote_repo_root,
            "local_snapshot": str(destination),
            "updated_at": _now_iso(),
        },
    )
    return destination


def _repair_framework_repo_root(parsed: dict[str, str], text: str) -> None:
    repo_root = parsed.get("repo_root")
    if not repo_root:
        return
    root = Path(repo_root).expanduser()
    if not root.exists():
        return

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        label, _ = _split_context_label(stripped)
        if label or _looks_like_shell_command(stripped):
            continue
        candidate = (root / stripped).expanduser()
        if _looks_like_training_repo(candidate):
            parsed["repo_root"] = str(candidate)
            return

    train_command = parsed.get("train_command")
    if not train_command:
        return
    for part in shlex.split(train_command):
        path = Path(part).expanduser()
        if not path.is_absolute() or not path.exists():
            continue
        for parent in path.parents:
            try:
                parent.relative_to(root)
            except ValueError:
                break
            if _looks_like_training_repo(parent):
                parsed["repo_root"] = str(parent)
                return


def _looks_like_training_repo(path: Path) -> bool:
    return path.is_dir() and any(
        (path / marker).exists()
        for marker in ("README.md", "readme.md", ".git", "pyproject.toml", "setup.py", "tools", "act")
    )


def _split_context_label(line: str) -> tuple[str | None, str]:
    for sep in (":", "="):
        if sep not in line:
            continue
        label, value = line.split(sep, 1)
        label_norm = _normalize_context_label(label)
        if label_norm in _CONTEXT_LABELS:
            return label_norm, value.strip()
    return None, line


def _normalize_context_label(value: str) -> str:
    value = value.strip().strip("-*#`").lower()
    value = value.replace("_", " ").replace("-", " ")
    return " ".join(value.split())


def _clean_context_value(value: str) -> str:
    value = value.strip().strip("`")
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1].strip()
    return value


def _looks_like_shell_command(value: str) -> bool:
    first = value.split(maxsplit=1)[0] if value.split() else ""
    return first in {"python", "python3", "conda", "mamba", "micromamba", "bash", "sh"} or first.startswith("./")


def _maybe_prefix_conda_run(command: str, conda_env: str) -> str:
    parts = shlex.split(command)
    if not parts:
        return command
    first = parts[0]
    if first in {"conda", "mamba", "micromamba"} or "conda activate" in command:
        return command
    return f"conda run -n {shlex.quote(conda_env)} {command}"


def _normalize_script_command(command: str) -> str:
    parts = shlex.split(command)
    if not parts:
        return command
    first = parts[0]
    if first.endswith(".py") and first not in {"python", "python3"}:
        return "python " + command
    if first.endswith(".sh") and first not in {"bash", "sh"}:
        return "bash " + command
    return command


def _normalize_remote_repo_command(command: Any, remote_repo_root: str, local_repo_root: Path) -> str:
    parts = shlex.split(str(command))
    if not parts:
        return str(command)
    normalized: list[str] = []
    remote_root = str(remote_repo_root).rstrip("/")
    for part in parts:
        replacement = part
        if part.startswith("/") and not part.startswith(("{", "/tmp/")):
            candidate: str | None = None
            if remote_root and part == remote_root:
                candidate = "."
            elif remote_root and part.startswith(remote_root + "/"):
                candidate = part[len(remote_root) + 1 :]
            else:
                stripped = part.lstrip("/")
                if (local_repo_root / stripped).exists():
                    candidate = stripped
            if candidate:
                replacement = candidate
        normalized.append(replacement)
    return shlex.join(normalized)


def _context_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _raw_artifacts_status(rollout_dir: Path) -> dict[str, Any]:
    raw_dir = rollout_dir / "raw"
    manifest = raw_dir / "raw_manifest.json"
    bag_dir = raw_dir / "rosbag2"
    if manifest.exists():
        payload = _read_json(manifest) or {}
        configured_bag = payload.get("bag_dir")
        if configured_bag:
            bag_dir = Path(str(configured_bag))
    missing: list[str] = []
    if not manifest.exists():
        missing.append("raw/raw_manifest.json")
    bag_present = bag_dir.exists() and any(bag_dir.iterdir())
    if not bag_present:
        missing.append("raw/rosbag2/*")
    return {
        "present": not missing,
        "raw_format": "rosbag2",
        "raw_dir": str(raw_dir),
        "manifest_path": str(manifest),
        "bag_dir": str(bag_dir),
        "bag_present": bag_present,
        "missing": missing,
    }


def _post_review_complete(rollout_dir: Path) -> bool:
    return (
        (rollout_dir / "rollout_summary.json").exists()
        and (rollout_dir / "dataset_admission.json").exists()
    )


def _eval_review_complete(rollout_dir: Path) -> bool:
    return (rollout_dir / "policy_evaluation.json").exists()


def _read_task_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _task_config_metadata_for_path(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    candidate_indexes = [
        resolved.parent / "task_config_index.json",
        resolved.parent / "task_configs" / "task_config_index.json",
        resolved.parent.parent / "task_config_index.json",
    ]
    try:
        resolved_str = str(resolved.resolve())
    except OSError:
        resolved_str = str(resolved)
    for index_path in candidate_indexes:
        index = _read_json(index_path)
        if not index:
            continue
        for entry in index.get("entries") or []:
            paths = [
                entry.get("version_path"),
                entry.get("latest_path"),
                entry.get("compatibility_path"),
            ]
            normalized: list[str] = []
            for raw in paths:
                if not raw:
                    continue
                try:
                    normalized.append(str(Path(str(raw)).expanduser().resolve()))
                except OSError:
                    normalized.append(str(raw))
            if resolved_str in normalized:
                return {
                    **entry,
                    "index_path": str(index_path),
                }
    return {
        "version_id": None,
        "version_path": str(resolved),
        "created_at": None,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
    except (OSError, json.JSONDecodeError):
        return rows
    return rows


def _latest_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") == event_name:
            return event
    return {}


def _master_review_ref(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {
        "status": "completed",
        "state_path": str(getattr(result, "state_path", "")),
        "review_path": str(getattr(result, "review_path", "")),
        "understanding_path": str(getattr(result, "understanding_path", "")),
    }


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_json_atomic(path: Path, data: dict[str, Any]) -> Path:
    return _write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _payload_shape(payload: Any) -> list[int] | None:
    shape = getattr(payload, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except TypeError:
        return None


def _status_shape(status: dict[str, Any] | None) -> list[int] | None:
    if not status:
        return None
    shape = status.get("shape") or status.get("payload_shape")
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def _status_age_sec(status: dict[str, Any] | None) -> float | None:
    if not status or status.get("host_mono_ns") is None:
        return None
    try:
        age = (time.monotonic_ns() - int(status["host_mono_ns"])) / 1_000_000_000
    except (TypeError, ValueError):
        return None
    return round(max(0.0, age), 3)


def _rollout_memory_stats(run: _OnlineRolloutRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "rollout_id": run.rollout_id,
        "rollout_dir": str(run.rollout_dir),
        "thread_alive": run.thread.is_alive(),
        "capture_stopped_at": run.capture_stopped_at,
        "analysis_completed_at": run.analysis_completed_at,
        "last_error": run.last_error,
        "pipeline": _pipeline_memory_stats(run.pipeline),
    }


def _pipeline_memory_stats(pipeline: Any) -> dict[str, Any]:
    if pipeline is None:
        return {}
    stats: dict[str, Any] = {
        "dropped_arm_before_cam": getattr(pipeline, "dropped_arm_before_cam", 0),
        "dropped_vlm_windows": getattr(pipeline, "_dropped_vlm_windows", 0),
        "materialized_vsa_windows": getattr(pipeline, "_materialized_vsa_windows", 0),
        "materialized_keyframes": getattr(pipeline, "_materialized_keyframes", 0),
        "materialized_keyframe_bytes": getattr(
            pipeline,
            "_materialized_keyframe_bytes",
            0,
        ),
    }
    analysis_q = getattr(pipeline, "_analysis_queue", None)
    if analysis_q is not None:
        stats["analysis_queue_size"] = _queue_size(analysis_q)
        stats["analysis_queue_maxsize"] = getattr(analysis_q, "maxsize", None)
    completed_q = getattr(pipeline, "_completed_snapshots", None)
    if completed_q is not None:
        stats["completed_snapshots_queue_size"] = _queue_size(completed_q)

    frame_buffer = getattr(pipeline, "frame_buffer", None)
    if frame_buffer is not None:
        stats["frame_buffer_len"] = _safe_len(frame_buffer)
        stats["frame_buffer_capacity"] = getattr(frame_buffer, "capacity", None)
        dropped_count = getattr(frame_buffer, "dropped_count", None)
        if callable(dropped_count):
            try:
                stats["frame_buffer_dropped"] = dropped_count()
            except Exception:
                stats["frame_buffer_dropped"] = None
        stats["frame_buffer_payload_bytes"] = _frame_buffer_payload_bytes(frame_buffer)
    return stats


def _queue_size(q: Any) -> int | None:
    try:
        return int(q.qsize())
    except Exception:
        return None


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))
    except Exception:
        return None


def _frame_buffer_payload_bytes(frame_buffer: Any) -> int | None:
    frames = getattr(frame_buffer, "_frames", None)
    if frames is None:
        return None
    lock = getattr(frame_buffer, "_lock", None)

    def _sum_bytes() -> int:
        total = 0
        for rec in frames.values():
            total += _payload_nbytes(getattr(rec, "bgr", None))
        return total

    try:
        if lock is not None:
            with lock:
                return _sum_bytes()
        return _sum_bytes()
    except Exception:
        return None


def _payload_nbytes(payload: Any) -> int:
    nbytes = getattr(payload, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes)
        except (TypeError, ValueError):
            pass
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    return 0


def _current_rss_mb() -> float | None:
    status = Path("/proc/self/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024, 2)
    except OSError:
        pass

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(rss / (1024 * 1024), 2)
        return round(rss / 1024, 2)
    except Exception:
        return None


def _extract_ROBOLINEAGE_pose_vector(payload: Any) -> tuple[list[float], list[float], float]:
    values = payload.tolist() if hasattr(payload, "tolist") else list(payload)
    if len(values) < 27:
        raise ValueError(f"RoboLineage pose vector must contain >=27 values; got {len(values)}")
    eef_xyz = [round(float(values[i]), 6) for i in range(21, 24)]
    eef_rxyz = [round(float(values[i]), 6) for i in range(24, 27)]
    gripper = round(float(values[6]), 6)
    return eef_xyz, eef_rxyz, gripper


def _profile_gripper_close_rule(state: dict[str, Any]) -> dict[str, Any] | None:
    gripper = state.get("gripper")
    if isinstance(gripper, dict) and isinstance(gripper.get("close_rule"), dict):
        rule = gripper["close_rule"]
    elif isinstance(state.get("gripper_close_rule"), dict):
        rule = state["gripper_close_rule"]
    else:
        return None
    operator = str(rule.get("operator") or "").strip()
    if not operator or rule.get("value") is None:
        return None
    try:
        value = float(rule["value"])
    except (TypeError, ValueError):
        return None
    return {"operator": operator, "value": value}


def _apply_gripper_close_rule(value: float, rule: dict[str, Any]) -> str | None:
    operator = str(rule.get("operator") or "")
    threshold = float(rule["value"])
    if operator in {"<=", "le"}:
        closed = value <= threshold
    elif operator in {"<", "lt"}:
        closed = value < threshold
    elif operator in {">=", "ge"}:
        closed = value >= threshold
    elif operator in {">", "gt"}:
        closed = value > threshold
    elif operator in {"==", "eq"}:
        closed = value == threshold
    else:
        return None
    return "closed" if closed else "open"


def _resolve_child(root: Path, child: str) -> Path:
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / child).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise FileNotFoundError(f"path is outside root: {child}")
    if not candidate.exists():
        raise FileNotFoundError(f"path not found: {candidate}")
    return candidate


def _safe_slug(value: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug[:60] or "task"


def _dir_child_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len([item for item in path.iterdir() if item.is_dir() and not item.name.startswith(".")])


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def _read_task_description(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("task_description:"):
                return line.split(":", 1)[1].strip().strip("'\"") or None
    except OSError:
        return None
    return None


def _dedupe_str(values: list[str]) -> list[str]:
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
