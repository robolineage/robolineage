from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, Optional

from robolineage_contracts.agents import SnapshotAssessment
from robolineage_data_source.config.schema import ArmTopicSpec

from ..agent import PreparedInference, VisualSnapshotAgent
from ..exceptions import VLMInferenceError
from ..types import ActionEvent, ActionGuidedWindow, TaskConfig
from ..vlm_runner import BaseVLMRunner
from .frame_ring_buffer import FrameRingBuffer
from .pending_window_queue import PendingWindowQueue
from .streaming_event_detector import StreamingEventDetector
from .streaming_signal_builder import StreamingSignalBuilder
from .streaming_window_builder import StreamingWindowBuilder
from .types import RealtimeActionRecord, RealtimeFrameRecord
from .window_scheduler import WindowScheduler


class StreamingRuntimePipeline:
    """In-process realtime VSA pipeline fed by RealtimeFrameRecord/RealtimeActionRecord.

    The realtime dispatch path never runs VLM. It only updates buffers,
    detects events, builds ActionGuidedWindow objects, and enqueues them. A
    single analysis worker consumes those windows linearly, so each prompt is
    built after the previous VLM result has updated rollout memory.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        vlm_runner: BaseVLMRunner,
        rollout_dir: Path,
        output_jsonl: Path,
        context_frames: int = 15,
        max_keyframes: int = 3,
        ring_capacity: int = 120,
        still_min_frames: int = 15,
        heartbeat_interval: float = 5.0,
        periodic_interval_sec: float = 2.0,
        merge_window_sec: float = 1.0,
        final_settle_sec: float = 1.0,
        max_vlm_windows_per_rollout: int | None = None,
        gripper_close_threshold: float = -1.0,
        still_threshold: float = 3e-4,
        rotation_weight: float = 0.2,
        smoothing_window: int = 10,
        motion_resume_threshold: float = 8e-4,
        min_same_event_interval: float = 3.0,
        vlm_workers: int = 1,
        max_pending_vlm: int = 16,
        strong_prior_margin: float = 0.35,
        prior_sticky_frames: int = 2,
        on_snapshot: Callable[[SnapshotAssessment], None] | None = None,
    ):
        self.rollout_dir = Path(rollout_dir)
        self.frame_buffer = FrameRingBuffer(capacity=ring_capacity)
        self.signal_builder = StreamingSignalBuilder(
            gripper_close_threshold=gripper_close_threshold,
            rotation_weight=rotation_weight,
            smoothing_window=smoothing_window,
            still_threshold=still_threshold,
        )
        self.event_detector = StreamingEventDetector(
            still_min_frames=still_min_frames,
            heartbeat_interval=heartbeat_interval,
            periodic_interval_sec=periodic_interval_sec,
            motion_resume_threshold=motion_resume_threshold,
            min_same_event_interval=min_same_event_interval,
        )
        self.pending = PendingWindowQueue(context_frames=context_frames)
        self.window_scheduler = WindowScheduler(
            context_frames=context_frames,
            merge_window_sec=merge_window_sec,
        )
        self.window_builder = StreamingWindowBuilder(
            self.frame_buffer,
            context_frames=context_frames,
            max_keyframes=max_keyframes,
        )
        self.agent = VisualSnapshotAgent(
            task_config=task_config,
            vlm_runner=vlm_runner,
            rollout_dir=self.rollout_dir,
            output_jsonl=output_jsonl,
            strong_prior_margin=strong_prior_margin,
            prior_sticky_frames=prior_sticky_frames,
        )
        self.rollout_id = self.rollout_dir.name
        self.on_snapshot = on_snapshot
        self._sequence_started = False
        self.dropped_arm_before_cam: int = 0
        self._keyframe_root = self.rollout_dir / "vsa_windows"
        self._keyframe_manifest_path = self._keyframe_root / "manifest.jsonl"
        self._keyframe_manifest_lock = threading.Lock()
        self._window_sequence = 0
        self._materialized_vsa_windows = 0
        self._materialized_keyframes = 0
        self._materialized_keyframe_bytes = 0
        self._final_observation_enqueued = False
        self._final_settle_sec = max(0.0, float(final_settle_sec))
        self._last_release_event: ActionEvent | None = None
        self._max_vlm_windows_per_rollout = (
            max(1, int(max_vlm_windows_per_rollout))
            if max_vlm_windows_per_rollout is not None
            else None
        )
        self._enqueued_vlm_windows = 0
        self._enqueued_non_terminal_vlm_windows = 0
        self._enqueued_terminal_vlm_windows = 0
        self._raw_trigger_count = 0
        self._scheduled_window_count = 0
        self._covered_coverage_event_count = 0
        self._gripper_burst_count = 0
        self._contact_transition_count = 0
        self._final_observation_reasons: dict[str, int] = {}
        self._vlm_call_count = 0

        self._configured_vlm_workers = max(1, vlm_workers)
        if self._configured_vlm_workers > 1:
            _LOG.warning(
                "Online VSA ignores vlm_workers=%s and runs VLM linearly so each "
                "prompt sees the previous phase decision.",
                self._configured_vlm_workers,
            )
        self._max_pending_vlm = max(1, max_pending_vlm)
        self._dropped_vlm_windows: int = 0
        self._analysis_sentinel = object()
        self._analysis_queue: queue.Queue[ActionGuidedWindow | object] = queue.Queue(
            maxsize=self._max_pending_vlm
        )
        self._completed_snapshots: queue.Queue[SnapshotAssessment] = queue.Queue()
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop,
            name=f"VSAAnalysis.{self.rollout_id}",
            daemon=True,
        )
        self._closed = False
        self._analysis_thread.start()

    def process_frame(self, frame: RealtimeFrameRecord) -> None:
        self.frame_buffer.put(frame)

    def process_action(self, action: RealtimeActionRecord, flush: bool = True) -> list[SnapshotAssessment]:
        _, signal = self.signal_builder.feed(action)
        if not self._sequence_started:
            self._sequence_started = True
            self.pending.enqueue(ActionEvent(
                event_type="sequence_start",
                anchor_frame=action.frame_index,
                timestamp_sec=action.host_mono_ns / 1_000_000_000,
                confidence=1.0,
            ))
            self._raw_trigger_count += 1
        events = self.event_detector.feed(signal)
        for event in events:
            self.pending.enqueue(event)
        self._raw_trigger_count += len(events)
        self._update_release_settle(signal, events)
        return self.process_ready_windows()

    def process_ready_windows(self, *, force: bool = False) -> list[SnapshotAssessment]:
        """Release ready windows into the analysis queue.

        Implementation note.
        Implementation note.
        """
        latest = self.frame_buffer.latest_frame_index()

        events = self.pending.pop_all() if force else self.pending.pop_ready(latest)
        watermark_timestamp = (
            self.signal_builder.signals[-1].timestamp_sec
            if self.signal_builder.signals
            else None
        )
        events = self.window_scheduler.schedule(
            events,
            watermark_timestamp=watermark_timestamp,
            force=force,
        )
        self._record_scheduled_events(events)
        for event in events:
            window = self.window_builder.build(
                event,
                self.signal_builder.records,
                self.signal_builder.signals,
                rollout_id=self.rollout_id,
            )
            self._enqueue_window(window)

        return self._drain_completed_snapshots()

    def drain(self) -> list[SnapshotAssessment]:
        """Flush queued windows and write every pending VLM result.

        Called when the operator stops a rollout. It stops waiting for more
        context frames and processes every pending event in the same linear
        VLM/apply order used during normal streaming.
        """
        snapshots = self.process_ready_windows(force=True)
        self._enqueue_final_observation_window()
        released_frames = self.frame_buffer.clear()
        if released_frames:
            _LOG.info(
                "Online VSA released frame ring buffer after keyframe materialization "
                "(rollout_id=%s released_frames=%d materialized_windows=%d "
                "materialized_keyframes=%d)",
                self.rollout_id,
                released_frames,
                self._materialized_vsa_windows,
                self._materialized_keyframes,
            )
        self._analysis_queue.join()
        snapshots.extend(self._drain_completed_snapshots())
        return snapshots

    def _enqueue_final_observation_window(self) -> None:
        if self._final_observation_enqueued:
            return
        latest_frame = self.frame_buffer.latest_frame_index()
        if latest_frame is None or not self.signal_builder.records:
            return
        record_frames = {record.frame_index for record in self.signal_builder.records}
        anchor_frame = latest_frame if latest_frame in record_frames else self.signal_builder.records[-1].frame_index
        record_by_frame = {record.frame_index: record for record in self.signal_builder.records}
        anchor_record = record_by_frame.get(anchor_frame)
        if anchor_record is None:
            return
        event = ActionEvent(
            event_type="final_observation",
            anchor_frame=anchor_frame,
            timestamp_sec=anchor_record.timestamp_sec,
            confidence=1.0,
            details={"reason": "rollout_stop"},
        )
        self._raw_trigger_count += 1
        window = self.window_builder.build(
            event,
            self.signal_builder.records,
            self.signal_builder.signals,
            rollout_id=self.rollout_id,
        )
        self._final_observation_enqueued = True
        self._record_scheduled_events([event])
        self._enqueue_window(window)

    def _update_release_settle(self, signal, events: list[ActionEvent]) -> None:
        if self._final_observation_enqueued:
            return
        for event in events:
            if event.event_type == "gripper_open":
                self._last_release_event = event

        release = self._last_release_event
        if release is None or self._final_settle_sec <= 0:
            return
        if signal.frame_index <= release.anchor_frame:
            return

        settle_after_sec = signal.timestamp_sec - release.timestamp_sec
        if settle_after_sec < 0:
            return
        if signal.gripper_state != "open":
            self._last_release_event = None
            return
        if settle_after_sec < self._final_settle_sec:
            return
        if signal.motion_energy > self.signal_builder.still_threshold:
            return

        self.pending.enqueue(
            ActionEvent(
                event_type="final_observation",
                anchor_frame=signal.frame_index,
                timestamp_sec=signal.timestamp_sec,
                confidence=1.0,
                details={
                    "reason": "release_settle",
                    "release_frame": release.anchor_frame,
                    "release_timestamp_sec": release.timestamp_sec,
                    "settle_after_sec": round(settle_after_sec, 4),
                },
            )
        )
        self._raw_trigger_count += 1
        self._final_observation_enqueued = True
        self._last_release_event = None

    def _record_scheduled_events(self, events: list[ActionEvent]) -> None:
        self._scheduled_window_count += len(events)
        for event in events:
            covered = event.details.get("covered_events")
            if isinstance(covered, list):
                self._covered_coverage_event_count += len(covered)
            if event.event_type == "gripper_burst":
                self._gripper_burst_count += 1
            elif event.event_type == "contact_transition":
                self._contact_transition_count += 1

    def _enqueue_window(self, window: ActionGuidedWindow) -> None:
        if self._rollout_window_cap_reached(window):
            self._drop_window_for_cap(window)
            return
        if self._analysis_queue.full():
            self._dropped_vlm_windows += 1
            self._discard_window_images(window)
            _LOG.warning(
                "Online VSA analysis queue full; dropping event %s at frame %s "
                "(queue_size=%d)",
                window.event_type,
                window.anchor_frame_id,
                self._analysis_queue.qsize(),
            )
            return
        self._materialize_window_keyframes(window)
        try:
            self._analysis_queue.put_nowait(window)
            self._record_enqueued_window(window)
        except queue.Full:
            self._dropped_vlm_windows += 1
            self._discard_window_images(window)
            _LOG.warning(
                "Online VSA analysis queue full; dropping event %s at frame %s "
                "(queue_size=%d)",
                window.event_type,
                window.anchor_frame_id,
                self._analysis_queue.qsize(),
            )

    def _rollout_window_cap_reached(self, window: ActionGuidedWindow) -> bool:
        cap = self._max_vlm_windows_per_rollout
        if cap is None:
            return False
        if window.event_type == "final_observation":
            return self._enqueued_terminal_vlm_windows >= 1 or self._enqueued_vlm_windows >= cap
        non_terminal_cap = max(0, cap - 1)
        return self._enqueued_non_terminal_vlm_windows >= non_terminal_cap

    def _drop_window_for_cap(self, window: ActionGuidedWindow) -> None:
        self._dropped_vlm_windows += 1
        self._discard_window_images(window)
        _LOG.warning(
            "Online VSA rollout window cap reached; dropping event %s at frame %s "
            "(max_vlm_windows_per_rollout=%d reserved_final_observation=%s)",
            window.event_type,
            window.anchor_frame_id,
            self._max_vlm_windows_per_rollout,
            window.event_type != "final_observation",
        )

    def _record_enqueued_window(self, window: ActionGuidedWindow) -> None:
        self._enqueued_vlm_windows += 1
        if window.event_type == "final_observation":
            self._enqueued_terminal_vlm_windows += 1
            reason = str(window.event_details.get("reason") or "unknown")
            self._final_observation_reasons[reason] = self._final_observation_reasons.get(reason, 0) + 1
        else:
            self._enqueued_non_terminal_vlm_windows += 1

    def _materialize_window_keyframes(self, window: ActionGuidedWindow) -> None:
        if window.keyframe_image_paths:
            self._discard_window_images(window)
            return
        images = list(window.color_frames)
        if not images:
            return

        window_id = self._next_window_id(window)
        window_dir = self._keyframe_root / window_id
        image_paths: list[Path] = []
        relative_paths: list[str] = []

        try:
            import cv2
            import numpy as np

            window_dir.mkdir(parents=True, exist_ok=True)
            for idx, image in enumerate(images):
                if image is None:
                    continue
                frame_id = (
                    window.keyframe_ids[idx]
                    if idx < len(window.keyframe_ids)
                    else window.anchor_frame_id
                )
                final_path = window_dir / f"kf_{idx:02d}_frame_{frame_id}.png"
                tmp_path = final_path.with_suffix(".tmp.png")
                array = np.ascontiguousarray(image)
                if array.ndim == 3 and array.shape[2] == 3:
                    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(
                    str(tmp_path),
                    array,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
                if not ok:
                    raise RuntimeError(f"cv2.imwrite failed for {tmp_path}")
                tmp_path.replace(final_path)
                image_paths.append(final_path)
                relative_paths.append(str(final_path.relative_to(self.rollout_dir)))

            if not image_paths:
                return

            window.keyframe_window_id = window_id
            window.keyframe_image_paths = [str(path) for path in image_paths]
            self._materialized_vsa_windows += 1
            self._materialized_keyframes += len(image_paths)
            self._materialized_keyframe_bytes += sum(
                path.stat().st_size for path in image_paths if path.exists()
            )
            self._append_keyframe_manifest(window, relative_paths)
            self._discard_window_images(window)
        except Exception:
            _LOG.exception(
                "Failed to materialize online VSA keyframes; keeping in-memory "
                "frames for rollout_id=%s event=%s anchor_frame=%s",
                self.rollout_id,
                window.event_type,
                window.anchor_frame_id,
            )

    def _append_keyframe_manifest(
        self,
        window: ActionGuidedWindow,
        relative_paths: list[str],
    ) -> None:
        entry = {
            "schema_version": "RoboLineage.vsa_window_manifest.v1",
            "window_id": window.keyframe_window_id,
            "rollout_id": window.rollout_id,
            "event_type": window.event_type,
            "anchor_frame_id": window.anchor_frame_id,
            "end_frame_id": window.end_frame_id,
            "end_timestamp": window.end_timestamp,
            "camera_name": window.camera_name,
            "keyframe_ids": list(window.keyframe_ids),
            "image_paths": relative_paths,
            "image_format": "png",
            "source_video_name": window.source_video_name,
            "action_summary": window.action_summary,
            "event_details": window.event_details,
        }
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        with self._keyframe_manifest_lock:
            self._keyframe_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with self._keyframe_manifest_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _next_window_id(self, window: ActionGuidedWindow) -> str:
        self._window_sequence += 1
        event_type = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_"
            for c in window.event_type
        )
        return f"{self._window_sequence:06d}_{window.anchor_frame_id:06d}_{event_type}"

    @staticmethod
    def _discard_window_images(window: ActionGuidedWindow) -> None:
        window.color_frames = []
        window.depth_frames = []

    def _analysis_loop(self) -> None:
        while True:
            item = self._analysis_queue.get()
            try:
                if item is self._analysis_sentinel:
                    return
                window = item
                assert isinstance(window, ActionGuidedWindow)
                prepared = self.agent.prepare(window)
                snapshot = self._run_prepared(prepared)
                if snapshot is not None:
                    self._completed_snapshots.put(snapshot)
            except Exception:
                _LOG.exception("Online VSA analysis worker failed")
            finally:
                self._analysis_queue.task_done()

    def _drain_completed_snapshots(self) -> list[SnapshotAssessment]:
        snapshots: list[SnapshotAssessment] = []
        while True:
            try:
                snapshots.append(self._completed_snapshots.get_nowait())
            except queue.Empty:
                break
        return snapshots

    def _run_prepared(self, prepared: PreparedInference) -> SnapshotAssessment | None:
        event_type = getattr(prepared.window, "event_type", "window")
        if prepared.fallback_parsed is not None:
            raw_response = prepared.fallback_raw_response or "skipped:no_images"
        else:
            t0 = time.perf_counter()
            try:
                self._vlm_call_count += 1
                raw_response = self.agent.vlm_runner.run(prepared.prompt, prepared.images)
            except VLMInferenceError as exc:
                _LOG.warning(
                    "VLM inference failed for event %s, using fallback: %s",
                    event_type,
                    exc,
                )
                raw_response = prepared.fallback_raw_response or "error:vlm_inference_failed"
            prepared.step_timings["vlm"] = time.perf_counter() - t0

        try:
            snapshot = self.agent.apply(prepared, raw_response)
        except VLMInferenceError as exc:
            _LOG.warning(
                "VLM response unparseable for event %s, skipping snapshot: %s",
                event_type,
                exc,
            )
            return None
        self.agent.write_assessment(snapshot)
        if self.on_snapshot is not None:
            self.on_snapshot(snapshot)
        return snapshot

    def close(self) -> None:
        if self._closed:
            return
        self._analysis_queue.join()
        self.frame_buffer.clear()
        self._analysis_queue.put(self._analysis_sentinel)
        self._analysis_thread.join(timeout=5.0)
        if self._analysis_thread.is_alive():
            _LOG.warning("Online VSA analysis worker did not stop within timeout")
        self.agent.close()
        self._closed = True

    def metrics(self) -> dict[str, object]:
        return {
            "raw_trigger_count": self._raw_trigger_count,
            "scheduled_window_count": self._scheduled_window_count,
            "covered_coverage_event_count": self._covered_coverage_event_count,
            "gripper_burst_count": self._gripper_burst_count,
            "contact_transition_count": self._contact_transition_count,
            "final_observation_reasons": dict(self._final_observation_reasons),
            "vlm_call_count": self._vlm_call_count,
            "enqueued_vlm_windows": self._enqueued_vlm_windows,
            "enqueued_non_terminal_vlm_windows": self._enqueued_non_terminal_vlm_windows,
            "enqueued_terminal_vlm_windows": self._enqueued_terminal_vlm_windows,
            "dropped_vlm_windows": self._dropped_vlm_windows,
            "materialized_vsa_windows": self._materialized_vsa_windows,
            "materialized_keyframes": self._materialized_keyframes,
            "materialized_keyframe_bytes": self._materialized_keyframe_bytes,
        }


def run_action_guided_stream(
    frame_messages: Iterable[RealtimeFrameRecord],
    action_messages: Iterable[RealtimeActionRecord],
    task_config: TaskConfig,
    vlm_runner: BaseVLMRunner,
    rollout_dir: Path,
    output_jsonl: Path,
    **kwargs,
) -> list[SnapshotAssessment]:
    """Deterministic helper for tests/rehearsal with pre-collected message iterables."""
    pipeline = StreamingRuntimePipeline(
        task_config=task_config,
        vlm_runner=vlm_runner,
        rollout_dir=rollout_dir,
        output_jsonl=output_jsonl,
        **kwargs,
    )
    snapshots: list[SnapshotAssessment] = []
    try:
        frame_iter = iter(frame_messages)
        for action in action_messages:
            for frame in frame_iter:
                pipeline.process_frame(frame)
                if frame.frame_index >= action.frame_index:
                    break
            snapshots.extend(pipeline.process_action(action))
        snapshots.extend(pipeline.drain())
    finally:
        pipeline.close()
    return snapshots


_LOG = logging.getLogger(__name__)


def run_ros_topic_stream(
    *,
    camera_topic: str,
    arm_topic: str,
    arm_spec: ArmTopicSpec,
    pipeline: StreamingRuntimePipeline,
    ros_domain_id: int = 0,
    max_events: Optional[int] = None,
    idle_timeout: float = 10.0,
    poll_interval: float = 0.05,
    stop_event: Event | None = None,
) -> list[SnapshotAssessment]:
    """Block until pipeline emits ``max_events`` SnapshotAssessments or idles.

    Threading model (post-review): two background threads inside
    Ros2TopicConsumer subscribes to ROS2 and pushes decoded
    RealtimeFrameRecord / RealtimeArmSample into thread-safe ``queue.Queue``
    instances. This main loop drains both queues and dispatches via
    ``pipeline.process_frame`` / ``pipeline.process_action``, so all
    signal_builder / event_detector / pending state mutation happens on a
    single thread. No cross-thread races on the inner pipeline objects.

    frame_index alignment: each arm sample is bound to the latest camera
    frame_index this main loop has dispatched. Arm samples that arrive
    before any camera sample are dropped silently — counted in
    ``pipeline.dropped_arm_before_cam`` (Phase 5+) for observability.
    """
    from .ros2_consumer import Ros2TopicConsumer

    consumer = Ros2TopicConsumer(
        camera_topic=camera_topic,
        arm_topic=arm_topic,
        arm_spec=arm_spec,
        ros_domain_id=ros_domain_id,
    )
    snapshots: list[SnapshotAssessment] = []
    last_activity = time.monotonic()
    latest_frame_index: Optional[int] = None
    # Phase 5 T3 — observability: how many arm samples we silently dropped
    # because the camera stream had not started yet. Surfaced on the pipeline
    # so the /health endpoint or callers can read it back. Logged at DEBUG
    # rate-limited to one line per 100 drops to avoid log flood during
    # startup races.
    pipeline.dropped_arm_before_cam = 0
    consumer.start()
    _LOG.info(
        "run_ros_topic_stream started (cam=%s arm=%s idle_timeout=%.1fs)",
        camera_topic, arm_topic, idle_timeout,
    )
    frames_seen = 0
    arms_seen = 0
    def _drain_consumer_queues() -> bool:
        nonlocal latest_frame_index, frames_seen, arms_seen, snapshots
        saw_data = False

        # 1. Drain frames first so latest_frame_index is up to date for
        #    any subsequent arm samples.
        frame = consumer.pop_frame()
        while frame is not None:
            pipeline.process_frame(frame)
            latest_frame_index = frame.frame_index
            saw_data = True
            frames_seen += 1
            if frames_seen == 1:
                _LOG.info("run_ros_topic_stream: first camera frame received (frame_index=%s)", latest_frame_index)
            frame = consumer.pop_frame()

        # 2. Drain arm samples; bind each to the current latest_frame_index.
        arm = consumer.pop_arm()
        while arm is not None:
            saw_data = True
            if latest_frame_index is not None:
                rec = RealtimeActionRecord(
                    frame_index=latest_frame_index,
                    host_mono_ns=arm.host_mono_ns,
                    eef_xyz=arm.eef_xyz,
                    eef_rxyz=arm.eef_rxyz,
                    gripper=arm.gripper,
                )
                snapshots.extend(pipeline.process_action(rec))
                arms_seen += 1
                if arms_seen == 1:
                    _LOG.info("run_ros_topic_stream: first arm sample received")
            else:
                pipeline.dropped_arm_before_cam += 1
                # Rate-limited DEBUG log: one line per 100 drops.
                if pipeline.dropped_arm_before_cam % 100 == 1:
                    _LOG.debug(
                        "run_ros_topic_stream dropped arm sample (no camera frame "
                        "seen yet); cumulative drops = %d",
                        pipeline.dropped_arm_before_cam,
                    )
            arm = consumer.pop_arm()

        # 3. Even when no arm sample arrived, ready windows may have
        #    completed because frames extended ring buffer coverage.
        new = pipeline.process_ready_windows()
        if new:
            snapshots.extend(new)
            saw_data = True

        return saw_data

    try:
        while max_events is None or len(snapshots) < max_events:
            if stop_event is not None and stop_event.is_set():
                _LOG.info("run_ros_topic_stream: stop requested; draining queued samples")
                break
            saw_data = False

            saw_data = _drain_consumer_queues()

            if saw_data:
                last_activity = time.monotonic()
            elif time.monotonic() - last_activity > idle_timeout:
                _LOG.warning(
                    "run_ros_topic_stream: idle_timeout=%.1fs reached with no data "
                    "(frames_seen=%d arms_seen=%d); exiting",
                    idle_timeout, frames_seen, arms_seen,
                )
                break
            else:
                time.sleep(poll_interval)
    finally:
        consumer.stop()
        # After unsubscribing from ROS2 topics, drain samples already copied
        # into the consumer queues, then flush pending windows and VLM futures.
        while _drain_consumer_queues():
            pass
        snapshots.extend(pipeline.drain())
        pipeline.close()
    _LOG.info(
        "run_ros_topic_stream finished (frames=%d arms=%d snapshots=%d)",
        frames_seen, arms_seen, len(snapshots),
    )
    return snapshots
