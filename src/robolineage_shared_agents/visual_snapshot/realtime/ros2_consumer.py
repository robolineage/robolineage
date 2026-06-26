"""ROS2 topic consumer for realtime VSA."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

import numpy as np

from robolineage_data_source.config.schema import ArmTopicSpec
from robolineage_data_source.adapters.ros2_profile import _robot_state_to_27_vec

from .types import RealtimeArmSample, RealtimeFrameRecord

_LOG = logging.getLogger(__name__)


class Ros2TopicConsumer:
    """Subscribe camera + robot-state ROS2 topics and expose thread-safe queues."""

    def __init__(
        self,
        *,
        camera_topic: str,
        arm_topic: str,
        arm_spec: ArmTopicSpec,
        ros_domain_id: int = 0,
        max_queue: int = 64,
    ) -> None:
        if max_queue <= 0:
            raise ValueError(f"max_queue must be > 0, got {max_queue}")
        self._camera_topic = camera_topic
        self._arm_topic = arm_topic
        self._arm_spec = arm_spec
        self._ros_domain_id = int(ros_domain_id)
        self._frame_q: queue.Queue[RealtimeFrameRecord] = queue.Queue(maxsize=max_queue)
        self._arm_q: queue.Queue[RealtimeArmSample] = queue.Queue(maxsize=max_queue)
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._started = False
        self._owns_rclpy_context = False
        self._stats_lock = threading.Lock()
        self._dropped_frames = 0
        self._dropped_arm_samples = 0
        self._frame_index = 0

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Ros2TopicConsumer already started")
        try:
            import rclpy
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import CompressedImage
            from rosidl_runtime_py.utilities import get_message
        except ImportError as exc:
            raise RuntimeError("ROS2 Python packages are required for realtime VSA ROS topic mode") from exc

        if _rclpy_context_ok(rclpy):
            _LOG.info("Ros2TopicConsumer reusing active default rclpy context")
            self._owns_rclpy_context = False
        else:
            rclpy.init(args=None, domain_id=self._ros_domain_id)
            self._owns_rclpy_context = True
        try:
            callback_group = ReentrantCallbackGroup()
            node = Node("ROBOLINEAGE_vsa_ros2_topic_consumer")
            node.create_subscription(
                CompressedImage,
                self._camera_topic,
                self._on_camera,
                10,
                callback_group=callback_group,
            )
            if not self._arm_spec.msg_type:
                raise RuntimeError("active robot state stream must define msg_type for realtime VSA")
            arm_cls = get_message(self._arm_spec.msg_type)
            node.create_subscription(
                arm_cls,
                self._arm_topic,
                self._on_arm,
                10,
                callback_group=callback_group,
            )
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(node)
            self._node = node
            self._executor = executor
            self._spin_thread = threading.Thread(
                target=executor.spin,
                name="Ros2TopicConsumer.spin",
                daemon=True,
            )
            self._spin_thread.start()
            self._started = True
        except Exception:
            if self._node is not None:
                self._node.destroy_node()
            if self._owns_rclpy_context:
                rclpy.shutdown()
                self._owns_rclpy_context = False
            self._node = None
            self._executor = None
            raise

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._executor is not None:
                self._executor.shutdown(timeout_sec=2.0)
            if self._node is not None:
                self._node.destroy_node()
        finally:
            try:
                import rclpy

                if self._owns_rclpy_context:
                    rclpy.shutdown()
            except Exception:
                pass
            self._owns_rclpy_context = False
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=3.0)
                self._spin_thread = None
            self._executor = None
            self._node = None
            self._started = False

    def pop_frame(self, timeout: float = 0.0) -> Optional[RealtimeFrameRecord]:
        try:
            if timeout > 0:
                return self._frame_q.get(timeout=timeout)
            return self._frame_q.get_nowait()
        except queue.Empty:
            return None

    def pop_arm(self, timeout: float = 0.0) -> Optional[RealtimeArmSample]:
        try:
            if timeout > 0:
                return self._arm_q.get(timeout=timeout)
            return self._arm_q.get_nowait()
        except queue.Empty:
            return None

    def queue_stats(self) -> dict[str, int]:
        with self._stats_lock:
            dropped_frames = self._dropped_frames
            dropped_arm_samples = self._dropped_arm_samples
        return {
            "frame_queue_size": self._frame_q.qsize(),
            "arm_queue_size": self._arm_q.qsize(),
            "dropped_frames": dropped_frames,
            "dropped_arm_samples": dropped_arm_samples,
        }

    def _on_camera(self, msg: Any) -> None:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            _LOG.warning("failed to decode compressed camera frame from %s", self._camera_topic)
            return
        frame_index = self._frame_index
        self._frame_index += 1
        self._put_latest(
            self._frame_q,
            RealtimeFrameRecord(
                frame_index=frame_index,
                host_mono_ns=time.monotonic_ns(),
                bgr=bgr,
            ),
            kind="frame",
            label=f"frame(frame_index={frame_index})",
        )

    def _on_arm(self, msg: Any) -> None:
        vec = _robot_state_to_27_vec(msg, self._arm_spec)
        eef_xyz, eef_rxyz, gripper = _extract_pose(vec)
        self._put_latest(
            self._arm_q,
            RealtimeArmSample(
                host_mono_ns=time.monotonic_ns(),
                eef_xyz=eef_xyz,
                eef_rxyz=eef_rxyz,
                gripper=gripper,
            ),
            kind="arm",
            label="arm",
        )

    def _record_queue_drop(self, kind: str) -> int:
        with self._stats_lock:
            if kind == "frame":
                self._dropped_frames += 1
                return self._dropped_frames
            self._dropped_arm_samples += 1
            return self._dropped_arm_samples

    def _put_latest(
        self,
        q: queue.Queue,
        item: object,
        *,
        kind: str,
        label: str,
    ) -> None:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            q.get_nowait()
        except queue.Empty:
            pass
        else:
            dropped = self._record_queue_drop(kind)
            if dropped == 1 or dropped % 100 == 0:
                _LOG.warning("%s queue full; dropped oldest sample (total_dropped=%s)", label, dropped)

        try:
            q.put_nowait(item)
        except queue.Full:
            dropped = self._record_queue_drop(kind)
            if dropped == 1 or dropped % 100 == 0:
                _LOG.warning("%s queue still full; dropped newest sample (total_dropped=%s)", label, dropped)


def _rclpy_context_ok(rclpy: Any) -> bool:
    ok = getattr(rclpy, "ok", None)
    if not callable(ok):
        return False
    try:
        return bool(ok())
    except Exception:
        return False


def _extract_pose(
    vec: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    if len(values) < 27:
        raise ValueError(f"pose payload must contain >=27 values; got {len(values)}")
    eef_xyz = (_clean_float(values[21]), _clean_float(values[22]), _clean_float(values[23]))
    eef_rxyz = (_clean_float(values[24]), _clean_float(values[25]), _clean_float(values[26]))
    gripper = _clean_float(values[6])
    return eef_xyz, eef_rxyz, gripper


def _clean_float(value: object) -> float:
    return round(float(value), 6)
