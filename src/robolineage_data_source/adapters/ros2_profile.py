"""ROS2 robot-profile source supervisor.

The ROS2 dependencies are lazy-imported so this module remains importable on
macOS development machines. Only `start()` builds the rclpy node and spins it.

Robot profiles provide ``msg_type`` and field mappings in ``ArmTopicSpec`` so
the runtime can validate the same 27-value pose vector expected by realtime
VSA without hard-coding one robot family.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from robolineage_data_source.adapters.base import DeviceAdapter
from robolineage_data_source.config.schema import (
    ArmTopicSpec,
    CameraTopicSpec,
    Ros2AdapterConfig,
)
from robolineage_data_source.sample import HealthState, HealthStatus


class Ros2ProfileAdapter(DeviceAdapter):
    """Subscribe ROS2 camera/state topics for health and decode validation."""

    def __init__(self, *, config: Ros2AdapterConfig) -> None:
        self.config = config
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.RLock()
        self._health = HealthStatus(state=HealthState.NOT_STARTED, message="")
        self._frame_indices: dict[str, int] = {}
        self._latest_camera_frames: dict[str, Any] = {}
        self._latest_camera_meta: dict[str, dict[str, Any]] = {}
        self._latest_arm_vectors: dict[str, Any] = {}
        self._latest_arm_meta: dict[str, dict[str, Any]] = {}
        # Phase 5 T3 — per-stream JPEG decode failure counter; surfaced via
        # health().meta["jpeg_decode_failures"] for /health endpoint + ops alerting.
        self._jpeg_decode_failures: dict[str, int] = {}

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Ros2ProfileAdapter already started")
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
        except ImportError as exc:
            raise RuntimeError(
                "rclpy is not installed. The ROS2 profile adapter requires ROS2 "
                "Humble packages on the robot host. macOS dev machines should "
                "run with data_source disabled or a mock adapter."
            ) from exc

        rclpy.init(args=None, domain_id=self.config.ros_domain_id)
        try:
            self._node = self._build_node()
            self._executor = MultiThreadedExecutor(num_threads=self.config.spin_threads)
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                name="Ros2ProfileAdapter.spin",
                daemon=True,
            )
            self._spin_thread.start()
            self._started = True
            with self._lock:
                self._health = HealthStatus(state=HealthState.OK, message="spinning")
        except Exception:
            if self._node is not None:
                self._node.destroy_node()
                self._node = None
            rclpy.shutdown()
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
                rclpy.shutdown()
            except Exception:
                pass
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=3.0)
                self._spin_thread = None
            self._started = False
            with self._lock:
                self._health = HealthStatus(
                    state=HealthState.NOT_STARTED,
                    message="",
                )

    def health(self) -> HealthStatus:
        """Return current health snapshot.

        Phase 5+: ``meta["jpeg_decode_failures"]`` carries a per-stream
        running counter (``{"cam/camera_h/color": 12, ...}``) so the /health
        endpoint can show degradation depth, not just a binary state.
        """
        with self._lock:
            failures = dict(self._jpeg_decode_failures)
            base = self._health
            meta = dict(base.meta)
            camera_streams = {
                key: dict(value) for key, value in self._latest_camera_meta.items()
            }
            arm_streams = {
                key: dict(value) for key, value in self._latest_arm_meta.items()
            }
        meta["jpeg_decode_failures"] = failures
        meta["camera_streams"] = camera_streams
        meta["arm_streams"] = arm_streams
        return HealthStatus(
            state=base.state,
            message=base.message,
            last_sample_mono_ns=base.last_sample_mono_ns,
            meta=meta,
        )

    def latest_camera_frame(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> Any | None:
        """Return a copy of the latest decoded BGR camera frame."""
        key = self._camera_lookup_key(stream_id=stream_id, topic=topic)
        if key is None:
            return None
        with self._lock:
            frame = self._latest_camera_frames.get(key)
        return frame.copy() if frame is not None and hasattr(frame, "copy") else frame

    def camera_status(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> dict[str, Any] | None:
        key = self._camera_lookup_key(stream_id=stream_id, topic=topic)
        if key is None:
            return None
        with self._lock:
            meta = self._latest_camera_meta.get(key)
        return dict(meta) if meta is not None else None

    def latest_arm_vector(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> Any | None:
        key = self._arm_lookup_key(stream_id=stream_id, topic=topic)
        if key is None:
            return None
        with self._lock:
            vec = self._latest_arm_vectors.get(key)
        return vec.copy() if vec is not None and hasattr(vec, "copy") else vec

    def arm_status(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> dict[str, Any] | None:
        key = self._arm_lookup_key(stream_id=stream_id, topic=topic)
        if key is None:
            return None
        with self._lock:
            meta = self._latest_arm_meta.get(key)
        return dict(meta) if meta is not None else None

    def _build_node(self) -> Any:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage

        node = Node("ROBOLINEAGE_ros2_profile")
        callback_group = ReentrantCallbackGroup()

        for spec in self.config.cameras.values():
            if spec.transport != "compressed":
                raise ValueError(
                    "Ros2ProfileAdapter supports only compressed camera "
                    f"topics; got {spec.transport!r} for {spec.topic!r}"
                )
            self._frame_indices.setdefault(spec.stream_id, 0)
            node.create_subscription(
                CompressedImage,
                spec.topic,
                lambda msg, s=spec: self._on_camera(msg, s),
                self._make_qos(spec.qos),
                callback_group=callback_group,
            )

        for arm_name, spec in self.config.arms.items():
            qos = self._make_qos(spec.qos)
            msg_cls = _message_class_for_arm(spec)
            node.create_subscription(
                msg_cls,
                spec.slave_status,
                lambda msg, s=spec, a=arm_name: self._on_arm_pose(msg, s, a),
                qos,
                callback_group=callback_group,
            )
            if spec.master_command and self.config.master_overlay_topic:
                node.create_subscription(
                    msg_cls,
                    spec.master_command,
                    lambda msg, a=arm_name: self._on_master_command(msg, a),
                    qos,
                    callback_group=callback_group,
                )

        return node

    def _on_camera(self, msg: Any, spec: CameraTopicSpec) -> None:
        import cv2
        import numpy as np

        host_mono_ns = time.monotonic_ns()
        jpeg = bytes(msg.data)
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            with self._lock:
                self._jpeg_decode_failures[spec.stream_id] = (
                    self._jpeg_decode_failures.get(spec.stream_id, 0) + 1
                )
            self._mark_degraded(f"jpeg decode failed: {spec.stream_id}")
            return

        shape = [int(item) for item in bgr.shape]
        ros_stamp_ns = _stamp_to_ns(msg.header.stamp)

        with self._lock:
            frame_index = self._frame_indices.get(spec.stream_id, 0)
            self._frame_indices[spec.stream_id] = frame_index + 1
            camera_meta = {
                "topic": spec.topic,
                "stream_id": spec.stream_id,
                "camera_name": spec.camera_name,
                "frame_index": frame_index,
                "stamp_ns": ros_stamp_ns,
                "host_mono_ns": host_mono_ns,
                "shape": shape,
            }
            self._latest_camera_frames[spec.stream_id] = bgr
            self._latest_camera_meta[spec.stream_id] = camera_meta
            self._health = HealthStatus(
                state=self._health.state,
                message=self._health.message,
                last_sample_mono_ns=host_mono_ns,
                meta={
                    **dict(self._health.meta),
                    "last_camera_topic": spec.topic,
                    "last_camera_stream_id": spec.stream_id,
                    "last_camera_frame_index": frame_index,
                    "last_camera_stamp_ns": ros_stamp_ns,
                    "last_camera_shape": shape,
                },
            )

    def _on_arm_pose(self, msg: Any, spec: ArmTopicSpec, arm_name: str) -> None:
        host_mono_ns = time.monotonic_ns()
        vec = _robot_state_to_27_vec(msg, spec)
        ros_stamp_ns = _stamp_to_ns(msg.header.stamp)

        with self._lock:
            arm_meta = {
                "topic": spec.slave_status,
                "stream_id": spec.state_stream_id,
                "arm_name": arm_name,
                "stamp_ns": ros_stamp_ns,
                "host_mono_ns": host_mono_ns,
                "vector_len": int(getattr(vec, "shape", [len(vec)])[0]),
            }
            self._latest_arm_vectors[spec.state_stream_id] = (
                vec.copy() if hasattr(vec, "copy") else vec
            )
            self._latest_arm_meta[spec.state_stream_id] = arm_meta
            self._health = HealthStatus(
                state=self._health.state,
                message=self._health.message,
                last_sample_mono_ns=host_mono_ns,
                meta={
                    **dict(self._health.meta),
                    "last_arm": arm_name,
                    "last_arm_topic": spec.slave_status,
                    "last_arm_stamp_ns": ros_stamp_ns,
                    "last_arm_vector_len": int(getattr(vec, "shape", [len(vec)])[0]),
                },
            )

    def _on_master_command(self, msg: Any, arm_name: str) -> None:
        host_mono_ns = time.monotonic_ns()
        spec = self.config.arms.get(arm_name)
        vec = _robot_state_to_27_vec(msg, spec) if spec is not None else _robot_status_to_27_vec(msg)
        ros_stamp_ns = _stamp_to_ns(msg.header.stamp)
        with self._lock:
            self._health = HealthStatus(
                state=self._health.state,
                message=self._health.message,
                last_sample_mono_ns=host_mono_ns,
                meta={
                    **dict(self._health.meta),
                    "last_master_arm": arm_name,
                    "last_master_stamp_ns": ros_stamp_ns,
                    "last_master_vector_len": int(getattr(vec, "shape", [len(vec)])[0]),
                },
            )

    def _make_qos(self, name: str) -> Any:
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        if name == "sensor_data":
            return QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

    def _mark_degraded(self, message: str) -> None:
        with self._lock:
            self._health = HealthStatus(
                state=HealthState.DEGRADED,
                message=message,
            )

    def _camera_lookup_key(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> str | None:
        stream_id = str(stream_id).strip() if stream_id else None
        topic = str(topic).strip() if topic else None
        if stream_id and stream_id in self._latest_camera_frames:
            return stream_id
        for name, spec in self.config.cameras.items():
            aliases = {name, spec.stream_id, spec.topic}
            if stream_id and stream_id in aliases:
                return spec.stream_id
            if topic and topic == spec.topic:
                return spec.stream_id
        return None

    def _arm_lookup_key(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> str | None:
        stream_id = str(stream_id).strip() if stream_id else None
        topic = str(topic).strip() if topic else None
        if stream_id and stream_id in self._latest_arm_vectors:
            return stream_id
        for name, spec in self.config.arms.items():
            aliases = {name, spec.state_stream_id, spec.slave_status}
            if stream_id and stream_id in aliases:
                return spec.state_stream_id
            if topic and topic == spec.slave_status:
                return spec.state_stream_id
        return None


def _stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _camera_name_from_stream_id(topic: str) -> str | None:
    parts = str(topic or "").split("/")
    if len(parts) >= 3 and parts[-1] == "color":
        return parts[-2]
    return None


def _message_class_for_arm(spec: ArmTopicSpec) -> Any:
    if spec.msg_type:
        try:
            from rosidl_runtime_py.utilities import get_message

            return get_message(spec.msg_type)
        except Exception as exc:
            raise RuntimeError(f"cannot resolve ROS2 message type {spec.msg_type!r}") from exc

    raise RuntimeError(
        "arm msg_type is missing. Set streams.robot_states.<name>.msg_type "
        "in the robot profile."
    )


def _robot_state_to_27_vec(msg: Any, spec: ArmTopicSpec) -> "np.ndarray":  # noqa: F821
    if _looks_like_joint_end_state(msg):
        return _robot_status_to_27_vec(msg)

    if spec.eef_position_field and spec.eef_orientation_field and spec.gripper_field:
        import numpy as np

        vec = np.zeros(27, dtype=np.float32)
        vec[21:24] = _extract_numeric_list(msg, spec.eef_position_field, expected=3)
        vec[24:27] = _extract_numeric_list(msg, spec.eef_orientation_field, expected=3)
        vec[6] = _extract_numeric_scalar(msg, spec.gripper_field)
        return vec

    raise ValueError(
        "cannot convert robot state to RoboLineage 27-vector; provide a message with "
        "joint_pos/joint_vel/joint_cur/end_pos or set "
        "eef_position_field/eef_orientation_field/gripper_field in the robot profile"
    )


def _looks_like_joint_end_state(msg: Any) -> bool:
    return hasattr(msg, "joint_pos") and hasattr(msg, "joint_vel") and hasattr(msg, "joint_cur") and hasattr(msg, "end_pos")


def _robot_status_to_27_vec(msg: Any) -> "np.ndarray":  # noqa: F821
    """Convert a joint/end-effector robot status message to the 27-dim RoboLineage vector.

    Layout:
      [0..6]   joint_pos[0..6], with joint_pos[6] as gripper
      [7..13]  joint_vel[0..6]
      [14..20] joint_cur[0..6]
      [21..26] end_pos[0..5] as x, y, z, rx, ry, rz
    """
    import numpy as np

    vec = np.zeros(27, dtype=np.float32)
    vec[0:7] = list(msg.joint_pos)[:7]
    vec[7:14] = list(msg.joint_vel)[:7]
    vec[14:21] = list(msg.joint_cur)[:7]
    vec[21:27] = list(msg.end_pos)[:6]
    return vec


def _extract_numeric_list(msg: Any, expression: str, *, expected: int) -> list[float]:
    value = _extract_expression(msg, expression)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{expression!r} did not resolve to a sequence")
    result = [float(item) for item in value[:expected]]
    if len(result) != expected:
        raise ValueError(f"{expression!r} must contain {expected} values; got {len(result)}")
    return result


def _extract_numeric_scalar(msg: Any, expression: str) -> float:
    value = _extract_expression(msg, expression)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{expression!r} resolved to an empty sequence")
        value = value[0]
    return float(value)


def _extract_expression(msg: Any, expression: str) -> Any:
    value = msg
    for part in expression.split("."):
        if not part:
            continue
        name, selector = _split_selector(part)
        value = getattr(value, name)
        if selector is not None:
            value = _apply_selector(value, selector)
    return value


def _split_selector(part: str) -> tuple[str, str | None]:
    if "[" not in part:
        return part, None
    name, rest = part.split("[", 1)
    if not rest.endswith("]"):
        raise ValueError(f"invalid field selector: {part!r}")
    return name, rest[:-1]


def _apply_selector(value: Any, selector: str) -> Any:
    if ":" in selector:
        start_raw, end_raw = selector.split(":", 1)
        start = int(start_raw) if start_raw else None
        end = int(end_raw) if end_raw else None
        return list(value)[start:end]
    return list(value)[int(selector)]
