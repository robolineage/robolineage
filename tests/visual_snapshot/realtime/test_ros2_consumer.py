from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from robolineage_data_source.config.schema import ArmTopicSpec


class _FakeRclpy(ModuleType):
    def __init__(self, *, already_ok: bool) -> None:
        super().__init__("rclpy")
        self.__path__ = []  # make import machinery treat this as a package
        self._ok = already_ok
        self.init_calls: list[dict[str, Any]] = []
        self.shutdown_calls = 0

    def ok(self) -> bool:
        return self._ok

    def init(self, *, args: Any = None, domain_id: int | None = None) -> None:
        if self._ok:
            raise RuntimeError("Context.init() must only be called once")
        self.init_calls.append({"args": args, "domain_id": domain_id})
        self._ok = True

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._ok = False


class _FakeCallbackGroup:
    pass


class _FakeExecutor:
    instances: list["_FakeExecutor"] = []

    def __init__(self, *, num_threads: int) -> None:
        self.num_threads = num_threads
        self.nodes: list[Any] = []
        self.shutdown_calls = 0
        _FakeExecutor.instances.append(self)

    def add_node(self, node: Any) -> None:
        self.nodes.append(node)

    def spin(self) -> None:
        return

    def shutdown(self, *, timeout_sec: float) -> None:
        self.shutdown_calls += 1


class _FakeNode:
    instances: list["_FakeNode"] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.subscriptions: list[tuple[Any, str, Any, Any]] = []
        self.destroyed = False
        _FakeNode.instances.append(self)

    def create_subscription(
        self,
        msg_cls: Any,
        topic: str,
        callback: Any,
        qos: Any,
        *,
        callback_group: Any,
    ) -> None:
        self.subscriptions.append((msg_cls, topic, callback, callback_group))

    def destroy_node(self) -> None:
        self.destroyed = True


def _install_fake_ros2_modules(monkeypatch: Any, *, already_ok: bool) -> _FakeRclpy:
    _FakeExecutor.instances.clear()
    _FakeNode.instances.clear()
    rclpy = _FakeRclpy(already_ok=already_ok)

    callback_groups = ModuleType("rclpy.callback_groups")
    callback_groups.ReentrantCallbackGroup = _FakeCallbackGroup
    executors = ModuleType("rclpy.executors")
    executors.MultiThreadedExecutor = _FakeExecutor
    node = ModuleType("rclpy.node")
    node.Node = _FakeNode
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs.__path__ = []
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.CompressedImage = type("CompressedImage", (), {})
    rosidl_runtime_py = ModuleType("rosidl_runtime_py")
    rosidl_runtime_py.__path__ = []
    utilities = ModuleType("rosidl_runtime_py.utilities")
    utilities.get_message = lambda msg_type: type("RobotState", (), {})

    for name, module in {
        "rclpy": rclpy,
        "rclpy.callback_groups": callback_groups,
        "rclpy.executors": executors,
        "rclpy.node": node,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "rosidl_runtime_py": rosidl_runtime_py,
        "rosidl_runtime_py.utilities": utilities,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return rclpy


def _arm_spec() -> ArmTopicSpec:
    return ArmTopicSpec(
        slave_status="/arm_state",
        state_stream_id="robot/arm/state",
        msg_type="example_msgs/msg/RobotState",
    )


def test_ros2_topic_consumer_reuses_existing_rclpy_context(monkeypatch):
    rclpy = _install_fake_ros2_modules(monkeypatch, already_ok=True)

    from robolineage_shared_agents.visual_snapshot.realtime.ros2_consumer import Ros2TopicConsumer

    consumer = Ros2TopicConsumer(
        camera_topic="/camera/compressed",
        arm_topic="/arm_state",
        arm_spec=_arm_spec(),
        ros_domain_id=17,
    )

    consumer.start()
    try:
        assert rclpy.init_calls == []
        assert _FakeNode.instances[-1].name == "ROBOLINEAGE_vsa_ros2_topic_consumer"
        assert [item[1] for item in _FakeNode.instances[-1].subscriptions] == [
            "/camera/compressed",
            "/arm_state",
        ]
    finally:
        consumer.stop()

    assert rclpy.shutdown_calls == 0


def test_ros2_topic_consumer_shuts_down_context_it_started(monkeypatch):
    rclpy = _install_fake_ros2_modules(monkeypatch, already_ok=False)

    from robolineage_shared_agents.visual_snapshot.realtime.ros2_consumer import Ros2TopicConsumer

    consumer = Ros2TopicConsumer(
        camera_topic="/camera/compressed",
        arm_topic="/arm_state",
        arm_spec=_arm_spec(),
        ros_domain_id=17,
    )

    consumer.start()
    assert rclpy.init_calls == [{"args": None, "domain_id": 17}]

    consumer.stop()

    assert rclpy.shutdown_calls == 1
