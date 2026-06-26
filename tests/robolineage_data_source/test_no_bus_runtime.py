from __future__ import annotations

from pathlib import Path


FORBIDDEN_RUNTIME_TOKENS = (
    "InProcessBus",
    "DeviceBus",
    "RolloutBusRecorder",
    "robolineage_data_source.bus",
)

FORBIDDEN_FRONTEND_TOKENS = (
    "RoboLineage bus",
    "bus_topic",
)


def test_production_runtime_code_no_longer_depends_on_in_process_bus():
    root = Path("src")
    checked_roots = (
        root / "robolineage_app",
        root / "robolineage_ar",
        root / "robolineage_shared_agents" / "visual_snapshot" / "realtime",
        root / "robolineage_data_source",
    )
    offenders: list[str] = []
    for checked_root in checked_roots:
        for path in checked_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_RUNTIME_TOKENS:
                if token in text:
                    offenders.append(f"{path}:{token}")
    assert offenders == []


def test_frontend_robot_onboard_uses_ros2_source_language():
    checked_files = (
        Path("frontend/src/views/RobotOnboardView.vue"),
        Path("frontend/src/stores/robot.ts"),
    )
    offenders: list[str] = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_FRONTEND_TOKENS:
            if token in text:
                offenders.append(f"{path}:{token}")
    assert offenders == []
