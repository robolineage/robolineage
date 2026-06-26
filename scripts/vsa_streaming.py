#!/usr/bin/env python3
"""Run realtime Visual Snapshot Agent directly from ROS2 topics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from robolineage_shared_agents.visual_snapshot import MockVLMRunner, OpenAIVLMRunner, Qwen2VLRunner, TaskConfig
from robolineage_shared_agents.visual_snapshot.realtime.runtime_pipeline import (
    StreamingRuntimePipeline,
    run_ros_topic_stream,
)
from robolineage_data_source.config.loader import load_config
from robolineage_data_source.config.schema import ArmTopicSpec, Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="RoboLineage YAML with an adapter section, e.g. configs/arx_one.yaml",
    )
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--vlm", required=True, help="mock, openai:<model>, or qwen:<model>")
    parser.add_argument("--camera-topic", help="ROS2 compressed image topic; defaults to the first configured camera")
    parser.add_argument("--arm-topic", help="ROS2 robot-state topic; defaults to the first configured arm")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--idle-timeout", type=float, default=10.0)
    parser.add_argument("--context-frames", type=int, default=15)
    parser.add_argument("--max-keyframes", type=int, default=3)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.adapter is None or cfg.adapter.type not in {"ros2_arx_one", "ros2_profile"}:
        print(
            "vsa_streaming: config must have adapter.type=ros2_profile or ros2_arx_one",
            file=sys.stderr,
        )
        return 2
    camera_topic = args.camera_topic or _default_camera_topic(cfg)
    arm_topic = args.arm_topic or _default_arm_topic(cfg)
    arm_spec = _arm_spec_for_topic(cfg, arm_topic)

    task_config = _load_task_config(Path(args.task_config))
    pipeline = StreamingRuntimePipeline(
        task_config=task_config,
        vlm_runner=_build_vlm_runner(args.vlm, task_config),
        rollout_dir=Path(args.rollout_dir),
        output_jsonl=Path(args.output_jsonl),
        context_frames=args.context_frames,
        max_keyframes=args.max_keyframes,
    )
    snapshots = run_ros_topic_stream(
        camera_topic=camera_topic,
        arm_topic=arm_topic,
        arm_spec=arm_spec,
        ros_domain_id=cfg.adapter.ros_domain_id,
        pipeline=pipeline,
        max_events=args.max_events,
        idle_timeout=args.idle_timeout,
    )

    print(
        json.dumps(
            {"snapshots": len(snapshots), "output_jsonl": args.output_jsonl},
            ensure_ascii=False,
        )
    )
    return 0


def _load_task_config(path: Path) -> TaskConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"task config must be a mapping: {path}")
    return TaskConfig(**raw)


def _default_camera_topic(cfg: Config) -> str:
    if cfg.vsa is not None and cfg.vsa.camera_topic and str(cfg.vsa.camera_topic).startswith("/"):
        return str(cfg.vsa.camera_topic)
    if cfg.adapter is not None and cfg.adapter.cameras:
        return next(iter(cfg.adapter.cameras.values())).topic
    raise ValueError("no ROS2 camera topic configured")


def _default_arm_topic(cfg: Config) -> str:
    if cfg.vsa is not None and cfg.vsa.arm_topic and str(cfg.vsa.arm_topic).startswith("/"):
        return str(cfg.vsa.arm_topic)
    if cfg.adapter is not None and cfg.adapter.arms:
        return next(iter(cfg.adapter.arms.values())).slave_status
    raise ValueError("no ROS2 arm topic configured")


def _arm_spec_for_topic(cfg: Config, topic: str) -> ArmTopicSpec:
    if cfg.adapter is None:
        raise ValueError("adapter section is required")
    for spec in cfg.adapter.arms.values():
        if spec.slave_status == topic:
            return spec
    raise ValueError(f"arm topic is not declared in adapter.arms: {topic}")


def _build_vlm_runner(spec: str, task_config: TaskConfig):
    if spec == "mock":
        return MockVLMRunner(valid_phases=task_config.phases, latency=0.0)
    if spec.startswith("openai:"):
        return OpenAIVLMRunner(model_name=spec.split(":", 1)[1])
    if spec.startswith("qwen:"):
        return Qwen2VLRunner(model_name=spec.split(":", 1)[1])
    raise ValueError(f"Unsupported --vlm value: {spec}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vsa_streaming.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
