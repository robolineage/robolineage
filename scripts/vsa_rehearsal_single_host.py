#!/usr/bin/env python3
"""Single-host realtime VSA rehearsal with synthetic Realtime{Frame,Action}Record streams."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from robolineage_shared_agents.visual_snapshot import MockVLMRunner, TaskConfig
from robolineage_shared_agents.visual_snapshot.realtime import (
    RealtimeActionRecord,
    RealtimeFrameRecord,
    run_action_guided_stream,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", default="/tmp/vsa_rehearsal_snapshots.jsonl")
    parser.add_argument("--rollout-dir", default="/tmp/vsa_rehearsal_rollout")
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args(argv)

    output = Path(args.output_jsonl)
    if output.exists():
        output.unlink()
    rollout_dir = Path(args.rollout_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    task_config = TaskConfig(
        task_description="Synthetic single-host realtime rehearsal.",
        phases=["approach", "grasp", "transfer", "place"],
        phase_action_hints={
            "grasp": {"gripper_state": "closed", "event_type": ["gripper_close"]},
            "place": {"gripper_state": "open", "event_type": ["gripper_open"]},
        },
    )
    frames, actions = _make_messages(args.frames)
    snapshots = run_action_guided_stream(
        frame_messages=frames,
        action_messages=actions,
        task_config=task_config,
        vlm_runner=MockVLMRunner(valid_phases=task_config.phases, latency=0.0),
        rollout_dir=rollout_dir,
        output_jsonl=output,
        context_frames=1,
        max_keyframes=3,
        still_min_frames=4,
        heartbeat_interval=0,
    )
    print(json.dumps({"snapshots": len(snapshots), "output_jsonl": str(output)}, ensure_ascii=False))
    return 0


def _make_messages(count: int) -> tuple[list[RealtimeFrameRecord], list[RealtimeActionRecord]]:
    now = time.monotonic_ns()
    bgr = _bgr_image()
    frames: list[RealtimeFrameRecord] = []
    actions: list[RealtimeActionRecord] = []
    for i in range(count):
        frames.append(
            RealtimeFrameRecord(
                frame_index=i,
                host_mono_ns=now + i * 10_000_000,
                bgr=bgr,
            )
        )
        gripper = -1.5 if 5 <= i < 16 else 0.0
        actions.append(
            RealtimeActionRecord(
                frame_index=i,
                host_mono_ns=now + i * 10_000_000,
                eef_xyz=(i * 0.001, 0.0, 0.0),
                eef_rxyz=(0.0, 0.0, 0.0),
                gripper=gripper,
            )
        )
    return frames, actions


def _bgr_image() -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :] = np.array((60, 40, 20), dtype=np.uint8)
    return img


if __name__ == "__main__":
    raise SystemExit(main())
