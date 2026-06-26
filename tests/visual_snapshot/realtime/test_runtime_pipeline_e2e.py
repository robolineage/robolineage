import json
import time

import numpy as np

from tests.visual_snapshot.realtime.helpers import action_msg, frame_msg
from robolineage_shared_agents.visual_snapshot import ActionEvent, MockVLMRunner, TaskConfig
from robolineage_shared_agents.visual_snapshot.exceptions import VLMInferenceError
from robolineage_shared_agents.visual_snapshot.realtime import (
    StreamingRuntimePipeline,
    run_action_guided_stream,
)
from robolineage_contracts.agents import SnapshotAssessment


_FIXED_RESPONSE = json.dumps(
    {
        "phase": "grasp",
        "progress": "advancing",
        "risk_level": "low",
        "confidence": 0.9,
    }
)


class _RaiseAfterFirstRunner:
    model_name = "raise-after-first"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "phase": "approach",
                    "progress": "advancing",
                    "risk_level": "low",
                    "confidence": 0.9,
                }
            )
        raise VLMInferenceError("simulated timeout")


class _PromptRecordingRunner:
    model_name = "prompt-recorder"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        self.prompts.append(prompt)
        phase = "approach" if len(self.prompts) == 1 else "grasp"
        return json.dumps(
            {
                "phase": phase,
                "progress": "advancing",
                "risk_level": "low",
                "confidence": 0.9,
            }
        )


class _SlowRunner:
    model_name = "slow-runner"

    def __init__(self, latency: float = 0.25) -> None:
        self.latency = latency
        self.calls = 0

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        self.calls += 1
        time.sleep(self.latency)
        return _FIXED_RESPONSE


class _ImageCountingRunner:
    model_name = "image-counter"

    def __init__(self) -> None:
        self.image_counts: list[int] = []

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        self.image_counts.append(len(images))
        return _FIXED_RESPONSE


def _trigger_value(snapshot: SnapshotAssessment) -> str | None:
    trigger = snapshot.trigger
    return trigger.value if hasattr(trigger, "value") else trigger


def test_runtime_pipeline_e2e_mock_vlm(tmp_path):
    frames = [frame_msg(i) for i in range(8)]
    actions = [action_msg(i, gripper=-1.5 if i >= 3 else 0.0) for i in range(8)]
    output = tmp_path / "snapshots.jsonl"

    snapshots = run_action_guided_stream(
        frame_messages=frames,
        action_messages=actions,
        task_config=TaskConfig(task_description="pick", phases=["approach", "grasp"]),
        vlm_runner=MockVLMRunner(fixed_response=_FIXED_RESPONSE, latency=0.0),
        rollout_dir=tmp_path,
        output_jsonl=output,
        context_frames=1,
        max_keyframes=3,
        heartbeat_interval=0,
    )

    assert snapshots
    rows = [SnapshotAssessment(**json.loads(line)) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(snapshots)
    assert any(_trigger_value(row) == "gripper_close" for row in rows)


def test_runtime_pipeline_emits_periodic_samples(tmp_path):
    frames = [frame_msg(i) for i in range(14)]
    actions = [action_msg(i, gripper=0.0) for i in range(14)]
    output = tmp_path / "snapshots.jsonl"

    snapshots = run_action_guided_stream(
        frame_messages=frames,
        action_messages=actions,
        task_config=TaskConfig(task_description="observe", phases=["approach", "place"]),
        vlm_runner=MockVLMRunner(fixed_response=_FIXED_RESPONSE, latency=0.0),
        rollout_dir=tmp_path,
        output_jsonl=output,
        context_frames=1,
        max_keyframes=3,
        still_min_frames=99,
        heartbeat_interval=0,
        periodic_interval_sec=0.3,
    )

    assert snapshots
    rows = [SnapshotAssessment(**json.loads(line)) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(_trigger_value(row) == "periodic_sample" for row in rows)


def test_vlm_error_holds_last_phase_instead_of_action_fallback(tmp_path):
    frames = [frame_msg(i) for i in range(8)]
    actions = [action_msg(i, gripper=-1.5 if i >= 3 else 0.0) for i in range(8)]
    output = tmp_path / "snapshots.jsonl"

    snapshots = run_action_guided_stream(
        frame_messages=frames,
        action_messages=actions,
        task_config=TaskConfig(
            task_description="pick",
            phases=["approach", "grasp"],
            phase_action_hints={
                "approach": {"event_type": ["sequence_start"], "gripper_state": "open"},
                "grasp": {"event_type": ["gripper_close"], "gripper_state": "closed"},
            },
        ),
        vlm_runner=_RaiseAfterFirstRunner(),
        rollout_dir=tmp_path,
        output_jsonl=output,
        context_frames=1,
        max_keyframes=3,
        still_min_frames=99,
        heartbeat_interval=0,
        periodic_interval_sec=0,
    )

    gripper_close = [
        snapshot for snapshot in snapshots
        if _trigger_value(snapshot) == "gripper_close"
    ]
    assert gripper_close
    assert gripper_close[0].phase == "approach"
    assert gripper_close[0].needs_review is True
    assert gripper_close[0].confidence <= 0.1


def test_ready_events_update_rollout_memory_before_next_prompt(tmp_path):
    runner = _PromptRecordingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="pick", phases=["approach", "grasp"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=1,
        max_keyframes=3,
        heartbeat_interval=0,
        periodic_interval_sec=0,
        vlm_workers=2,
    )

    try:
        for i in range(4):
            pipeline.process_frame(frame_msg(i))
            pipeline.signal_builder.feed(action_msg(i))
        pipeline.pending.enqueue(ActionEvent("sequence_start", anchor_frame=1, timestamp_sec=1.0, confidence=1.0))
        pipeline.pending.enqueue(ActionEvent("periodic_sample", anchor_frame=2, timestamp_sec=2.0, confidence=1.0))

        snapshots = pipeline.process_ready_windows(force=True)
        snapshots.extend(pipeline.drain())
    finally:
        pipeline.close()

    assert [snapshot.phase for snapshot in snapshots[:2]] == ["approach", "grasp"]
    assert _trigger_value(snapshots[-1]) == "final_observation"
    assert len(runner.prompts) == 3
    assert "Online rollout memory from previous assessments" not in runner.prompts[0]
    assert "last_confirmed_phase=approach" in runner.prompts[1]


def test_vlm_analysis_does_not_block_realtime_dispatch(tmp_path):
    runner = _SlowRunner(latency=0.25)
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="pick", phases=["approach", "grasp"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=1,
        max_keyframes=3,
        heartbeat_interval=0,
        periodic_interval_sec=0,
    )

    try:
        pipeline.process_frame(frame_msg(0))
        pipeline.process_action(action_msg(0))
        pipeline.process_frame(frame_msg(1))
        start = time.monotonic()
        pipeline.process_action(action_msg(1))
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    assert runner.calls >= 1
    assert snapshots


def test_drain_materializes_keyframes_to_png_and_releases_ring_buffer(tmp_path):
    runner = _ImageCountingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="pick", phases=["approach", "grasp"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=1,
        max_keyframes=3,
        heartbeat_interval=0,
        periodic_interval_sec=0,
    )

    try:
        for i in range(5):
            pipeline.process_frame(frame_msg(i))
            pipeline.signal_builder.feed(action_msg(i, gripper=-1.5 if i >= 2 else 0.0))
        pipeline.pending.enqueue(
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=2,
                timestamp_sec=0.2,
                confidence=1.0,
            )
        )

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    assert snapshots
    assert runner.image_counts
    assert all(count > 0 for count in runner.image_counts)
    assert len(pipeline.frame_buffer) == 0
    pngs = sorted((tmp_path / "vsa_windows").rglob("*.png"))
    assert pngs
    assert pipeline._materialized_vsa_windows >= 1
    assert pipeline._materialized_keyframes == len(pngs)
    manifest = tmp_path / "vsa_windows" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[0]["image_format"] == "png"
    assert all(path.endswith(".png") for path in rows[0]["image_paths"])


def test_drain_emits_final_observation_window_from_latest_frames(tmp_path):
    runner = _ImageCountingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="generic manipulation", phases=["start", "finish"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=1,
        max_keyframes=3,
        heartbeat_interval=0,
        periodic_interval_sec=0,
    )

    try:
        for i in range(5):
            pipeline.process_frame(frame_msg(i))
            pipeline.signal_builder.feed(action_msg(i, gripper=0.0))

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    assert snapshots
    assert _trigger_value(snapshots[-1]) == "final_observation"
    assert snapshots[-1].frame_id == 4
    assert runner.image_counts
    assert runner.image_counts[-1] > 0
    manifest = tmp_path / "vsa_windows" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event_type"] == "final_observation"


def test_drain_anchors_final_observation_on_release_settle_when_available(tmp_path):
    runner = _ImageCountingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="generic manipulation", phases=["start", "finish"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=1,
        max_keyframes=3,
        still_min_frames=2,
        heartbeat_interval=0,
        periodic_interval_sec=0,
        final_settle_sec=0.2,
        still_threshold=1e-6,
    )

    try:
        positions = {
            0: 0.0,
            1: 0.01,
            2: 0.02,
            3: 0.03,
            4: 0.03,
            5: 0.03,
            6: 0.03,
            7: 0.03,
        }
        for i in range(8):
            pipeline.process_frame(frame_msg(i))
            gripper = -1.5 if i < 3 else 0.0
            pipeline.process_action(action_msg(i, gripper=gripper, x=positions[i]))

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    assert snapshots
    assert _trigger_value(snapshots[-1]) == "final_observation"
    assert snapshots[-1].frame_id == 5
    manifest = tmp_path / "vsa_windows" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event_type"] == "final_observation"
    assert rows[-1]["anchor_frame_id"] == 5
    assert rows[-1]["event_details"]["reason"] == "release_settle"


def test_release_settle_does_not_emit_before_configured_delay(tmp_path):
    runner = _ImageCountingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="generic manipulation", phases=["start", "finish"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=1,
        max_keyframes=3,
        still_min_frames=2,
        heartbeat_interval=0,
        periodic_interval_sec=0,
        final_settle_sec=0.5,
        still_threshold=1e-6,
    )

    try:
        positions = {
            0: 0.0,
            1: 0.01,
            2: 0.02,
            3: 0.03,
            4: 0.03,
            5: 0.03,
            6: 0.03,
            7: 0.03,
        }
        for i in range(8):
            pipeline.process_frame(frame_msg(i))
            gripper = -1.5 if i < 3 else 0.0
            pipeline.process_action(action_msg(i, gripper=gripper, x=positions[i]))

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    assert snapshots
    assert _trigger_value(snapshots[-1]) == "final_observation"
    assert snapshots[-1].frame_id == 7
    manifest = tmp_path / "vsa_windows" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event_type"] == "final_observation"
    assert rows[-1]["anchor_frame_id"] == 7
    assert rows[-1]["event_details"]["reason"] == "rollout_stop"


def test_drain_coalesces_periodic_window_when_high_information_event_shares_anchor(tmp_path):
    runner = _ImageCountingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="generic manipulation", phases=["start", "finish"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=1,
        max_keyframes=3,
        heartbeat_interval=0,
        periodic_interval_sec=0,
    )

    try:
        for i in range(5):
            pipeline.process_frame(frame_msg(i))
            pipeline.signal_builder.feed(action_msg(i, gripper=-1.5 if i >= 2 else 0.0))
        pipeline.pending.enqueue(
            ActionEvent(
                event_type="periodic_sample",
                anchor_frame=2,
                timestamp_sec=0.2,
                confidence=0.5,
            )
        )
        pipeline.pending.enqueue(
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=2,
                timestamp_sec=0.2,
                confidence=1.0,
            )
        )

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    assert [_trigger_value(row) for row in snapshots] == ["gripper_close", "final_observation"]
    manifest = tmp_path / "vsa_windows" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == ["gripper_close", "final_observation"]


def test_rollout_window_cap_reserves_slot_for_final_observation(tmp_path):
    runner = _ImageCountingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="generic manipulation", phases=["start", "finish"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=0,
        max_keyframes=3,
        heartbeat_interval=0,
        periodic_interval_sec=0,
        merge_window_sec=0,
        max_vlm_windows_per_rollout=2,
    )

    try:
        for i in range(5):
            pipeline.process_frame(frame_msg(i))
            pipeline.signal_builder.feed(action_msg(i, gripper=-1.5 if i >= 2 else 0.0))
        pipeline.pending.enqueue(
            ActionEvent(
                event_type="periodic_sample",
                anchor_frame=1,
                timestamp_sec=0.1,
                confidence=0.5,
            )
        )
        pipeline.pending.enqueue(
            ActionEvent(
                event_type="gripper_close",
                anchor_frame=3,
                timestamp_sec=0.3,
                confidence=1.0,
            )
        )

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    triggers = [_trigger_value(row) for row in snapshots]
    assert triggers[-1] == "final_observation"
    assert len(triggers) == 2
    assert pipeline.metrics()["enqueued_vlm_windows"] == 2
    assert pipeline.metrics()["dropped_vlm_windows"] == 1

    manifest = tmp_path / "vsa_windows" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows][-1] == "final_observation"
    assert len(rows) == 2


def test_runtime_pipeline_uses_stateful_scheduler_across_ready_batches(tmp_path):
    runner = _ImageCountingRunner()
    pipeline = StreamingRuntimePipeline(
        task_config=TaskConfig(task_description="generic manipulation", phases=["start", "finish"]),
        vlm_runner=runner,
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
        context_frames=0,
        max_keyframes=3,
        heartbeat_interval=0,
        periodic_interval_sec=0,
    )

    try:
        grippers = {
            0: 0.0,
            1: 0.0,
            2: -1.5,
            3: 0.0,
            4: 0.0,
        }
        for i in range(5):
            pipeline.process_frame(frame_msg(i))
            pipeline.process_action(action_msg(i, gripper=grippers[i]))

        snapshots = pipeline.drain()
    finally:
        pipeline.close()

    triggers = [_trigger_value(row) for row in snapshots]
    assert "gripper_burst" in triggers
    assert "gripper_close" not in triggers
    assert "gripper_open" not in triggers
    assert triggers[-1] == "final_observation"
    assert pipeline.metrics()["gripper_burst_count"] == 1
