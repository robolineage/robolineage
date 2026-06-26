import json

import numpy as np

from robolineage_shared_agents.visual_snapshot import MockVLMRunner, TaskConfig, VisualSnapshotAgent
from robolineage_shared_agents.visual_snapshot.phase_fusion import PhaseFusion
from robolineage_shared_agents.visual_snapshot.phase_state_machine import PhaseStateMachine
from robolineage_shared_agents.visual_snapshot.prompt_builder import PromptBuilder
from robolineage_shared_agents.visual_snapshot.temporal_stabilizer import TemporalStabilizer
from robolineage_shared_agents.visual_snapshot.types import (
    ActionGuidedWindow,
    PhasePriorResult,
    VisualObservationWindow,
)


def _task() -> TaskConfig:
    return TaskConfig(
        task_description="move object to target",
        phases=["approach", "grasp", "lift", "move_to_target", "place"],
        phase_definitions={
            "approach": "move the end effector toward the object",
            "grasp": "contact and secure the object",
            "lift": "lift the object away from support",
            "move_to_target": "transport the object toward the target",
            "place": "place and release the object at the target",
        },
    )


def test_prompt_treats_action_as_auxiliary_not_trusted_prior():
    window = ActionGuidedWindow(
        rollout_id="r1",
        frame_ids=[1],
        timestamps=[0.1],
        camera_name="camera_h",
        color_frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
        depth_frames=[],
        end_frame_id=1,
        end_timestamp=0.1,
        anchor_frame_id=1,
        event_type="gripper_close",
        keyframe_ids=[1],
        action_summary={"gripper_state_after": "closed", "position_delta": (0.0, 0.0, 0.0)},
    )
    prior = PhasePriorResult(
        phase_scores={"grasp": 0.9, "place": 0.1},
        top_phase="grasp",
        top_margin=0.8,
        prior_reason="top=grasp event=gripper_close",
    )

    prompt, images = PromptBuilder().build(window, _task(), prior=prior)

    assert images
    assert "Use visual evidence as the primary evidence" in prompt
    assert "Measured robot action evidence (auxiliary, not a phase prior)" in prompt
    assert "Trusted phase prior" not in prompt
    assert "top_phase_prior" not in prompt


def test_phase_fusion_keeps_visual_when_action_disagrees():
    fusion = PhaseFusion(strong_prior_margin=0.35)
    prior = PhasePriorResult(
        phase_scores={"grasp": 0.9, "place": 0.1},
        top_phase="grasp",
        top_margin=0.8,
        prior_reason="top=grasp event=gripper_close",
    )

    phase, reason = fusion.fuse(prior, "place", 0.62, _task())

    assert phase == "place"
    assert reason.startswith("visual_primary")


def test_phase_fusion_uses_action_only_for_low_confidence_visual_fallback():
    fusion = PhaseFusion(strong_prior_margin=0.35)
    prior = PhasePriorResult(
        phase_scores={"grasp": 0.9, "place": 0.1},
        top_phase="grasp",
        top_margin=0.8,
        prior_reason="top=grasp event=gripper_close",
    )

    phase, reason = fusion.fuse(prior, "place", 0.2, _task())

    assert phase == "grasp"
    assert reason.startswith("low_conf_visual_action_assist")


def test_phase_state_machine_allows_confident_forward_jump():
    sm = PhaseStateMachine(forward_jump_confidence=0.60)
    task = _task()

    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert sm.apply("move_to_target", task, confidence=0.62) == "move_to_target"
    assert sm.last_reason.startswith("visual_forward_jump")


def test_phase_state_machine_keeps_current_on_unsupported_forward_jump():
    sm = PhaseStateMachine(forward_jump_confidence=0.60)
    task = _task()

    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert sm.apply("move_to_target", task, event_type="periodic_sample", confidence=0.4) == "grasp"
    assert sm.last_reason.startswith("forward_jump_pending")


def test_phase_state_machine_never_invents_intermediate_phase_on_forward_skip():
    sm = PhaseStateMachine(forward_jump_confidence=0.60)
    task = _task()

    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    result = sm.apply("place", task, event_type="periodic_sample", confidence=0.4)

    assert result in {"grasp", "place"}
    assert result != "lift"
    assert result != "move_to_target"


def test_phase_state_machine_accepts_boundary_event_forward_skip():
    sm = PhaseStateMachine(forward_jump_confidence=0.60)
    task = _task()

    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert sm.apply("move_to_target", task, event_type="still_start", confidence=0.55) == "move_to_target"
    assert sm.last_reason.startswith("event_forward_jump")


def test_phase_state_machine_accepts_high_risk_forward_skip():
    sm = PhaseStateMachine(forward_jump_confidence=0.60)
    task = _task()

    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert (
        sm.apply(
            "move_to_target",
            task,
            event_type="periodic_sample",
            confidence=0.52,
            risk_level="high",
            imminent_failure=True,
        )
        == "move_to_target"
    )
    assert sm.last_reason.startswith("risk_forward_jump")


def test_phase_state_machine_holds_risky_terminal_skip_from_early_phase():
    sm = PhaseStateMachine(forward_jump_confidence=0.60)
    task = _task()

    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert (
        sm.apply(
            "place",
            task,
            event_type="gripper_open",
            confidence=0.62,
            risk_level="high",
            imminent_failure=True,
        )
        == "grasp"
    )
    assert sm.last_reason.startswith("terminal_failure_hold")


def test_phase_state_machine_requires_repeated_clean_terminal_skip():
    sm = PhaseStateMachine(
        forward_jump_confidence=0.60,
        terminal_forward_jump_confidence=0.80,
        terminal_forward_jump_repeat=2,
    )
    task = _task()

    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert sm.apply("place", task, confidence=0.84, risk_level="low") == "grasp"
    assert sm.last_reason.startswith("terminal_skip_pending")

    assert sm.apply("place", task, confidence=0.84, risk_level="low") == "place"
    assert sm.last_reason.startswith("repeated_terminal_jump")


def test_phase_state_machine_allows_retry_after_terminal_with_failure_evidence():
    sm = PhaseStateMachine(retry_after_terminal_confidence=0.70)
    task = _task()

    assert "retry" not in task.phases
    assert sm.apply("approach", task, confidence=0.9) == "approach"
    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert sm.apply("lift", task, confidence=0.9) == "lift"
    assert sm.apply("move_to_target", task, confidence=0.9) == "move_to_target"
    assert sm.apply("place", task, confidence=0.9) == "place"

    assert (
        sm.apply(
            "approach",
            task,
            event_type="gripper_close",
            confidence=0.78,
            risk_level="high",
            imminent_failure=True,
        )
        == "approach"
    )
    assert sm.last_reason.startswith("retry_after_terminal")


def test_phase_state_machine_blocks_terminal_regression_without_failure_evidence():
    sm = PhaseStateMachine(retry_after_terminal_confidence=0.70)
    task = _task()

    assert sm.apply("approach", task, confidence=0.9) == "approach"
    assert sm.apply("grasp", task, confidence=0.9) == "grasp"
    assert sm.apply("lift", task, confidence=0.9) == "lift"
    assert sm.apply("move_to_target", task, confidence=0.9) == "move_to_target"
    assert sm.apply("place", task, confidence=0.9) == "place"

    assert (
        sm.apply(
            "move_to_target",
            task,
            event_type="gripper_close",
            confidence=0.85,
            risk_level="low",
            imminent_failure=False,
        )
        == "place"
    )
    assert sm.last_reason.startswith("regression_pending")


def test_phase_state_machine_allows_generic_terminal_retry_on_regressing_progress():
    sm = PhaseStateMachine(retry_after_terminal_confidence=0.70)
    task = TaskConfig(
        task_description="generic closed-loop manipulation",
        phases=["reach", "engage", "transport", "release", "verify"],
    )

    assert sm.apply("reach", task, confidence=0.9) == "reach"
    assert sm.apply("engage", task, confidence=0.9) == "engage"
    assert sm.apply("transport", task, confidence=0.9) == "transport"
    assert sm.apply("release", task, confidence=0.9) == "release"
    assert sm.apply("verify", task, confidence=0.9) == "verify"

    assert (
        sm.apply(
            "transport",
            task,
            event_type="periodic_sample",
            confidence=0.76,
            risk_level="low",
            imminent_failure=False,
            progress="regressing",
        )
        == "transport"
    )
    assert sm.last_reason.startswith("retry_after_terminal")


def test_phase_state_machine_allows_generic_terminal_retry_on_contact_transition():
    sm = PhaseStateMachine(retry_after_terminal_confidence=0.70)
    task = TaskConfig(
        task_description="generic manipulation with contact-rich recovery",
        phases=["start", "contact", "move", "finish"],
    )

    assert sm.apply("start", task, confidence=0.9) == "start"
    assert sm.apply("contact", task, confidence=0.9) == "contact"
    assert sm.apply("move", task, confidence=0.9) == "move"
    assert sm.apply("finish", task, confidence=0.9) == "finish"

    assert (
        sm.apply(
            "contact",
            task,
            event_type="contact_transition",
            confidence=0.74,
            risk_level="low",
            imminent_failure=False,
        )
        == "contact"
    )
    assert sm.last_reason.startswith("retry_after_terminal")


def test_temporal_stabilizer_releases_boundary_event_phase_without_global_threshold():
    stabilizer = TemporalStabilizer()
    task = _task()

    first = stabilizer.stabilize(
        {"phase": "grasp", "progress": "advancing", "risk_level": "low", "confidence": 0.9},
        task,
        event_type="gripper_close",
    )
    assert first["phase"] == "grasp"

    changed = stabilizer.stabilize(
        {"phase": "place", "progress": "advancing", "risk_level": "medium", "confidence": 0.55},
        task,
        event_type="still_start",
    )

    assert changed["phase"] == "place"


def test_temporal_stabilizer_releases_high_risk_phase_quickly():
    stabilizer = TemporalStabilizer()
    task = _task()

    stabilizer.stabilize(
        {"phase": "grasp", "progress": "advancing", "risk_level": "low", "confidence": 0.9},
        task,
        event_type="gripper_close",
    )
    changed = stabilizer.stabilize(
        {
            "phase": "place",
            "progress": "stalled",
            "risk_level": "high",
            "imminent_failure": True,
            "confidence": 0.52,
        },
        task,
        event_type="periodic_sample",
    )

    assert changed["phase"] == "place"
    assert changed["risk_level"] == "high"


def test_agent_injects_same_rollout_memory_after_first_assessment(tmp_path):
    agent = VisualSnapshotAgent(
        task_config=_task(),
        vlm_runner=MockVLMRunner(latency=0.0),
        rollout_dir=tmp_path,
        output_jsonl=tmp_path / "snapshots.jsonl",
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    first = VisualObservationWindow(
        rollout_id="r1",
        frame_ids=[1],
        timestamps=[0.1],
        camera_name="camera_h",
        color_frames=[image],
        depth_frames=[],
        end_frame_id=1,
        end_timestamp=0.1,
    )
    prepared = agent.prepare(first)
    agent.apply(
        prepared,
        json.dumps(
            {
                "phase": "grasp",
                "progress": "advancing",
                "risk_level": "low",
                "confidence": 0.9,
            }
        ),
    )

    second = VisualObservationWindow(
        rollout_id="r1",
        frame_ids=[2],
        timestamps=[0.2],
        camera_name="camera_h",
        color_frames=[image],
        depth_frames=[],
        end_frame_id=2,
        end_timestamp=0.2,
    )
    next_prepared = agent.prepare(second)

    assert "Online rollout memory" in next_prepared.prompt
    assert "last_confirmed_phase=grasp" in next_prepared.prompt
