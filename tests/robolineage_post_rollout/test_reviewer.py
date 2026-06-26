from __future__ import annotations

import json

from robolineage_post_rollout import PostRolloutReviewAgent
from robolineage_shared_agents.visual_snapshot.exceptions import VLMInferenceError
from robolineage_schemas import validate


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_png(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(220, 20, 20)).save(path)


class _FakeRunner:
    def __init__(self, response: dict | list[dict]):
        self.responses = list(response) if isinstance(response, list) else [response]
        self.calls = []

    def run(self, prompt, images):
        self.calls.append({"prompt": prompt, "image_count": len(images)})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return json.dumps(self.responses[index])


class _FailingPacketRunner:
    def __init__(self, responses: list[dict], fail_on_call: int):
        self.responses = list(responses)
        self.fail_on_call = fail_on_call
        self.calls = []

    def run(self, prompt, images):
        self.calls.append({"prompt": prompt, "image_count": len(images)})
        if len(self.calls) == self.fail_on_call:
            raise VLMInferenceError("packet timeout")
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return json.dumps(self.responses[index])


def _assert_schema_valid(payload: dict, schema_name: str) -> None:
    errors = [issue for issue in validate(payload, schema_name) if issue.severity == "error"]
    assert errors == []


def test_post_review_splits_post_final_last_window_frames_before_early_context(tmp_path):
    rollout_dir = tmp_path / "rollout_terminal_tail_selection"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: stack red block on blue block",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 0.3,
                "frame_id": 8,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "gripper_close",
            },
            {
                "timestamp": 0.6,
                "frame_id": 16,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "still_start",
            },
            {
                "timestamp": 3.0,
                "frame_id": 168,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
            {
                "timestamp": 3.2,
                "frame_id": 183,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
            {
                "timestamp": 3.4,
                "frame_id": 198,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
        ],
    )
    manifest_rows = []
    for window_id, event_type, frame_ids in [
        ("000001_000001_sequence_start", "sequence_start", [1]),
        ("000002_000008_gripper_close", "gripper_close", [8]),
        ("000003_000016_still_start", "still_start", [16]),
        ("000004_000183_final_observation", "final_observation", [168, 183, 198]),
        ("000005_000216_contact_transition", "contact_transition", [208, 216, 238]),
    ]:
        image_paths = []
        for index, frame_id in enumerate(frame_ids):
            image_path = rollout_dir / "vsa_windows" / window_id / f"kf_{index:02d}_frame_{frame_id}.png"
            _write_png(image_path)
            image_paths.append(str(image_path.relative_to(rollout_dir)))
        manifest_rows.append(
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": window_id,
                "rollout_id": rollout_dir.name,
                "event_type": event_type,
                "anchor_frame_id": frame_ids[len(frame_ids) // 2],
                "end_frame_id": frame_ids[-1],
                "keyframe_ids": frame_ids,
                "image_paths": image_paths,
            }
        )
    _write_jsonl(rollout_dir / "vsa_windows" / "manifest.jsonl", manifest_rows)
    runner = _FakeRunner(
        {
            "final_success": True,
            "success_confidence": 0.9,
            "final_phase": "place",
            "retry_or_failure_evidence": [],
            "reasoning": "terminal images show stable placement",
        }
    )

    PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=6).run(rollout_dir)

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    review_packets = json.loads((rollout_dir / "review_packets.json").read_text(encoding="utf-8"))
    selected = annotation["vlm_review"]["image_frames"]
    assert selected == [168, 183, 198]
    assert review_packets[0]["purpose"] == "terminal_focus"
    assert review_packets[0]["image_frames"] == [168, 183, 198]
    assert review_packets[1]["purpose"] == "post_terminal_context"
    assert review_packets[1]["image_frames"] == [208, 216, 238]
    assert 1 not in selected


def test_post_review_vlm_failure_without_episode_end_coverage_is_uncertain(tmp_path):
    rollout_dir = tmp_path / "rollout_vlm_missing_terminal_tail"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: complete a generic manipulation task",
                "phases:",
                "  - approach",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.86,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "contact_transition",
            },
        ],
    )
    final_png = rollout_dir / "vsa_windows" / "000001_000010_final_observation" / "kf_00_frame_10.png"
    _write_png(final_png)
    _write_jsonl(
        rollout_dir / "vsa_windows" / "manifest.jsonl",
        [
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": "000001_000010_final_observation",
                "rollout_id": rollout_dir.name,
                "event_type": "final_observation",
                "anchor_frame_id": 10,
                "end_frame_id": 10,
                "keyframe_ids": [10],
                "image_paths": [str(final_png.relative_to(rollout_dir))],
            },
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": "000002_000020_contact_transition",
                "rollout_id": rollout_dir.name,
                "event_type": "contact_transition",
                "anchor_frame_id": 20,
                "end_frame_id": 20,
                "keyframe_ids": [20],
                "image_paths": [],
            },
        ],
    )
    runner = _FakeRunner(
        {
            "final_success": False,
            "success": False,
            "success_confidence": 0.88,
            "final_phase": "approach",
            "retry_or_failure_evidence": [
                {
                    "failure_type": "release_failure",
                    "phase": "finish",
                    "start_frame": 10,
                    "end_frame": 10,
                    "evidence_frames": [10],
                    "reasoning": "the selected image does not show stable completion",
                }
            ],
            "reasoning": "terminal failure in selected image",
        }
    )

    PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=4).run(rollout_dir)

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    summary = json.loads((rollout_dir / "rollout_summary.json").read_text(encoding="utf-8"))
    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert annotation["vlm_review"]["image_frames"] == [10]
    assert summary["final_success"] is True
    assert summary["success_status"] == "uncertain"
    assert summary["final_phase"] == "finish"
    assert summary["success_reasoning"] == "offline_vlm_failure_without_terminal_image_coverage"
    assert admission["decision"] == "needs_review"


def test_post_rollout_review_writes_final_artifacts_from_snapshots_jsonl(tmp_path):
    rollout_dir = tmp_path / "rollout_a"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: stack red block on blue block",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
                "failure_signals:",
                "  - object dropped",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "stalled",
                "risk_level": "high",
                "phase": "grasp",
                "imminent_failure": True,
                "confidence": 0.7,
                "needs_review": True,
                "raw_response": "{}",
                "trigger": "gripper_open",
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "still_start",
            },
        ],
    )

    result = PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    assert result.status == "completed"
    assert result.used_vlm is False
    for artifact in [
        "rollout_summary.json",
        "evidence_index.json",
        "annotation.final.json",
        "phase_timeline.final.jsonl",
        "failure_analysis.json",
        "dataset_admission.json",
        "review_report.md",
        "post_review_status.json",
    ]:
        assert (rollout_dir / artifact).exists()

    summary = json.loads((rollout_dir / "rollout_summary.json").read_text())
    assert summary["success_likely"] is True
    assert summary["final_phase"] == "place"

    failure = json.loads((rollout_dir / "failure_analysis.json").read_text())
    assert failure["candidate_count"] == 1
    assert failure["candidate_segments"][0]["phase"] == "grasp"
    _assert_schema_valid(failure, "failure_analysis")

    admission = json.loads((rollout_dir / "dataset_admission.json").read_text())
    assert admission["decision"] == "accepted"
    assert admission["admission_class"] == "accepted_with_labels"
    assert "recovery_training" in admission["data_use"]
    _assert_schema_valid(admission, "dataset_admission")

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text())
    assert annotation["outcome"]["final_success"] is True
    assert annotation["l1_annotation"]["phases"] == ["approach", "grasp", "place"]
    _assert_schema_valid(annotation, "annotation_final")


def test_post_rollout_review_tracks_retry_without_making_retry_a_phase(tmp_path):
    rollout_dir = tmp_path / "rollout_retry"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: place red block on blue block",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": '{"phase":"approach","confidence":0.8}',
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "stalled",
                "risk_level": "high",
                "phase": "place",
                "imminent_failure": True,
                "confidence": 0.55,
                "needs_review": True,
                "raw_response": '{"phase":"place","confidence":0.7}',
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "medium",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.52,
                "needs_review": True,
                "raw_response": '{"phase":"approach","confidence":0.8}',
            },
            {
                "timestamp": 3.0,
                "frame_id": 30,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": '{"phase":"grasp","confidence":0.8}',
            },
            {
                "timestamp": 4.0,
                "frame_id": 40,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": '{"phase":"place","confidence":0.9}',
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text())
    phases = [segment["phase"] for segment in annotation["phase_timeline"]]
    assert "retry" not in phases
    assert annotation["retry_events"]

    failure = json.loads((rollout_dir / "failure_analysis.json").read_text())
    assert failure["status"] == "recovered_failure_found"
    assert failure["recovered_count"] >= 1


def test_post_rollout_review_applies_vlm_final_annotation(tmp_path):
    rollout_dir = tmp_path / "rollout_vlm"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: pick and place",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.7,
                "needs_review": False,
                "raw_response": "{}",
            },
        ],
    )
    final_png = rollout_dir / "vsa_windows" / "000001_000020_final_observation" / "kf_00_frame_20.png"
    _write_png(final_png)
    _write_jsonl(
        rollout_dir / "vsa_windows" / "manifest.jsonl",
        [
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": "000001_000020_final_observation",
                "rollout_id": rollout_dir.name,
                "event_type": "final_observation",
                "anchor_frame_id": 20,
                "end_frame_id": 20,
                "keyframe_ids": [20],
                "image_paths": [str(final_png.relative_to(rollout_dir))],
            }
        ],
    )
    runner = _FakeRunner(
        {
            "success": True,
            "success_confidence": 0.91,
            "final_phase": "place",
            "phase_corrections": [
                {"start_frame": 20, "end_frame": 20, "phase": "place", "reason": "object placed"}
            ],
            "failure_events": [],
            "reasoning": "final image shows the task completed",
        }
    )

    result = PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=1).run(rollout_dir)

    assert result.used_vlm is True
    assert runner.calls
    summary = json.loads((rollout_dir / "rollout_summary.json").read_text())
    assert summary["final_success"] is True
    assert summary["success_confidence"] == 0.91
    assert summary["final_phase"] == "place"

    timeline = [
        json.loads(line)
        for line in (rollout_dir / "phase_timeline.final.jsonl").read_text().splitlines()
    ]
    assert timeline[-1]["phase"] == "place"
    assert timeline[-1]["corrected_by_review"] is True


def test_post_rollout_vlm_dataset_decision_cannot_reject_trainable_rollout(tmp_path):
    rollout_dir = tmp_path / "rollout_vlm_cannot_reject"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: generic supervised collection",
                "phases:",
                "  - approach",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
            },
        ],
    )
    final_png = rollout_dir / "vsa_windows" / "000001_000010_final_observation" / "kf_00_frame_10.png"
    _write_png(final_png)
    _write_jsonl(
        rollout_dir / "vsa_windows" / "manifest.jsonl",
        [
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": "000001_000010_final_observation",
                "rollout_id": rollout_dir.name,
                "event_type": "final_observation",
                "anchor_frame_id": 10,
                "end_frame_id": 10,
                "keyframe_ids": [10],
                "image_paths": [str(final_png.relative_to(rollout_dir))],
            }
        ],
    )
    runner = _FakeRunner(
        {
            "final_success": True,
            "success_confidence": 0.9,
            "final_phase": "finish",
            "label_quality": "uncertain",
            "training_usability": False,
            "dataset_decision": "rejected",
            "reasoning": "labels need review",
        }
    )

    PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=1).run(rollout_dir)

    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert admission["decision"] == "accepted"
    assert admission["accepted_for_training"] is True
    assert admission["label_quality"] == "uncertain"

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    assert "dataset_decision" not in annotation["vlm_review"]
    assert annotation["vlm_review"]["ignored_legacy_fields"] == ["dataset_decision"]


def test_post_rollout_vlm_without_images_cannot_invent_terminal_failure(tmp_path):
    rollout_dir = tmp_path / "rollout_no_image_vlm"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: generic manipulation",
                "phases:",
                "  - approach",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
            },
        ],
    )
    runner = _FakeRunner(
        {
            "final_success": False,
            "success": False,
            "final_phase": "approach",
            "failure_events": [
                {
                    "failure_type": "slip",
                    "phase": "finish",
                    "start_frame": 10,
                    "end_frame": 10,
                    "reasoning": "object slipped",
                }
            ],
            "reasoning": "terminal failure",
        }
    )

    PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=4).run(rollout_dir)

    summary = json.loads((rollout_dir / "rollout_summary.json").read_text(encoding="utf-8"))
    failure = json.loads((rollout_dir / "failure_analysis.json").read_text(encoding="utf-8"))
    assert summary["final_success"] is True
    assert summary["success_reasoning"] == "terminal_phase_confirmed_without_final_high_risk"
    assert failure["candidate_count"] == 0
    assert failure["status"] == "no_failure_candidates"


def test_post_review_terminal_final_observation_overrides_later_phase_lag(tmp_path):
    rollout_dir = tmp_path / "rollout_terminal_evidence_lag"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: pick and place",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "gripper_close",
            },
            {
                "timestamp": 2.0,
                "frame_id": 30,
                "progress": "advancing",
                "risk_level": "medium",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.62,
                "needs_review": True,
                "raw_response": (
                    "The release has occurred and the object is on the target.\n"
                    '{"phase":"place","progress":"advancing","risk_level":"medium",'
                    '"imminent_failure":false,"needs_review":true,"confidence":0.62}'
                ),
                "trigger": "final_observation",
            },
            {
                "timestamp": 3.0,
                "frame_id": 40,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.85,
                "needs_review": False,
                "raw_response": '{"phase":"grasp","progress":"advancing","risk_level":"low","confidence":0.85}',
                "trigger": "contact_transition",
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    summary = json.loads((rollout_dir / "rollout_summary.json").read_text(encoding="utf-8"))
    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert summary["final_success"] is True
    assert summary["final_phase"] == "place"
    assert summary["success_status"] == "uncertain"
    assert summary["success_reasoning"] == "terminal_phase_supported_by_terminal_evidence"
    assert annotation["outcome"]["terminal_evidence"]["frame_id"] == 30
    assert admission["decision"] == "needs_review"
    assert admission["admission_class"] == "successful_but_ambiguous"


def test_post_review_terminal_high_risk_is_uncertain_success_not_hard_failure(tmp_path):
    rollout_dir = tmp_path / "rollout_terminal_high_risk"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: finish a generic manipulation",
                "phases:",
                "  - approach",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "high",
                "phase": "finish",
                "imminent_failure": True,
                "confidence": 0.7,
                "needs_review": True,
                "raw_response": '{"phase":"finish","progress":"advancing","risk_level":"medium","confidence":0.7}',
                "trigger": "final_observation",
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    summary = json.loads((rollout_dir / "rollout_summary.json").read_text(encoding="utf-8"))
    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert summary["final_success"] is True
    assert summary["success_status"] == "uncertain"
    assert summary["success_reasoning"] == "terminal_phase_reached_with_unresolved_final_risk"
    assert admission["decision"] == "needs_review"
    assert admission["admission_class"] == "successful_but_ambiguous"


def test_post_rollout_review_indexes_vsa_window_keyframes(tmp_path):
    rollout_dir = tmp_path / "rollout_with_vsa_windows"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: complete a generic manipulation task",
                "phases:",
                "  - approach",
                "  - interact",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
                "frame_index_range": [0, 2],
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
                "frame_index_range": [18, 20],
            },
        ],
    )
    first_png = rollout_dir / "vsa_windows" / "000001_000001_sequence_start" / "kf_00_frame_1.png"
    final_png = rollout_dir / "vsa_windows" / "000002_000020_final_observation" / "kf_00_frame_20.png"
    _write_png(first_png)
    _write_png(final_png)
    _write_jsonl(
        rollout_dir / "vsa_windows" / "manifest.jsonl",
        [
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": "000001_000001_sequence_start",
                "rollout_id": rollout_dir.name,
                "event_type": "sequence_start",
                "anchor_frame_id": 1,
                "end_frame_id": 1,
                "keyframe_ids": [1],
                "image_paths": [str(first_png.relative_to(rollout_dir))],
            },
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": "000002_000020_final_observation",
                "rollout_id": rollout_dir.name,
                "event_type": "final_observation",
                "anchor_frame_id": 20,
                "end_frame_id": 20,
                "keyframe_ids": [20],
                "image_paths": [str(final_png.relative_to(rollout_dir))],
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    evidence = json.loads((rollout_dir / "evidence_index.json").read_text(encoding="utf-8"))
    assert evidence["image_count"] == 2
    assert evidence["image_frame_count"] == 2
    assert evidence["vsa_window_count"] == 2
    assert evidence["final_observation_frames"] == [20]
    assert any(item["event_type"] == "final_observation" for item in evidence["key_frames"])


def test_dataset_admission_keeps_final_failure_out_of_policy_training(tmp_path):
    rollout_dir = tmp_path / "rollout_failure_not_policy_trainable"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: generic supervised collection",
                "phases:",
                "  - approach",
                "  - interact",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "stalled",
                "risk_level": "medium",
                "phase": "interact",
                "imminent_failure": False,
                "confidence": 0.4,
                "needs_review": True,
                "raw_response": "{}",
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert admission["decision"] == "needs_review"
    assert admission["accepted_for_training"] is False
    assert admission["label_quality"] == "uncertain"
    assert admission["review_reason"]
    assert admission["recommended_split"] is None
    _assert_schema_valid(admission, "dataset_admission")


def test_dataset_admission_keeps_uncertain_terminal_success_policy_trainable(tmp_path):
    rollout_dir = tmp_path / "rollout_uncertain_success_trainable"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: generic manipulation",
                "phases:",
                "  - approach",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "high",
                "phase": "finish",
                "imminent_failure": True,
                "confidence": 0.7,
                "needs_review": True,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert admission["decision"] == "needs_review"
    assert admission["admission_class"] == "successful_but_ambiguous"
    assert admission["accepted_for_training"] is True
    assert admission["recommended_split"] == "train"
    assert "success_trajectory" in admission["data_use"]
    _assert_schema_valid(admission, "dataset_admission")


def test_post_review_multichunk_aggregates_terminal_success_after_tail_contact(tmp_path):
    rollout_dir = tmp_path / "rollout_multichunk_terminal_success"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: stack red block on blue block",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 40,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "gripper_close",
            },
            {
                "timestamp": 2.0,
                "frame_id": 100,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.72,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
            {
                "timestamp": 3.0,
                "frame_id": 130,
                "progress": "advancing",
                "risk_level": "medium",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.62,
                "needs_review": True,
                "raw_response": "{}",
                "trigger": "contact_transition",
            },
        ],
    )
    manifest_rows = []
    for window_id, event_type, frame_ids in [
        ("000001_000001_sequence_start", "sequence_start", [1, 10, 20]),
        ("000002_000040_gripper_close", "gripper_close", [35, 40, 45]),
        ("000003_000100_final_observation", "final_observation", [90, 100, 110]),
        ("000004_000130_contact_transition", "contact_transition", [120, 130, 140]),
    ]:
        image_paths = []
        for index, frame_id in enumerate(frame_ids):
            image_path = rollout_dir / "vsa_windows" / window_id / f"kf_{index:02d}_frame_{frame_id}.png"
            _write_png(image_path)
            image_paths.append(str(image_path.relative_to(rollout_dir)))
        manifest_rows.append(
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": window_id,
                "rollout_id": rollout_dir.name,
                "event_type": event_type,
                "anchor_frame_id": frame_ids[len(frame_ids) // 2],
                "end_frame_id": frame_ids[-1],
                "keyframe_ids": frame_ids,
                "image_paths": image_paths,
            }
        )
    _write_jsonl(rollout_dir / "vsa_windows" / "manifest.jsonl", manifest_rows)
    runner = _FakeRunner(
        [
            {
                "packet_terminal_state": "success",
                "terminal_intact_at_packet_end": True,
                "post_terminal_status": "intact",
                "evidence_frames": [100, 110],
                "success_confidence": 0.86,
                "reasoning": "terminal packet shows stable stack",
            },
            {
                "packet_terminal_state": "not_visible",
                "terminal_intact_at_packet_end": None,
                "post_terminal_status": "not_applicable",
                "evidence_frames": [1, 40],
                "reasoning": "setup and grasp context",
            },
            {
                "packet_terminal_state": "success",
                "terminal_intact_at_packet_end": True,
                "post_terminal_status": "intact",
                "evidence_frames": [120, 130, 140],
                "success_confidence": 0.82,
                "reasoning": "tail contact does not break terminal stack",
            },
        ]
    )

    PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=4).run(rollout_dir)

    assert len(runner.calls) >= 2
    assert (rollout_dir / "review_packets.json").exists()
    assert (rollout_dir / "vlm_packet_reviews.jsonl").exists()
    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert annotation["vlm_review"]["status"] == "packet_aggregated"
    assert annotation["vlm_review"]["final_success"] is True
    assert annotation["outcome"]["success_status"] == "success"
    assert admission["decision"] == "accepted"
    assert admission["accepted_for_training"] is True


def test_post_review_multichunk_keeps_terminal_review_when_later_packet_fails(tmp_path):
    rollout_dir = tmp_path / "rollout_multichunk_partial_packet_failure"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: stack red block on blue block",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 40,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "gripper_close",
            },
            {
                "timestamp": 2.0,
                "frame_id": 100,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.72,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
            {
                "timestamp": 3.0,
                "frame_id": 130,
                "progress": "advancing",
                "risk_level": "medium",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.62,
                "needs_review": True,
                "raw_response": "{}",
                "trigger": "contact_transition",
            },
        ],
    )
    manifest_rows = []
    for window_id, event_type, frame_ids in [
        ("000001_000001_sequence_start", "sequence_start", [1, 10, 20]),
        ("000002_000040_gripper_close", "gripper_close", [35, 40, 45]),
        ("000003_000100_final_observation", "final_observation", [90, 100, 110]),
        ("000004_000130_contact_transition", "contact_transition", [120, 130, 140]),
    ]:
        image_paths = []
        for index, frame_id in enumerate(frame_ids):
            image_path = rollout_dir / "vsa_windows" / window_id / f"kf_{index:02d}_frame_{frame_id}.png"
            _write_png(image_path)
            image_paths.append(str(image_path.relative_to(rollout_dir)))
        manifest_rows.append(
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": window_id,
                "rollout_id": rollout_dir.name,
                "event_type": event_type,
                "anchor_frame_id": frame_ids[len(frame_ids) // 2],
                "end_frame_id": frame_ids[-1],
                "keyframe_ids": frame_ids,
                "image_paths": image_paths,
            }
        )
    _write_jsonl(rollout_dir / "vsa_windows" / "manifest.jsonl", manifest_rows)
    runner = _FailingPacketRunner(
        [
            {
                "packet_terminal_state": "success",
                "terminal_intact_at_packet_end": True,
                "post_terminal_status": "intact",
                "evidence_frames": [100, 110],
                "success_confidence": 0.86,
                "reasoning": "terminal packet shows stable stack",
            },
            {
                "packet_terminal_state": "not_visible",
                "terminal_intact_at_packet_end": None,
                "post_terminal_status": "not_applicable",
                "reasoning": "early context",
            },
        ],
        fail_on_call=2,
    )

    result = PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=4).run(rollout_dir)

    assert len(runner.calls) >= 2
    assert result.used_vlm is True
    assert result.vlm_error is None
    packet_reviews = [
        json.loads(line)
        for line in (rollout_dir / "vlm_packet_reviews.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert annotation["vlm_review"]["status"] == "packet_partially_aggregated"
    assert annotation["vlm_review"]["final_success"] is True
    assert annotation["vlm_review"]["failed_packet_count"] == 1
    assert packet_reviews[1]["status"] == "failed"
    assert packet_reviews[1]["purpose"] == "post_terminal_context"
    assert "packet timeout" in packet_reviews[1]["error"]
    assert admission["accepted_for_training"] is True


def test_post_review_multichunk_splits_terminal_and_post_terminal_packets(tmp_path):
    rollout_dir = tmp_path / "rollout_multichunk_split_terminal_tail"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: stack red block on blue block",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 40,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "gripper_close",
            },
            {
                "timestamp": 2.0,
                "frame_id": 100,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
            {
                "timestamp": 3.0,
                "frame_id": 130,
                "progress": "advancing",
                "risk_level": "medium",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.62,
                "needs_review": True,
                "raw_response": "{}",
                "trigger": "gripper_close",
            },
        ],
    )
    manifest_rows = []
    for window_id, event_type, frame_ids in [
        ("000001_000001_sequence_start", "sequence_start", [1, 10, 20]),
        ("000002_000040_gripper_close", "gripper_close", [35, 40, 45]),
        ("000003_000100_final_observation", "final_observation", [90, 100, 110]),
        ("000004_000130_gripper_close", "gripper_close", [120, 130, 140]),
    ]:
        image_paths = []
        for index, frame_id in enumerate(frame_ids):
            image_path = rollout_dir / "vsa_windows" / window_id / f"kf_{index:02d}_frame_{frame_id}.png"
            _write_png(image_path)
            image_paths.append(str(image_path.relative_to(rollout_dir)))
        manifest_rows.append(
            {
                "schema_version": "RoboLineage.vsa_window_manifest.v1",
                "window_id": window_id,
                "rollout_id": rollout_dir.name,
                "event_type": event_type,
                "anchor_frame_id": frame_ids[len(frame_ids) // 2],
                "end_frame_id": frame_ids[-1],
                "keyframe_ids": frame_ids,
                "image_paths": image_paths,
            }
        )
    _write_jsonl(rollout_dir / "vsa_windows" / "manifest.jsonl", manifest_rows)
    runner = _FakeRunner(
        [
            {
                "packet_terminal_state": "success",
                "terminal_intact_at_packet_end": True,
                "post_terminal_status": "intact",
                "success_confidence": 0.9,
                "reasoning": "terminal packet shows stable stack",
            },
            {
                "packet_terminal_state": "failure",
                "terminal_intact_at_packet_end": None,
                "post_terminal_status": "uncertain",
                "success_confidence": 0.4,
                "retry_or_failure_evidence": [
                    {
                        "failure_type": "uncertain",
                        "phase": "grasp",
                        "start_frame": 120,
                        "end_frame": 140,
                        "recovered": False,
                        "evidence_frames": [120, 130, 140],
                        "reasoning": "tail packet is ambiguous but does not prove broken terminal state",
                    }
                ],
                "reasoning": "post-terminal motion does not prove the terminal state was destroyed",
            },
            {
                "packet_terminal_state": "not_visible",
                "terminal_intact_at_packet_end": None,
                "post_terminal_status": "not_applicable",
                "reasoning": "early context",
            },
        ]
    )

    PostRolloutReviewAgent(vlm_runner=runner, use_vlm=True, max_review_images=4).run(rollout_dir)

    review_packets = json.loads((rollout_dir / "review_packets.json").read_text(encoding="utf-8"))
    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    admission = json.loads((rollout_dir / "dataset_admission.json").read_text(encoding="utf-8"))
    assert review_packets[0]["purpose"] == "terminal_focus"
    assert review_packets[0]["image_frames"] == [90, 100, 110]
    assert review_packets[1]["purpose"] == "post_terminal_context"
    assert review_packets[1]["image_frames"] == [120, 130, 140]
    assert annotation["vlm_review"]["final_success"] is True
    assert admission["accepted_for_training"] is True


def test_retry_requires_evidence_beyond_low_risk_phase_regression(tmp_path):
    rollout_dir = tmp_path / "rollout_phase_jitter_not_retry"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: generic manipulation",
                "phases:",
                "  - approach",
                "  - interact",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.75,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.1,
                "frame_id": 11,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "interact",
                "imminent_failure": False,
                "confidence": 0.55,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.85,
                "needs_review": False,
                "raw_response": "{}",
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    failure = json.loads((rollout_dir / "failure_analysis.json").read_text(encoding="utf-8"))
    assert [segment["phase"] for segment in annotation["phase_timeline"]] == ["approach", "finish"]
    assert annotation["retry_events"] == []
    assert failure["status"] == "no_failure_candidates"


def test_post_review_merges_zero_duration_segment_without_evidence(tmp_path):
    rollout_dir = tmp_path / "rollout_zero_duration_cleanup"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: generic manipulation",
                "phases:",
                "  - approach",
                "  - interact",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "interact",
                "imminent_failure": False,
                "confidence": 0.54,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "periodic_sample",
            },
            {
                "timestamp": 1.0,
                "frame_id": 11,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.82,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "periodic_sample",
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    assert [segment["phase"] for segment in annotation["phase_timeline"]] == ["approach", "finish"]
    assert annotation["phase_timeline"][0]["zero_duration_reason"] == "sequence_start"


def test_post_review_retains_zero_duration_segment_with_failure_evidence(tmp_path):
    rollout_dir = tmp_path / "rollout_zero_duration_failure"
    rollout_dir.mkdir()
    (rollout_dir / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: generic manipulation",
                "phases:",
                "  - approach",
                "  - interact",
                "  - finish",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout_dir / "snapshots.jsonl",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "sequence_start",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "stalled",
                "risk_level": "high",
                "phase": "interact",
                "imminent_failure": True,
                "confidence": 0.7,
                "needs_review": True,
                "raw_response": "{}",
                "trigger": "contact_transition",
            },
            {
                "timestamp": 1.0,
                "frame_id": 11,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "finish",
                "imminent_failure": False,
                "confidence": 0.82,
                "needs_review": False,
                "raw_response": "{}",
                "trigger": "final_observation",
            },
        ],
    )

    PostRolloutReviewAgent(use_vlm=False).run(rollout_dir)

    annotation = json.loads((rollout_dir / "annotation.final.json").read_text(encoding="utf-8"))
    interact = [segment for segment in annotation["phase_timeline"] if segment["phase"] == "interact"][0]
    assert interact["duration_sec"] == 0.0
    assert interact["zero_duration_reason"] == "failure_evidence"
