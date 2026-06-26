from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from robolineage_train.dataset_adapters.rosbag_act_hdf5 import build_rosbag_act_hdf5_dataset


cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")


def _jpeg_b64(value: int) -> str:
    frame = np.full((6, 8, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _vector(value: float) -> list[float]:
    vec = [0.0] * 27
    vec[:7] = [value] * 7
    vec[7:14] = [value + 100.0] * 7
    vec[14:21] = [value + 200.0] * 7
    vec[21:27] = [value + 300.0] * 6
    return vec


def _write_jsonl_rosbag_fixture(raw_dir: Path) -> None:
    bag_dir = raw_dir / "rosbag2"
    bag_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for i, stamp_ns in enumerate((0, 100_000_000, 200_000_000, 300_000_000)):
        records.extend(
            [
                {
                    "topic": "/cam/head/image/compressed",
                    "stamp_ns": stamp_ns,
                    "type": "sensor_msgs/msg/CompressedImage",
                    "data": {"format": "jpeg", "bytes_b64": _jpeg_b64(20 + i)},
                },
                {
                    "topic": "/cam/right_wrist/image/compressed",
                    "stamp_ns": stamp_ns,
                    "type": "sensor_msgs/msg/CompressedImage",
                    "data": {"format": "jpeg", "bytes_b64": _jpeg_b64(80 + i)},
                },
                {
                    "topic": "/arm/left/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": _vector(i / 3.0)},
                },
                {
                    "topic": "/arm/right/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": _vector(10.0 + (2.0 * i / 3.0))},
                },
            ]
        )
    with (bag_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    (raw_dir / "raw_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "RoboLineage.raw_rosbag_manifest.v1",
                "status": "closed",
                "bag_dir": str(bag_dir),
                "topics": [
                    "/cam/head/image/compressed",
                    "/cam/right_wrist/image/compressed",
                    "/arm/left/state",
                    "/arm/right/state",
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_fixed_rate_interpolation_fixture(raw_dir: Path) -> None:
    bag_dir = raw_dir / "rosbag2"
    bag_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    camera_stamps = (0, 33_333_333, 66_666_667, 100_000_000)
    for i, stamp_ns in enumerate(camera_stamps):
        records.extend(
            [
                {
                    "topic": "/cam/head/image/compressed",
                    "stamp_ns": stamp_ns,
                    "type": "sensor_msgs/msg/CompressedImage",
                    "data": {"format": "jpeg", "bytes_b64": _jpeg_b64(20 + i)},
                },
                {
                    "topic": "/cam/right_wrist/image/compressed",
                    "stamp_ns": stamp_ns,
                    "type": "sensor_msgs/msg/CompressedImage",
                    "data": {"format": "jpeg", "bytes_b64": _jpeg_b64(80 + i)},
                },
            ]
        )
    for stamp_ns, left_value, right_value in (
        (0, 0.0, 10.0),
        (100_000_000, 1.0, 12.0),
    ):
        records.extend(
            [
                {
                    "topic": "/arm/left/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": _vector(left_value)},
                },
                {
                    "topic": "/arm/right/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": _vector(right_value)},
                },
            ]
        )
    with (bag_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda row: (int(row["stamp_ns"]), str(row["topic"]))):
            f.write(json.dumps(record, sort_keys=True) + "\n")
    (raw_dir / "raw_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "RoboLineage.raw_rosbag_manifest.v1",
                "status": "closed",
                "bag_dir": str(bag_dir),
            }
        ),
        encoding="utf-8",
    )


def _write_sparse_camera_fixture(raw_dir: Path) -> None:
    bag_dir = raw_dir / "rosbag2"
    bag_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for i, stamp_ns in enumerate((0, 33_333_333, 66_666_667, 100_000_000, 133_333_333, 166_666_667, 200_000_000)):
        records.append(
            {
                "topic": "/cam/head/image/compressed",
                "stamp_ns": stamp_ns,
                "type": "sensor_msgs/msg/CompressedImage",
                "data": {"format": "jpeg", "bytes_b64": _jpeg_b64(20 + i)},
            }
        )
    for i, stamp_ns in enumerate((0, 200_000_000)):
        records.append(
            {
                "topic": "/cam/right_wrist/image/compressed",
                "stamp_ns": stamp_ns,
                "type": "sensor_msgs/msg/CompressedImage",
                "data": {"format": "jpeg", "bytes_b64": _jpeg_b64(80 + i)},
            }
        )
    for stamp_ns, left_value, right_value in (
        (0, 0.0, 10.0),
        (200_000_000, 2.0, 14.0),
    ):
        records.extend(
            [
                {
                    "topic": "/arm/left/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": _vector(left_value)},
                },
                {
                    "topic": "/arm/right/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": _vector(right_value)},
                },
            ]
        )
    with (bag_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda row: (int(row["stamp_ns"]), str(row["topic"]))):
            f.write(json.dumps(record, sort_keys=True) + "\n")
    (raw_dir / "raw_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "RoboLineage.raw_rosbag_manifest.v1",
                "status": "closed",
                "bag_dir": str(bag_dir),
            }
        ),
        encoding="utf-8",
    )


def test_rosbag_act_hdf5_adapter_resamples_to_fixed_30hz_and_interpolates_state(tmp_path):
    rollout = tmp_path / "rollouts" / "r1"
    raw_dir = rollout / "raw"
    raw_dir.mkdir(parents=True)
    _write_fixed_rate_interpolation_fixture(raw_dir)
    selected = tmp_path / "selected_rollouts.json"
    selected.write_text(
        json.dumps(
            {
                "schema_version": "RoboLineage.selected_rollouts.v1",
                "dataset_version": "v1",
                "selected_rollouts": [
                    {"rollout_id": "r1", "rollout_dir": str(rollout), "decision": "accepted"}
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_rosbag_act_hdf5_dataset(
        selected_rollouts_path=selected,
        output_dir=tmp_path / "dataset",
        camera_topics={
            "head": "/cam/head/image/compressed",
            "right_wrist": "/cam/right_wrist/image/compressed",
        },
        left_state_topic="/arm/left/state",
        right_state_topic="/arm/right/state",
        camera_names=("head", "right_wrist"),
        target_hz=30.0,
        jpeg_quality=50,
        overwrite=True,
    )

    assert report["schema_version"] == "RoboLineage.rosbag_act_hdf5_dataset.v1"
    assert report["exported_episode_count"] == 1
    assert report["target_hz"] == 30.0
    assert report["robot_state_semantics"]["policy"] == "driver_native_passthrough"
    assert report["robot_state_semantics"]["unit_conversion"] == "none"
    assert report["robot_state_semantics"]["coordinate_transform"] == "none"
    assert not (raw_dir / "frames.csv").exists()
    assert not (raw_dir / "videos").exists()
    episode = tmp_path / "dataset" / "episode_0.hdf5"
    with h5py.File(episode, "r") as root:
        assert root.attrs["sim"] == np.False_
        assert root.attrs["frame_rate"] == 30
        assert root.attrs["robot_state_semantics"] == "driver_native_passthrough"
        assert root.attrs["robot_state_unit_conversion"] == "none"
        assert root.attrs["robot_state_coordinate_transform"] == "none"
        assert root["/action"].shape == (4, 14)
        assert root["/observations/qpos"].shape == (4, 14)
        assert root["/observations/qvel"].shape == (4, 14)
        assert root["/observations/eef"].shape == (4, 14)
        assert root["/observations/effort"].shape == (4, 14)
        qpos = root["/observations/qpos"][()]
        assert qpos[:, 0].tolist() == pytest.approx([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], abs=1e-5)
        assert qpos[:, 7].tolist() == pytest.approx([10.0, 10.0 + 2.0 / 3.0, 10.0 + 4.0 / 3.0, 12.0], abs=1e-5)
        assert set(root["/observations/images"].keys()) == {"head", "right_wrist"}
        for camera in ("head", "right_wrist"):
            first = root[f"/observations/images/{camera}"][0]
            assert cv2.imdecode(first, cv2.IMREAD_COLOR) is not None


def test_sparse_camera_frames_are_hold_last_filled_and_reported(tmp_path):
    rollout = tmp_path / "rollouts" / "r1"
    raw_dir = rollout / "raw"
    raw_dir.mkdir(parents=True)
    _write_sparse_camera_fixture(raw_dir)
    selected = tmp_path / "selected_rollouts.json"
    selected.write_text(
        json.dumps(
            {
                "schema_version": "RoboLineage.selected_rollouts.v1",
                "dataset_version": "v1",
                "selected_rollouts": [
                    {"rollout_id": "r1", "rollout_dir": str(rollout), "decision": "accepted"}
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_rosbag_act_hdf5_dataset(
        selected_rollouts_path=selected,
        output_dir=tmp_path / "dataset",
        camera_topics={
            "head": "/cam/head/image/compressed",
            "right_wrist": "/cam/right_wrist/image/compressed",
        },
        left_state_topic="/arm/left/state",
        right_state_topic="/arm/right/state",
        camera_names=("head", "right_wrist"),
        target_hz=30.0,
        jpeg_quality=50,
        overwrite=True,
    )

    episode = tmp_path / "dataset" / "episode_0.hdf5"
    with h5py.File(episode, "r") as root:
        assert root["/action"].shape == (7, 14)
        assert cv2.imdecode(root["/observations/images/right_wrist"][3], cv2.IMREAD_COLOR) is not None
    right_stats = report["episodes"][0]["image_alignment"]["right_wrist"]
    assert right_stats["stale_frame_count"] == 3
    assert right_stats["max_abs_dt_sec"] == pytest.approx(4.0 / 30.0, abs=1e-6)
    assert right_stats["fill_policy"] == "nearest_with_hold_last_for_stale"
