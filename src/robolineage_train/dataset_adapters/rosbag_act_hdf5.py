from __future__ import annotations

import argparse
import base64
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "RoboLineage.rosbag_act_hdf5_dataset.v1"
DEFAULT_CAMERAS = ("head", "right_wrist")
ROBOT_STATE_SEMANTICS = {
    "policy": "driver_native_passthrough",
    "unit_conversion": "none",
    "coordinate_transform": "none",
    "column_order": "RobotStatus 27-vector as recorded in rosbag2",
    "resampling": "linear interpolation over time for numeric state/action arrays",
}
CAMERA_ALIASES = {
    "head": ("head", "camera_h", "front", "primary"),
    "left_wrist": ("left_wrist", "camera_l", "left"),
    "right_wrist": ("right_wrist", "camera_r", "right"),
    "camera_h": ("camera_h", "head", "front", "primary"),
    "camera_l": ("camera_l", "left_wrist", "left"),
    "camera_r": ("camera_r", "right_wrist", "right"),
}


@dataclass(frozen=True)
class RosbagRecord:
    topic: str
    stamp_ns: int
    msg_type: str
    data: Any


@dataclass(frozen=True)
class _ImageSample:
    timestamp: float
    bgr: Any


@dataclass(frozen=True)
class _VectorSample:
    timestamp: float
    vector: Any


@dataclass(frozen=True)
class _ImageSelection:
    samples: list[_ImageSample]
    stats: dict[str, Any]


@dataclass(frozen=True)
class _EpisodeWriteResult:
    frame_count: int
    image_alignment: dict[str, dict[str, Any]]


def build_rosbag_act_hdf5_dataset(
    *,
    selected_rollouts_path: Path,
    output_dir: Path,
    camera_topics: dict[str, str] | None = None,
    left_state_topic: str | None = None,
    right_state_topic: str | None = None,
    camera_names: tuple[str, ...] = DEFAULT_CAMERAS,
    target_hz: float = 30.0,
    jpeg_quality: int = 50,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert selected RoboLineage rosbag2 raw rollouts directly to ACT HDF5.

    The HDF5 layout intentionally matches the previous generated ACT adapter:
    contiguous ``episode_*.hdf5`` files, padded JPEG byte rows under
    ``/observations/images/{camera}``, 14-wide qpos/qvel/eef/effort/action
    arrays, and zero-filled base fields.
    """

    import h5py  # noqa: F401
    import numpy as np  # noqa: F401

    selected_rollouts_path = Path(selected_rollouts_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output_dir is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_payload = json.loads(selected_rollouts_path.read_text(encoding="utf-8"))
    selected = selected_payload.get("selected_rollouts")
    if not isinstance(selected, list):
        raise ValueError("selected_rollouts JSON must contain a selected_rollouts list")

    episodes: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            skipped.append({"rollout_id": f"index_{index}", "reason": "invalid_selected_rollout_entry"})
            continue
        rollout_id = str(item.get("rollout_id") or f"episode_{index}")
        rollout_dir = Path(str(item.get("rollout_dir") or ""))
        raw_dir = rollout_dir / "raw" if (rollout_dir / "raw").exists() else rollout_dir
        output_path = output_dir / f"episode_{len(episodes)}.hdf5"
        try:
            episode_result = write_episode(
                raw_dir=raw_dir,
                output_path=output_path,
                camera_names=camera_names,
                target_hz=target_hz,
                jpeg_quality=jpeg_quality,
                camera_topics=camera_topics,
                left_state_topic=left_state_topic,
                right_state_topic=right_state_topic,
            )
        except Exception as exc:
            skipped.append({"rollout_id": rollout_id, "reason": repr(exc)})
            continue
        episodes.append(
            {
                "episode_index": len(episodes),
                "rollout_id": rollout_id,
                "source_rollout_dir": str(rollout_dir),
                "source_raw_dir": str(raw_dir),
                "episode_path": str(output_path),
                "frame_count": episode_result.frame_count,
                "image_alignment": episode_result.image_alignment,
            }
        )

    if not episodes:
        raise ValueError(f"no rosbag ACT HDF5 episodes exported; skipped={skipped}")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "camera_names": list(camera_names),
        "target_hz": float(target_hz),
        "robot_state_semantics": dict(ROBOT_STATE_SEMANTICS),
        "source_dataset_version": selected_payload.get("dataset_version"),
        "selected_rollout_count": len(selected),
        "exported_episode_count": len(episodes),
        "episodes": episodes,
        "skipped_rollouts": skipped,
        "validation": validate_dataset(output_dir, camera_names),
    }
    _write_json(output_dir / "ROBOLINEAGE_generated_dataset_report.json", report)
    _write_json(output_dir / "dataset_adapter_report.json", report)
    return report


def write_episode(
    *,
    raw_dir: Path,
    output_path: Path,
    camera_names: tuple[str, ...],
    target_hz: float,
    jpeg_quality: int,
    camera_topics: dict[str, str] | None = None,
    left_state_topic: str | None = None,
    right_state_topic: str | None = None,
) -> _EpisodeWriteResult:
    import cv2
    import h5py
    import numpy as np

    records = list(iter_rosbag_records(raw_dir))
    if not records:
        raise ValueError(f"rosbag contains no records: {raw_dir}")
    topic_names = sorted({record.topic for record in records})
    resolved_camera_topics = {
        camera: _resolve_camera_topic(camera, camera_topics, topic_names)
        for camera in camera_names
    }
    left_topic = _resolve_state_topic(left_state_topic, topic_names, side="left")
    right_topic = _resolve_state_topic(right_state_topic, topic_names, side="right")

    images: dict[str, list[_ImageSample]] = {camera: [] for camera in camera_names}
    left: list[_VectorSample] = []
    right: list[_VectorSample] = []
    for record in records:
        ts = float(record.stamp_ns) / 1_000_000_000.0
        for camera, topic in resolved_camera_topics.items():
            if record.topic == topic:
                images[camera].append(_ImageSample(timestamp=ts, bgr=_decode_image_bgr(record)))
                break
        if record.topic == left_topic:
            left.append(_VectorSample(timestamp=ts, vector=_decode_vector(record)))
        elif record.topic == right_topic:
            right.append(_VectorSample(timestamp=ts, vector=_decode_vector(record)))

    for camera, rows in images.items():
        rows.sort(key=lambda row: row.timestamp)
        if not rows:
            raise ValueError(f"missing camera samples for {camera} topic={resolved_camera_topics[camera]}")
    left.sort(key=lambda row: row.timestamp)
    right.sort(key=lambda row: row.timestamp)
    if not left or not right:
        raise ValueError("left/right robot-state samples are required")

    target_timestamps = _build_target_timestamps(images, left, right, camera_names, target_hz)
    encoded_by_camera: dict[str, list[Any]] = {}
    image_alignment: dict[str, dict[str, Any]] = {}
    max_width = 0
    for camera in camera_names:
        selection = _select_image_samples(
            images[camera],
            target_timestamps,
            stale_threshold=1.0 / float(target_hz),
        )
        selected = selection.samples
        image_alignment[camera] = selection.stats
        encoded: list[Any] = []
        for sample in selected:
            ok, jpeg = cv2.imencode(".jpg", sample.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                raise RuntimeError(f"failed to encode JPEG for camera {camera}")
            encoded.append(jpeg)
            max_width = max(max_width, len(jpeg))
        encoded_by_camera[camera] = encoded

    numeric = _numeric_from_vectors(
        left=_interpolate_vectors(left, target_timestamps),
        right=_interpolate_vectors(right, target_timestamps),
    )
    frame_count = len(target_timestamps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["frame_rate"] = int(round(float(target_hz)))
        root.attrs["robot_state_semantics"] = ROBOT_STATE_SEMANTICS["policy"]
        root.attrs["robot_state_unit_conversion"] = ROBOT_STATE_SEMANTICS["unit_conversion"]
        root.attrs["robot_state_coordinate_transform"] = ROBOT_STATE_SEMANTICS["coordinate_transform"]
        obs = root.create_group("observations")
        image_group = obs.create_group("images")
        for camera in camera_names:
            padded = [np.pad(jpeg, (0, max_width - len(jpeg)), constant_values=0) for jpeg in encoded_by_camera[camera]]
            image_group.create_dataset(camera, data=np.asarray(padded, dtype=np.uint8), chunks=(1, max_width))
        for key in ("qpos", "qvel", "eef", "effort", "robot_base", "base_velocity"):
            obs.create_dataset(key, data=numeric[key])
        for key in ("action", "action_eef", "action_base", "action_velocity"):
            root.create_dataset(key, data=numeric[key])
    return _EpisodeWriteResult(frame_count=frame_count, image_alignment=image_alignment)


def iter_rosbag_records(raw_dir_or_bag_dir: Path) -> Iterable[RosbagRecord]:
    bag_dir = _resolve_bag_dir(Path(raw_dir_or_bag_dir))
    jsonl = bag_dir / "messages.jsonl"
    if jsonl.exists():
        yield from _iter_jsonl_records(jsonl)
        return
    yield from _iter_rosbag2_records(bag_dir)


def validate_dataset(dataset_dir: Path, camera_names: tuple[str, ...]) -> dict[str, Any]:
    import cv2
    import h5py

    episodes = sorted(dataset_dir.glob("episode_*.hdf5"), key=lambda path: int(path.stem.split("_")[1]))
    checked: list[dict[str, Any]] = []
    for expected_index, path in enumerate(episodes):
        if path.name != f"episode_{expected_index}.hdf5":
            raise ValueError(f"non-contiguous episode file: {path.name}")
        with h5py.File(path, "r") as root:
            frame_count = root["/action"].shape[0]
            for key in (
                "/action",
                "/action_eef",
                "/observations/qpos",
                "/observations/qvel",
                "/observations/eef",
                "/observations/effort",
            ):
                if root[key].shape != (frame_count, 14):
                    raise ValueError(f"{path}:{key} has invalid shape {root[key].shape}")
            for camera in camera_names:
                data = root[f"/observations/images/{camera}"]
                if data.shape[0] != frame_count:
                    raise ValueError(f"{path}: camera {camera} frame mismatch")
                if cv2.imdecode(data[0], cv2.IMREAD_COLOR) is None:
                    raise ValueError(f"{path}: camera {camera} first frame is not JPEG-decodable")
        checked.append({"episode": path.name, "frame_count": int(frame_count)})
    return {"status": "passed", "checked_episode_count": len(checked), "episodes": checked}


def _iter_jsonl_records(path: Path) -> Iterable[RosbagRecord]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            yield RosbagRecord(
                topic=str(row["topic"]),
                stamp_ns=int(row["stamp_ns"]),
                msg_type=str(row.get("type") or ""),
                data=row.get("data"),
            )


def _iter_rosbag2_records(bag_dir: Path) -> Iterable[RosbagRecord]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover - depends on ROS host
        raise RuntimeError(
            "rosbag2_py/rclpy are required to read native rosbag2 directories. "
            "For unit tests, use a rosbag2/messages.jsonl fixture."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    msg_classes: dict[str, Any] = {}
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        msg_type = topic_types.get(topic, "")
        if msg_type not in msg_classes:
            msg_classes[msg_type] = get_message(msg_type)
        yield RosbagRecord(
            topic=str(topic),
            stamp_ns=int(stamp_ns),
            msg_type=msg_type,
            data=deserialize_message(payload, msg_classes[msg_type]),
        )


def _resolve_bag_dir(path: Path) -> Path:
    if (path / "messages.jsonl").exists() or (path / "metadata.yaml").exists():
        return path
    manifest = path / "raw_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        bag_dir = Path(str(payload.get("bag_dir") or ""))
        if bag_dir:
            return bag_dir
    return path / "rosbag2"


def _resolve_camera_topic(
    camera: str,
    explicit: dict[str, str] | None,
    topics: list[str],
) -> str:
    if explicit and explicit.get(camera):
        return str(explicit[camera])
    aliases = CAMERA_ALIASES.get(camera, (camera,))
    candidates = [
        topic for topic in topics
        if any(alias in topic for alias in aliases)
        and any(token in topic.lower() for token in ("image", "color", "camera", "compressed"))
    ]
    if not candidates:
        raise KeyError(f"cannot infer camera topic for {camera!r}; available={topics}")
    return candidates[0]


def _resolve_state_topic(explicit: str | None, topics: list[str], *, side: str) -> str:
    if explicit:
        return str(explicit)
    aliases = ("left", "_l", "/l/") if side == "left" else ("right", "_r", "/r/", "active")
    candidates = [
        topic for topic in topics
        if any(alias in topic.lower() for alias in aliases)
        and not any(token in topic.lower() for token in ("image", "camera", "compressed", "color"))
    ]
    if not candidates:
        raise KeyError(f"cannot infer {side} state topic; available={topics}")
    return candidates[0]


def _decode_image_bgr(record: RosbagRecord) -> Any:
    import cv2
    import numpy as np

    payload = record.data
    if isinstance(payload, dict):
        if payload.get("bytes_b64"):
            raw = base64.b64decode(str(payload["bytes_b64"]))
        elif payload.get("data_b64"):
            raw = base64.b64decode(str(payload["data_b64"]))
        elif isinstance(payload.get("data"), list):
            raw = bytes(int(item) for item in payload["data"])
        else:
            raise ValueError(f"JSON image record for {record.topic} lacks bytes_b64/data")
    elif hasattr(payload, "data"):
        raw = bytes(payload.data)
    else:
        raw = bytes(payload)
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode compressed image from {record.topic}")
    return bgr


def _decode_vector(record: RosbagRecord) -> Any:
    import numpy as np

    payload = record.data
    if isinstance(payload, dict):
        if "vector" in payload:
            values = payload["vector"]
        elif {"joint_pos", "joint_vel", "joint_cur", "end_pos"} <= set(payload):
            values = [
                *list(payload["joint_pos"])[:7],
                *list(payload["joint_vel"])[:7],
                *list(payload["joint_cur"])[:7],
                *list(payload["end_pos"])[:6],
            ]
        else:
            raise ValueError(f"JSON state record for {record.topic} lacks vector")
    elif _looks_like_joint_end_state(payload):
        values = [
            *list(payload.joint_pos)[:7],
            *list(payload.joint_vel)[:7],
            *list(payload.joint_cur)[:7],
            *list(payload.end_pos)[:6],
        ]
    elif hasattr(payload, "position") and hasattr(payload, "velocity") and hasattr(payload, "effort"):
        values = [
            *list(payload.position)[:7],
            *list(payload.velocity)[:7],
            *list(payload.effort)[:7],
            *([0.0] * 6),
        ]
    else:
        values = list(payload)
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] < 27:
        array = np.pad(array, (0, 27 - array.shape[0]), constant_values=0.0)
    return array[:27].astype(np.float32)


def _looks_like_joint_end_state(msg: Any) -> bool:
    return hasattr(msg, "joint_pos") and hasattr(msg, "joint_vel") and hasattr(msg, "joint_cur") and hasattr(msg, "end_pos")


def _build_target_timestamps(
    images: dict[str, list[_ImageSample]],
    left: list[_VectorSample],
    right: list[_VectorSample],
    camera_names: tuple[str, ...],
    target_hz: float,
) -> list[float]:
    if target_hz <= 0:
        raise ValueError(f"target_hz must be > 0, got {target_hz}")
    camera_start = max(images[name][0].timestamp for name in camera_names)
    camera_end = min(images[name][-1].timestamp for name in camera_names)
    start = max(camera_start, left[0].timestamp, right[0].timestamp)
    end = min(camera_end, left[-1].timestamp, right[-1].timestamp)
    if end < start:
        raise ValueError(f"camera/state streams do not overlap: start={start:.6f} end={end:.6f}")
    period = 1.0 / float(target_hz)
    count = int(((end - start) / period) + 1e-6) + 1
    if count <= 0:
        raise ValueError("no fixed-rate samples inside common camera/state overlap")
    return [start + (i * period) for i in range(count)]


def _select_image_samples(
    rows: list[_ImageSample],
    target_timestamps: list[float],
    *,
    stale_threshold: float,
) -> _ImageSelection:
    selected: list[Any] = []
    nearest_cursor = 0
    hold_cursor = 0
    last = len(rows) - 1
    stale_count = 0
    max_abs_dt = 0.0
    for target in target_timestamps:
        while nearest_cursor < last and abs(rows[nearest_cursor + 1].timestamp - target) <= abs(rows[nearest_cursor].timestamp - target):
            nearest_cursor += 1
        nearest_dt = abs(rows[nearest_cursor].timestamp - target)
        if nearest_dt <= stale_threshold + 1e-9:
            sample = rows[nearest_cursor]
            selected_dt = nearest_dt
        else:
            stale_count += 1
            while hold_cursor < last and rows[hold_cursor + 1].timestamp <= target + 1e-9:
                hold_cursor += 1
            sample = rows[hold_cursor]
            selected_dt = abs(sample.timestamp - target)
        max_abs_dt = max(max_abs_dt, float(selected_dt))
        selected.append(sample)
    return _ImageSelection(
        samples=selected,
        stats={
            "sample_count": len(selected),
            "source_sample_count": len(rows),
            "stale_threshold_sec": float(stale_threshold),
            "stale_frame_count": stale_count,
            "max_abs_dt_sec": max_abs_dt,
            "fill_policy": "nearest_with_hold_last_for_stale",
        },
    )


def _interpolate_vectors(rows: list[_VectorSample], target_timestamps: list[float]) -> list[Any]:
    import numpy as np

    source_ts = np.asarray([sample.timestamp for sample in rows], dtype=np.float64)
    source_vec = np.asarray([sample.vector for sample in rows], dtype=np.float32)
    if source_vec.ndim != 2:
        raise ValueError("robot-state vectors must be a 2D array after decoding")
    result: list[Any] = []
    for target in target_timestamps:
        idx = int(np.searchsorted(source_ts, target, side="left"))
        if idx <= 0:
            result.append(source_vec[0].astype(np.float32))
            continue
        if idx >= len(source_ts):
            result.append(source_vec[-1].astype(np.float32))
            continue
        t0 = float(source_ts[idx - 1])
        t1 = float(source_ts[idx])
        if t1 <= t0:
            result.append(source_vec[idx].astype(np.float32))
            continue
        alpha = float((target - t0) / (t1 - t0))
        result.append(((1.0 - alpha) * source_vec[idx - 1] + alpha * source_vec[idx]).astype(np.float32))
    return result


def _numeric_from_vectors(*, left: list[Any], right: list[Any]) -> dict[str, Any]:
    import numpy as np

    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    # Preserve the driver-native RobotStatus vector convention. These slices
    # change layout only; they do not convert units or coordinate frames.
    left_qpos = left_vec[:, :7]
    right_qpos = right_vec[:, :7]
    qpos = np.concatenate((left_qpos, right_qpos), axis=1).astype(np.float32)
    qvel = np.concatenate((left_vec[:, 7:14], right_vec[:, 7:14]), axis=1).astype(np.float32)
    effort = np.concatenate((left_vec[:, 14:21], right_vec[:, 14:21]), axis=1).astype(np.float32)
    eef = np.concatenate((left_vec[:, 21:27], left_qpos[:, 6:7], right_vec[:, 21:27], right_qpos[:, 6:7]), axis=1).astype(np.float32)
    frame_count = qpos.shape[0]
    return {
        "qpos": qpos,
        "qvel": qvel,
        "eef": eef,
        "effort": effort,
        "robot_base": np.zeros((frame_count, 6), dtype=np.float32),
        "base_velocity": np.zeros((frame_count, 4), dtype=np.float32),
        "action": qpos.copy(),
        "action_eef": eef.copy(),
        "action_base": np.zeros((frame_count, 6), dtype=np.float32),
        "action_velocity": np.zeros((frame_count, 4), dtype=np.float32),
    }


def _parse_camera_topic(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--camera-topic values must use name=/ros/topic")
        name, topic = item.split("=", 1)
        result[name.strip()] = topic.strip()
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RoboLineage rosbag2 raw -> ACT episode_*.hdf5")
    parser.add_argument("--selected-rollouts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-names", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--camera-topic", action="append", default=[], help="Map a camera name to a ROS topic: head=/topic")
    parser.add_argument("--left-state-topic")
    parser.add_argument("--right-state-topic")
    parser.add_argument("--target-hz", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = build_rosbag_act_hdf5_dataset(
        selected_rollouts_path=args.selected_rollouts,
        output_dir=args.output_dir,
        camera_topics=_parse_camera_topic(args.camera_topic) or None,
        left_state_topic=args.left_state_topic,
        right_state_topic=args.right_state_topic,
        camera_names=tuple(str(item) for item in args.camera_names),
        target_hz=args.target_hz,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    print(
        "rosbag_act_hdf5_dataset "
        + json.dumps(
            {"episodes": report["exported_episode_count"], "output_dir": str(args.output_dir)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
