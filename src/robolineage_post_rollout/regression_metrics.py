from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "post_review.regression_metrics.v1"


def summarize_post_review_regression(
    rollouts_dir: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(rollouts_dir)
    rollout_dirs = _discover_rollout_dirs(root, limit=limit)
    rows = [_rollout_metrics(path) for path in rollout_dirs]
    return {
        "schema_version": SCHEMA_VERSION,
        "rollouts_dir": str(root),
        "rollout_count": len(rows),
        "metrics": _aggregate(rows),
        "rollouts": rows,
    }


def _discover_rollout_dirs(root: Path, *, limit: int | None) -> list[Path]:
    if not root.exists():
        return []
    dirs = [
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and _looks_like_reviewed_rollout(path)
    ]
    if limit is None or limit <= 0:
        return dirs
    return dirs[:limit]


def _looks_like_reviewed_rollout(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in (
            "evidence_index.json",
            "annotation.final.json",
            "dataset_admission.json",
            "rollout_summary.json",
        )
    )


def _rollout_metrics(rollout_dir: Path) -> dict[str, Any]:
    evidence = _read_json(rollout_dir / "evidence_index.json") or {}
    annotation = _read_json(rollout_dir / "annotation.final.json") or {}
    summary = _read_json(rollout_dir / "rollout_summary.json") or {}
    admission = _read_json(rollout_dir / "dataset_admission.json") or {}

    final_observation_frames = _int_list(evidence.get("final_observation_frames"))
    vsa_windows = evidence.get("vsa_windows")
    if not isinstance(vsa_windows, list):
        vsa_windows = []

    annotation_success = _bool_or_none((annotation.get("outcome") or {}).get("final_success"))
    summary_success = _bool_or_none(summary.get("final_success"))
    final_success_aligned = (
        annotation_success == summary_success
        if annotation_success is not None and summary_success is not None
        else None
    )

    return {
        "rollout_id": rollout_dir.name,
        "path": str(rollout_dir),
        "image_count": int(evidence.get("image_count") or 0),
        "vsa_window_count": int(evidence.get("vsa_window_count") or len(vsa_windows)),
        "final_observation_frames": final_observation_frames,
        "final_observation_frame_count": len(final_observation_frames),
        "duplicate_window_count": _duplicate_window_count(vsa_windows),
        "zero_duration_phase_count": _zero_duration_phase_count(annotation),
        "accepted_for_training": _accepted_for_training(admission),
        "dataset_decision": str(admission.get("decision") or ""),
        "label_quality": admission.get("label_quality"),
        "annotation_final_success": annotation_success,
        "summary_final_success": summary_success,
        "final_success_aligned": final_success_aligned,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compared = sum(1 for row in rows if row["final_success_aligned"] is not None)
    aligned = sum(1 for row in rows if row["final_success_aligned"] is True)
    decisions = Counter(str(row.get("dataset_decision") or "missing") for row in rows)
    return {
        "image_count_total": sum(int(row["image_count"]) for row in rows),
        "final_observation_frame_count": sum(int(row["final_observation_frame_count"]) for row in rows),
        "duplicate_window_count": sum(int(row["duplicate_window_count"]) for row in rows),
        "zero_duration_phase_count": sum(int(row["zero_duration_phase_count"]) for row in rows),
        "accepted_for_training_count": sum(1 for row in rows if row["accepted_for_training"] is True),
        "final_success_compared_count": compared,
        "final_success_aligned_count": aligned,
        "final_success_alignment_rate": round(aligned / compared, 4) if compared else None,
        "dataset_decision_counts": dict(sorted(decisions.items())),
    }


def _duplicate_window_count(vsa_windows: list[Any]) -> int:
    keys: list[tuple[str, int | None]] = []
    for item in vsa_windows:
        if not isinstance(item, dict):
            continue
        keys.append((str(item.get("event_type") or "unknown"), _int_or_none(item.get("anchor_frame_id"))))
    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def _zero_duration_phase_count(annotation: dict[str, Any]) -> int:
    timeline = annotation.get("phase_timeline")
    if not isinstance(timeline, list):
        return 0
    count = 0
    for segment in timeline:
        if not isinstance(segment, dict):
            continue
        duration = _float_or_none(segment.get("duration_sec"))
        if duration is not None and duration <= 0.0:
            count += 1
    return count


def _accepted_for_training(admission: dict[str, Any]) -> bool | None:
    value = admission.get("accepted_for_training")
    if isinstance(value, bool):
        return value
    decision = str(admission.get("decision") or "")
    if decision:
        return decision == "accepted"
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        parsed = _int_or_none(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize post-review regression metrics for rollout artifacts.")
    parser.add_argument("rollouts_dir", help="Directory containing rollout subdirectories.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rollout directories to scan.")
    parser.add_argument("--output", help="Optional JSON output path. Defaults to stdout only.")
    args = parser.parse_args(argv)

    summary = summarize_post_review_regression(args.rollouts_dir, limit=args.limit)
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
