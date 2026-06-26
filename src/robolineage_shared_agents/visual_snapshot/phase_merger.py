from __future__ import annotations


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    """
    Remove duplicated assessments that share the same frame_id.

    The UI layer keeps only the first assessment for a frame because the same
    anchor frame may be triggered by multiple nearby events.
    """
    seen: set[int] = set()
    result: list[dict] = []
    for row in rows:
        frame_id = row.get("frame_id")
        if frame_id not in seen:
            seen.add(frame_id)
            result.append(row)
    return result


def merge_phase_segments(rows: list[dict]) -> list[dict]:
    """
    Build timeline segments directly from SnapshotAssessments.

    Rule:
    - For the interval between assessment i and assessment i+1,
      assign that whole interval to the phase of assessment i+1.
    - Then merge consecutive intervals with the same assigned phase.

    This matches the UI rule:
    the phase between two keyframes belongs to the next keyframe's phase.
    """
    if not rows:
        return []

    if len(rows) == 1:
        timestamp = float(rows[0]["timestamp"])
        return [_build_segment(rows, timestamp, timestamp)]

    intervals = _build_intervals_from_next_keyframe(rows)
    groups: list[list[dict]] = []
    current: list[dict] = [intervals[0]]

    for interval in intervals[1:]:
        if interval["phase"] == current[0]["phase"]:
            current.append(interval)
        else:
            groups.append(current)
            current = [interval]
    groups.append(current)

    segments: list[dict] = []
    for group in groups:
        start_time = float(group[0]["start_time"])
        end_time = float(group[-1]["end_time"])
        segments.append(_build_segment(group, start_time, end_time))
    return segments


def _build_intervals_from_next_keyframe(rows: list[dict]) -> list[dict]:
    intervals: list[dict] = []
    for index in range(len(rows) - 1):
        current = rows[index]
        nxt = rows[index + 1]
        intervals.append(
            {
                "phase": nxt["phase"],
                "progress": nxt.get("progress", "unknown"),
                "risk_level": nxt.get("risk_level", "unknown"),
                "confidence": float(nxt.get("confidence", 0.0)),
                "start_time": float(current["timestamp"]),
                "end_time": float(nxt["timestamp"]),
            }
        )
    return intervals


def _build_segment(rows: list[dict], start_time: float, end_time: float) -> dict:
    progress_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    confidence_sum = 0.0

    for row in rows:
        progress = row.get("progress", "unknown")
        risk = row.get("risk_level", "unknown")
        progress_counts[progress] = progress_counts.get(progress, 0) + 1
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        confidence_sum += float(row.get("confidence", 0.0))

    return {
        "phase": rows[0]["phase"],
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "duration": round(end_time - start_time, 3),
        "assessment_count": len(rows),
        "dominant_progress": max(progress_counts, key=progress_counts.__getitem__),
        "dominant_risk": max(risk_counts, key=risk_counts.__getitem__),
        "avg_confidence": round(confidence_sum / len(rows), 3),
    }
