from tests.visual_snapshot.realtime.helpers import action_msg
from robolineage_shared_agents.visual_snapshot.realtime import StreamingEventDetector, StreamingSignalBuilder


def _signal_builder():
    return StreamingSignalBuilder(smoothing_window=1, still_threshold=1e-6)


def test_first_frame_no_event_and_gripper_crossing_event():
    builder = _signal_builder()
    detector = StreamingEventDetector(still_min_frames=3, heartbeat_interval=0)
    _, first = builder.feed(action_msg(0, gripper=0.0))
    assert detector.feed(first) == []

    _, second = builder.feed(action_msg(1, gripper=-1.5))
    events = detector.feed(second)
    assert [event.event_type for event in events] == ["gripper_close"]
    assert events[0].anchor_frame == 1


def test_still_start_and_heartbeat():
    builder = StreamingSignalBuilder(smoothing_window=1, still_threshold=1.0)
    detector = StreamingEventDetector(still_min_frames=2, heartbeat_interval=0.3)
    emitted = []
    for idx in range(5):
        _, signal = builder.feed(action_msg(idx, gripper=0.0, x=0.0))
        emitted.extend(detector.feed(signal))

    assert any(event.event_type == "still_start" for event in emitted)
    assert any(event.event_type == "heartbeat" for event in emitted)


def test_periodic_sample_emits_at_fixed_interval_when_no_action_event():
    builder = _signal_builder()
    detector = StreamingEventDetector(
        still_min_frames=99,
        heartbeat_interval=0,
        periodic_interval_sec=0.3,
    )
    emitted = []
    for idx in range(8):
        _, signal = builder.feed(action_msg(idx, gripper=0.0))
        emitted.extend(detector.feed(signal))

    periodic = [event for event in emitted if event.event_type == "periodic_sample"]
    assert [event.anchor_frame for event in periodic] == [3, 6]
    assert periodic[0].details["interval_sec"] == 0.3


def test_action_event_takes_priority_and_resets_periodic_timer():
    builder = _signal_builder()
    detector = StreamingEventDetector(
        still_min_frames=99,
        heartbeat_interval=0,
        periodic_interval_sec=0.3,
    )
    emitted = []
    for idx in range(7):
        gripper = -1.5 if idx >= 3 else 0.0
        _, signal = builder.feed(action_msg(idx, gripper=gripper))
        emitted.extend(detector.feed(signal))

    by_type = [(event.event_type, event.anchor_frame) for event in emitted]
    assert ("gripper_close", 3) in by_type
    assert ("periodic_sample", 3) not in by_type
    assert ("periodic_sample", 6) in by_type
