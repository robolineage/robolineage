from tests.visual_snapshot.realtime.helpers import action_msg, frame_msg
from robolineage_shared_agents.visual_snapshot.realtime import FrameRingBuffer, StreamingSignalBuilder, StreamingWindowBuilder
from robolineage_shared_agents.visual_snapshot.types import ActionEvent


def test_streaming_window_builder_fields_complete():
    ring = FrameRingBuffer(capacity=10)
    signal_builder = StreamingSignalBuilder()
    for idx in range(4):
        ring.put(frame_msg(idx))
        signal_builder.feed(action_msg(idx, gripper=-1.5 if idx >= 2 else 0.0))

    event = ActionEvent(event_type="gripper_close", anchor_frame=2, timestamp_sec=0.2, confidence=1.0)
    window = StreamingWindowBuilder(ring, context_frames=1, max_keyframes=3).build(
        event,
        signal_builder.records,
        signal_builder.signals,
        rollout_id="live_test",
    )

    assert window.rollout_id == "live_test"
    assert window.event_type == "gripper_close"
    assert window.anchor_frame_id == 2
    assert window.frame_ids == [1, 2, 3]
    assert window.color_frames
    assert window.action_summary["event_type"] == "gripper_close"
