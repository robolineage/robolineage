from tests.visual_snapshot.realtime.helpers import action_msg
from robolineage_shared_agents.visual_snapshot.realtime import StreamingSignalBuilder


def test_streaming_signal_builder_detects_edges_and_smoothed_motion():
    msgs = [
        action_msg(0, gripper=0.0, x=0.0),
        action_msg(1, gripper=-1.5, x=0.01),
        action_msg(2, gripper=-1.5, x=0.02),
    ]
    streaming = StreamingSignalBuilder(smoothing_window=2)
    stream_signals = [streaming.feed(msg)[1] for msg in msgs]

    assert [s.gripper_edge for s in stream_signals] == ["none", "closing_edge", "none"]
    assert [round(s.motion_energy_avg, 6) for s in stream_signals] == [0.0, 0.005, 0.01]
    assert [record.frame_index for record in streaming.records] == [0, 1, 2]
