from tests.visual_snapshot.realtime.helpers import frame_msg
from robolineage_shared_agents.visual_snapshot.realtime import FrameRingBuffer


def test_put_get_decode_and_latest():
    ring = FrameRingBuffer(capacity=2)
    ring.put(frame_msg(1))
    assert ring.get(1).frame_index == 1
    assert ring.latest_frame_index() == 1
    assert ring.get_rgb(1).shape == (16, 16, 3)


def test_evict_and_missing():
    ring = FrameRingBuffer(capacity=2)
    ring.put(frame_msg(1))
    ring.put(frame_msg(2))
    dropped = ring.put(frame_msg(3))
    assert dropped == 1
    assert ring.dropped_count() == 1
    assert ring.get(1) is None
    assert ring.get(3).frame_index == 3
