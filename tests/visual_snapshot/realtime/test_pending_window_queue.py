from robolineage_shared_agents.visual_snapshot.realtime import PendingWindowQueue
from robolineage_shared_agents.visual_snapshot.types import ActionEvent


def _event(frame: int) -> ActionEvent:
    return ActionEvent(event_type="heartbeat", anchor_frame=frame, timestamp_sec=frame / 10, confidence=0.5)


def test_fifo_release_when_context_arrives():
    queue = PendingWindowQueue(context_frames=2)
    queue.enqueue(_event(10))
    assert queue.pop_ready(11) == []
    ready = queue.pop_ready(12)
    assert [event.anchor_frame for event in ready] == [10]


def test_batch_release_multiple_ready_events():
    queue = PendingWindowQueue(context_frames=1)
    queue.enqueue(_event(1))
    queue.enqueue(_event(2))
    assert [event.anchor_frame for event in queue.pop_ready(3)] == [1, 2]
