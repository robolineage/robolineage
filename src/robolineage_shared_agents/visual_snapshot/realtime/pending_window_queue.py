from __future__ import annotations

from collections import deque

from ..types import ActionEvent


class PendingWindowQueue:
    """FIFO queue that releases events after post-context frames arrive."""

    def __init__(self, context_frames: int = 15, max_wait_frames: int | None = None):
        self.context_frames = max(0, context_frames)
        self.max_wait_frames = max_wait_frames
        self._events: deque[ActionEvent] = deque()

    def enqueue(self, event: ActionEvent) -> None:
        self._events.append(event)

    def pop_ready(self, latest_frame_index: int | None) -> list[ActionEvent]:
        if latest_frame_index is None:
            return []

        ready: list[ActionEvent] = []
        while self._events:
            event = self._events[0]
            target = event.anchor_frame + self.context_frames
            timeout_target = (
                event.anchor_frame + self.max_wait_frames
                if self.max_wait_frames is not None
                else target
            )
            if latest_frame_index < min(target, timeout_target):
                break
            ready.append(self._events.popleft())
        return ready

    def pop_all(self) -> list[ActionEvent]:
        """Release every pending event, used when a rollout is stopping."""
        ready = list(self._events)
        self._events.clear()
        return ready

    def __len__(self) -> int:
        return len(self._events)
