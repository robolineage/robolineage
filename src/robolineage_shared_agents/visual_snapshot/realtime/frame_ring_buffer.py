"""Thread-safe frame ring buffer keyed by frame_index.

Phase 3 update: stores BGR ndarrays directly (no lazy JPEG decode); jpeg
decoding moves to the ROS2 profile adapter.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from .types import RealtimeFrameRecord


class FrameRingBuffer:
    def __init__(self, capacity: int = 120):
        self.capacity = max(1, capacity)
        self._frames: OrderedDict[int, RealtimeFrameRecord] = OrderedDict()
        self._dropped = 0
        self._lock = threading.Lock()

    def put(self, frame: RealtimeFrameRecord) -> int:
        dropped_now = 0
        with self._lock:
            if frame.frame_index in self._frames:
                self._frames.pop(frame.frame_index)
            self._frames[frame.frame_index] = frame
            while len(self._frames) > self.capacity:
                self._frames.popitem(last=False)
                self._dropped += 1
                dropped_now += 1
        return dropped_now

    def get(self, frame_index: int) -> Optional[RealtimeFrameRecord]:
        with self._lock:
            return self._frames.get(frame_index)

    def get_rgb(self, frame_index: int) -> Optional[np.ndarray]:
        with self._lock:
            rec = self._frames.get(frame_index)
        if rec is None:
            return None
        import cv2
        return cv2.cvtColor(rec.bgr, cv2.COLOR_BGR2RGB).copy()

    def latest_frame_index(self) -> Optional[int]:
        with self._lock:
            if not self._frames:
                return None
            return next(reversed(self._frames))

    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    def clear(self) -> int:
        with self._lock:
            count = len(self._frames)
            self._frames.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
