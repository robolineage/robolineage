from __future__ import annotations

import time

import numpy as np

from robolineage_shared_agents.visual_snapshot.realtime.types import (
    RealtimeActionRecord,
    RealtimeFrameRecord,
)


def bgr_image(color: tuple[int, int, int] = (60, 40, 20), size: int = 16) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = np.array(color, dtype=np.uint8)
    return img


def frame_msg(frame_index: int) -> RealtimeFrameRecord:
    return RealtimeFrameRecord(
        frame_index=frame_index,
        host_mono_ns=frame_index * 100_000_000,
        bgr=bgr_image(),
    )


def action_msg(frame_index: int, gripper: float = 0.0, x: float | None = None) -> RealtimeActionRecord:
    x = frame_index * 0.001 if x is None else x
    return RealtimeActionRecord(
        frame_index=frame_index,
        host_mono_ns=frame_index * 100_000_000,
        eef_xyz=(x, 0.0, 0.0),
        eef_rxyz=(0.0, 0.0, 0.0),
        gripper=gripper,
    )


def wait_for(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    return predicate()
