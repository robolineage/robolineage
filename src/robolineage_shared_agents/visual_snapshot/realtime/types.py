"""Internal realtime record types for the VSA realtime sub-pipeline.

These records are never serialized; they exist only inside the realtime VSA
process bridging ROS2 topic callbacks → ring buffer / signal builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RealtimeFrameRecord:
    frame_index: int
    host_mono_ns: int
    bgr: np.ndarray


@dataclass(frozen=True)
class RealtimeActionRecord:
    frame_index: int
    host_mono_ns: int
    eef_xyz: Tuple[float, float, float]
    eef_rxyz: Tuple[float, float, float]
    gripper: float


@dataclass(frozen=True)
class RealtimeArmSample:
    """Raw arm pose data extracted from a ROS2 message, awaiting frame_index
    alignment.

    Ros2TopicConsumer's arm thread builds these and pushes them into a queue.
    The main loop in ``run_ros_topic_stream`` pairs each sample with the most
    recent camera frame_index seen on the main thread to construct the final
    RealtimeActionRecord. Keeping alignment on the main thread eliminates
    cross-thread mutation of signal_builder / event_detector / pending state.
    """
    host_mono_ns: int
    eef_xyz: Tuple[float, float, float]
    eef_rxyz: Tuple[float, float, float]
    gripper: float
