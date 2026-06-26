"""Backward-compatible import path for the profile-driven ROS2 adapter.

New code should import :mod:`robolineage_data_source.adapters.ros2_profile`. This
module remains so older configs/tests that import ``RosArxOneAdapter`` keep
working while ARX-specific behavior lives in robot profiles.
"""
from __future__ import annotations

from robolineage_data_source.adapters.ros2_profile import (
    Ros2ProfileAdapter,
    _robot_status_to_27_vec,
)

RosArxOneAdapter = Ros2ProfileAdapter

__all__ = ["Ros2ProfileAdapter", "RosArxOneAdapter", "_robot_status_to_27_vec"]
