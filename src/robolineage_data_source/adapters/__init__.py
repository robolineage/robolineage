"""Device adapters.

Currently bundled:
- MockAdapter (mock.py)
- RealSenseAdapter (cameras/realsense.py, optional pyrealsense2)
- Ros2ProfileAdapter (ros2_profile.py, optional rclpy for ROS2 robots)
"""
from robolineage_data_source.adapters.base import DeviceAdapter, UnsupportedSyncError
from robolineage_data_source.adapters.cameras.realsense import RealSenseAdapter
from robolineage_data_source.adapters.mock import MockAdapter

__all__ = [
    "DeviceAdapter",
    "UnsupportedSyncError",
    "RealSenseAdapter",
    "MockAdapter",
]
