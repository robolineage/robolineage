# Data Source Adapters

Adapters implement the `DeviceAdapter` lifecycle: `start()`, `stop()`, and `health()`.

- `mock.py`: synthetic unit-test source.
- `cameras/realsense.py`: RealSense camera source.
- `ros2_profile.py`: Profile-driven ROS2 source for robot profiles.
- `ros2_arx_one.py`: compatibility shim that re-exports the profile-driven adapter.