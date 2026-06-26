# robolineage_data_source

Data-source layer for cameras, ROS2 robot state, synchronization, and raw rollout capture. Production rollout capture writes source-of-truth data under `raw/rosbag2` through `RosbagRawRecorder` while online VSA runs independently.

Important modules:

- `orchestrator.py`: starts configured adapters.
- `rosbag/recorder.py`: `RosbagRawRecorder` for per-rollout raw recording.
- `adapters/ros2_profile.py`: Profile-driven ROS2 adapter for camera and end-effector state topics.
- `config/loader.py` and `config/schema.py`: runtime YAML parsing.

The package can import `robolineage_contracts` and `robolineage_schemas`, but should not depend on higher-level agents.