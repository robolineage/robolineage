# Realtime VSA

Realtime VSA subscribes to ROS2 camera and end-effector state topics declared by the active robot profile. It builds event windows, runs VLM calls on a linear analysis thread, and writes snapshots without blocking raw rosbag2 recording.

Entry points:

- `scripts/vsa_streaming.py` for robot workstation streaming.
- `scripts/vsa_rehearsal_single_host.py` for local rehearsal.