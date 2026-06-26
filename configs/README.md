# Runtime Configuration

This directory stores YAML files used by the RoboLineage runtime.

- `robolineage_default.yaml` starts the console without binding to a specific robot.
- `arx_one.yaml` is a validated profile-driven ROS2 configuration.
- `arx_one_rs3.yaml` is the same style of configuration with RealSense input.
- `robot_profiles/` contains reusable robot profile YAML files for onboarding.

Start with:

```bash
ROBOLINEAGE_CONFIG=configs/robolineage_default.yaml ./run.sh
ROBOLINEAGE_CONFIG=configs/arx_one.yaml ./run.sh
```

Robot-specific network settings are usually edited through environment variables, `cyclonedds.xml`, and the selected profile. The important fields are ROS domain, namespace, camera topics, end-effector state topics, action topics, message types, stream ids, and health thresholds.

Use `python -m robolineage_robot.topic_probe` on a ROS2 workstation to list candidate topics before finalizing a new profile. The probe is advisory; the selected profile remains the lifecycle artifact used by the runtime.

The parser lives in `src/robolineage_data_source/config/loader.py` and `src/robolineage_data_source/config/schema.py`. The profile-driven ROS2 adapter lives in `src/robolineage_data_source/adapters/ros2_profile.py`.
