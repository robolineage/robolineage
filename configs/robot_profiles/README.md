# Robot Profiles

Robot profiles describe the ROS2 topics, message types, streams, health checks, and bindings needed by RoboLineage. Profiles are consumed by the Robot Onboarding Agent and the profile-driven ROS2 adapter.

This directory contains one deployed ARX-style profile and two sanitized templates for Realman and GALBOT G1 workflows. The sanitized templates intentionally use placeholder message packages and public example topic names; replace them with the lab deployment values before collection. Use `robolineage_robot.topic_probe.probe_ros2_topics()` on a ROS2 workstation to list candidate camera, state, and action/control topics before finalizing a profile.

Edit these files when moving to a new robot or ROS namespace.
