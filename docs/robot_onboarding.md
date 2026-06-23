# Robot Onboarding

Robot onboarding maps a local robot setup into RoboLineage lifecycle streams.
The goal is not to hide robot differences, but to translate them into common
artifact fields that later stages can rely on.

## Inputs

- Robot name and embodiment description.
- Camera streams and calibration notes.
- End-effector or robot-state streams.
- Action command interface.
- Time synchronization assumptions.
- Workspace and safety notes.

## Output

The output is a `robot_profile` artifact. It binds local stream names to
canonical lifecycle roles such as `primary_camera`, `wrist_camera`,
`end_effector_state`, `action_command`, and `episode_clock`.

For ROS2 robots, topic discovery is profile-guided: the system can inspect the
graph and propose candidate bindings, while the operator validates the profile
before collection.

## Why It Matters

Every downstream artifact depends on consistent stream identity. If camera,
state, and action channels are not grounded at the beginning, review evidence,
training adapters, and evaluation summaries cannot be reliably compared across
robots or iterations.
