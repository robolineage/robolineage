# Release Status

RoboLineage is released as a research runtime and lifecycle artifact interface.
The repository is meant to be useful both as runnable code on an Ubuntu/ROS2
robot workstation and as a reference for the artifact contract.

## Included Now

- Lifecycle artifact schemas.
- Prompt contracts for artifact-producing agents.
- Mini lifecycle artifact examples.
- Robot profiles and profile-guided ROS2 topic binding.
- Raw rollout capture hooks and online VSA.
- Asynchronous post-rollout review and dataset governance.
- Dataset locks, training integration utilities, evaluation artifacts, and
  deployment recommendation records.
- FastAPI backend and Vue lifecycle console.
- Static checks and tests for the runtime contracts.

## Coming Next

- Replay examples with mock model routes.
- Broader adapter examples and packaged scoring utilities.
- Additional real-robot examples.
- More local-model route examples.

The release will keep the lifecycle contract stable while expanding the runtime
around more robots, training stacks, and model routes.
