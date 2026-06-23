# Release Plan

RoboLineage is released progressively so the public interface stays stable while
runtime components are cleaned and documented.

## Available Now

- Lifecycle artifact schemas.
- Prompt contracts for artifact-producing agents.
- Minimal artifact examples.
- Design notes for robot onboarding, review, data governance, training
  integration, evaluation, and recollection.

## Planned Additions

- Schema validators and artifact scoring scripts.
- Replay examples with mock model routes.
- ROS2/runtime integration and frontend console components.
- Training adapters and evaluation tools.
- Additional real-robot examples.

The release plan follows the same principle as the system: keep the lifecycle
contract stable, then expand the runtime around it.
