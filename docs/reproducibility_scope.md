# Reproducibility Boundary

RoboLineage separates artifact-level reproduction from hardware-level
reproduction.

## Artifact-Level Reproduction

The public release exposes runtime code, schemas, prompt contracts, example
artifacts, figures, and lightweight checks. These materials make the lifecycle
contract inspectable without requiring private robot videos, local network
settings, or lab-specific robot workstations.

## Semantic-Agent Reproduction

Re-running semantic agents requires an equivalent model route and access to the
same style of visual evidence. Different VLM or LLM backends can be attached
behind the same artifact interface, but absolute review accuracy may change with
model capability, prompt compatibility, and sensor visibility.

## Hardware-Level Reproduction

Re-running robot collection, policy training, and physical evaluation requires
the corresponding robot, workspace, ROS2 setup, training repository, and safety
procedure. The lifecycle interface is designed so those local systems can be
connected without changing the artifact semantics.
