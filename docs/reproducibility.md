# Reproducibility Scope

The public release supports three levels of inspection.

1. Artifact-level reproduction: schemas, sample artifacts, frozen evidence packets, prompts, and scoring scripts can be inspected without private robot data.
2. Semantic-agent reproduction: requires an equivalent model route, API access, and sampled visual evidence.
3. Hardware reproduction: requires the robot, ROS2 workspace, task objects, cameras, training repository, and lab safety procedures.

The Windows copy is not expected to execute ROS2 tests. Run hardware and integration tests on the Ubuntu robot workstation.

Robot and training portability are reproduced through editable profiles rather
than through a fixed hosted robot stack. The release includes one deployed
ARX-style profile, sanitized Realman and GALBOT G1 profile templates, and
training framework profile examples for common imitation-learning and
vision-language-action workflows.
