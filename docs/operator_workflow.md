# Operator Workflow

The normal RoboLineage workflow is intentionally small.

1. Start the runtime on the Ubuntu robot workstation.
2. Select or import a robot profile.
3. Define the task contract and phase vocabulary.
4. Collect rollouts through the lab's normal robot control stack.
5. Inspect routed review cases when the system marks uncertainty or risk.
6. Confirm dataset decisions before training.
7. Launch the configured training job.
8. Review evaluation summaries and deployment recommendations.
9. Use the next-collection brief to guide the next data round.

The operator still controls the robot and approves consequential decisions. RoboLineage removes routine bookkeeping: evidence indexing, review packets, dataset locks, policy ancestry, evaluation summaries, and next-collection briefs.
