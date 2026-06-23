# RoboLineage Overview

RoboLineage defines a lifecycle layer for robot policy iteration. Its purpose is
to keep the relationship between data collection, review, training, evaluation,
and recollection explicit as robots, tasks, and policy learners change.

The interface is built around typed lifecycle artifacts. A rollout is not only a
video or trajectory file; it is linked to a robot profile, task contract,
semantic evidence, review decision, dataset decision, training record,
evaluation summary, and next-collection brief. This turns policy iteration into
a traceable sequence rather than a set of disconnected files.

## Main Stages

1. **Robot and task grounding.** A robot profile and task contract map local
   camera streams, robot state, action channels, task phases, success criteria,
   and risk events into common artifact fields.
2. **Rollout capture.** Raw rollout evidence is recorded as the source of truth.
   Semantic interpretation is allowed to run beside capture but does not block
   the raw stream.
3. **Visual snapshot and post-rollout review.** Online snapshots write sparse
   task relevant anchors. Post-rollout review reads raw evidence, snapshots, and
   terminal observations to produce a governed review artifact.
4. **Dataset governance.** Review outcomes are converted into dataset decisions,
   failure pools, exclusion records, and dataset locks.
5. **Training integration.** Accepted data is adapted to the selected training
   stack and linked to the resulting training run and policy metadata.
6. **Evaluation and recollection.** Evaluation summaries and data-health state
   become next-collection briefs for targeted improvement.

## Design Principle

Agents perform semantic and integration work. Lifecycle state is carried by
artifacts. This separation lets the system use multimodal model interpretation
without turning free-form model output into hidden training or deployment state.
