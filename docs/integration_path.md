# Integration Path

This document describes how an external robot-learning workflow can connect to
the RoboLineage lifecycle interface.

## Step 1: Define a Robot Profile

Start by mapping local robot streams into a `robot_profile` artifact:

- primary camera;
- optional wrist or secondary camera;
- robot or end-effector state;
- action command stream;
- episode timing;
- workspace and safety notes.

For ROS2 robots, this usually starts from topic discovery and operator
validation. For non-ROS systems, the same artifact can be written from local log
or streaming APIs.

## Step 2: Define the Task Contract

Write a `task_config` artifact for each task. The contract should include:

- task goal;
- phase vocabulary;
- visual success criteria;
- common failure modes;
- risk events;
- terminal observation checks.

The task contract is what lets VSA, post-rollout review, dataset decisions, and
evaluation summaries use the same language.

## Step 3: Record Rollout Manifests

Each episode should produce a `rollout_manifest` artifact that links raw
evidence to the robot profile and task config. The manifest should store hashes
or stable references to raw video, state traces, action logs, and operator notes.

## Step 4: Write Review Artifacts

During collection, online VSA can write `vsa_snapshot` artifacts for important
windows. After the rollout closes, post-rollout review writes a
`post_rollout_review` artifact with outcome, failure phase, evidence, and
admission recommendation.

## Step 5: Govern Dataset Admission

Use `dataset_decision` artifacts to decide whether a rollout belongs in the
primary training dataset, human review queue, failure pool, or exclusion list.
When a training round is ready, write a `dataset_lock` artifact that lists the
accepted rollouts and adapter profile.

## Step 6: Connect Training

Create a framework profile for the target learner. The adapter should convert
accepted rollout evidence into the learner's expected format and write a
`training_run` artifact. The resulting checkpoint is recorded through
`policy_metadata`.

## Step 7: Close the Loop

Evaluation writes an `evaluation_summary`. Recollection writes a
`next_collection_brief`. Deployment review writes a
`deployment_recommendation`. These artifacts make the next iteration depend on
recorded evidence rather than memory.
