# RoboLineage Overview

RoboLineage treats robot policy iteration as a data lifecycle. Each consequential transition is written as a typed artifact: rollouts become reviewed evidence, reviewed evidence becomes dataset decisions, dataset updates become locks, locks bind training runs, training runs produce policy metadata, evaluations produce summaries, and deployment recommendations become the next lifecycle transition.

Agents are lifecycle workers. They interpret visual evidence, task contracts, logs, and evaluation summaries, but their outputs become state only through validated artifacts. Raw robot data remains the source of truth.

## Main Lanes

1. Raw rollout capture records source-of-truth data and never waits for VLM calls.
2. Online VSA writes sparse semantic anchors during collection.
3. Post-rollout review drains evidence asynchronously and prepares governed outputs.

## Main Agents

- Robot Onboarding Agent translates a robot profile into runtime bindings; an optional ROS2 topic probe helps operators build new profiles.
- Task Config Agent converts a task description into phase, success, and risk contracts.
- Online Visual Snapshot Agent records phase, risk, and event anchors.
- Post-Rollout Review Agent performs packetized review after a rollout closes.
- Data Governance Agent separates review outcome from training eligibility.
- Data Health Agent summarizes dataset readiness and coverage gaps.
- Framework Discovery and Dataset Adapter agents connect accepted data to existing learners through external command profiles.
- Training Monitor and Version Governance agents bind training runs to dataset locks and policy metadata.
- Policy Evaluation and Deployment Governance agents summarize evaluation evidence.
- Master Agent maintains cross-iteration summaries and next-action state.
