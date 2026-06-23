# Agent Roles

RoboLineage uses lifecycle agents as artifact-producing workers. Each agent
reads existing artifacts, performs a bounded interpretation or integration task,
and writes a typed artifact for the next stage.

## Core Agents

- **Robot Onboarding Agent:** translates a robot environment into a profile that
  RoboLineage can use consistently.
- **Task Config Agent:** turns task goals, phases, success criteria, and risk
  events into a task contract.
- **Online Visual Snapshot Agent:** writes sparse semantic anchors during
  rollout collection.
- **Post-Rollout Review Agent:** reviews packets of evidence after a rollout is
  closed and writes a final review artifact.
- **Data Governance Agent:** converts review evidence into training eligibility,
  failure-pool routing, exclusion records, and dataset decisions.
- **Data Health Agent:** summarizes dataset readiness, coverage gaps, and
  failure distributions.
- **Training Integration Agents:** prepare framework profiles, adapter records,
  training runs, dataset locks, and policy metadata.
- **Policy Evaluation Agent:** links evaluation evidence to policy-level
  outcomes and regressions.
- **Deployment Governance Agent:** records release recommendations and remaining
  risks as artifacts.
- **Master Agent:** summarizes cross-iteration state and turns local artifacts
  into operator-facing lifecycle guidance.

## Boundary

Agents do not replace the robot controller, the policy learner, or the lab's
safety procedure. Their role is to make the surrounding lifecycle legible:
interpret evidence, prepare structured records, and preserve the links that
matter for the next iteration.
