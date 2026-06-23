# Artifact Walkthrough

The mini lifecycle example shows how a single rollout can move through the
RoboLineage artifact contract. It is not a benchmark or runtime demo. It is a
compact format reference.

File: [examples/mini_lifecycle/artifact_trace.example.json](../examples/mini_lifecycle/artifact_trace.example.json)

## 1. Robot Profile

`robot_profile.arx_tabletop.v1` describes the local robot setup in lifecycle
terms. The important part is the stream binding: local camera, state, and action
streams are mapped into canonical roles that later stages can read.

This artifact lets the rest of the lifecycle avoid robot-specific assumptions.
The post-review stage does not need to know the original topic names; it reads
evidence through the profile.

## 2. Task Config

`task_config.stack_red_on_blue.v1` defines the task goal, phases, success
criteria, and expected failure modes. It gives semantic structure to the
rollout. For example, a late release error is meaningful only because the task
contract defines release and settle as distinct phases.

## 3. Rollout Manifest

`rollout.r042` records the source-of-truth evidence for one episode. It links to
both the robot profile and task config. Its payload points to raw evidence such
as video and state traces through URIs and hashes.

The manifest is intentionally separate from semantic review. Raw evidence
remains available even if a model call fails or a review is later revised.

## 4. VSA Snapshot

`vsa.r042.window03` is an online semantic anchor. It records what the Visual
Snapshot Agent saw in a short window: the likely phase, visible evidence, and
uncertainty. It links back to the rollout manifest.

The snapshot helps later review find relevant moments. It does not decide
whether the rollout enters training.

## 5. Post-Rollout Review

`review.r042` reads the rollout and VSA snapshot, then writes a final review
artifact. In the example, the rollout fails because the red cube remains in the
gripper after the release phase.

This artifact contains outcome, failure phase, admission recommendation, and
supporting evidence. It is the semantic review record, not the dataset decision.

## 6. Dataset Decision

`decision.r042` converts the review into dataset state. The failed rollout is
not eligible for the primary imitation dataset, but it is routed to the failure
pool because it contains useful release-failure evidence.

This separation is central to RoboLineage: failed or uncertain rollouts can
still be valuable without silently entering the main training manifest.

## 7. Next-Collection Brief

`brief.stack_red_on_blue.round2` turns the failure into a collection action. It
asks for successful release-timing variants after confirmed alignment and cites
the review and VSA artifacts as support.

The brief is also an artifact. A later iteration can check whether this advice
was followed and whether the resulting policy improved.
