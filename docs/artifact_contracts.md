# Artifact Contracts

RoboLineage uses typed artifacts as the lifecycle boundary between agents, deterministic validators, and humans.

Common artifacts:

- `raw/rosbag2/`: source-of-truth robot recording.
- `snapshots.jsonl`: online VSA semantic anchors.
- `annotation.final.json`: post-rollout review outcome.
- `failure_analysis.json`: failure phase and evidence summary.
- `dataset_decision.json`: public dataset decision alias.
- `dataset_admission.json`: compatibility artifact used by the current runtime.
- `dataset_health_report.json`: dataset readiness and collection gaps.
- `dataset.lock`: immutable dataset version.
- `training_status.json`: external training run status.
- `policy.meta.json`: checkpoint ancestry and dataset binding.
- `policy_evaluation.json`: evaluation rollout summary.
- `deployment_decision.json`: advisory deployment governance artifact.
- `next_collection_brief.json`: targeted recollection plan.
- `master_review.json`: cross-iteration summary.

Schemas live in `src/robolineage_schemas/`. Python contracts live in `src/robolineage_contracts/`.
