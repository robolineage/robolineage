# Prompt Contracts

RoboLineage prompts are bounded by task contracts and JSON schemas. Model outputs are not trusted as lifecycle state until they are parsed, normalized, validated, and written as artifacts.

The main prompt routes are:

- `VSA_VLM_*`: online phase, risk, progress, and visual evidence anchors.
- `POST_REVIEW_VLM_*`: post-rollout packet review and final observations.
- `ROBOT_ONBOARDING_LLM_*`: optional robot profile and topic-binding understanding after deterministic profile normalization and topic probing.
- `ROBOLINEAGE_DISCOVERY_LLM_*`: optional training repository understanding.
- `TRAINING_MONITOR_LLM_*`: optional log and failure diagnosis.
- `DATASET_HEALTH_LLM_*`: optional dataset health explanation.
- `POLICY_EVAL_VLM_*`: evaluation rollout review.
- `DEPLOYMENT_GOVERNANCE_LLM_*`: advisory deployment summary.
- `MASTER_LLM_*`: cross-iteration memory and next-action brief.

Each route can be moved to another OpenAI-compatible backend by editing `.env`.
