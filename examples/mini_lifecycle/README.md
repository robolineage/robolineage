# Mini Lifecycle Example

This example mirrors the RoboLineage lifecycle artifact chain. It is intentionally small and synthetic: no private robot video or lab network setting is included.

Order:

1. `rollout_001/raw_manifest.json`
2. `rollout_001/snapshots.jsonl`
3. `rollout_001/annotation.final.json`
4. `rollout_001/failure_analysis.json`
5. `rollout_001/dataset_decision.json`
6. `dataset_health_report.json`
7. `dataset.lock`
8. `policy.meta.json`
9. `eval.summary.json`
10. `deployment_decision.json`
11. `next_collection_brief.json`
12. `master_review.json`

The values are illustrative and should be used for schema and documentation inspection, not for policy training.
