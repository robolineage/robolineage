"""RoboLineage shared agents namespace.

This file intentionally contains no imports and no exports. It exists only
to mark `src/robolineage_shared_agents` as a Python package. It is not the complete
set of RoboLineage agents; domain-local agents live beside their runtime domain
(`robolineage_robot`, `robolineage_post_rollout`, `robolineage_train`, `robolineage_eval`).

Sub-packages:

    visual_snapshot/   — online/offline Visual Snapshot Agent
    master/            — global lifecycle review and memory agent

Consumers must import directly from the sub-package, e.g.

    from robolineage_shared_agents.visual_snapshot import VisualSnapshotAgent

Cross-module type usage stays inside `robolineage_contracts.agents.*`; this namespace
init remains empty.
"""
