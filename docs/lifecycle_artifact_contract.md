# Lifecycle Artifact Contract

The lifecycle artifact contract is the stable interface exposed by
RoboLineage. It lets different robots and training stacks share the same
governance semantics even when their local topics, data formats, and training
commands differ.

## Common Fields

Every artifact should include:

- `artifact_id`: stable identifier for this artifact.
- `artifact_type`: artifact kind, such as `rollout_manifest` or
  `dataset_decision`.
- `schema_version`: schema version used for validation.
- `created_at`: timestamp in ISO 8601 format.
- `producer`: component or agent that wrote the artifact.
- `parents`: artifact identifiers that this artifact depends on.
- `content_hash`: hash of the artifact content or associated payload.
- `content`: typed artifact payload.

Parent links are the lineage boundary. They allow a policy checkpoint to be
traced back to the dataset lock, the dataset lock to admitted rollout decisions,
and each rollout decision to the evidence and review packet that justified it.

## Why Typed Artifacts

Robot learning workflows often involve videos, trajectory logs, annotations,
dataset folders, training scripts, checkpoints, evaluation tables, and local
notes. Without typed transitions, the reason one artifact led to the next is
usually reconstructed after the fact. RoboLineage makes those transitions part
of the record.

Typed artifacts also make agent output safer to use. A model can provide
semantic evidence or a structured suggestion, but downstream lifecycle state is
written only through an explicit artifact type.
