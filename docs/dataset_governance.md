# Dataset Governance and Data Health

Dataset governance converts rollout-level review into data lifecycle state. A
review artifact can say that a rollout succeeded, failed, needs human review, or
contains useful failure evidence. A dataset decision decides where that rollout
belongs.

## Dataset Decisions

Dataset decisions separate several fields that are often collapsed in manual
workflows:

- training eligibility;
- review priority;
- failure-pool routing;
- exclusion reason;
- operator override;
- downstream dataset split.

This separation matters because `needs_review` is not the same as rejection, and
a failed rollout may still be valuable as failure evidence or recovery context.

## Dataset Locks

A dataset lock records exactly which artifacts enter a training dataset. It
stores accepted rollout references, hashes, adapter settings, split membership,
and parent artifact identifiers. Policy metadata then points to the dataset lock
instead of a loose folder path.

## Data Health

The Data Health Agent summarizes data-readiness signals such as accepted rollout
counts, phase coverage, failure concentration, distribution gaps, and
unresolved review queues. These summaries help decide whether the next action is
more collection, more review, adapter repair, or training.
