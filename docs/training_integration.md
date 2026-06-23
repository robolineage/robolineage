# Training Integration

Training integration lets RoboLineage connect governed datasets to existing
policy-learning stacks without replacing the learner.

## Framework Profile

A framework profile records the target repository, expected input format,
dataset conversion command, training command, evaluation command, relevant
config files, and output locations. The profile can be written manually or
prepared by a discovery agent from the user's normal training commands.

## Dataset Adapter

The dataset adapter materializes accepted rollouts into the format expected by
the target learner. The adapter output is validated before training and linked
to the dataset lock. This makes training reproducible at the lifecycle level:
the checkpoint points back to the exact data, adapter, command, and code
revision used to produce it.

## Policy Metadata

Policy metadata links a checkpoint to:

- parent policy;
- dataset lock;
- adapter record;
- training command;
- code revision;
- training metrics;
- evaluation summary.

This is the bridge between data governance and policy iteration.
