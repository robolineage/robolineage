# Evaluation and Recollection

Evaluation closes the loop between a trained policy and the next data action.
RoboLineage records evaluation outcomes as artifacts, then connects failures to
dataset health and policy ancestry.

## Evaluation Summary

An evaluation summary captures the policy under test, task, robot, trial count,
success rate, failure phases, representative evidence, and regressions relative
to a parent policy. It is linked to the policy metadata and the rollout evidence
used during evaluation.

## Next-Collection Brief

A next-collection brief turns failure evidence into data-collection guidance.
It can point to under-covered task phases, repeated failure modes, changed object
positions, missing recovery examples, or release-timing issues. The brief is an
artifact, so later iterations can inspect whether the recommendation was
followed and whether it helped.

## Deployment Recommendation

Deployment recommendation artifacts record whether a policy should be held,
rolled back, evaluated further, collected against, or marked as a release
candidate. The artifact preserves remaining risks and supporting evidence; it
does not replace the lab's existing release procedure.
