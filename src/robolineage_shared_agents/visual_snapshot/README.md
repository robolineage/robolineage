# Visual Snapshot Agent

Online VSA writes sparse task-aware semantic anchors during rollout collection. Raw capture is independent and remains the source of truth. VSA output is reviewed downstream by post-rollout review and never directly admits data to training.

Key behavior:

- event-guided windows from camera and end-effector state;
- linear VLM priority so online feedback does not block raw recording;
- terminal observations for post-rollout review;
- append-only `snapshots.jsonl` artifacts.