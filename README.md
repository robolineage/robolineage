# RoboLineage

[Project page](https://robolineage.github.io/) |
[arXiv](https://arxiv.org/abs/2606.22142)

RoboLineage is an agent-native data lifecycle governance system for robot policy iteration. It turns the routine loop behind robot learning into typed, inspectable artifacts: rollout evidence, online visual snapshots, post-rollout reviews, dataset decisions, dataset locks, training records, policy metadata, evaluation summaries, deployment recommendations, and next-collection briefs.

The system is a lightweight lifecycle layer. It does not replace the robot controller, policy learner, training framework, or lab safety procedure. Agents interpret evidence and prepare structured artifacts; deterministic validators, schemas, and operator confirmation keep lifecycle state checkable.

## Release Status

This repository contains the RoboLineage research runtime and lifecycle artifact interface. The current public release includes the FastAPI/Vue console, ROS2-oriented robot profiles and topic binding, raw rollout capture hooks, online VSA, asynchronous post-rollout review, dataset governance, dataset locks, training integration utilities, evaluation/deployment artifacts, prompt contracts, schemas, examples, and tests.

The runtime targets Ubuntu/ROS2 robot workstations. Windows is useful for editing and static review, but hardware-facing ROS2 paths should be run on the robot workstation.

Future releases will continue to add replay examples with mock model routes, broader adapter examples, additional real-robot task traces, and more packaged scoring utilities.

## What Is Included

- Robot onboarding and Task Config setup through profile-guided ROS2 topic binding, with an optional ROS2 topic probe for building new profiles.
- Raw rollout recording through rosbag2 while online VSA runs in the background.
- Online Visual Snapshot Agent (VSA) for sparse task-aware semantic anchors.
- Asynchronous post-rollout review with packetized evidence and terminal observations.
- Data Governance and Data Health agents for admission, failure pools, and dataset readiness.
- Framework discovery, dataset adaptation, training lifecycle execution, policy metadata, and version governance.
- Policy evaluation and deployment recommendation artifacts.
- Master Agent summaries that connect local artifacts across iterations.
- A Vue/FastAPI lifecycle console.

## Repository Layout

```text
configs/                         Runtime YAML files, robot profiles, and training framework examples.
docs/                            Runtime, artifact, and integration documentation.
examples/mini_lifecycle/          Small artifact trace showing the lifecycle format.
frontend/                         Vue lifecycle console.
robot_profiles/                   Generated or imported robot profile files.
scripts/                          Check and smoke scripts.
src/robolineage_app/              Unified runtime and API orchestration.
src/robolineage_robot/            Robot profiles and onboarding.
src/robolineage_data_source/      ROS2 sources, camera adapters, rosbag2 raw recorder.
src/robolineage_shared_agents/    Master Agent, LLM routes, and online VSA.
src/robolineage_post_rollout/     Post-rollout review, failure analysis, data governance.
src/robolineage_dataset/          Dataset locks and dataset version utilities.
src/robolineage_train/            Framework discovery, adapters, monitor, policy metadata.
src/robolineage_eval/             Policy evaluation and deployment governance.
src/robolineage_contracts/        Typed Python contracts shared across stages.
src/robolineage_schemas/          JSON schemas and validation helpers.
tests/                            Unit and integration tests for the Ubuntu/ROS target.
```

## Ubuntu / ROS2 Setup

This repository is intended to run on an Ubuntu robot workstation used for data collection. The Windows copy is useful for editing and static review, but ROS2 and hardware-facing tests should be run on the Ubuntu machine.

```bash
cd RoboLineage
python3 -m venv .venv-ros
source .venv-ros/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,contracts,data-source,vsa,session,agents,dataset,train]"
cd frontend
npm install
cd ..
```

ROS2 and `rclpy` are installed through the robot workstation's ROS distribution, not through pip. Hardware extras are optional:

```bash
python -m pip install -e ".[realsense]"
python -m pip install -e ".[vlm]"
```

## Configure Model Routes and Network Settings

If the robot workstation already exports model routes through the shell or
systemd, no repository-local `.env` is required. Otherwise copy `.env.example`
to `.env` and edit the endpoint, key, and model for each route.

```bash
cp .env.example .env
chmod 600 .env
```

The important customization points are:

- `ROBOT_ONBOARDING_LLM_*` for robot profile and binding understanding.
- `TASK_LLM_*` for optional task configuration assistance.
- `VSA_VLM_*` for online Visual Snapshot Agent calls.
- `POST_REVIEW_VLM_*` for asynchronous post-rollout review.
- `ROBOLINEAGE_DISCOVERY_LLM_*` for training repository discovery.
- `TRAINING_MONITOR_LLM_*` for training log interpretation.
- `POLICY_EVAL_VLM_*` for evaluation rollout review.
- `DEPLOYMENT_GOVERNANCE_LLM_*` for deployment recommendation summaries.
- `DATASET_HEALTH_LLM_*` for dataset health guidance.
- `MASTER_LLM_*` for cross-iteration Master Agent summaries.
- `OPENAI_*` as a compatibility fallback for OpenAI-compatible gateways.

Network and ROS settings live in environment variables, `cyclonedds.xml`, and
`configs/*.yaml`. Typical robot-level fields are `ROS_DOMAIN_ID`, ROS namespace,
camera topics, end-effector state topics, action topics, message types, and
profile health thresholds.

## Robot Profiles and Topic Probing

RoboLineage uses profile-guided ROS2 topic binding. A robot profile maps local
camera and robot-state topics into canonical lifecycle streams used by raw
capture, online VSA, post-review, training conversion, and evaluation. The
checked-in profiles are:

- `configs/robot_profiles/arx_one_default.yaml`: deployed ARX-style profile.
- `configs/robot_profiles/realman_default.yaml`: sanitized Realman template.
- `configs/robot_profiles/galbot_g1_default.yaml`: sanitized GALBOT G1 template.

On a ROS2 workstation, probe the current graph before editing a new profile:

```bash
python -m robolineage_robot.topic_probe --ros-domain-id "${ROS_DOMAIN_ID:-0}"
```

The probe is advisory. The operator still reviews the generated profile and
validates stream health before collection.

## Training Framework Profiles

Training integration is profile-based. RoboLineage stages accepted rollouts,
then calls the target repository's normal dataset, train, and evaluation
commands. Sanitized examples are in `configs/training_frameworks/`:

- `act_hdf5.example.yaml`
- `diffusion_policy.example.yaml`
- `lerobot_vla.example.yaml`

The Framework Discovery Agent can generate the same profile shape from a
repository tree and the operator's normal commands. The Dataset Adapter Agent
then materializes accepted rollouts for the selected profile and validates the
adapter output before training.

## Start the Runtime

The default runtime starts with the neutral console config. A robot profile can then be selected or activated from the frontend.

```bash
export ROBOLINEAGE_CONFIG=configs/robolineage_default.yaml
./run.sh
```

For a validated ARX profile:

```bash
ROBOLINEAGE_CONFIG=configs/arx_one.yaml ./run.sh
```

The backend listens on `http://localhost:8080` and the health service on `http://localhost:8081`. The frontend dev server listens on `http://localhost:5173` unless `ROBOLINEAGE_NO_FRONTEND=1` is set.

## Operator Workflow

1. Select or import a robot profile.
2. Define the task contract and success/failure phases.
3. Collect rollouts. Raw capture never waits for VLM calls.
4. Let online VSA write sparse semantic anchors during collection.
5. Review post-rollout packets and confirm dataset decisions when needed.
6. Build or update the dataset lock.
7. Launch the configured training command.
8. Inspect policy evaluation and deployment recommendation artifacts.
9. Follow the next-collection brief for targeted recollection.

## Artifact Names

RoboLineage writes machine-readable JSON artifacts and human-readable reports under the task/session directory. Common files include:

- `raw/rosbag2/`
- `snapshots.jsonl`
- `annotation.final.json`
- `failure_analysis.json`
- `dataset_decision.json` and `dataset_admission.json`
- `dataset_health_report.json`
- `dataset.lock`
- `training_status.json`
- `policy.meta.json`
- `policy_evaluation.json`
- `deployment_decision.json`
- `next_collection_brief.json`
- `master_review.json`

RoboLineage uses a lifecycle artifact contract for this boundary: model outputs become lifecycle state only after schema validation, deterministic checks, and traceable writes.

## Reproduction Scope

The public release supports artifact-level inspection and reproduction: schemas, prompts, sample artifacts, scoring scripts, and frozen evidence packets can be inspected without redistributing private full-resolution robot videos. Re-running semantic agents requires an equivalent model route and API access. Re-running policy training or robot evaluation requires the corresponding robot, ROS2 workspace, and training repository.

## Static Checks

On the Ubuntu machine, use the normal test suite. On Windows, prefer static checks only.

```bash
rg -n -i "legacy_project_token" .
rg -n "[\u4e00-\u9fff]" .
PYTHONPATH=src python scripts/check_contracts_imports.py
PYTHONPATH=src python scripts/check_ownership.py --from-git HEAD
```

## License and Release Notes

The release contains code, schemas, prompts, sample artifacts, and scripts. Keep private API keys, lab-specific network endpoints, private robot topics, and private robot logs outside git; use local environment variables and private robot profiles for deployment-specific values.
