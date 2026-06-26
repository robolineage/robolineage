<h1>
  <img src="assets/brand/robolineage_logo_180.png" alt="RoboLineage logo" width="42">
  RoboLineage
</h1>

RoboLineage is an agent-native data lifecycle governance system for robot policy
iteration. It provides a portable lifecycle interface that links rollout
evidence, review decisions, dataset updates, training records, evaluations, and
next-collection plans through typed artifacts. The same interface can sit behind
different robot embodiments, data streams, and policy learners while keeping the
iteration traceable across time.

[Project page](https://robolineage.github.io/) |
[arXiv](https://arxiv.org/abs/2606.22142) |
[PDF](https://arxiv.org/pdf/2606.22142)

<p align="center">
  <img src="assets/figures/fig1.png" alt="RoboLineage teaser" width="600">
</p>

## Core Idea

Existing robot policy iteration often leaves the work between training runs to
expert reconstruction: which evidence mattered, why data changed, which
checkpoint used which dataset, and what should be collected next. RoboLineage
turns those transitions into linked lifecycle artifacts. Agents interpret
robotic evidence and prepare structured outputs, while schemas and artifact
boundaries keep lifecycle state inspectable, versioned, and reusable.

![RoboLineage lifecycle overview](assets/figures/fig2.png)

## Release Status

The public repository now includes the research runtime, frontend console,
artifact contracts, prompt contracts, example traces, and tests. The runtime is
intended for Ubuntu/ROS2 robot workstations; Windows is useful for editing and
static review.

- [x] Lifecycle artifact schemas, prompt contracts, and mini artifact examples.
- [x] FastAPI backend and Vue lifecycle console.
- [x] Profile-guided robot onboarding and ROS2 topic binding.
- [x] Raw rollout capture hooks and online Visual Snapshot Agent (VSA).
- [x] Asynchronous post-rollout review and dataset governance.
- [x] Dataset locks, Data Health, training integration, policy metadata, and
      evaluation/deployment artifacts.
- [x] Static checks and runtime contract tests.
- [ ] Mock replay pipeline for users without robot hardware.
- [ ] Packaged scoring scripts and benchmark-style artifact replay.
- [ ] Broader robot/profile templates and local model route examples.
- [ ] More training adapters and real-robot lifecycle examples.
- [ ] Upgraded agent modules for reward/progress signals, failure memory, and
      closed-loop data improvement.

## Quickstart

```bash
git clone https://github.com/robolineage/robolineage.git
cd robolineage

python3 -m venv .venv-ros
source .venv-ros/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,contracts,data-source,vsa,session,agents,dataset,train]"

cd frontend
npm install
cd ..

export ROBOLINEAGE_CONFIG=configs/robolineage_default.yaml
./run.sh
```

The backend listens on `http://localhost:8080`, the health service on
`http://localhost:8081`, and the frontend dev server on
`http://localhost:5173`. ROS2 and `rclpy` should come from the robot
workstation's ROS distribution, not from pip.

<details>
<summary>Model routes, robot profiles, and runtime configuration</summary>

If the robot workstation already exports model routes through the shell or
systemd, no repository-local `.env` is required. Otherwise copy `.env.example`
to `.env` and edit the endpoint, key, and model for each route:

```bash
cp .env.example .env
chmod 600 .env
```

Important route groups include `ROBOT_ONBOARDING_LLM_*`, `TASK_LLM_*`,
`VSA_VLM_*`, `POST_REVIEW_VLM_*`, `ROBOLINEAGE_DISCOVERY_LLM_*`,
`TRAINING_MONITOR_LLM_*`, `POLICY_EVAL_VLM_*`,
`DEPLOYMENT_GOVERNANCE_LLM_*`, `DATASET_HEALTH_LLM_*`, and `MASTER_LLM_*`.

RoboLineage uses profile-guided ROS2 topic binding. Checked-in examples live in
`configs/robot_profiles/` and `configs/training_frameworks/`. On a ROS2
workstation, probe the graph before editing a new robot profile:

```bash
python -m robolineage_robot.topic_probe --ros-domain-id "${ROS_DOMAIN_ID:-0}"
```

For a validated profile:

```bash
ROBOLINEAGE_CONFIG=configs/arx_one.yaml ./run.sh
```

</details>

## Repository Layout

```text
assets/figures/                 Paper and project-page figures.
configs/                        Runtime YAML files, robot profiles, and training examples.
docs/                           Runtime, artifact, and integration documentation.
examples/mini_lifecycle/         Small artifact trace showing the lifecycle format.
frontend/                       Vue lifecycle console.
prompts/                         Prompt contracts for artifact-producing agents.
schemas/                         JSON schemas for typed lifecycle artifacts.
scripts/                         Smoke checks and runtime helpers.
src/                             RoboLineage Python runtime packages.
tests/                           Unit and integration tests for the Ubuntu/ROS target.
```

## Lifecycle Artifacts

RoboLineage treats robot policy iteration as a sequence of typed transitions:

```text
robot + task + collection context
  -> rollout evidence
  -> visual snapshots and post-rollout review
  -> dataset decision and dataset lock
  -> training record and policy metadata
  -> evaluation summary
  -> deployment recommendation and next-collection brief
```

Each artifact carries an identifier, parent links, producer metadata,
timestamps, and typed content. This lets downstream stages read from a stable
interface instead of relying on ad hoc folders, review sheets, local scripts, or
expert memory.

## Where to Start

- System view: [docs/overview.md](docs/overview.md),
  [docs/agent_roles.md](docs/agent_roles.md), and
  [docs/operator_workflow.md](docs/operator_workflow.md).
- Artifact contract: [docs/lifecycle_artifact_contract.md](docs/lifecycle_artifact_contract.md),
  [docs/artifact_contracts.md](docs/artifact_contracts.md),
  [schemas/](schemas/), and [examples/mini_lifecycle](examples/mini_lifecycle).
- Robot and task setup: [docs/robot_onboarding.md](docs/robot_onboarding.md),
  [configs/robot_profiles/](configs/robot_profiles/), and
  [configs/training_frameworks/](configs/training_frameworks/).
- Review pipeline: [docs/vsa_post_review.md](docs/vsa_post_review.md),
  [docs/dataset_governance.md](docs/dataset_governance.md), and
  [prompts/](prompts/).
- Training and iteration: [docs/training_integration.md](docs/training_integration.md),
  [docs/evaluation_and_recollection.md](docs/evaluation_and_recollection.md),
  and [docs/reproducibility.md](docs/reproducibility.md).
- Runtime entry points: [src/](src/), [frontend/README.md](frontend/README.md),
  [run.sh](run.sh), and [scripts/](scripts/).
- Additional references: [docs/artifact_walkthrough.md](docs/artifact_walkthrough.md),
  [docs/prompt_contracts.md](docs/prompt_contracts.md),
  [docs/integration_path.md](docs/integration_path.md),
  [docs/reproducibility_scope.md](docs/reproducibility_scope.md),
  [docs/deployment/](docs/deployment/), [AGENTS.md](AGENTS.md), and
  [CITATION.cff](CITATION.cff).

## Static Checks

On the Ubuntu machine, use the normal test suite. On Windows, prefer static
checks only.

```bash
rg -n -i "legacy_project_token" .
rg -n "[\u4e00-\u9fff]" .
PYTHONPATH=src python scripts/check_contracts_imports.py
PYTHONPATH=src python scripts/check_ownership.py --from-git HEAD
```

## Citation

```bibtex
@article{luo2026robolineage,
  title={RoboLineage: Agent-Native Data Lifecycle Governance Across Robot Policy Iterations},
  author={Luo, Qian and Guo, Wentao and Qin, Zhennan and Guo, Nanchun and Zhao, Yunhan and Ma, Yi and Yang, Yanchao},
  journal={arXiv preprint arXiv:2606.22142},
  year={2026}
}
```

## License

This release is provided under the MIT License.
