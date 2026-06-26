# Repository Guide for Agents

RoboLineage is organized as a set of bounded lifecycle agents around typed artifacts. Keep public text in English and keep the repository free of legacy project names.

## Boundaries

- Cross-stage types belong in `src/robolineage_contracts/` or `src/robolineage_schemas/`.
- Runtime orchestration belongs in `src/robolineage_app/` and `src/robolineage_session/`.
- Robot profiles and onboarding belong in `src/robolineage_robot/`.
- ROS2 sources, cameras, robot state, and raw rosbag2 capture belong in `src/robolineage_data_source/`.
- Online VSA and Master Agent code belongs in `src/robolineage_shared_agents/`.
- Post-rollout review and data governance belong in `src/robolineage_post_rollout/`.
- Dataset locks belong in `src/robolineage_dataset/`.
- Training framework integration and data health belong in `src/robolineage_train/`.
- Evaluation and deployment recommendations belong in `src/robolineage_eval/`.

## Development Commands

```bash
python -m pip install -e ".[dev,contracts,data-source,vsa,session,agents,dataset,train]"
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/check_contracts_imports.py
PYTHONPATH=src python scripts/check_ownership.py --from-git HEAD
cd frontend && npm run build
```

The Windows copy is not expected to run ROS2 or hardware tests. Run the full test suite on the Ubuntu robot workstation.

## Editing Rules

- Keep API and network route templates in `.env.example`; do not commit a real `.env`.
- Preserve `ROBOLINEAGE_*`, `VSA_VLM_*`, `POST_REVIEW_VLM_*`, and related route names because the Ubuntu deployment depends on them.
- Do not let VLM/LLM output directly mutate raw data, deploy policies, or bypass schema validation.
- When adding artifacts, write JSON first, validate it, then expose summaries to the frontend or Master Agent.
- Keep all public files in English.
