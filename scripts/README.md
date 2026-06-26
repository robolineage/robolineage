# Scripts

Repository-level scripts used during development and deployment.

- `check_contracts_imports.py` checks cross-domain import boundaries.
- `check_ownership.py` checks file ownership rules.
- `vsa_streaming.py` starts the online VSA stream against ROS2 topics on the robot workstation.
- `vsa_rehearsal_single_host.py` runs a local rehearsal with synthetic publishers and the real VSA runtime.

Typical checks:

```bash
PYTHONPATH=src python scripts/check_contracts_imports.py
PYTHONPATH=src python scripts/check_ownership.py --from-git HEAD
```

The VSA streaming scripts require the Ubuntu ROS2 environment and the relevant robot profile. They are not expected to run on the Windows editing copy.
