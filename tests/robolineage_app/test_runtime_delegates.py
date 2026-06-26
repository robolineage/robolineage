from __future__ import annotations

from robolineage_app.runtime import UnifiedRuntime
from robolineage_data_source.config.schema import (
    Config,
    HealthConfig,
    RecorderConfig,
    RolloutConfig,
    ServicesToggle,
    TuningConfig,
    VlmConfig,
    VsaConfig,
)


def _runtime(tmp_path) -> UnifiedRuntime:
    return UnifiedRuntime(
        Config(
            rollout=RolloutConfig(task_id="t", operator_id="op"),
            recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
            services=ServicesToggle(data_source=False, session=False, vsa=False, health_check=False),
            tuning=TuningConfig(),
            vsa=VsaConfig(),
            vlm=VlmConfig(),
            health=HealthConfig(port=8081),
        )
    )


def test_unified_runtime_exposes_focused_internal_delegates(tmp_path):
    runtime = _runtime(tmp_path)

    assert runtime.robot_runtime.domain == "robot"
    assert runtime.master_runtime.domain == "master"
    assert runtime.training_runtime.domain == "training"
    assert runtime.rollout_runtime.domain == "rollout"

    assert runtime.robot_profiles() == runtime.robot_runtime.profiles()
    assert runtime.training_framework_status() == runtime.training_runtime.status()
    assert runtime.rollout_session_state() == runtime.rollout_runtime.session_state()
