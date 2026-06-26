from __future__ import annotations

from robolineage_shared_agents.visual_snapshot.vlm_runner import OpenAIVLMRunner, make_vlm_runner_from_env


def _clear_route(monkeypatch, prefix: str) -> None:
    for suffix in (
        "BACKEND",
        "API_KEY",
        "MODEL",
        "BASE_URL",
        "TIMEOUT",
        "MAX_TOKENS",
    ):
        monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)


def test_openai_compatible_vlm_route_defaults_to_current_sonnet(monkeypatch) -> None:
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    _clear_route(monkeypatch, "POST_REVIEW_VLM")
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POST_REVIEW_VLM_API_KEY", "post-review-key")
    monkeypatch.setenv("POST_REVIEW_VLM_BASE_URL", "https://gateway.example/v1")

    runner = make_vlm_runner_from_env("POST_REVIEW_VLM")

    assert isinstance(runner, OpenAIVLMRunner)
    assert runner.api_key == "post-review-key"
    assert runner.base_url == "https://gateway.example/v1"
    assert runner.model_name == "anthropic/claude-sonnet-4.6"
    assert runner.timeout == 60.0
    assert runner.max_output_tokens == 4096


def test_policy_eval_vlm_route_uses_its_own_model_before_openai_fallback(monkeypatch) -> None:
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    _clear_route(monkeypatch, "POLICY_EVAL_VLM")
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "openai-fallback-model")
    monkeypatch.setenv("POLICY_EVAL_VLM_API_KEY", "policy-eval-key")
    monkeypatch.setenv("POLICY_EVAL_VLM_MODEL", "anthropic/claude-sonnet-4.6")

    runner = make_vlm_runner_from_env("POLICY_EVAL_VLM")

    assert isinstance(runner, OpenAIVLMRunner)
    assert runner.api_key == "policy-eval-key"
    assert runner.model_name == "anthropic/claude-sonnet-4.6"
