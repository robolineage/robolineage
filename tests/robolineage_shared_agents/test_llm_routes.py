from __future__ import annotations


def _clear_ai_env(monkeypatch) -> None:
    prefixes = (
        "ROBOT_ONBOARDING_LLM",
        "TASK_LLM",
        "VSA_VLM",
        "POST_REVIEW_VLM",
        "ROBOLINEAGE_DISCOVERY_LLM",
        "TRAINING_MONITOR_LLM",
        "POLICY_EVAL_VLM",
        "DEPLOYMENT_GOVERNANCE_LLM",
        "DATASET_HEALTH_LLM",
        "MASTER_LLM",
        "ROBOLINEAGE_AGENT",
        "OPENAI",
    )
    suffixes = (
        "BACKEND",
        "API_KEY",
        "BASE_URL",
        "MODEL",
        "TIMEOUT",
        "TIMEOUT_S",
        "TIMEOUT_SEC",
        "MAX_TOKENS",
    )
    for prefix in prefixes:
        for suffix in suffixes:
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)


def test_resolve_ai_route_uses_explicit_route_before_openai_fallback(monkeypatch) -> None:
    from robolineage_shared_agents import llm_routes

    monkeypatch.setattr(llm_routes, "_load_dotenv_if_available", lambda: None)
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model")
    monkeypatch.setenv("TASK_LLM_API_KEY", "task-key")
    monkeypatch.setenv("TASK_LLM_BASE_URL", "https://task.example/v1")

    route = llm_routes.resolve_ai_route("TASK_LLM", fallback_prefixes=("OPENAI",))

    assert route.api_key == "task-key"
    assert route.model == "openai-model"
    assert route.base_url == "https://task.example/v1"
    assert route.sources["api_key"] == "TASK_LLM_API_KEY"
    assert route.sources["model"] == "OPENAI_MODEL"


def test_discovery_route_helper_does_not_fallback_to_vsa(monkeypatch) -> None:
    from robolineage_shared_agents import llm_routes

    monkeypatch.setattr(llm_routes, "_load_dotenv_if_available", lambda: None)
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("VSA_VLM_API_KEY", "vsa-key")
    monkeypatch.setenv("VSA_VLM_MODEL", "vsa-model")

    route = llm_routes.resolve_ai_route(
        "ROBOLINEAGE_DISCOVERY_LLM",
        fallback_prefixes=("ROBOLINEAGE_AGENT", "TASK_LLM", "OPENAI"),
    )

    assert route.api_key is None
    assert route.model == "anthropic/claude-sonnet-4.6"
    assert route.configured is False


def test_all_ai_route_statuses_include_agent_routes_without_secrets(monkeypatch) -> None:
    from robolineage_shared_agents import llm_routes

    monkeypatch.setattr(llm_routes, "_load_dotenv_if_available", lambda: None)
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("TRAINING_MONITOR_LLM_API_KEY", "monitor-placeholder-key")
    monkeypatch.setenv("TRAINING_MONITOR_LLM_BASE_URL", "https://gateway.example/v1")

    payload = llm_routes.all_ai_route_statuses()

    monitor = payload["routes"]["TRAINING_MONITOR_LLM"]
    assert monitor["implemented"] is True
    assert monitor["configured"] is True
    assert monitor["api_key_configured"] is True
    assert monitor["model"] == "anthropic/claude-sonnet-4.6"
    assert "api_key" not in monitor
    assert "monitor-placeholder-key" not in str(payload)
    assert payload["routes"]["ROBOT_ONBOARDING_LLM"]["implemented"] is True
    assert payload["routes"]["DEPLOYMENT_GOVERNANCE_LLM"]["implemented"] is True
    assert payload["routes"]["DATASET_HEALTH_LLM"]["implemented"] is True
    assert "TRAINING_LIFECYCLE_LLM" not in payload["routes"]
