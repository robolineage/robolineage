from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_OPENAI_COMPAT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_OPENAI_COMPAT_BASE_URL = "https://api.openai.com/v1"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


@dataclass(frozen=True)
class AIRouteSpec:
    prefix: str
    kind: str
    label: str
    implemented: bool
    fallback_prefixes: tuple[str, ...] = ()
    backend_default: str = "openai"
    base_url_default: str | None = None
    timeout_default: float | None = None
    max_tokens_default: int | None = None


@dataclass(frozen=True)
class AIRouteConfig:
    prefix: str
    kind: str
    label: str
    implemented: bool
    backend: str
    api_key: str | None
    model: str
    base_url: str | None
    timeout_s: float | None
    max_tokens: int | None
    sources: dict[str, str | None]
    direct_configured: bool

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def to_status(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "kind": self.kind,
            "label": self.label,
            "implemented": self.implemented,
            "configured": self.configured,
            "direct_configured": self.direct_configured,
            "api_key_configured": bool(self.api_key),
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "base_url_configured": self.base_url is not None,
            "timeout_s": self.timeout_s,
            "max_tokens": self.max_tokens,
            "sources": dict(self.sources),
        }


AI_ROUTE_SPECS: tuple[AIRouteSpec, ...] = (
    AIRouteSpec(
        prefix="ROBOT_ONBOARDING_LLM",
        kind="llm",
        label="Robot Onboarding LLM",
        implemented=True,
        base_url_default=DEFAULT_OPENAI_COMPAT_BASE_URL,
        timeout_default=30.0,
    ),
    AIRouteSpec(
        prefix="TASK_LLM",
        kind="vlm",
        label="Task Config LLM",
        implemented=True,
        fallback_prefixes=("OPENAI",),
    ),
    AIRouteSpec(
        prefix="VSA_VLM",
        kind="vlm",
        label="Online VSA VLM",
        implemented=True,
        fallback_prefixes=("OPENAI",),
    ),
    AIRouteSpec(
        prefix="POST_REVIEW_VLM",
        kind="vlm",
        label="PostRolloutReview VLM",
        implemented=True,
        fallback_prefixes=("OPENAI",),
        timeout_default=60.0,
        max_tokens_default=4096,
    ),
    AIRouteSpec(
        prefix="ROBOLINEAGE_DISCOVERY_LLM",
        kind="llm",
        label="Framework Discovery LLM",
        implemented=True,
        fallback_prefixes=("ROBOLINEAGE_AGENT", "TASK_LLM", "OPENAI"),
        base_url_default=DEFAULT_OPENAI_COMPAT_BASE_URL,
        timeout_default=60.0,
    ),
    AIRouteSpec(
        prefix="TRAINING_MONITOR_LLM",
        kind="llm",
        label="Training Monitor LLM",
        implemented=True,
        base_url_default=DEFAULT_OPENAI_COMPAT_BASE_URL,
        timeout_default=30.0,
    ),
    AIRouteSpec(
        prefix="POLICY_EVAL_VLM",
        kind="vlm",
        label="Policy Evaluation VLM",
        implemented=True,
        fallback_prefixes=("OPENAI",),
        max_tokens_default=512,
    ),
    AIRouteSpec(
        prefix="DEPLOYMENT_GOVERNANCE_LLM",
        kind="llm",
        label="Deployment Governance LLM",
        implemented=True,
        base_url_default=DEFAULT_OPENAI_COMPAT_BASE_URL,
        timeout_default=30.0,
    ),
    AIRouteSpec(
        prefix="DATASET_HEALTH_LLM",
        kind="llm",
        label="Dataset Health LLM",
        implemented=True,
        base_url_default=DEFAULT_OPENAI_COMPAT_BASE_URL,
        timeout_default=30.0,
    ),
    AIRouteSpec(
        prefix="MASTER_LLM",
        kind="llm",
        label="Master Agent LLM",
        implemented=True,
        fallback_prefixes=("ROBOLINEAGE_AGENT", "TASK_LLM", "OPENAI"),
        base_url_default=DEFAULT_OPENAI_COMPAT_BASE_URL,
        timeout_default=30.0,
    ),
)

_SPECS_BY_PREFIX = {spec.prefix: spec for spec in AI_ROUTE_SPECS}


def route_spec(prefix: str) -> AIRouteSpec:
    return _SPECS_BY_PREFIX.get(
        prefix,
        AIRouteSpec(prefix=prefix, kind="llm", label=prefix, implemented=False),
    )


def resolve_ai_route(
    prefix: str,
    *,
    fallback_prefixes: tuple[str, ...] | None = None,
    backend_default: str | None = None,
    default_model: str = DEFAULT_OPENAI_COMPAT_MODEL,
    base_url_default: str | None = None,
    timeout_default: float | None = None,
    max_tokens_default: int | None = None,
    load_dotenv: bool = True,
) -> AIRouteConfig:
    if load_dotenv:
        _load_dotenv_if_available()

    spec = route_spec(prefix)
    fallbacks = spec.fallback_prefixes if fallback_prefixes is None else fallback_prefixes
    candidates = (prefix, *fallbacks)
    backend = (os.environ.get(f"{prefix}_BACKEND") or backend_default or spec.backend_default).lower()
    api_key, api_key_source = _first_prefixed_env(candidates, "API_KEY")
    model, model_source = _first_prefixed_env(candidates, "MODEL")
    base_url, base_url_source = _first_prefixed_env(candidates, "BASE_URL")
    timeout_s, timeout_source = _first_timeout(candidates)
    max_tokens, max_tokens_source = _first_int(candidates, "MAX_TOKENS")

    return AIRouteConfig(
        prefix=prefix,
        kind=spec.kind,
        label=spec.label,
        implemented=spec.implemented,
        backend=backend,
        api_key=api_key,
        model=model or default_model,
        base_url=base_url or base_url_default or spec.base_url_default,
        timeout_s=timeout_s if timeout_s is not None else (
            timeout_default if timeout_default is not None else spec.timeout_default
        ),
        max_tokens=max_tokens if max_tokens is not None else (
            max_tokens_default if max_tokens_default is not None else spec.max_tokens_default
        ),
        sources={
            "api_key": api_key_source,
            "model": model_source,
            "base_url": base_url_source,
            "timeout": timeout_source,
            "max_tokens": max_tokens_source,
        },
        direct_configured=bool(os.environ.get(f"{prefix}_API_KEY")),
    )


def all_ai_route_statuses(*, load_dotenv: bool = True) -> dict[str, Any]:
    if load_dotenv:
        _load_dotenv_if_available()
    routes = {
        spec.prefix: resolve_ai_route(spec.prefix, load_dotenv=False).to_status()
        for spec in AI_ROUTE_SPECS
    }
    return {
        "schema_version": "RoboLineage.ai_routes.v1",
        "default_model": DEFAULT_OPENAI_COMPAT_MODEL,
        "routes": routes,
        "configured_count": sum(1 for route in routes.values() if route["configured"]),
        "implemented_count": sum(1 for route in routes.values() if route["implemented"]),
    }


def _first_prefixed_env(prefixes: tuple[str, ...], suffix: str) -> tuple[str | None, str | None]:
    for prefix in prefixes:
        key = f"{prefix}_{suffix}"
        value = os.environ.get(key)
        if value:
            return value, key
    return None, None


def _first_timeout(prefixes: tuple[str, ...]) -> tuple[float | None, str | None]:
    for suffix in ("TIMEOUT_S", "TIMEOUT_SEC", "TIMEOUT"):
        value, source = _first_prefixed_env(prefixes, suffix)
        if value is None:
            continue
        try:
            return float(value), source
        except ValueError:
            return None, source
    return None, None


def _first_int(prefixes: tuple[str, ...], suffix: str) -> tuple[int | None, str | None]:
    value, source = _first_prefixed_env(prefixes, suffix)
    if value is None:
        return None, None
    try:
        return int(value), source
    except ValueError:
        return None, source
