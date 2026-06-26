from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .llm_routes import DEFAULT_OPENAI_COMPAT_BASE_URL, DEFAULT_OPENAI_COMPAT_MODEL, resolve_ai_route


@dataclass(frozen=True)
class OpenAICompatibleJSONClient:
    """Small OpenAI-compatible chat client for optional JSON-only understanding."""

    api_key: str
    model: str
    system_prompt: str
    base_url: str = DEFAULT_OPENAI_COMPAT_BASE_URL
    timeout_s: float = 30.0
    temperature: float = 0.2

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str,
        system_prompt: str,
        fallback_prefixes: tuple[str, ...] = (),
        timeout_default: float = 30.0,
        load_dotenv: bool = True,
    ) -> "OpenAICompatibleJSONClient | None":
        route = resolve_ai_route(
            prefix,
            fallback_prefixes=fallback_prefixes,
            base_url_default=DEFAULT_OPENAI_COMPAT_BASE_URL,
            timeout_default=timeout_default,
            load_dotenv=load_dotenv,
        )
        if not route.api_key:
            return None
        return cls(
            api_key=route.api_key,
            model=route.model or DEFAULT_OPENAI_COMPAT_MODEL,
            system_prompt=system_prompt,
            base_url=(route.base_url or DEFAULT_OPENAI_COMPAT_BASE_URL).rstrip("/"),
            timeout_s=float(route.timeout_s or timeout_default),
        )

    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # nosec B310 - configured endpoint
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.model} HTTP {exc.code}: {detail[:500]}") from exc
        content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        parsed = parse_json_object(str(content))
        if not parsed:
            raise RuntimeError("LLM response did not contain a JSON object")
        return parsed


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None
