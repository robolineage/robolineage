"""
Implementation note.

Implementation note.
    Implementation note.
    Implementation note.
    Implementation note.

Implementation note.
"""

from __future__ import annotations

import json
import logging
import re

from .exceptions import VLMInferenceError
from .types import TaskConfig

logger = logging.getLogger(__name__)

# Internal implementation note.
_JSON_BLOCK_RE = re.compile(r'\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\}', re.DOTALL)


class ResponseParser:
    """Implementation note."""

    EXPECTED_KEYS = {"phase", "progress", "risk_level", "confidence"}

    def parse(self, raw_response: str, task_config: TaskConfig) -> dict:
        """
        Implementation note.

        Args:
            Implementation note.
            Implementation note.

        Returns:
            Implementation note.
            Implementation note.

        Raises:
            Implementation note.
        """
        if not raw_response or not raw_response.strip():
            raise VLMInferenceError("VLM returned empty or whitespace-only response.")

        # Internal implementation note.
        if raw_response.startswith(("skipped:", "error:")):
            logger.warning("Sentinel response received (%r), returning conservative fallback.", raw_response)
            return {
                "phase": "",
                "progress": "unknown",
                "risk_level": "unknown",
                "confidence": 0.1,
                "imminent_failure": False,
                "needs_review": True,
            }

        # Internal implementation note.
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Internal implementation note.
            inner = lines[1:] if lines[0].startswith("```") else lines
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            text = "\n".join(inner).strip()
        else:
            text = raw_response

        # Internal implementation note.
        parsed = self._try_direct_json(text)
        if parsed is not None:
            return parsed

        # Internal implementation note.
        parsed = self._try_extract_json(text)
        if parsed is not None:
            return parsed

        # Internal implementation note.
        snippet = raw_response[:200].replace("\n", " ")
        raise VLMInferenceError(
            f"Cannot parse VLM response as JSON. Raw (truncated): {snippet!r}"
        )

    # ------------------------------------------------------------------
    # Internal implementation note.
    # ------------------------------------------------------------------

    def _try_direct_json(self, text: str) -> dict | None:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return None

    def _try_extract_json(self, text: str) -> dict | None:
        # Internal implementation note.
        candidates = _JSON_BLOCK_RE.findall(text)
        # Internal implementation note.
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and self.EXPECTED_KEYS.issubset(obj.keys()):
                    logger.debug("Extracted JSON block from VLM response wrapper.")
                    return obj
            except json.JSONDecodeError:
                continue
        # Internal implementation note.
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and any(k in obj for k in self.EXPECTED_KEYS):
                    logger.debug("Extracted partial JSON block from VLM response.")
                    return obj
            except json.JSONDecodeError:
                continue
        return None
