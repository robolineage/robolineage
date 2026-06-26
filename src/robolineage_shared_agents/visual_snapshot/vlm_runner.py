"""
VLM inference backends.

Layers:
    BaseVLMRunner      - shared interface
    OpenAIVLMRunner    - OpenAI-compatible chat-completions (openai SDK)
    GoogleVLMRunner    - Native Google Generative AI (google-generativeai SDK)
    AnthropicVLMRunner - Native Anthropic Claude (anthropic SDK)
    Qwen2VLRunner      - Local Qwen2-VL for offline / GPU deployment
    MockVLMRunner      - Fixed/random responses for tests

Factory:
    make_vlm_runner_from_env(prefix, valid_phases, vlm_cfg)
        Reads {prefix}_BACKEND / {prefix}_MODEL / {prefix}_API_KEY /
        {prefix}_BASE_URL from env (falls back to OPENAI_*) and returns
        the appropriate runner. Used by runtime.py and session/api.py.

Failure behavior:
    Timeout / garbled output / backend outage -> raise VLMInferenceError
    Retry / pause behavior is handled by the upper agent layer.
"""

from __future__ import annotations

import abc
import base64
import io
import json
import logging
import os
import random
import time
from typing import Optional

import numpy as np

from robolineage_shared_agents.llm_routes import DEFAULT_OPENAI_COMPAT_MODEL, resolve_ai_route

from .exceptions import VLMInferenceError

logger = logging.getLogger(__name__)


def _load_dotenv_if_available() -> None:
    """
    Load variables from a local .env file when python-dotenv is installed.

    This keeps the runner compatible with plain environment variables while
    also supporting a repo-local .env workflow.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


class BaseVLMRunner(abc.ABC):
    """Abstract base class for multimodal inference backends."""

    @abc.abstractmethod
    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        """
        Execute inference and return the raw text response.

        Args:
            prompt: structured prompt text
            images: RGB uint8 numpy arrays

        Returns:
            Raw model text output.

        Raises:
            VLMInferenceError: timeout, service failure, or empty output
        """


class OpenAIVLMRunner(BaseVLMRunner):
    """
    OpenAI-compatible chat-completions backend.

    This runner intentionally uses `chat.completions.create(...)` because the
    user's validated gateway configuration supports that endpoint. Some
    third-party OpenAI-compatible gateways do not implement `/responses`.

    Environment variables:
        OPENAI_API_KEY   required unless api_key is passed explicitly
        OPENAI_BASE_URL  optional, for proxies / gateways / compatible endpoints
        OPENAI_MODEL     optional, defaults to anthropic/claude-sonnet-4.6
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        timeout: float = 20.0,
        max_output_tokens: int = 256,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client_max_retries: int = 0,
    ):
        _load_dotenv_if_available()
        self.model_name = model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_COMPAT_MODEL)
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client_max_retries = client_max_retries
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise VLMInferenceError(
                "OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key explicitly."
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            raise VLMInferenceError(
                f"Required package not installed for OpenAIVLMRunner: {e}. "
                "Run: pip install openai"
            ) from e

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or None,
            timeout=self.timeout,
            max_retries=self.client_max_retries,
        )
        return self._client

    def _encode_image(self, image: np.ndarray, max_side: int = 768, jpeg_quality: int = 85) -> str:
        from PIL import Image as PILImage

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        pil_img = PILImage.fromarray(image)

        # Internal implementation note.
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                PILImage.BILINEAR,
            )

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_chat_content(self, prompt: str, images: list[np.ndarray]) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._encode_image(image),
                    },
                }
            )
        return content

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        client = self._get_client()
        encode_start = time.perf_counter()
        content = self._build_chat_content(prompt, images)
        encode_elapsed = time.perf_counter() - encode_start

        try:
            api_start = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                temperature=0.0,
                max_tokens=self.max_output_tokens,
            )
            api_elapsed = time.perf_counter() - api_start
        except Exception as e:
            raise VLMInferenceError(f"OpenAI VLM inference failed: {e}") from e

        output = ""
        try:
            output = (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError, TypeError):
            output = ""

        if not output:
            try:
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", "unknown")
                refusal = getattr(choice.message, "refusal", None)
                logger.error(
                    "VLM empty response — full response: %s",
                    response.model_dump() if hasattr(response, "model_dump") else repr(response),
                )
            except Exception:
                finish_reason, refusal = "unknown", None
            raise VLMInferenceError(
                f"OpenAI response had no text content "
                f"(model={self.model_name!r}, finish_reason={finish_reason!r}"
                + (f", refusal={refusal!r}" if refusal else "")
                + ")"
            )

        logger.info(
            "OpenAI VLM timing: model=%s images=%d encode=%.3fs api=%.3fs total=%.3fs",
            self.model_name,
            len(images),
            encode_elapsed,
            api_elapsed,
            encode_elapsed + api_elapsed,
        )
        return output


class Qwen2VLRunner(BaseVLMRunner):
    """
    Local Qwen2-VL backend (transformers + torch).

    Preserved intentionally as the local fallback implementation. The project
    can now call external OpenAI APIs through OpenAIVLMRunner, but this class
    remains available so local GPU deployment can be re-enabled without
    recovering old code from history.

    Dependencies:
        pip install torch transformers accelerate qwen-vl-utils
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        timeout: float = 20.0,
        max_new_tokens: int = 256,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map

        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            logger.info(f"Loading VLM: {self.model_name} (device_map={self.device_map})")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device_map,
            )
            logger.info("VLM loaded successfully.")
        except ImportError as e:
            raise VLMInferenceError(
                f"Required packages not installed for Qwen2VLRunner: {e}. "
                "Run: pip install torch transformers accelerate qwen-vl-utils"
            ) from e
        except Exception as e:
            raise VLMInferenceError(f"Failed to load VLM model '{self.model_name}': {e}") from e

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        import torch

        self._load_model()

        try:
            from PIL import Image as PILImage
            from qwen_vl_utils import process_vision_info

            # Keep the original local-model path intact for fallback use.
            prep_start = time.perf_counter()
            content: list[dict] = []
            for img in images:
                pil_img = PILImage.fromarray(img.astype(np.uint8))
                content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)
            prep_elapsed = time.perf_counter() - prep_start

            infer_start = time.perf_counter()
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )
            infer_elapsed = time.perf_counter() - infer_start

            if infer_elapsed > self.timeout:
                raise VLMInferenceError(
                    f"VLM inference timed out: {infer_elapsed:.1f}s > {self.timeout}s"
                )

            decode_start = time.perf_counter()
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            output = self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            decode_elapsed = time.perf_counter() - decode_start

            if not output:
                raise VLMInferenceError("VLM returned empty response.")

            logger.info(
                "Qwen2-VL timing: model=%s images=%d prepare=%.3fs infer=%.3fs decode=%.3fs total=%.3fs",
                self.model_name,
                len(images),
                prep_elapsed,
                infer_elapsed,
                decode_elapsed,
                prep_elapsed + infer_elapsed + decode_elapsed,
            )
            return output

        except VLMInferenceError:
            raise
        except Exception as e:
            raise VLMInferenceError(f"VLM inference failed: {e}") from e


class MockVLMRunner(BaseVLMRunner):
    """
    Mock VLM for tests without GPU or model files.

    Args:
        fixed_response: if provided, always return this string
        valid_phases: used when generating random legal responses
        failure_rate: simulated backend failure probability in [0, 1)
    """

    def __init__(
        self,
        fixed_response: Optional[str] = None,
        valid_phases: Optional[list[str]] = None,
        failure_rate: float = 0.0,
        latency: float = 0.1,
    ):
        self.fixed_response = fixed_response
        self.valid_phases = valid_phases or ["phase_1"]
        self.failure_rate = failure_rate
        self.latency = latency

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        start = time.perf_counter()
        time.sleep(self.latency)

        if self.failure_rate > 0 and random.random() < self.failure_rate:
            raise VLMInferenceError("MockVLMRunner: simulated inference failure.")

        if self.fixed_response is not None:
            elapsed = time.perf_counter() - start
            logger.info(
                "Mock VLM timing: images=%d fixed_response=true total=%.3fs",
                len(images),
                elapsed,
            )
            return self.fixed_response

        response = {
            "phase": random.choice(self.valid_phases),
            "progress": random.choice(["advancing", "stalled", "regressing"]),
            "risk_level": random.choice(["low", "medium", "high"]),
            "confidence": round(random.uniform(0.55, 0.95), 2),
        }
        elapsed = time.perf_counter() - start
        logger.info(
            "Mock VLM timing: images=%d fixed_response=false total=%.3fs",
            len(images),
            elapsed,
        )
        return json.dumps(response)


class GoogleVLMRunner(BaseVLMRunner):
    """Native Google Generative AI backend (google-generativeai SDK).

    Env vars (read at construction time via _load_dotenv_if_available):
        VSA_VLM_API_KEY  or  GOOGLE_API_KEY
        VSA_VLM_MODEL    default: gemini-2.0-flash
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 20.0,
        max_output_tokens: int = 256,
    ):
        _load_dotenv_if_available()
        self.model_name = model_name or os.getenv("VSA_VLM_MODEL", "gemini-2.0-flash")
        self.api_key = api_key or os.getenv("VSA_VLM_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def _to_pil(self, image: np.ndarray, max_side: int = 768):
        from PIL import Image as PILImage
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        pil_img = PILImage.fromarray(image)
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                PILImage.BILINEAR,
            )
        return pil_img

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise VLMInferenceError(
                f"google-generativeai not installed: {exc}. Run: pip install google-generativeai"
            ) from exc

        if not self.api_key:
            raise VLMInferenceError(
                "Google API key missing. Set VSA_VLM_API_KEY or GOOGLE_API_KEY."
            )

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        parts: list = [prompt] + [self._to_pil(img) for img in images]

        try:
            api_start = time.perf_counter()
            response = model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_output_tokens,
                    temperature=0.0,
                ),
                request_options={"timeout": self.timeout},
            )
            api_elapsed = time.perf_counter() - api_start
        except Exception as exc:
            raise VLMInferenceError(f"Google VLM inference failed: {exc}") from exc

        try:
            output = (response.text or "").strip()
        except Exception:
            output = ""

        if not output:
            raise VLMInferenceError(
                f"Google VLM returned no text "
                f"(model={self.model_name!r}, "
                f"finish_reason={getattr(response.candidates[0] if response.candidates else None, 'finish_reason', 'unknown')!r})"
            )

        logger.info(
            "Google VLM timing: model=%s images=%d api=%.3fs",
            self.model_name, len(images), api_elapsed,
        )
        return output


class AnthropicVLMRunner(BaseVLMRunner):
    """Native Anthropic Claude backend (anthropic SDK).

    Env vars (read at construction time via _load_dotenv_if_available):
        VSA_VLM_API_KEY  or  ANTHROPIC_API_KEY
        VSA_VLM_BASE_URL (optional, for proxy)
        VSA_VLM_MODEL    default: claude-3-5-sonnet-20241022
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 20.0,
        max_output_tokens: int = 256,
    ):
        _load_dotenv_if_available()
        self.model_name = model_name or os.getenv("VSA_VLM_MODEL", "claude-3-5-sonnet-20241022")
        self.api_key = api_key or os.getenv("VSA_VLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.getenv("VSA_VLM_BASE_URL")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def _encode_image(self, image: np.ndarray, max_side: int = 768, jpeg_quality: int = 85) -> str:
        from PIL import Image as PILImage
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        pil_img = PILImage.fromarray(image)
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                PILImage.BILINEAR,
            )
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise VLMInferenceError(
                f"anthropic not installed: {exc}. Run: pip install anthropic"
            ) from exc

        if not self.api_key:
            raise VLMInferenceError(
                "Anthropic API key missing. Set VSA_VLM_API_KEY or ANTHROPIC_API_KEY."
            )

        kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = anthropic.Anthropic(**kwargs)

        content: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": self._encode_image(img),
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})

        try:
            api_start = time.perf_counter()
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_output_tokens,
                messages=[{"role": "user", "content": content}],
            )
            api_elapsed = time.perf_counter() - api_start
        except Exception as exc:
            raise VLMInferenceError(f"Anthropic VLM inference failed: {exc}") from exc

        try:
            output = (response.content[0].text or "").strip()
        except Exception:
            output = ""

        if not output:
            raise VLMInferenceError(
                f"Anthropic VLM returned no text "
                f"(model={self.model_name!r}, stop_reason={getattr(response, 'stop_reason', 'unknown')!r})"
            )

        logger.info(
            "Anthropic VLM timing: model=%s images=%d api=%.3fs",
            self.model_name, len(images), api_elapsed,
        )
        return output


def make_vlm_runner_from_env(
    prefix: str = "VSA_VLM",
    *,
    valid_phases: Optional[list[str]] = None,
    timeout_default: float | None = None,
    max_output_tokens_default: int | None = None,
    min_timeout_s: float | None = None,
    min_output_tokens: int | None = None,
) -> "BaseVLMRunner":
    """Build a VLM runner from environment variables.

    Priority for each field: {prefix}_* > OPENAI_* fallback.
    If no API key is found, returns MockVLMRunner (safe offline default).

    Args:
        prefix: env var prefix, e.g. "VSA_VLM" or "TASK_LLM"
        valid_phases: passed to MockVLMRunner when falling back to mock
        timeout_default / max_output_tokens_default: used when env or route spec do not set them
        min_timeout_s: optional timeout floor applied after env/default resolution
        min_output_tokens: optional floor applied after env/default resolution
    """
    route = resolve_ai_route(
        prefix,
        fallback_prefixes=("OPENAI",),
        timeout_default=timeout_default,
        max_tokens_default=max_output_tokens_default,
    )
    backend = route.backend
    api_key = route.api_key
    model = route.model
    base_url = route.base_url
    timeout = route.timeout_s or timeout_default or 20.0
    if min_timeout_s is not None:
        timeout = max(timeout, min_timeout_s)
    max_tokens = route.max_tokens or max_output_tokens_default or 256
    if min_output_tokens is not None:
        max_tokens = max(max_tokens, min_output_tokens)

    if not api_key:
        logger.warning(
            "%s_API_KEY not set — falling back to MockVLMRunner", prefix
        )
        return MockVLMRunner(valid_phases=valid_phases or [], latency=0.0)

    if backend == "google":
        return GoogleVLMRunner(
            model_name=model,
            api_key=api_key,
            timeout=timeout,
            max_output_tokens=max_tokens,
        )
    if backend == "anthropic":
        return AnthropicVLMRunner(
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_output_tokens=max_tokens,
        )
    # default: openai-compatible
    return OpenAIVLMRunner(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_output_tokens=max_tokens,
    )
