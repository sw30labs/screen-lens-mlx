"""Small OpenAI-compatible inference client used by ScreenLens.

Both vLLM on DGX Spark and oMLX on Apple Silicon expose the same
``/v1/chat/completions`` contract, including OpenAI-style vision inputs.  The
legacy module and ``OMLXClient`` names remain as compatibility aliases.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit

from .config import (
    CaptionBackend,
    CaptioningConfig,
    InferenceBackend,
    load_dotenv_if_present,
)

logger = logging.getLogger("screenlens.inference")


DEFAULT_OMLX_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_MODEL = "nvidia/Qwen3.6-27B-NVFP4"
DEFAULT_VLLM_CONTEXT = 32768
_VLLM_CONTEXT_ERROR_PARAMS = {"input_tokens", "input_text"}
_TOKENIZE_CHAT_FIELDS = {
    "model",
    "messages",
    "add_generation_prompt",
    "continue_final_message",
    "chat_template",
    "chat_template_kwargs",
    "media_io_kwargs",
    "mm_processor_kwargs",
    "tools",
}
_API_KEY_PLACEHOLDERS = {
    "your-api-key",
    "your-api-key-here",
    "your-omlx-api-key",
    "your-omlx-api-key-here",
    "hf_replace_me",
}
_OMLX_KEY_PLACEHOLDERS = _API_KEY_PLACEHOLDERS
# Known text-only families. A vision marker (below) always overrides — so e.g.
# "MiniMax-VL" is still treated as vision even though "minimax" is listed here.
_KNOWN_TEXT_ONLY_PATTERNS = (
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-v4",
    "gpt-oss",
    "minimax-m1",
    "minimax-m2",
    "minimax-m3",
    "minimax-text",
    "kimi-k2",
    "nemotron",
    "glm-5-1",
)
_KNOWN_VISION_MARKERS = ("vl", "vision", "omni", "janus")
# Unified/multimodal model families whose names DON'T contain a vision marker
# but which do accept image input (verified June 2026). Matched on the
# normalized id (non-alphanumerics → '-'). VL/vision markers above still win.
_KNOWN_VISION_PATTERNS = (
    "gemma-4", "gemma-3",          # Gemma 3/4 are natively multimodal (OCR/doc/screen)
    "qwen3-6", "qwen3-5",          # Qwen3.5/3.6 are unified multimodal
    "qwen2-5-vl", "qwen3-vl",
    "pixtral", "internvl", "minicpm-v", "llava", "molmo", "kimi-vl",
)
# Draft/speculative-decode helpers — not standalone OCR models. "mtp" only
# counts as a standalone token (so "MTPLX-Optimized" — a real served model — is
# NOT flagged), via is_draft_model().
_DRAFT_MARKERS = ("dflash", "draft", "eagle")
_DRAFT_RE = re.compile(r"(^|-)mtp(-|$)")


class InferenceTruncatedError(RuntimeError):
    """Raised when a caller requires a complete generation but receives a prefix."""

    def __init__(self, backend: str, completion_limit: int | str):
        self.backend = backend
        self.completion_limit = completion_limit
        super().__init__(
            f"{backend} truncated the response at {completion_limit} "
            "(finish_reason=length); incomplete output was discarded"
        )


def is_draft_model(model_id: str | None) -> bool:
    """True for speculative-decode draft models (not usable standalone)."""
    if not model_id:
        return False
    n = normalized_model_id(model_id)
    return any(m in n for m in _DRAFT_MARKERS) or bool(_DRAFT_RE.search(n))


def _env_value(*names: str, ignore_placeholders: bool = False) -> str | None:
    load_dotenv_if_present()
    for name in names:
        value = os.getenv(name)
        if not value:
            continue
        value = value.strip()
        if ignore_placeholders and value.lower() in _API_KEY_PLACEHOLDERS:
            continue
        return value
    return None


def normalize_api_base_url(url: str) -> str:
    """Normalize a root, dashboard, or API URL to an OpenAI ``/v1`` base."""
    parsed = urlsplit(url)
    if parsed.path in ("", "/") or parsed.path.startswith("/admin"):
        return urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", ""))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def normalize_omlx_base_url(url: str) -> str:
    """Compatibility alias for :func:`normalize_api_base_url`."""
    return normalize_api_base_url(url)


def resolve_omlx_base_url(config: CaptioningConfig) -> str:
    """Resolve oMLX base URL with Scriptorium-compatible env aliases."""
    env_url = _env_value("MLX_BASE_URL", "OMLX_BASE_URL")
    configured = config.omlx_base_url
    if configured and configured != DEFAULT_OMLX_BASE_URL:
        return normalize_api_base_url(configured)
    return normalize_api_base_url(env_url or configured or DEFAULT_OMLX_BASE_URL)


def resolve_omlx_api_key(config: CaptioningConfig) -> str | None:
    """Resolve oMLX API key with MLX_* and OMLX_* aliases."""
    return config.omlx_api_key or _env_value(
        "MLX_API_KEY",
        "OMLX_API_KEY",
        ignore_placeholders=True,
    )


def resolve_omlx_model(config: CaptioningConfig) -> str:
    """Resolve the model id to send to oMLX."""
    return (
        config.omlx_model
        or _env_value("MLX_MODEL", "OMLX_MODEL", "LLM_MODEL")
        or "default"
    )


def resolve_vllm_base_url(config: CaptioningConfig) -> str:
    """Resolve a direct vLLM endpoint with explicit config taking precedence."""
    configured = config.vllm_base_url
    env_url = _env_value("VLLM_BASE_URL")
    if configured and configured != DEFAULT_VLLM_BASE_URL:
        return normalize_api_base_url(configured)
    return normalize_api_base_url(env_url or configured or DEFAULT_VLLM_BASE_URL)


def resolve_vllm_api_key(config: CaptioningConfig) -> str:
    """Resolve vLLM auth; the bundled loopback service accepts ``local``."""
    return config.vllm_api_key or _env_value(
        "VLLM_API_KEY", ignore_placeholders=True
    ) or "local"


def resolve_vllm_model(config: CaptioningConfig) -> str:
    """Resolve the model id served by vLLM on DGX Spark."""
    return config.vllm_model or _env_value("VLLM_MODEL") or DEFAULT_VLLM_MODEL


def resolve_inference_backend(config: CaptioningConfig) -> InferenceBackend:
    """Return the direct backend selected for a captioning config."""
    backend = getattr(config.backend, "value", config.backend)
    if backend == CaptionBackend.ollama.value:
        raise ValueError("Ollama does not use the direct OpenAI-compatible client")
    return InferenceBackend(backend)


def resolve_inference_base_url(config: CaptioningConfig) -> str:
    """Resolve the selected direct provider's API root."""
    if resolve_inference_backend(config) == InferenceBackend.vllm:
        return resolve_vllm_base_url(config)
    return resolve_omlx_base_url(config)


def resolve_inference_api_key(config: CaptioningConfig) -> str | None:
    """Resolve the selected direct provider's bearer token."""
    if resolve_inference_backend(config) == InferenceBackend.vllm:
        return resolve_vllm_api_key(config)
    return resolve_omlx_api_key(config)


def resolve_inference_model(config: CaptioningConfig) -> str:
    """Resolve the selected direct provider's model id."""
    if resolve_inference_backend(config) == InferenceBackend.vllm:
        return resolve_vllm_model(config)
    return resolve_omlx_model(config)


def resolve_inference_context(config: CaptioningConfig) -> int:
    """Return the configured context size used for prompt chunk planning."""
    if resolve_inference_backend(config) == InferenceBackend.vllm:
        env_context = _env_value("VLLM_MAX_MODEL_LEN")
        if config.vllm_model_context == DEFAULT_VLLM_CONTEXT and env_context:
            try:
                parsed = int(env_context)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        return config.vllm_model_context
    return config.omlx_model_context


def resolve_role_backend(config: Any) -> InferenceBackend:
    """Resolve an OCR/reconstruction config's direct backend."""
    backend = getattr(config, "backend", None)
    value = getattr(backend, "value", backend) or InferenceBackend.omlx.value
    return InferenceBackend(value)


def resolve_role_base_url(config: Any) -> str:
    """Resolve an OCR/reconstruction endpoint with provider env aliases."""
    configured = getattr(config, "base_url", None)
    backend = resolve_role_backend(config)
    env_url = _env_value("VLLM_BASE_URL") if backend == InferenceBackend.vllm else _env_value(
        "MLX_BASE_URL", "OMLX_BASE_URL"
    )
    default = DEFAULT_VLLM_BASE_URL if backend == InferenceBackend.vllm else DEFAULT_OMLX_BASE_URL
    if configured and configured != default:
        return normalize_api_base_url(configured)
    return normalize_api_base_url(env_url or configured or default)


def resolve_role_context(config: Any) -> int:
    """Resolve role-specific context planning against the vLLM serving limit."""
    configured = int(getattr(config, "model_context", DEFAULT_VLLM_CONTEXT))
    if (
        resolve_role_backend(config) == InferenceBackend.vllm
        and configured == DEFAULT_VLLM_CONTEXT
    ):
        env_context = _env_value("VLLM_MAX_MODEL_LEN")
        if env_context:
            try:
                parsed = int(env_context)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
    return configured


def resolve_role_api_key(config: Any, *preferred_names: str) -> str | None:
    """Resolve OCR/reconstruction auth without conflating vLLM and MLX envs."""
    configured = getattr(config, "api_key", None)
    if configured:
        return configured
    backend = resolve_role_backend(config)
    if backend == InferenceBackend.vllm:
        vllm_names = tuple(name for name in preferred_names if name.startswith("VLLM_"))
        return _env_value(
            *vllm_names, "VLLM_API_KEY", ignore_placeholders=True
        ) or "local"
    omlx_names = tuple(name for name in preferred_names if not name.startswith("VLLM_"))
    return _env_value(
        *omlx_names,
        "MLX_API_KEY",
        "OMLX_API_KEY",
        ignore_placeholders=True,
    )


# Default oMLX models per role. The vLLM path uses the single model served by
# Spark for both roles; on Apple Silicon the roles are normally two different
# checkpoints, because a vision model is wasted on text reasoning and a text
# model cannot see frames at all.
RECOMMENDED_OCR_MODEL = "Qwen3.6-27B-bf16"
RECOMMENDED_TEXT_MODEL = "DeepSeek-V4-Flash-0731-MLX"


def resolve_ocr_model(config) -> str:
    """Resolve the OCR (vision) model id from an OCRConfig-like object."""
    if resolve_role_backend(config) == InferenceBackend.vllm:
        return (
            getattr(config, "model", None)
            or _env_value("VLLM_OCR_MODEL", "VLLM_MODEL")
            or DEFAULT_VLLM_MODEL
        )
    return (
        getattr(config, "model", None)
        or _env_value("OCR_MODEL", "MLX_VISION_MODEL", "MLX_OCR_MODEL")
        or RECOMMENDED_OCR_MODEL
    )


def resolve_llm_model(config) -> str:
    """Resolve the text-LLM id from a ReconstructionConfig-like object."""
    if resolve_role_backend(config) == InferenceBackend.vllm:
        return (
            getattr(config, "model", None)
            or _env_value("VLLM_LLM_MODEL", "VLLM_MODEL")
            or DEFAULT_VLLM_MODEL
        )
    return (
        getattr(config, "model", None)
        or _env_value("LLM_MODEL", "MLX_MODEL", "OMLX_MODEL")
        or RECOMMENDED_TEXT_MODEL
    )


def _urlopen(req: request.Request, timeout: float):
    """Open an API request, bypassing inherited proxies for loopback servers."""
    host = (urlsplit(req.full_url).hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        opener = request.build_opener(request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)
    return request.urlopen(req, timeout=timeout)


def list_models(base_url: str, api_key: str | None = None, timeout: float = 30.0) -> list[str]:
    """Return served model ids from an OpenAI-compatible ``/v1/models`` endpoint."""
    base = normalize_api_base_url(base_url)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(f"{base}/models", headers=headers, method="GET")
    try:
        with _urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except (HTTPError, URLError) as exc:  # pragma: no cover - network
        raise RuntimeError(f"Could not list inference models at {base}/models: {exc}") from exc
    items = data.get("data") or data.get("models") or []
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append(it.get("id") or it.get("name") or "")
        else:
            out.append(str(it))
    return [m for m in out if m]


def normalized_model_id(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower())


def is_known_vision_model(model_id: str | None) -> bool:
    """Return True if the id is a known vision/multimodal model."""
    if not model_id:
        return False
    n = normalized_model_id(model_id)
    if any(m in n for m in _KNOWN_VISION_MARKERS) or "ocr" in n:
        return True
    return any(p in n for p in _KNOWN_VISION_PATTERNS)


def is_known_text_only_model(model_id: str | None) -> bool:
    """Return True for served model ids that are known not to accept images."""
    if not model_id:
        return False
    if is_known_vision_model(model_id):
        return False
    normalized = normalized_model_id(model_id)
    return any(pattern in normalized for pattern in _KNOWN_TEXT_ONLY_PATTERNS)


def validate_omlx_vision_model(model_id: str) -> None:
    """Compatibility alias for :func:`validate_vision_model`."""
    validate_vision_model(model_id)


def validate_vision_model(model_id: str) -> None:
    """Raise an actionable error if the selected model is known text-only."""
    if is_known_text_only_model(model_id):
        raise ValueError(
            f"{model_id} is a text-only model. ScreenLens captioning sends "
            "image inputs, so choose a vision-capable model such as a VL, vision, "
            "omni, or Janus model."
        )


def strip_thinking(text: str) -> str:
    """Remove Qwen/DeepSeek-style thinking blocks from final user-visible text.

    Handles three shapes:
      * complete ``<think>…</think>`` blocks,
      * a dangling ``</think>`` (opening tag was a prompt prefix) — keep what
        follows,
      * a dangling ``<think>`` with no close — generation was truncated mid-
        reasoning, so everything after it is thinking with no answer; drop it.
    """
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    elif "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def _image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "output_text"):
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


class OpenAICompatibleClient:
    """Minimal chat client shared by vLLM and oMLX."""

    def __init__(self, config: CaptioningConfig):
        self.config = config
        self.backend = resolve_inference_backend(config)
        self.base_url = resolve_inference_base_url(config)
        self.model = resolve_inference_model(config)
        self.api_key = resolve_inference_api_key(config)
        self.timeout = (
            config.vllm_timeout_seconds
            if self.backend == InferenceBackend.vllm
            else config.omlx_timeout_seconds
        )
        self.context_size = resolve_inference_context(config)
        self._default_max_tokens = config.max_tokens
        self._default_temperature = config.temperature

    @classmethod
    def from_endpoint(
        cls,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        backend: str | InferenceBackend = InferenceBackend.omlx,
        timeout: float = 600.0,
        context_size: int = 32768,
        default_max_tokens: int = 32768,
        default_temperature: float = 0.0,
    ) -> "OpenAICompatibleClient":
        """Build a client directly from endpoint params (no CaptioningConfig).

        Used by the verbatim OCR pass (vision model) and the reconstruction pass
        (text model), which keep their own config objects.
        """
        self = cls.__new__(cls)
        self.config = None
        self.backend = InferenceBackend(getattr(backend, "value", backend))
        self.base_url = normalize_api_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.context_size = context_size
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        return self

    def model_supports_vision(self) -> bool | None:
        """Best-effort: is this client's model vision-capable?

        Returns True/False from the name heuristic, or None if unknown. Used to
        fail loudly before sending images to a text-only model.
        """
        if is_known_text_only_model(self.model):
            return False
        if is_known_vision_model(self.model):
            return True
        return None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
        require_complete: bool = False,
    ) -> str:
        if images:
            validate_vision_model(self.model)

        user_content: str | list[dict[str, Any]]
        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(
                {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
                for path in images
            )
        else:
            user_content = user_prompt

        requested_max_tokens = (
            max_tokens if max_tokens is not None else self._default_max_tokens
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature if temperature is not None else self._default_temperature,
            "stream": False,
        }
        # vLLM's context limit includes the chat template, prompt, image tokens,
        # and completion. Reserving the entire 32K window as max_tokens leaves
        # zero room for input and is rejected. Omitting the field at that ceiling
        # makes vLLM compute max_model_len - actual_input_length, which is the
        # largest valid completion budget for each image. Explicit smaller caps
        # remain exact, and oMLX retains its existing explicit-field behavior.
        if not (
            self.backend == InferenceBackend.vllm
            and requested_max_tokens >= self.context_size
        ):
            payload["max_tokens"] = requested_max_tokens
        # Pass-through sampler controls (repetition_penalty, no_repeat_ngram_size,
        # etc.). Both the bundled vLLM recipe and current oMLX accept these.
        if extra:
            payload.update({k: v for k, v in extra.items() if v is not None})
        if require_complete:
            return strip_thinking(
                self._post_chat(payload, require_complete=True)
            )
        return strip_thinking(self._post_chat(payload))

    def _tokenize_url(self) -> str:
        """Return vLLM's root-level tokenization endpoint URL."""
        parsed = urlsplit(self.base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/tokenize", "", ""))

    def _tokenize_chat(self, payload: dict[str, Any]) -> tuple[int, int] | None:
        """Ask vLLM for the exact rendered chat token count, if supported."""
        tokenize_payload = {
            key: value
            for key, value in payload.items()
            if key in _TOKENIZE_CHAT_FIELDS
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            self._tokenize_url(),
            data=json.dumps(tokenize_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with _urlopen(req, timeout=self.timeout) as resp:
                response = json.load(resp)
            return int(response["count"]), int(response["max_model_len"])
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError):
            return None

    def _context_retry_payload(
        self,
        payload: dict[str, Any],
        status_code: int,
        detail: str,
    ) -> dict[str, Any] | None:
        """Build one exact-budget retry for a structured vLLM context error."""
        if self.backend != InferenceBackend.vllm or status_code != 400:
            return None
        try:
            decoded = json.loads(detail)
        except json.JSONDecodeError:
            return None
        error = decoded.get("error", decoded) if isinstance(decoded, dict) else {}
        if not isinstance(error, dict):
            return None
        message = str(error.get("message", ""))
        if (
            error.get("param") not in _VLLM_CONTEXT_ERROR_PARAMS
            or "maximum context length" not in message.lower()
        ):
            return None

        requested = payload.get("max_completion_tokens", payload.get("max_tokens"))
        if requested is None:
            return None

        tokenized = self._tokenize_chat(payload)
        retry = dict(payload)
        if tokenized is None:
            # Older vLLM builds may not expose /tokenize. Omitting both output
            # fields lets vLLM allocate the exact remaining context itself.
            retry.pop("max_tokens", None)
            retry.pop("max_completion_tokens", None)
            logger.warning(
                "vLLM rejected the explicit completion reservation; /tokenize "
                "is unavailable, retrying once with the remaining context."
            )
            return retry

        prompt_tokens, max_model_len = tokenized
        available = max_model_len - prompt_tokens
        if available <= 0:
            raise RuntimeError(
                f"vllm prompt uses {prompt_tokens:,} tokens, exceeding the "
                f"server's {max_model_len:,}-token context before any response "
                "can be generated. Reduce or split the prompt."
            )

        retry_tokens = min(int(requested), available)
        if retry_tokens >= int(requested):
            return None
        if "max_completion_tokens" in retry:
            retry["max_completion_tokens"] = retry_tokens
            retry.pop("max_tokens", None)
        else:
            retry["max_tokens"] = retry_tokens
        logger.warning(
            "vLLM prompt uses %s of %s context tokens; retrying once with "
            "max_tokens=%s instead of %s.",
            prompt_tokens,
            max_model_len,
            retry_tokens,
            requested,
        )
        return retry

    def _post_chat(
        self,
        payload: dict[str, Any],
        *,
        allow_context_retry: bool = True,
        require_complete: bool = False,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with _urlopen(req, timeout=self.timeout) as resp:
                response = json.load(resp)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            if allow_context_retry:
                retry_payload = self._context_retry_payload(payload, exc.code, detail)
                if retry_payload is not None:
                    return self._post_chat(
                        retry_payload,
                        allow_context_retry=False,
                        require_complete=require_complete,
                    )
            hint = ""
            if exc.code == 401:
                hint = (
                    " Set VLLM_API_KEY for vLLM or MLX_API_KEY/OMLX_API_KEY for oMLX."
                )
            raise RuntimeError(
                f"{self.backend.value} chat completion failed with HTTP "
                f"{exc.code}: {detail}{hint}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"{self.backend.value} chat completion timed out after "
                f"{self.timeout:g} seconds. Increase the configured request "
                "timeout if this local model needs longer."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to {self.backend.value} at {self.base_url}. "
                "Start the local inference service or pass --inference-url."
            ) from exc

        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"{self.backend.value} response contained no choices: {response}"
            )

        first = choices[0]
        if isinstance(first, dict) and first.get("finish_reason") == "length":
            completion_limit = payload.get(
                "max_completion_tokens",
                payload.get("max_tokens"),
            )
            if completion_limit is None:
                completion_limit = (
                    f"remaining space in the {self.context_size}-token context"
                )
            if require_complete:
                raise InferenceTruncatedError(
                    self.backend.value,
                    completion_limit,
                )
            logger.warning(
                "%s truncated the response at %s (finish_reason=length); "
                "the returned text may be incomplete. Reduce the prompt or "
                "increase the completion budget within the model context.",
                self.backend.value,
                completion_limit,
            )
        if isinstance(first, dict):
            if "message" in first and isinstance(first["message"], dict):
                return _message_text(first["message"].get("content"))
            if "text" in first:
                return str(first["text"])
        return str(first)


# Compatibility names for callers and serialized workflows from the oMLX-only
# releases. New code should prefer ``InferenceClient``.
InferenceClient = OpenAICompatibleClient
OMLXClient = OpenAICompatibleClient
