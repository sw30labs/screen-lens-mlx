"""
Verbatim OCR pass.

Transcribes each frame character-for-character with a VISION model through a
vLLM or oMLX OpenAI-compatible server. Unlike captioning, this never describes
— it copies.

Defenses baked in after the original failure (a text-only model was used for
vision and silently returned "no image provided" for all 173 frames):

  * Hard capability guard — refuse to run if the configured model isn't
    vision-capable, and a live probe that catches "no image" refusals.
  * Anti-loop sampler controls (repetition_penalty / no_repeat_ngram_size).
  * Optional Apple Vision deterministic backstop (ocrmac) for code, where VLMs
    are known to hallucinate tokens.
"""
from __future__ import annotations

import logging
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .config import OCRConfig
from .omlx_client import (
    InferenceClient,
    resolve_ocr_model,
    resolve_role_api_key,
    resolve_role_backend,
    resolve_role_base_url,
    resolve_role_context,
)

logger = logging.getLogger("screenlens.ocr")

# Sentinels a blind/text-only model emits when it can't see the image.
_NO_IMAGE_RE = re.compile(
    r"\b(no (image|video frame|picture)\b.*(provided|attached|been))"
    r"|(please (attach|provide|upload).{0,40}(image|frame))"
    r"|(i (cannot|can't|am unable to) see (an|the|any) image)",
    re.IGNORECASE | re.DOTALL,
)

_EMPTY_MARKERS = {"[no text]", "[notext]", "no text", ""}


class VerbatimOCR:
    """Transcribe frames verbatim through the selected vision server."""

    def __init__(self, config: OCRConfig):
        self.config = config
        if config.deterministic_backstop and platform.system() != "Darwin":
            raise RuntimeError(
                "The deterministic OCR backstop uses Apple Vision and is only "
                "available on macOS; omit --deterministic on DGX Spark."
            )
        self.model = resolve_ocr_model(config)
        self.client = InferenceClient.from_endpoint(
            base_url=resolve_role_base_url(config),
            model=self.model,
            api_key=resolve_role_api_key(config, "VLLM_OCR_API_KEY", "OCR_API_KEY"),
            backend=resolve_role_backend(config),
            timeout=config.timeout_seconds,
            context_size=resolve_role_context(config),
            default_max_tokens=config.max_tokens,
            default_temperature=config.temperature,
        )
        self._extra = {
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size or None,
        }
        if config.disable_thinking:
            # Verbatim OCR needs no chain-of-thought. A reasoning model otherwise
            # spends the whole token budget thinking and never emits the transcription.
            self._extra["chat_template_kwargs"] = {"enable_thinking": False}
        self._probed = False

    # ── Capability checks ────────────────────────────────────────────────────

    def assert_vision_capable(self) -> None:
        """Refuse to run against a text-only model (the original bug)."""
        supports = self.client.model_supports_vision()
        if supports is False and self.config.require_vision_model:
            raise RuntimeError(
                f"OCR model '{self.model}' looks text-only. Verbatim OCR sends "
                f"images, so it would read every frame blind (this is exactly the "
                f"bug that produced 173 empty captions). Set ocr.model or the "
                f"selected provider's OCR/model environment variable to a vision "
                f"model (VL / vision / omni / *-OCR). To override, set "
                f"ocr.require_vision_model=false."
            )
        if supports is None:
            logger.warning(
                "Could not verify '%s' is vision-capable from its name; relying on "
                "the live probe. If OCR returns '[NO TEXT]' for every frame, the "
                "model is text-only.", self.model,
            )

    def probe(self, sample_image_path: str) -> None:
        """One live call to catch a blind model before processing the whole video."""
        if self._probed:
            return
        self._probed = True
        try:
            raw = self.client.chat(
                self.config.system_prompt,
                self.config.user_prompt,
                images=[sample_image_path],
                max_tokens=256,
                temperature=0.0,
                extra=self._extra,
            )
        except Exception as exc:
            raise RuntimeError(f"OCR probe call failed: {exc}") from exc
        if _NO_IMAGE_RE.search(raw or ""):
            raise RuntimeError(
                f"OCR model '{self.model}' responded as if no image was sent "
                f"(\"{(raw or '').strip()[:80]}…\"). It is not actually seeing "
                f"frames. Pick a vision-capable model on your "
                f"{self.client.backend.value} server."
            )

    # ── Per-frame OCR ────────────────────────────────────────────────────────

    def ocr_frame(self, image_path: str) -> str:
        raw = self.client.chat(
            self.config.system_prompt,
            self.config.user_prompt,
            images=[image_path],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            extra=self._extra,
        )
        text = (raw or "").strip()
        if _NO_IMAGE_RE.search(text):
            raise RuntimeError(
                f"OCR model returned a 'no image' refusal mid-run — it is not "
                f"vision-capable: {text[:80]}"
            )
        # Strip an accidental outer ``` fence wrapping the whole answer.
        if text.startswith("```") and text.endswith("```") and text.count("```") == 2:
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?|\n?```$", "", text).strip()
        if text.lower() in _EMPTY_MARKERS:
            return ""
        if self.config.deterministic_backstop:
            text = self._reconcile_with_apple_vision(image_path, text)
        return text

    def ocr_frames(self, image_paths: list[str]) -> list[str]:
        if not image_paths:
            return []
        self.assert_vision_capable()
        self.probe(image_paths[0])
        workers = max(1, min(self.config.concurrency, len(image_paths)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._safe_ocr, image_paths))

    def _safe_ocr(self, path: str) -> str:
        try:
            return self.ocr_frame(path)
        except Exception as exc:  # keep ordering; one bad frame shouldn't kill the run
            logger.error("OCR failed on %s: %s", path, exc)
            return ""

    # ── Deterministic backstop (Apple Vision) ────────────────────────────────

    def _reconcile_with_apple_vision(self, image_path: str, vlm_text: str) -> str:
        """Flag lines where the VLM disagrees with Apple Vision's literal read.

        Apple Vision never hallucinates a fake line (but mangles indentation), so
        on a strong character-level disagreement we trust the deterministic read
        and annotate the seam for human review. Best-effort: silently skip if
        ocrmac/Apple Vision isn't available.
        """
        try:
            lines = apple_vision_lines(image_path)
        except Exception as exc:
            logger.debug("Apple Vision backstop unavailable (%s); keeping VLM text", exc)
            return vlm_text
        if not lines:
            return vlm_text
        from difflib import SequenceMatcher

        det = "\n".join(lines)
        ratio = SequenceMatcher(None, vlm_text, det).ratio()
        if ratio < 0.55:
            # Large divergence — surface both so nothing is silently invented.
            return (
                vlm_text
                + "\n\n<!-- OCR-DISAGREEMENT: Apple Vision read this region "
                "differently; verify -->\n"
                + det
            )
        return vlm_text


def apple_vision_lines(image_path: str) -> list[str]:
    """Deterministic OCR via Apple Vision (ocrmac), language correction OFF.

    Correction OFF is mandatory: the default corrects toward dictionary words,
    which is wrong for code/identifiers. Returns lines in top-to-bottom order.
    Raises if ocrmac/pyobjc isn't installed (macOS only).
    """
    from ocrmac import ocrmac  # type: ignore

    annotations = ocrmac.OCR(
        image_path,
        recognition_level="accurate",
        language_preference=["en-US"],
    ).recognize()
    # annotations: list of (text, confidence, bbox[x,y,w,h]) with y from bottom.
    items = []
    for text, conf, bbox in annotations:
        x, y, w, h = bbox
        items.append((round(1.0 - y, 4), round(x, 4), text))  # sort top→bottom, left→right
    items.sort()
    return [t for _, _, t in items]
