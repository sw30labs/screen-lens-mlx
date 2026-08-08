"""Shared run/session helpers for the CLI, TUI, and web command deck.

Everything here is UI-agnostic: config loading, per-video slug allocation,
role-aware inference wiring, and read-only discovery of what already lives
under ``./data/``. The three front ends (``cli.py``, ``tui.py``,
``web/runner.py``) all build on these so a run started from the browser lands
in exactly the same layout as one started from the terminal.

Model roles
-----------
ScreenLens uses two distinct roles against the same OpenAI-compatible server:

* **vision** — captioning and verbatim OCR. MUST be vision-capable.
* **text**   — summarize, reconstruction plan/QA, transcript cleanup.

On DGX Spark both resolve to the single served vLLM checkpoint. On Apple
Silicon they are normally two different oMLX models (e.g. a Qwen3.6 VLM for
vision and DeepSeek for text), which is why the roles are resolved separately
rather than sharing one model id.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import (
    CaptioningConfig,
    CaptionBackend,
    InferenceBackend,
    ScreenLensConfig,
)
from .omlx_client import (
    is_known_text_only_model,
    is_known_vision_model,
    list_models,
    normalize_api_base_url,
    resolve_inference_api_key,
    resolve_inference_base_url,
    resolve_inference_model,
    resolve_llm_model,
    resolve_ocr_model,
    resolve_role_api_key,
    resolve_role_base_url,
    resolve_role_context,
)

__all__ = [
    "load_config",
    "apply_video_slug",
    "point_config_at_data_dir",
    "apply_direct_inference",
    "text_role_captioning_config",
    "model_roles",
    "endpoint_status",
    "discover_runs",
    "run_snapshot",
    "read_artifact",
    "base_slug",
    "VIDEO_SUFFIXES",
]

VIDEO_SUFFIXES = (".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v")

_SLUG_TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")

# Artifacts the reconstruct/transcribe pipelines drop into ``output/``.
_OUTPUT_PREVIEW_SUFFIXES = (".md", ".py", ".txt", ".json", ".html", ".csv")


# ── config plumbing ─────────────────────────────────────────────────────────

def load_config(config_path: str | Path | None = None) -> ScreenLensConfig:
    """Load a JSON config if it exists, otherwise return defaults."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return ScreenLensConfig(**json.load(f))
    return ScreenLensConfig()


def base_slug(name: str) -> str:
    """Strip the ``_YYYYMMDD_HHMMSS`` suffix from a run folder name."""
    return _SLUG_TIMESTAMP_RE.sub("", name)


def apply_video_slug(config: ScreenLensConfig, video: Path) -> str:
    """Point ``config`` at a fresh per-video subfolder and return its slug.

    Uses ``<video_stem>_<YYYYMMDD_HHMMSS>`` under the config's ``data_dir`` so
    repeated runs of the same video never clobber each other.
    """
    stem = video.stem.replace(" ", "_")
    slug = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.data_dir = config.data_dir / slug
    config.vector_db.persist_directory = str(config.data_dir / "chromadb")
    config.vector_db.collection_name = f"screenlens_{stem}"
    return slug


def point_config_at_data_dir(config: ScreenLensConfig, data_dir: Path | str) -> None:
    """Make ``data_dir`` and the vector DB path agree for read-side commands."""
    path = Path(data_dir)
    config.data_dir = path
    config.vector_db.persist_directory = str(path / "chromadb")


def apply_direct_inference(
    config: ScreenLensConfig,
    *,
    backend: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    vision_model: Optional[str] = None,
    text_model: Optional[str] = None,
    batch_size: Optional[int] = None,
    caption_max_tokens: Optional[int] = None,
) -> None:
    """Wire one OpenAI-compatible endpoint into every role of ``config``.

    ``vision_model`` drives captioning + OCR; ``text_model`` drives
    reconstruction/cleanup/summary. Either may be ``None`` to keep the
    environment-resolved default for that role.

    ``caption_max_tokens`` bounds how long one caption may run. It matters more
    than it looks: left at the full context, a dense frame keeps the model
    generating long past the real content and into a degenerate repetition
    loop, which costs tens of minutes per frame.
    """
    caption_backend = CaptionBackend(backend)
    config.captioning.backend = caption_backend
    if batch_size is not None:
        config.captioning.batch_size = batch_size
    if caption_max_tokens is not None:
        config.captioning.max_tokens = caption_max_tokens

    if caption_backend == CaptionBackend.ollama:
        # Captions come from Ollama; the direct roles keep their own endpoint.
        if vision_model is not None:
            config.captioning.ollama_model = vision_model
        if base_url is not None:
            config.captioning.ollama_base_url = base_url
        return

    direct = InferenceBackend(caption_backend.value)
    config.ocr.backend = direct
    config.reconstruction.backend = direct

    if base_url is not None:
        normalized = normalize_api_base_url(base_url)
        config.ocr.base_url = normalized
        config.reconstruction.base_url = normalized
        if direct == InferenceBackend.vllm:
            config.captioning.vllm_base_url = normalized
        else:
            config.captioning.omlx_base_url = normalized

    if api_key is not None:
        config.ocr.api_key = api_key
        config.reconstruction.api_key = api_key
        if direct == InferenceBackend.vllm:
            config.captioning.vllm_api_key = api_key
        else:
            config.captioning.omlx_api_key = api_key

    if vision_model is not None:
        config.ocr.model = vision_model
        if direct == InferenceBackend.vllm:
            config.captioning.vllm_model = vision_model
        else:
            config.captioning.omlx_model = vision_model

    if text_model is not None:
        config.reconstruction.model = text_model


def text_role_captioning_config(config: ScreenLensConfig) -> CaptioningConfig:
    """Return a ``CaptioningConfig`` shim bound to the TEXT role.

    ``InferenceClient`` is constructed from a ``CaptioningConfig``, so text-only
    work — search summaries, full-video summaries, reconstruction planning/QA,
    transcript cleanup — borrows that shape but fills it from
    ``config.reconstruction``. Without this the vision model would be asked to
    do the reasoning, which is both slower and (on Apple Silicon, where the two
    roles are different checkpoints) simply the wrong model.

    On DGX Spark both roles resolve to the single served vLLM checkpoint, so
    this is a no-op there.
    """
    reconstruction = config.reconstruction
    shim = config.captioning.model_copy(deep=True)
    shim.backend = CaptionBackend(reconstruction.backend.value)
    shim.max_tokens = reconstruction.max_tokens
    if shim.backend == CaptionBackend.vllm:
        shim.vllm_base_url = resolve_role_base_url(reconstruction)
        shim.vllm_model = resolve_llm_model(reconstruction)
        shim.vllm_api_key = resolve_role_api_key(
            reconstruction, "VLLM_LLM_API_KEY", "LLM_API_KEY"
        )
        shim.vllm_timeout_seconds = reconstruction.timeout_seconds
        shim.vllm_model_context = resolve_role_context(reconstruction)
    else:
        shim.omlx_base_url = resolve_role_base_url(reconstruction)
        shim.omlx_model = resolve_llm_model(reconstruction)
        shim.omlx_api_key = resolve_role_api_key(reconstruction, "LLM_API_KEY")
        shim.omlx_timeout_seconds = reconstruction.timeout_seconds
        shim.omlx_model_context = resolve_role_context(reconstruction)
    return shim


def model_roles(config: ScreenLensConfig) -> dict[str, Any]:
    """Describe the resolved vision and text roles, with capability flags."""
    captioning = config.captioning
    if captioning.backend == CaptionBackend.ollama:
        caption_model = captioning.ollama_model
        caption_endpoint = captioning.ollama_base_url
        caption_provider = "ollama"
    else:
        caption_model = resolve_inference_model(captioning)
        caption_endpoint = resolve_inference_base_url(captioning)
        caption_provider = captioning.backend.value

    ocr_model = resolve_ocr_model(config.ocr)
    text_model = resolve_llm_model(config.reconstruction)

    return {
        "caption": {
            "role": "vision",
            "provider": caption_provider,
            "model": caption_model,
            "base_url": caption_endpoint,
            "vision_ok": _vision_ok(caption_model),
        },
        "ocr": {
            "role": "vision",
            "provider": config.ocr.backend.value,
            "model": ocr_model,
            "base_url": resolve_role_base_url(config.ocr),
            "vision_ok": _vision_ok(ocr_model),
        },
        "text": {
            "role": "text",
            "provider": config.reconstruction.backend.value,
            "model": text_model,
            "base_url": resolve_role_base_url(config.reconstruction),
            "vision_ok": None,  # irrelevant for the text role
        },
    }


def _vision_ok(model_id: str | None) -> bool | None:
    """True/False when the model id is conclusive, None when unknown."""
    if not model_id:
        return None
    if is_known_vision_model(model_id):
        return True
    if is_known_text_only_model(model_id):
        return False
    return None


def endpoint_status(
    config: ScreenLensConfig,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Probe the configured direct endpoint and list its served models."""
    captioning = config.captioning
    if captioning.backend == CaptionBackend.ollama:
        base = captioning.ollama_base_url
        api_key = None
    else:
        base = resolve_inference_base_url(captioning)
        api_key = resolve_inference_api_key(captioning)
        if api_key is None:
            api_key = resolve_role_api_key(config.reconstruction, "LLM_API_KEY")

    try:
        models = list_models(base, api_key, timeout=timeout)
    except Exception as exc:  # network/auth failures are reported, not raised
        return {
            "reachable": False,
            "base_url": base,
            "provider": captioning.backend.value,
            "models": [],
            "detail": str(exc),
        }
    return {
        "reachable": True,
        "base_url": base,
        "provider": captioning.backend.value,
        "models": models,
        "vision_models": [m for m in models if is_known_vision_model(m)],
        "text_models": [m for m in models if not is_known_vision_model(m)],
        "detail": None,
    }


# ── run discovery (read-only) ───────────────────────────────────────────────

def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.glob(pattern) if p.is_file())


def _folder_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def discover_runs(data_dir: Path | str = "./data") -> list[dict[str, Any]]:
    """List run folders under ``data_dir``, newest first.

    A folder counts as a run when it holds any of the pipeline's own
    subdirectories, so partially-completed runs still show up.
    """
    root = Path(data_dir)
    if not root.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        markers = ("frames", "captions", "ocr", "chromadb", "output")
        if not any((folder / m).is_dir() for m in markers):
            continue
        runs.append(_run_summary(folder))
    runs.sort(key=lambda r: r["modified"], reverse=True)
    return runs


def _has_vector_store(folder: Path) -> bool:
    """True only once ChromaDB has actually written a store.

    ``ensure_dirs()`` creates an empty ``chromadb/`` at the start of every run,
    so directory existence would report a vector DB before a single embedding
    exists — and would label a still-captioning run "embedded".
    """
    return (folder / "chromadb" / "chroma.sqlite3").is_file()


def _run_summary(folder: Path) -> dict[str, Any]:
    frames = _count_files(folder / "frames")
    captions = _count_files(folder / "captions", "caption_*.json")
    ocr = _count_files(folder / "ocr", "ocr_*.json")
    outputs = sorted(
        p.name for p in (folder / "output").glob("*") if p.is_file()
    ) if (folder / "output").is_dir() else []
    embedded = _has_vector_store(folder)

    if captions and frames:
        stage = "embedded" if embedded else "captioned"
    elif ocr:
        stage = "transcribed"
    elif frames:
        stage = "frames"
    else:
        stage = "empty"

    return {
        "slug": folder.name,
        "base": base_slug(folder.name),
        "path": str(folder),
        "frames": frames,
        "captions": captions,
        "ocr": ocr,
        "outputs": outputs,
        "has_chromadb": embedded,
        "collection": f"screenlens_{base_slug(folder.name)}",
        "stage": stage,
        "modified": _folder_mtime(folder),
    }


def run_snapshot(
    folder: Path | str,
    *,
    frame_limit: int = 400,
) -> dict[str, Any] | None:
    """Return a detailed, JSON-safe snapshot of one run folder."""
    path = Path(folder)
    if not path.is_dir():
        return None

    snap = _run_summary(path)
    snap["frames_list"] = _frame_entries(path, limit=frame_limit)
    snap["captions_preview"] = _caption_previews(path, limit=frame_limit)
    snap["transcript"] = _transcript_meta(path)
    snap["reconstruction"] = _reconstruction_meta(path)
    return snap


def _frame_entries(folder: Path, *, limit: int) -> list[dict[str, Any]]:
    frames_dir = folder / "frames"
    if not frames_dir.is_dir():
        return []
    files = sorted(p for p in frames_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    return [{"name": p.name, "size": p.stat().st_size} for p in files[:limit]]


def _caption_previews(folder: Path, *, limit: int) -> list[dict[str, Any]]:
    combined = folder / "captions" / "all_captions.json"
    records: list[dict[str, Any]] = []
    if combined.exists():
        try:
            records = json.loads(combined.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            records = []
    else:
        caps = sorted((folder / "captions").glob("caption_*.json")) if (folder / "captions").is_dir() else []
        for cap in caps[:limit]:
            try:
                records.append(json.loads(cap.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue

    previews = []
    for rec in records[:limit]:
        if not isinstance(rec, dict):
            continue
        caption = str(rec.get("caption") or "")
        previews.append({
            "frame_id": rec.get("frame_id"),
            "timestamp_str": rec.get("timestamp_str"),
            "frame": Path(str(rec.get("path") or "")).name,
            "chars": len(caption),
            "preview": caption[:400],
        })
    return previews


def _transcript_meta(folder: Path) -> dict[str, Any] | None:
    out = folder / "output"
    meta_path = out / "transcribe_meta.json"
    transcript = out / "transcript.md"
    raw = out / "transcript.raw.md"
    if not meta_path.exists() and not transcript.exists():
        return None

    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    for name, p in (("transcript_chars", transcript), ("raw_chars", raw)):
        meta[name] = p.stat().st_size if p.exists() else 0
    meta["has_transcript"] = transcript.exists()
    meta["has_raw"] = raw.exists()
    return meta


def _reconstruction_meta(folder: Path) -> dict[str, Any] | None:
    meta_path = folder / "output" / "reconstruction_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_artifact(
    folder: Path | str,
    name: str,
    *,
    max_bytes: int = 400_000,
) -> dict[str, Any] | None:
    """Read one file out of a run's ``output/`` directory, safely.

    ``name`` is treated as a plain file name — any path separators or parent
    references are rejected rather than resolved, so this cannot escape the
    run folder.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    out_dir = (Path(folder) / "output").resolve()
    target = (out_dir / name).resolve()
    if target.parent != out_dir or not target.is_file():
        return None
    if target.suffix.lower() not in _OUTPUT_PREVIEW_SUFFIXES:
        return None

    data = target.read_bytes()[:max_bytes]
    return {
        "name": name,
        "size": target.stat().st_size,
        "truncated": target.stat().st_size > max_bytes,
        "text": data.decode("utf-8", errors="replace"),
    }
