"""Shared run/session helpers for the CLI and the web command deck.

Everything here is UI-agnostic: config loading, per-video slug allocation,
role-aware inference wiring, and read-only discovery of what already lives
under ``./data/``. Both front ends (``cli.py``, ``web/runner.py``) build on
these so a run started from the browser lands in exactly the same layout as
one started from the terminal.

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
    "safe_video_stem",
    "chroma_collection_name",
    "point_config_at_data_dir",
    "find_reusable_run",
    "reuse_video_run",
    "extraction_meta_matches",
    "load_cached_frames",
    "transcribe_run_matches",
    "apply_direct_inference",
    "text_role_captioning_config",
    "model_roles",
    "form_defaults",
    "endpoint_status",
    "discover_runs",
    "run_snapshot",
    "read_artifact",
    "base_slug",
    "VIDEO_SUFFIXES",
    "DEFAULT_INPUT_FOLDER",
    "resolve_video_queue",
    "list_videos_chronological",
    "video_recorded_at",
    "recording_stamp",
    "find_successful_run",
    "ingest_succeeded",
    "transcribe_succeeded",
    "run_belongs_to_video",
    "read_run_identity",
    "write_run_identity",
    "content_excerpt",
    "infer_content_slug",
    "text_slug_generate",
    "apply_content_name",
    "slugify_title",
]

VIDEO_SUFFIXES = (".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v")
DEFAULT_INPUT_FOLDER = Path("./input")

_SLUG_TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")
# macOS Screenshot / Screen Recording names, including a narrow no-break space
# before AM/PM: "Screen Recording 2026-08-17 at 9.29.20 AM.mov"
_SCREEN_REC_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})\s+at\s+(\d{1,2})[.:](\d{2})[.:](\d{2})\s*([APap][Mm])?",
)
# Compact rename: 20260817_092920.mov
_COMPACT_STAMP_RE = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?!\d)")
_UNSAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
# Chroma collection names: 3-512 chars of [a-zA-Z0-9._-], start/end alnum.
_CHROMA_NAME_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]{0,510}[a-zA-Z0-9])?$")
_GENERIC_SLUGS = {
    "recording", "screen_recording", "screenshot", "untitled",
    "video", "image", "clip", "text", "ui_elements", "tables",
}
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_FILE_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_.-]*\.(?:py|md|ts|tsx|js|jsx|html|pdf|json|txt|csv))"
)

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


def safe_video_stem(video: Path | str) -> str:
    """Filesystem- and Chroma-safe stem from a video path.

    macOS Screen Recording names carry spaces and a U+202F thin space before
    AM/PM. Chroma rejects those in collection names, so they collapse to the
    recording clock (``YYYYMMDD_HHMMSS``). Anything else is stripped to
    ``[A-Za-z0-9._-]``.
    """
    path = Path(video)
    if _SCREEN_REC_RE.search(path.name):
        return recording_stamp(path)
    cleaned = _UNSAFE_STEM_RE.sub("_", path.stem)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "video"


def chroma_collection_name(stem: str) -> str:
    """Build a Chroma-legal ``screenlens_<stem>`` collection name."""
    cleaned = _UNSAFE_STEM_RE.sub("_", stem or "")
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-") or "video"
    name = f"screenlens_{cleaned}"
    return name[:512]


def _valid_chroma_name(name: str) -> bool:
    return 3 <= len(name) <= 512 and bool(_CHROMA_NAME_RE.fullmatch(name))


def apply_video_slug(config: ScreenLensConfig, video: Path) -> str:
    """Point ``config`` at a fresh per-video subfolder and return its slug.

    Uses ``<video_stem>_<YYYYMMDD_HHMMSS>`` under the config's ``data_dir`` so
    repeated runs of the same video never clobber each other. After a
    successful pass the folder may be renamed to a content slug; identity is
    stored in ``output/run.json`` so skip/reuse still find it.
    """
    stem = safe_video_stem(video)
    slug = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.data_dir = config.data_dir / slug
    config.vector_db.persist_directory = str(config.data_dir / "chromadb")
    config.vector_db.collection_name = chroma_collection_name(stem)
    write_run_identity(config.data_dir, _video_identity(video, config))
    return slug


def point_config_at_data_dir(config: ScreenLensConfig, data_dir: Path | str) -> None:
    """Make ``data_dir`` and the vector DB path agree for read-side commands."""
    path = Path(data_dir)
    config.data_dir = path
    config.vector_db.persist_directory = str(path / "chromadb")


# ── video queue (empty picker = every file in ./input, oldest first) ─────────


def video_recorded_at(path: Path) -> datetime:
    """Best-effort recording time: filename clock, else file mtime."""
    match = _SCREEN_REC_RE.search(path.name)
    if match:
        year, month, day, hour, minute, second, ampm = match.groups()
        hour_i = int(hour)
        if ampm:
            mer = ampm.lower()
            if mer == "pm" and hour_i != 12:
                hour_i += 12
            elif mer == "am" and hour_i == 12:
                hour_i = 0
        try:
            return datetime(int(year), int(month), int(day), hour_i, int(minute), int(second))
        except ValueError:
            pass
    compact = _COMPACT_STAMP_RE.search(path.stem)
    if compact:
        year, month, day, hour, minute, second = compact.groups()
        try:
            return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.min


def recording_stamp(path: Path) -> str:
    """``YYYYMMDD_HHMMSS`` from the recording clock (filename or mtime)."""
    return video_recorded_at(path).strftime("%Y%m%d_%H%M%S")


def list_videos_chronological(folder: Path | str) -> list[Path]:
    """Video files in ``folder``, oldest recording first."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    videos = [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    ]
    videos.sort(key=lambda p: (video_recorded_at(p), p.name.lower()))
    return videos


def resolve_video_queue(
    source: str | Path | None,
    *,
    default_folder: str | Path = DEFAULT_INPUT_FOLDER,
) -> list[Path]:
    """One file, every video in a folder, or every video in ``default_folder``.

    ``None`` / empty string means "no file picked" — process ``default_folder``
    in chronological order.
    """
    if source is None or str(source).strip() == "":
        folder = Path(default_folder).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"input folder not found: {folder}")
        return list_videos_chronological(folder)
    path = Path(str(source)).expanduser()
    if path.is_dir():
        return list_videos_chronological(path)
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return [path.resolve()]
    if path.is_file():
        raise FileNotFoundError(f"not a supported video: {path}")
    raise FileNotFoundError(f"video not found: {path}")


# ── run identity (survives content-based folder rename) ──────────────────────


def _video_stat(video: Path) -> tuple[int | None, float | None]:
    try:
        stat = video.stat()
        return stat.st_size, stat.st_mtime
    except OSError:
        return None, None


def _video_identity(video: Path, config: ScreenLensConfig) -> dict[str, Any]:
    size, mtime = _video_stat(video)
    return {
        "video": str(video.resolve()),
        "video_size": size,
        "video_mtime": mtime,
        "source_name": video.name,
        "collection": config.vector_db.collection_name,
        "status": "running",
    }


def write_run_identity(run_dir: Path, data: dict[str, Any]) -> None:
    """Merge ``data`` into ``output/run.json``."""
    folder = Path(run_dir)
    (folder / "output").mkdir(parents=True, exist_ok=True)
    path = folder / "output" / "run.json"
    current = read_run_identity(folder) or {}
    current.update(data)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def read_run_identity(run_dir: Path) -> Optional[dict[str, Any]]:
    path = Path(run_dir) / "output" / "run.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _identity_from_run(run_dir: Path) -> dict[str, Any]:
    identity = read_run_identity(run_dir)
    if identity:
        return identity
    for rel in ("frames/frames_meta.json", "output/transcribe_meta.json"):
        path = Path(run_dir) / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and (data.get("video") or data.get("source_name")):
            return data
    return {}


def run_belongs_to_video(run_dir: Path, video: Path) -> bool:
    """True when this run folder was produced from ``video``.

    Prefers resolved path + size. Falls back to original filename + size so a
    content-renamed folder still matches after the operator moves the file
    inside ``./input``.
    """
    meta = _identity_from_run(run_dir)
    if not meta:
        return False
    size, _ = _video_stat(video)
    resolved = str(video.resolve())
    meta_size = meta.get("video_size")
    if meta.get("video") == resolved:
        if meta_size is None or size is None:
            return True
        try:
            return int(meta_size) == size
        except (TypeError, ValueError):
            return False
    source = meta.get("source_name") or Path(str(meta.get("video") or "")).name
    if source and source == video.name and meta_size is not None and size is not None:
        try:
            return int(meta_size) == size
        except (TypeError, ValueError):
            return False
    return False


def ingest_succeeded(run_dir: Path) -> bool:
    """Captions written and the vector store actually materialized.

    An error status is not success even when ``chroma.sqlite3`` exists —
    Chroma opens the sqlite file before it validates the collection name,
    so a rejected name still leaves an empty store on disk.
    """
    folder = Path(run_dir)
    if not (folder / "captions" / "all_captions.json").is_file():
        return False
    if not (folder / "chromadb" / "chroma.sqlite3").is_file():
        return False
    identity = read_run_identity(folder) or {}
    return identity.get("status") != "error"


def transcribe_succeeded(run_dir: Path) -> bool:
    """``transcribe_meta.json`` is only written after a complete pass."""
    return (Path(run_dir) / "output" / "transcribe_meta.json").is_file()


def find_runs_for_video(data_dir: Path | str, video: Path) -> list[Path]:
    """Run folders that belong to ``video``, newest name last."""
    base = Path(data_dir)
    if not base.is_dir():
        return []
    found = [
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".") and run_belongs_to_video(d, video)
    ]
    found.sort(key=lambda d: d.name)
    return found


def find_successful_run(
    data_dir: Path | str, video: Path, kind: str
) -> Optional[Path]:
    """Newest fully-successful run of ``video`` for ingest or transcribe."""
    check = ingest_succeeded if kind == "ingest" else transcribe_succeeded
    matches = [d for d in find_runs_for_video(data_dir, video) if check(d)]
    return matches[-1] if matches else None


# ── run reuse (write-side) ────────────────────────────────────────────────────
# Re-running a pipeline on the same video used to mint a fresh timestamped
# folder and re-pay the entire model cost. These helpers let the CLI and the
# web runner pick the newest prior run of the SAME video instead, so each
# stage can skip work its artifacts already cover. `--fresh` opts out.
# After a content rename the stem glob no longer matches; identity in
# run.json / frames_meta / transcribe_meta is the source of truth.


def find_reusable_run(
    config: ScreenLensConfig, video: Path, required: str
) -> Optional[Path]:
    """Newest run folder for ``video`` that already has ``required``.

    ``required`` is a path relative to the run folder (e.g. ``"ocr"`` or
    ``"frames/frames_meta.json"``). Identity match wins; the historical
    ``<stem>_*`` glob remains as a fallback for folders that predate
    ``output/run.json``.
    """
    base = Path(config.data_dir)
    if not base.is_dir():
        return None
    identified = [
        d for d in find_runs_for_video(base, video) if (d / required).exists()
    ]
    if identified:
        return identified[-1]
    stem = safe_video_stem(video)
    candidates = sorted(
        (d for d in base.glob(f"{stem}_*") if d.is_dir() and (d / required).exists()),
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def reuse_video_run(config: ScreenLensConfig, video: Path, run_dir: Path) -> str:
    """Point ``config`` at an existing run folder and return its slug.

    Keeps the collection name already stored on the run when present, so a
    content-renamed folder still searches the embeddings written into it.
    """
    run_dir = Path(run_dir)
    identity = read_run_identity(run_dir) or {}
    stem = safe_video_stem(video)
    config.data_dir = run_dir
    config.vector_db.persist_directory = str(run_dir / "chromadb")
    stored = str(identity.get("collection") or "")
    config.vector_db.collection_name = (
        stored if _valid_chroma_name(stored) else chroma_collection_name(stem)
    )
    return run_dir.name


def extraction_meta_matches(
    run_dir: Path, video: Path, config: ScreenLensConfig
) -> bool:
    """True when the run's stored extraction metadata matches this video+config.

    Without this guard a change of strategy/fps — or a replaced video file —
    would silently pair new frames with stale captions.
    """
    try:
        meta = json.loads(
            (Path(run_dir) / "frames" / "frames_meta.json").read_text(encoding="utf-8")
        )
    except Exception:
        return False
    if meta.get("video") != str(video.resolve()):
        return False
    try:
        size = meta.get("video_size")
        if size is not None and int(size) != video.stat().st_size:
            return False
    except (OSError, TypeError, ValueError):
        return False
    return meta.get("extraction") == config.frame_extraction.model_dump(mode="json")


def load_cached_frames(
    run_dir: Path, video: Path, config: ScreenLensConfig
) -> Optional[list[dict]]:
    """Return a prior run's extracted-frame metadata when it is safe to reuse."""
    if not extraction_meta_matches(run_dir, video, config):
        return None
    try:
        meta = json.loads(
            (Path(run_dir) / "frames" / "frames_meta.json").read_text(encoding="utf-8")
        )
        frames = meta["frames"]
        assert isinstance(frames, list)
    except Exception:
        return None
    if not frames or not all(Path(f.get("path", "")).is_file() for f in frames):
        return None
    return frames


def transcribe_run_matches(run_dir: Path, video: Path) -> bool:
    """True when a transcribe run folder belongs to this video.

    The meta file only exists after a fully successful run, so an OCR cache
    without it is still accepted — the stem match plus deterministic frame
    names keep the pairing safe; pass ``--fresh`` after replacing a video.
    """
    meta_path = Path(run_dir) / "output" / "transcribe_meta.json"
    if not meta_path.exists():
        return True
    return run_belongs_to_video(run_dir, video)


def slugify_title(text: str) -> str:
    """Filesystem-safe slug: lowercase, digits, underscores, max 60 chars."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:60]


def content_excerpt(run_dir: Path, *, limit: int = 4000) -> str:
    """A short text sample from captions or the stitched transcript."""
    folder = Path(run_dir)
    captions = folder / "captions" / "all_captions.json"
    if captions.is_file():
        try:
            records = json.loads(captions.read_text(encoding="utf-8"))
        except Exception:
            records = []
        parts = []
        for record in records[:8]:
            text = str((record or {}).get("caption") or "").strip()
            if text:
                parts.append(text)
        blob = "\n\n".join(parts)
        if blob:
            return blob[:limit]
    for name in ("transcript.md", "transcript.raw.md"):
        path = folder / "output" / name
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")[:limit]
            except OSError:
                continue
    return ""


def heuristic_content_slug(text: str) -> str:
    """Guess a slug from headings, filenames, or the first real line."""
    if not text or not text.strip():
        return "recording"
    heading = _HEADING_RE.search(text)
    if heading:
        slug = slugify_title(heading.group(1))
        if slug and slug not in _GENERIC_SLUGS:
            return slug
    filename = _FILE_RE.search(text)
    if filename:
        slug = slugify_title(Path(filename.group(1)).stem)
        if slug and slug not in _GENERIC_SLUGS:
            return slug
    for raw in text.splitlines():
        line = raw.strip().lstrip("#*- ").strip()
        if len(line) < 8:
            continue
        slug = slugify_title(line)
        if slug and slug not in _GENERIC_SLUGS:
            return slug
    return "recording"


def text_slug_generate(config: ScreenLensConfig):
    """Text-role callback for ``infer_content_slug``. ``None`` if it cannot load."""
    try:
        from .omlx_client import InferenceClient
        client = InferenceClient(text_role_captioning_config(config))
    except Exception:
        return None

    def generate(system: str, user: str) -> str:
        return client.chat(
            system,
            user,
            max_tokens=64,
            temperature=0.2,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )

    return generate


def infer_content_slug(text: str, generate=None) -> str:
    """Name the run from processed content.

    ``generate(system, user) -> str`` is the text-role model. On failure or a
    generic answer we fall back to a heading/filename heuristic so a down
    text model cannot block a finished ingest.
    """
    fallback = heuristic_content_slug(text)
    if generate is None or not text.strip():
        return fallback
    try:
        raw = generate(
            "You name screen-recording sessions. Reply with only a short "
            "filesystem slug: 3-8 lowercase words, digits and underscores. "
            "No quotes, no punctuation, no explanation.",
            "These excerpts describe one screen recording. Name the document, "
            "application, or task shown — never 'screen_recording' or a "
            f"timestamp.\n\n{text[:3500]}",
        )
    except Exception:
        return fallback
    slug = slugify_title(str(raw or "").splitlines()[0] if raw else "")
    if not slug or slug in _GENERIC_SLUGS:
        return fallback
    return slug


def apply_content_name(
    config: ScreenLensConfig, video: Path, slug_base: str
) -> str:
    """Rename the current run folder to ``<content>_<recording-clock>``.

    Generic slugs keep the existing folder name. Collisions get ``_2``, ``_3``.
    """
    current = Path(config.data_dir)
    if slug_base and slug_base not in _GENERIC_SLUGS:
        parent = current.parent
        dest = parent / f"{slug_base}_{recording_stamp(video)}"
        suffix = 2
        while dest.exists() and dest != current:
            dest = parent / f"{slug_base}_{recording_stamp(video)}_{suffix}"
            suffix += 1
        if dest != current:
            current.rename(dest)
            point_config_at_data_dir(config, dest)
            current = dest
    identity = read_run_identity(current) or _video_identity(video, config)
    identity.update(
        status="success",
        title=slug_base.replace("_", " ") if slug_base else None,
        content_slug=slug_base or None,
        collection=config.vector_db.collection_name,
    )
    write_run_identity(current, identity)
    return current.name


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


def form_defaults(config: ScreenLensConfig) -> dict[str, Any]:
    """Resolved deck/CLI form defaults — config plus ``.env`` / shell overrides."""
    captioning = config.captioning
    if captioning.backend == CaptionBackend.ollama:
        base_url = captioning.ollama_base_url
    else:
        base_url = resolve_inference_base_url(captioning)
    return {
        "backend": captioning.backend.value,
        "base_url": base_url,
        "strategy": config.frame_extraction.strategy.value,
        "fps": config.frame_extraction.fps,
        "sample_fps": config.frame_selection.sample_fps,
        "caption_max_tokens": config.captioning.max_tokens,
        "cleanup": config.reconstruction.enabled,
        "deterministic": config.ocr.deterministic_backstop,
    }


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

    identity = read_run_identity(folder) or {}
    return {
        "slug": folder.name,
        "base": identity.get("content_slug") or base_slug(folder.name),
        "path": str(folder),
        "frames": frames,
        "captions": captions,
        "ocr": ocr,
        "outputs": outputs,
        "has_chromadb": embedded,
        "collection": (
            identity.get("collection")
            if _valid_chroma_name(str(identity.get("collection") or ""))
            else chroma_collection_name(base_slug(folder.name))
        ),
        "title": identity.get("title"),
        "source_name": identity.get("source_name"),
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
