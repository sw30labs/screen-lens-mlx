"""Background job runner for the ScreenLens web command deck.

One job at a time. Every pipeline here is CPU/GPU-bound and shares a single
oMLX/vLLM endpoint plus one CLIP device, so running two concurrently would
only make both slower — and on DGX Spark it would breach the two-sequence
serving recipe. The HTTP layer polls job status; the worker thread drives the
LangGraph pipelines directly.

The pipelines report progress by printing and by ``logging``, so the worker
captures both into a ring buffer that the dashboard tails.

Author: Nic Cravino — ScreenLens
"""
from __future__ import annotations

import contextlib
import io
import logging
import re
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import ScreenLensConfig
from ..session import (
    VIDEO_SUFFIXES,
    apply_direct_inference,
    apply_video_slug,
    discover_runs,
    endpoint_status,
    extraction_meta_matches,
    find_reusable_run,
    load_config,
    model_roles,
    point_config_at_data_dir,
    read_artifact,
    reuse_video_run,
    run_snapshot,
    transcribe_run_matches,
)

__all__ = [
    "busy",
    "active_job_id",
    "get_job",
    "list_jobs",
    "live_events",
    "start_job",
    "current_run",
    "probe_endpoint",
    "roles",
    "list_runs",
    "snapshot",
    "artifact",
    "resolve_run",
    "list_videos",
    "search_now",
    "PIPELINES",
    "set_pipeline_override",
]

logger = logging.getLogger("screenlens.web")

_MAX_LIVE_EVENTS = 600
_MAX_JOBS = 25

PIPELINES = ("ingest", "transcribe", "reconstruct", "summarize", "assemble")

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_BUSY = threading.Lock()

_LIVE: dict[str, Any] = {
    "events": deque(maxlen=_MAX_LIVE_EVENTS),
    "active_job_id": None,
    "run_slug": None,
    "data_dir": None,
}
_LIVE_LOCK = threading.Lock()

# Tests inject fakes here so the dashboard can be exercised without a model.
_PIPELINE_OVERRIDE: dict[str, Callable[[dict[str, Any], Any], dict[str, Any]]] = {}


def set_pipeline_override(name: str | None, fn: Callable | None = None) -> None:
    """Test hook — replace one pipeline implementation, or clear all."""
    if name is None:
        _PIPELINE_OVERRIDE.clear()
        return
    if fn is None:
        _PIPELINE_OVERRIDE.pop(name, None)
    else:
        _PIPELINE_OVERRIDE[name] = fn


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ── read-side helpers (no job required) ─────────────────────────────────────

def busy() -> bool:
    return _BUSY.locked()


def active_job_id() -> str | None:
    with _LIVE_LOCK:
        return _LIVE.get("active_job_id")


def current_run() -> dict[str, Any]:
    with _LIVE_LOCK:
        return {"slug": _LIVE.get("run_slug"), "data_dir": _LIVE.get("data_dir")}


def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs(limit: int = _MAX_JOBS) -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.get("started_at") or "", reverse=True)
        return [dict(j) for j in jobs[:limit]]


def live_events(limit: int = 120) -> list[dict[str, Any]]:
    with _LIVE_LOCK:
        events = list(_LIVE["events"])
    return events[-limit:]


def probe_endpoint(config_path: str | None = None) -> dict[str, Any]:
    """Reachability + served models for the configured endpoint."""
    return endpoint_status(load_config(config_path))


def roles(config_path: str | None = None) -> dict[str, Any]:
    """Resolved vision/text model roles."""
    return model_roles(load_config(config_path))


def list_runs(data_dir: str | None = None) -> list[dict[str, Any]]:
    return discover_runs(data_dir or "./data")


def snapshot(slug: str, data_dir: str | None = None) -> dict[str, Any] | None:
    folder = resolve_run(slug, data_dir)
    return run_snapshot(folder) if folder else None


def artifact(slug: str, name: str, data_dir: str | None = None) -> dict[str, Any] | None:
    folder = resolve_run(slug, data_dir)
    return read_artifact(folder, name) if folder else None


def resolve_run(slug: str, data_dir: str | None = None) -> Path | None:
    """Map a run slug to its folder, refusing anything that escapes data_dir."""
    if not slug or "/" in slug or "\\" in slug or slug.startswith("."):
        return None
    root = Path(data_dir or "./data").resolve()
    folder = (root / slug).resolve()
    if folder.parent != root or not folder.is_dir():
        return None
    return folder


def list_videos(folder: str) -> list[dict[str, Any]]:
    """List video files in a folder so the dashboard can offer a picker."""
    path = Path(folder).expanduser()
    if not path.is_dir():
        return []
    out = []
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            out.append({"name": p.name, "path": str(p), "size": p.stat().st_size})
    return out


# ── event capture ───────────────────────────────────────────────────────────

_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*(.+)$")
# tqdm redraws in place with \r, e.g.
#   "Captioning frames:  40%|####      | 8/20 [08:23<12:38, 63.22s/it]"
# Without this the dashboard would show only the stage banner for the whole of
# a twenty-minute captioning pass.
_TQDM_RE = re.compile(r"^(?P<desc>.*?):?\s*\d+%\|.*?\|\s*(?P<n>\d+)/(?P<total>\d+)\s*\[(?P<timing>[^\]]*)\]")


def _emit(kind: str, note: str, **extra: Any) -> None:
    entry = {"kind": kind, "at": _now(), "note": note, **extra}
    with _LIVE_LOCK:
        _LIVE["events"].append(entry)
    job_id = active_job_id()
    if not job_id:
        return
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        progress = job.setdefault("progress", {})
        progress["events"] = int(progress.get("events") or 0) + 1
        if kind == "stage":
            progress["stage"] = note
        elif kind == "progress":
            progress["note"] = note
            if extra.get("steps"):
                progress["step"] = extra.get("step")
                progress["steps"] = extra.get("steps")
        elif kind == "error":
            progress["note"] = f"ERROR: {note}"
        else:
            progress["note"] = note


class _EventStream(io.TextIOBase):
    """Turn the pipelines' stdout/stderr chatter into dashboard events."""

    def __init__(self, mirror: io.TextIOBase | None = None) -> None:
        self._buf = ""
        self._mirror = mirror
        self._last_progress: tuple[str, int] | None = None

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if self._mirror is not None:
            with contextlib.suppress(Exception):
                self._mirror.write(s)
        self._buf += s
        # Split on \r as well as \n so in-place progress bars are seen at all.
        while True:
            cut = min((i for i in (self._buf.find("\n"), self._buf.find("\r")) if i >= 0),
                      default=-1)
            if cut < 0:
                break
            line, self._buf = self._buf[:cut], self._buf[cut + 1:]
            self._handle(line)
        return len(s)

    def flush(self) -> None:
        if self._mirror is not None:
            with contextlib.suppress(Exception):
                self._mirror.flush()

    def _handle(self, line: str) -> None:
        text = line.rstrip()
        if not text or set(text) <= {"=", "-", "─"}:
            return

        bar = _TQDM_RE.match(text)
        if bar:
            n, total = int(bar.group("n")), int(bar.group("total"))
            desc = (bar.group("desc") or "working").strip() or "working"
            # tqdm redraws many times per step; only the step changes matter.
            if (desc, n) == self._last_progress:
                return
            self._last_progress = (desc, n)
            _emit("progress", f"{desc} {n}/{total} [{bar.group('timing')}]",
                  step=n, steps=total)
            return

        match = _PROGRESS_RE.match(text)
        if match:
            _emit("stage", match.group(3).strip(), step=int(match.group(1)),
                  steps=int(match.group(2)))
        else:
            _emit("log", text[:400])


class _EventLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover — defensive
            return
        kind = "error" if record.levelno >= logging.ERROR else "log"
        _emit(kind, f"{record.name}: {message}"[:400])


@contextlib.contextmanager
def _captured_output():
    """Route pipeline print()/logging into the dashboard event ring."""
    handler = _EventLogHandler()
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    previous_level = root.level
    if previous_level > logging.INFO or previous_level == logging.NOTSET:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    stream = _EventStream(mirror=sys.__stdout__)
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


# ── job control ─────────────────────────────────────────────────────────────

def _build_config(params: dict[str, Any]) -> ScreenLensConfig:
    """Apply dashboard form values onto a freshly loaded config."""
    config = load_config(params.get("config_file") or None)

    backend = str(params.get("backend") or config.captioning.backend.value)
    apply_direct_inference(
        config,
        backend=backend,
        base_url=(str(params["base_url"]) if params.get("base_url") else None),
        api_key=(str(params["api_key"]) if params.get("api_key") else None),
        vision_model=(str(params["vision_model"]) if params.get("vision_model") else None),
        text_model=(str(params["text_model"]) if params.get("text_model") else None),
        batch_size=(int(params["batch_size"]) if params.get("batch_size") else None),
        caption_max_tokens=(
            int(params["caption_max_tokens"]) if params.get("caption_max_tokens") else None
        ),
    )
    return config


def start_job(params: dict[str, Any]) -> tuple[str | None, str | None]:
    """Launch one pipeline run. Returns ``(job_id, None)`` or ``(None, error)``."""
    pipeline = str(params.get("pipeline") or "").strip()
    if pipeline not in PIPELINES:
        return None, f"pipeline must be one of: {', '.join(PIPELINES)}"

    if pipeline in ("ingest", "transcribe"):
        video = str(params.get("video_path") or "").strip()
        if not video:
            return None, "video_path is required"
        if not Path(video).expanduser().is_file():
            return None, f"video not found: {video}"
    if pipeline == "summarize" and not str(params.get("run_slug") or "").strip():
        return None, "run_slug is required for summarize"

    if not _BUSY.acquire(blocking=False):
        return None, "another job is still running — wait for it to finish"

    job_id = f"sl-{uuid.uuid4().hex[:8]}"
    try:
        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "id": job_id,
                "pipeline": pipeline,
                "status": "running",
                "started_at": _now(),
                "finished_at": None,
                "result": None,
                "error": None,
                "params": {k: v for k, v in params.items() if k != "api_key"},
                "progress": {"stage": "starting", "note": "", "events": 0},
            }
            if len(_JOBS) > _MAX_JOBS:
                stale = sorted(_JOBS, key=lambda k: _JOBS[k]["started_at"])
                for old in stale[: len(_JOBS) - _MAX_JOBS]:
                    if _JOBS[old]["status"] != "running":
                        del _JOBS[old]
        with _LIVE_LOCK:
            _LIVE["active_job_id"] = job_id
            _LIVE["events"].clear()
            _LIVE["run_slug"] = None
            _LIVE["data_dir"] = None
        threading.Thread(target=_run_job, args=(job_id, dict(params)), daemon=True).start()
    except BaseException:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id].update(
                    status="error", error="failed to start job thread", finished_at=_now()
                )
        with _LIVE_LOCK:
            _LIVE["active_job_id"] = None
        _BUSY.release()
        raise
    return job_id, None


def _run_job(job_id: str, params: dict[str, Any]) -> None:
    t0 = time.time()
    try:
        try:
            with _captured_output():
                result = _dispatch(params)
            status, error = ("error", result.get("error")) if result.get("error") else ("done", None)
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            result, status, error = None, "error", f"{type(exc).__name__}: {exc}"
            _emit("error", str(exc)[:400])

        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id].update(
                    status=status,
                    result=result,
                    error=error,
                    finished_at=_now(),
                    elapsed_seconds=round(time.time() - t0, 1),
                )
        _emit("done" if status == "done" else "error",
              f"{params.get('pipeline')} {status} in {time.time() - t0:.1f}s")
    finally:
        with _LIVE_LOCK:
            if _LIVE.get("active_job_id") == job_id:
                _LIVE["active_job_id"] = None
        _BUSY.release()


def _dispatch(params: dict[str, Any]) -> dict[str, Any]:
    pipeline = str(params["pipeline"])
    config = _build_config(params)
    override = _PIPELINE_OVERRIDE.get(pipeline)
    if override is not None:
        return override(params, config)
    return _RUNNERS[pipeline](params, config)


def _publish_run(config: ScreenLensConfig) -> None:
    with _LIVE_LOCK:
        _LIVE["run_slug"] = Path(config.data_dir).name
        _LIVE["data_dir"] = str(config.data_dir)


def _prepare_run_folder(
    config: ScreenLensConfig,
    video: Path,
    *,
    fresh: bool,
    required: str,
    matches,
) -> tuple[str, bool]:
    """Reuse the newest matching prior run folder for ``video`` when possible.

    Same resume semantics as the CLI: stages whose artifacts already exist are
    skipped, so re-running a pipeline from the deck does not re-pay the model
    cost. Falls back to a fresh timestamped slug.
    """
    if not fresh:
        reuse_dir = find_reusable_run(config, video, required)
        if reuse_dir is not None and matches(reuse_dir, video, config):
            return reuse_video_run(config, video, reuse_dir), True
    return apply_video_slug(config, video), False


def _run_ingest(params: dict[str, Any], config: ScreenLensConfig) -> dict[str, Any]:
    from ..config import ExtractionStrategy
    from ..pipeline import build_ingest_graph

    video = Path(str(params["video_path"])).expanduser().resolve()
    if params.get("strategy"):
        config.frame_extraction.strategy = ExtractionStrategy(str(params["strategy"]))
    if params.get("fps"):
        config.frame_extraction.fps = float(params["fps"])
    if params.get("max_interval"):
        config.frame_extraction.max_interval_seconds = float(params["max_interval"])
    if params.get("device"):
        config.embedding.device = str(params["device"])

    slug, reused = _prepare_run_folder(
        config, video,
        fresh=bool(params.get("fresh")),
        required="frames/frames_meta.json",
        matches=extraction_meta_matches,
    )
    _publish_run(config)
    _emit("stage", f"ingest {video.name} → {slug}" + (" (resuming)" if reused else ""))

    result = build_ingest_graph().invoke(
        {"video_path": str(video), "config": config.model_dump()}
    )
    return {
        "run_slug": slug,
        "data_dir": str(config.data_dir),
        "collection": config.vector_db.collection_name,
        "num_frames": result.get("num_frames"),
        "embeddings_shape": result.get("embeddings_shape"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


def _run_transcribe(params: dict[str, Any], config: ScreenLensConfig) -> dict[str, Any]:
    from ..transcribe import transcribe_video

    video = Path(str(params["video_path"])).expanduser().resolve()
    if params.get("sample_fps"):
        config.frame_selection.sample_fps = float(params["sample_fps"])
    config.reconstruction.enabled = bool(params.get("cleanup"))
    config.ocr.deterministic_backstop = bool(params.get("deterministic"))

    slug, reused = _prepare_run_folder(
        config, video,
        fresh=bool(params.get("fresh")),
        required="ocr",
        matches=lambda run_dir, vid, _cfg: transcribe_run_matches(run_dir, vid),
    )
    _publish_run(config)
    _emit("stage", f"transcribe {video.name} → {slug}" + (" (resuming)" if reused else ""))

    result = transcribe_video(str(video), config, config.data_dir)
    return {"run_slug": slug, "data_dir": str(config.data_dir), **result}


def _run_reconstruct(params: dict[str, Any], config: ScreenLensConfig) -> dict[str, Any]:
    from ..reconstruct import reconstruct_folder

    data_dir = str(params.get("data_dir") or "./data")
    slug = str(params.get("run_slug") or "").strip()

    if slug:
        folders = [f for f in [resolve_run(slug, data_dir)] if f]
        if not folders:
            return {"error": f"unknown run: {slug}"}
    else:
        folders = [
            Path(r["path"]) for r in discover_runs(data_dir) if r["captions"] > 0
        ]
    if not folders:
        return {"error": "no folders with captions found — ingest a video first"}

    results = []
    for folder in folders:
        _emit("stage", f"reconstruct {folder.name}")
        with _LIVE_LOCK:
            _LIVE["run_slug"] = folder.name
            _LIVE["data_dir"] = str(folder)
        results.append({"folder": folder.name, "result": reconstruct_folder(str(folder), config)})

    failed = [r for r in results if (r["result"] or {}).get("error")]
    return {
        "folders": len(results),
        "results": results,
        "error": failed[0]["result"]["error"] if len(failed) == len(results) and failed else None,
    }


def _run_summarize(params: dict[str, Any], config: ScreenLensConfig) -> dict[str, Any]:
    from ..pipeline import summarize_all_node

    data_dir = str(params.get("data_dir") or "./data")
    folder = resolve_run(str(params["run_slug"]), data_dir)
    if folder is None:
        return {"error": f"unknown run: {params.get('run_slug')}"}

    point_config_at_data_dir(config, folder)
    with _LIVE_LOCK:
        _LIVE["run_slug"] = folder.name
        _LIVE["data_dir"] = str(folder)
    _emit("stage", f"summarize {folder.name}")

    result = summarize_all_node({"config": config.model_dump()})
    summary = result.get("summary", "")
    (folder / "output").mkdir(parents=True, exist_ok=True)
    (folder / "output" / "summary.md").write_text(summary, encoding="utf-8")
    return {
        "run_slug": folder.name,
        "summary": summary,
        "chars": len(summary),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


def _run_assemble(params: dict[str, Any], config: ScreenLensConfig) -> dict[str, Any]:
    from ..assemble import assemble_corpus

    data_dir = str(params.get("data_dir") or "./data")
    output_dir = str(params.get("output_dir") or "./assembled")
    _emit("stage", f"assemble {data_dir} → {output_dir}")
    return assemble_corpus(data_dir=data_dir, output_dir=output_dir, config=config)


_RUNNERS: dict[str, Callable[[dict[str, Any], ScreenLensConfig], dict[str, Any]]] = {
    "ingest": _run_ingest,
    "transcribe": _run_transcribe,
    "reconstruct": _run_reconstruct,
    "summarize": _run_summarize,
    "assemble": _run_assemble,
}


# ── synchronous search ──────────────────────────────────────────────────────

def search_now(params: dict[str, Any]) -> dict[str, Any]:
    """Run a CLIP/ChromaDB search inline — fast enough not to need a job."""
    from ..pipeline import build_search_graph, search_node

    query = str(params.get("query") or "").strip()
    if not query:
        return {"error": "query is required"}

    data_dir = str(params.get("data_dir") or "./data")
    folder = resolve_run(str(params.get("run_slug") or ""), data_dir)
    if folder is None:
        return {"error": "run_slug is required and must name a run folder"}

    config = _build_config(params)
    point_config_at_data_dir(config, folder)
    config.vector_db.collection_name = str(
        params.get("collection") or f"screenlens_{_base(folder.name)}"
    )
    if params.get("top_k"):
        config.search.top_k = int(params["top_k"])

    state = {"query": query, "config": config.model_dump()}
    with _captured_output():
        if params.get("summarize"):
            result = build_search_graph().invoke(state)
        else:
            result = search_node(state)
    return {
        "query": query,
        "run_slug": folder.name,
        "results": result.get("search_results", []),
        "summary": result.get("summary"),
    }


def _base(name: str) -> str:
    from ..session import base_slug

    return base_slug(name)
