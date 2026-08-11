"""
CLI Interface for ScreenLens.

Usage:
    python -m src.cli ingest VIDEO_PATH                     # Native backend + keyframes
    python -m src.cli ingest VIDEO_PATH --backend ollama    # Use Ollama instead
    python -m src.cli search "your query"                   # Search ingested video
    python -m src.cli run VIDEO_PATH "query"                # Ingest + search in one shot
    python -m src.cli batch FOLDER_PATH                     # Batch-ingest all videos in a folder
    python -m src.cli reconstruct                           # Reconstruct artifacts from captions
    python -m src.cli info                                  # Show vector store stats
"""
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import (
    ScreenLensConfig,
    CaptionBackend,
    ExtractionStrategy,
    InferenceBackend,
    default_caption_backend,
    default_embedding_device,
    default_inference_backend,
    default_inference_concurrency,
)
from .omlx_client import resolve_inference_model, resolve_llm_model, resolve_ocr_model
from .pipeline import build_ingest_graph, build_search_graph, build_full_graph, summarize_all_node
from .session import (
    apply_video_slug,
    extraction_meta_matches,
    find_reusable_run,
    load_config,
    reuse_video_run,
    transcribe_run_matches,
)

app = typer.Typer(
    name="screenlens",
    help="ScreenLens — Local video scene intelligence for DGX Spark and Apple Silicon",
    rich_markup_mode="rich",
)
console = Console()

DEFAULT_INFERENCE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_OMLX_URL = DEFAULT_INFERENCE_URL  # compatibility for external imports
DEFAULT_BACKEND = default_caption_backend().value
DEFAULT_INFERENCE_BACKEND = default_inference_backend().value
DEFAULT_DEVICE = default_embedding_device()
DEFAULT_BATCH_SIZE = default_inference_concurrency()


# Config/slug plumbing is shared with the web command deck so a run started
# from either front end lands in the same ./data/<slug>/ layout.
_load_config = load_config
_apply_video_slug = apply_video_slug


def _prepare_video_run(
    config: ScreenLensConfig,
    video: Path,
    *,
    fresh: bool,
    required: str,
    matches,
) -> tuple[str, bool]:
    """Pick the run folder for a video: reuse the newest matching prior run.

    Falls back to a fresh timestamped slug when there is nothing to reuse,
    when the prior run's metadata belongs to a different video/config, or when
    ``fresh`` was requested. Returns ``(slug, reused)``.
    """
    if not fresh:
        reuse_dir = find_reusable_run(config, video, required)
        if reuse_dir is not None and matches(reuse_dir, video, config):
            return reuse_video_run(config, video, reuse_dir), True
    return apply_video_slug(config, video), False


def _apply_captioning_options(
    config: ScreenLensConfig,
    *,
    backend: str = DEFAULT_BACKEND,
    ollama_model: str = "llama3.2-vision",
    ollama_url: str = "http://127.0.0.1:11434",
    batch_size: int = DEFAULT_BATCH_SIZE,
    omlx_url: str = DEFAULT_OMLX_URL,
    omlx_model: Optional[str] = None,
    omlx_api_key: Optional[str] = None,
) -> None:
    """Apply CLI captioning/inference flags to the config."""
    config.captioning.backend = CaptionBackend(backend)
    config.captioning.ollama_model = ollama_model
    config.captioning.ollama_base_url = ollama_url
    config.captioning.batch_size = batch_size
    if config.captioning.backend == CaptionBackend.vllm:
        config.captioning.vllm_base_url = omlx_url
        if omlx_model is not None:
            config.captioning.vllm_model = omlx_model
        if omlx_api_key is not None:
            config.captioning.vllm_api_key = omlx_api_key
    else:
        config.captioning.omlx_base_url = omlx_url
        if omlx_model is not None:
            config.captioning.omlx_model = omlx_model
        if omlx_api_key is not None:
            config.captioning.omlx_api_key = omlx_api_key


def _caption_model_display(config: ScreenLensConfig) -> str:
    """Return a short model label for CLI panels."""
    if config.captioning.backend in (CaptionBackend.vllm, CaptionBackend.omlx):
        provider = "vLLM" if config.captioning.backend == CaptionBackend.vllm else "oMLX"
        return f"{resolve_inference_model(config.captioning).split('/')[-1]} via {provider}"
    return f"{config.captioning.ollama_model} via Ollama"


@app.command()
def ingest(
    video_path: str = typer.Argument(..., help="Path to the video file (.mov, .mp4, etc.)"),
    # Extraction strategy
    strategy: str = typer.Option("keyframe", help="Extraction strategy: 'keyframe' (smart) or 'fixed_fps'"),
    fps: float = typer.Option(1.0, help="Frames per second (only for fixed_fps strategy)"),
    max_interval: float = typer.Option(4.0, help="Max seconds between keyframes (keyframe strategy)"),
    # Captioning backend
    backend: str = typer.Option(DEFAULT_BACKEND, help="Caption backend: vllm, omlx, or ollama"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_model: Optional[str] = typer.Option(
        None, "--inference-model", "--vllm-model", "--omlx-model",
        help="Served model id (defaults to provider environment)",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
    ollama_model: str = typer.Option("llama3.2-vision", help="Ollama vision model (if backend=ollama)"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", help="Ollama API URL"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, help="Concurrent caption requests"),
    # Other
    device: str = typer.Option(DEFAULT_DEVICE, help="Device for CLIP: mps, cuda, cpu"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore prior runs of this video and start a new run folder"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Ingest a video: extract keyframes, generate captions, create embeddings.

    Re-running the same video resumes the newest prior run instead of paying
    for extraction, captions, and embeddings again (use --fresh to start over).
    """
    video = Path(video_path)
    if not video.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    config = _load_config(config_file)

    # Frame extraction
    config.frame_extraction.strategy = ExtractionStrategy(strategy)
    config.frame_extraction.fps = fps
    config.frame_extraction.max_interval_seconds = max_interval

    _apply_captioning_options(
        config,
        backend=backend,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        batch_size=batch_size,
        omlx_url=omlx_url,
        omlx_model=omlx_model,
        omlx_api_key=omlx_api_key,
    )
    # Embedding
    config.embedding.device = device

    # Per-video data directory: resume the newest prior run of this video when
    # its extraction config matches, otherwise mint a fresh slug (as `batch`).
    slug, reused = _prepare_video_run(
        config, video, fresh=fresh,
        required="frames/frames_meta.json",
        matches=extraction_meta_matches,
    )
    if reused:
        console.print(f"[dim]Resuming run {slug} — completed stages are skipped (--fresh to start over)[/dim]")

    # Display config
    model_display = _caption_model_display(config)

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Video Ingestion[/bold green]\n"
        f"Video: {video.name} ({video.stat().st_size / (1024**2):.0f} MB)\n"
        f"Output: {config.data_dir}\n"
        f"Extraction: {strategy} | Captioning: {backend} ({model_display})\n"
        f"CLIP device: {device}",
        title="Configuration",
    ))

    pipeline = build_ingest_graph()
    initial_state = {
        "video_path": str(video.resolve()),
        "config": config.model_dump(),
    }

    t0 = time.time()
    result = pipeline.invoke(initial_state)
    total_time = time.time() - t0

    # Display results
    console.print(f"\n[bold green]Ingestion complete![/bold green]")

    table = Table(title="Pipeline Summary")
    table.add_column("Stage", style="cyan")
    table.add_column("Time (s)", justify="right")
    table.add_column("Details")

    elapsed = result.get("elapsed_seconds", {})
    table.add_row(
        "Frame Extraction",
        f"{elapsed.get('ingest', 0):.1f}",
        f"{result.get('num_frames', 0)} frames ({strategy})"
    )
    table.add_row(
        "Captioning",
        f"{elapsed.get('caption', 0):.1f}",
        f"{backend} ({model_display})"
    )
    emb_shape = result.get('embeddings_shape', ['?', '?'])
    table.add_row(
        "Embedding + Store",
        f"{elapsed.get('embed', 0):.1f}",
        f"dim={emb_shape[1] if len(emb_shape) > 1 else '?'}"
    )
    table.add_row("Total", f"{total_time:.1f}", "", style="bold")
    console.print(table)


def _best_collection_name(chromadb_path: Path, hint: str) -> str:
    """Find the best collection name in a ChromaDB directory.

    Tries ``hint`` first; if it has 0 items, falls back to the collection
    with the most items (handles legacy ingestions with different naming).
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chromadb_path))
        colls = client.list_collections()
        # Check if hint exists and has items
        for c in colls:
            if c.name == hint and c.count() > 0:
                return hint
        # Fallback: pick collection with most items
        best = max(colls, key=lambda c: c.count(), default=None)
        if best and best.count() > 0:
            return best.name
    except Exception:
        pass
    return hint


def _resolve_data_targets(data_dir: Optional[str], collection: Optional[str],
                          config: ScreenLensConfig) -> list[tuple[Path, str]]:
    """Resolve --data-dir / --collection into (chromadb_path, collection_name) pairs.

    When data_dir points to a parent folder with multiple video sub-folders,
    returns all of them.  Otherwise returns a single target.
    """
    import re as _re

    targets: list[tuple[Path, str]] = []

    if data_dir:
        dp = Path(data_dir)

        # Check if this is a parent directory containing multiple video folders
        sub_chromadbs = sorted(
            d for d in dp.iterdir()
            if d.is_dir() and (d / "chromadb").exists()
        ) if dp.is_dir() else []

        if sub_chromadbs:
            # Parent directory — search across all video sub-folders
            for sub in sub_chromadbs:
                base_slug = _re.sub(r'_\d{8}_\d{6}$', '', sub.name)
                cname = collection or f"screenlens_{base_slug}"
                cname = _best_collection_name(sub / "chromadb", cname)
                targets.append((sub / "chromadb", cname))
        elif (dp / "chromadb").exists():
            # Single video folder
            base_slug = _re.sub(r'_\d{8}_\d{6}$', '', dp.name)
            cname = collection or f"screenlens_{base_slug}"
            cname = _best_collection_name(dp / "chromadb", cname)
            targets.append((dp / "chromadb", cname))

    elif collection:
        # Auto-infer persist_directory from collection name
        if collection.startswith("screenlens_"):
            slug = collection[len("screenlens_"):]
            inferred = Path(f"./data/{slug}")
            if not (inferred / "chromadb").exists():
                candidates = sorted(Path("./data").glob(f"{slug}_*"), reverse=True)
                for c in candidates:
                    if (c / "chromadb").exists():
                        inferred = c
                        break
            if (inferred / "chromadb").exists():
                targets.append((inferred / "chromadb", collection))

    # Fallback: default config
    if not targets:
        targets.append((
            Path(config.vector_db.persist_directory),
            config.vector_db.collection_name,
        ))

    return targets


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(10, help="Number of results to return"),
    summarize: bool = typer.Option(True, help="Generate LLM summary of results"),
    collection: Optional[str] = typer.Option(None, help="ChromaDB collection name (e.g. screenlens_existinginvestment)"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory — a single video folder or the parent ./data/ for all"),
    ollama_url: str = typer.Option(
        "http://127.0.0.1:11434",
        help="Ollama API URL (used only when the loaded config selects Ollama)",
    ),
    device: str = typer.Option(DEFAULT_DEVICE, help="Device for CLIP: mps, cuda, cpu"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Search the ingested video with a natural language query."""
    config = _load_config(config_file)
    config.search.top_k = top_k
    config.search.base_url = ollama_url
    config.embedding.device = device

    targets = _resolve_data_targets(data_dir, collection, config)
    multi = len(targets) > 1

    console.print(f"\n[bold cyan]Searching:[/bold cyan] '{query}'")
    if multi:
        console.print(f"[dim]  Searching across {len(targets)} collections[/dim]")
    console.print()

    all_results = []
    summaries: list[tuple[str, str]] = []  # (source_name, summary_text)

    for chroma_path, coll_name in targets:
        cfg_copy = config.model_copy(deep=True)
        cfg_copy.vector_db.persist_directory = str(chroma_path)
        cfg_copy.vector_db.collection_name = coll_name

        pipeline = build_search_graph()
        state = {"query": query, "config": cfg_copy.model_dump()}
        result = pipeline.invoke(state)

        results = result.get("search_results", [])
        # Tag results with their source collection
        for r in results:
            r["_collection"] = coll_name
        all_results.extend(results)
        if result.get("summary"):
            source_label = coll_name.replace("screenlens_", "")
            summaries.append((source_label, result["summary"]))

    # When searching multiple collections, ensure representation from each source
    if multi and len(targets) > 1:
        # Guarantee at least min_per results from each collection
        min_per = max(2, top_k // len(targets))
        by_source: dict[str, list] = {}
        for r in all_results:
            by_source.setdefault(r.get("_collection", ""), []).append(r)
        # Sort each source by score
        for k in by_source:
            by_source[k].sort(key=lambda r: r.get("score", 0), reverse=True)
        # Build balanced result list
        balanced = []
        seen = set()
        # First pass: take min_per from each source
        for k, items in by_source.items():
            for item in items[:min_per]:
                balanced.append(item)
                seen.add(id(item))
        # Second pass: fill remaining slots by score
        remainder = [r for r in all_results if id(r) not in seen]
        remainder.sort(key=lambda r: r.get("score", 0), reverse=True)
        balanced.extend(remainder)
        all_results = balanced[:top_k]
    else:
        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        all_results = all_results[:top_k]

    if not all_results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Top {len(all_results)} Results")
    table.add_column("#", justify="right", width=3)
    if multi:
        table.add_column("Source", style="magenta", width=20)
    table.add_column("Time", style="cyan", width=12)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Caption", max_width=80)

    for i, r in enumerate(all_results):
        row = [str(i + 1)]
        if multi:
            row.append(r.get("_collection", "?").replace("screenlens_", ""))
        row.extend([
            r.get("timestamp_str", "?"),
            f"{r.get('score', 0):.3f}",
            r.get("caption", "")[:120] + "...",
        ])
        table.add_row(*row)
    console.print(table)

    if summarize and summaries:
        for source_label, summary_text in summaries:
            title = f"[bold]Summary — {source_label}[/bold]" if multi else "[bold]Summary[/bold]"
            console.print(Panel(summary_text, title=title, border_style="green"))


@app.command()
def run(
    video_path: str = typer.Argument(..., help="Path to the video file"),
    query: str = typer.Argument(..., help="Natural language search query"),
    strategy: str = typer.Option("keyframe", help="Extraction strategy: 'keyframe' or 'fixed_fps'"),
    backend: str = typer.Option(DEFAULT_BACKEND, help="Caption backend: vllm, omlx, or ollama"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_model: Optional[str] = typer.Option(
        None, "--inference-model", "--vllm-model", "--omlx-model",
        help="Served model id",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
    device: str = typer.Option(DEFAULT_DEVICE, help="Device for CLIP"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore prior runs of this video and start a new run folder"),
):
    """Ingest a video AND search it in one shot."""
    video = Path(video_path)
    if not video.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    config = ScreenLensConfig()
    config.frame_extraction.strategy = ExtractionStrategy(strategy)
    _apply_captioning_options(
        config,
        backend=backend,
        omlx_url=omlx_url,
        omlx_model=omlx_model,
        omlx_api_key=omlx_api_key,
    )
    config.embedding.device = device

    # Per-video data directory (resume semantics consistent with `ingest`)
    _prepare_video_run(
        config, video, fresh=fresh,
        required="frames/frames_meta.json",
        matches=extraction_meta_matches,
    )

    pipeline = build_full_graph()
    state = {
        "video_path": str(video.resolve()),
        "query": query,
        "config": config.model_dump(),
    }

    result = pipeline.invoke(state)

    if result.get("summary"):
        console.print(Panel(result["summary"], title="[bold]Answer[/bold]", border_style="green"))


@app.command()
def summarize(
    backend: str = typer.Option(DEFAULT_INFERENCE_BACKEND, help="Inference backend: vllm or omlx"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_model: Optional[str] = typer.Option(
        None, "--inference-model", "--vllm-model", "--omlx-model",
        help="Served model id",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Generate a full-video summary from all ingested captions."""
    config = _load_config(config_file)
    _apply_captioning_options(
        config,
        backend=backend,
        omlx_url=omlx_url,
        omlx_model=omlx_model,
        omlx_api_key=omlx_api_key,
    )

    model_display = _caption_model_display(config)
    console.print(Panel.fit(
        f"[bold green]ScreenLens — Video Summarization[/bold green]\n"
        f"Model: {model_display}\n"
        f"Captions dir: {config.data_dir / 'captions'}",
        title="Configuration",
    ))

    import time
    t0 = time.time()
    state = {"config": config.model_dump()}
    result = summarize_all_node(state)
    total_time = time.time() - t0

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    if result.get("summary"):
        console.print(Panel(
            result["summary"],
            title="[bold]Full Video Summary[/bold]",
            border_style="green",
        ))
        console.print(f"\n[dim]Generated in {total_time:.1f}s[/dim]")


VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


@app.command()
def batch(
    folder_path: str = typer.Argument(..., help="Path to a folder containing video files"),
    # Extraction strategy
    strategy: str = typer.Option("keyframe", help="Extraction strategy: 'keyframe' (smart) or 'fixed_fps'"),
    fps: float = typer.Option(1.0, help="Frames per second (only for fixed_fps strategy)"),
    max_interval: float = typer.Option(4.0, help="Max seconds between keyframes (keyframe strategy)"),
    # Captioning backend
    backend: str = typer.Option(DEFAULT_BACKEND, help="Caption backend: vllm, omlx, or ollama"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_model: Optional[str] = typer.Option(
        None, "--inference-model", "--vllm-model", "--omlx-model",
        help="Served model id",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
    ollama_model: str = typer.Option("llama3.2-vision", help="Ollama vision model (if backend=ollama)"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", help="Ollama API URL"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, help="Concurrent caption requests"),
    # Other
    device: str = typer.Option(DEFAULT_DEVICE, help="Device for CLIP: mps, cuda, cpu"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore prior runs and start new run folders"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Batch-ingest all videos in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        console.print(f"[red]Error: Not a directory: {folder_path}[/red]")
        raise typer.Exit(1)

    videos = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        console.print(f"[yellow]No video files found in {folder_path}[/yellow]")
        console.print(f"[dim]Supported extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}[/dim]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Batch Ingestion[/bold green]\n"
        f"Folder: {folder.resolve()}\n"
        f"Videos found: {len(videos)}\n"
        f"Backend: {backend} | Strategy: {strategy}",
        title="Batch Configuration",
    ))

    for i, video in enumerate(videos, 1):
        console.print(f"\n[bold cyan]({'='*50})[/bold cyan]")
        console.print(f"[bold cyan]  Video {i}/{len(videos)}: {video.name}[/bold cyan]")
        console.print(f"[bold cyan]({'='*50})[/bold cyan]")

        config = _load_config(config_file)
        config.frame_extraction.strategy = ExtractionStrategy(strategy)
        config.frame_extraction.fps = fps
        config.frame_extraction.max_interval_seconds = max_interval
        _apply_captioning_options(
            config,
            backend=backend,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
            batch_size=batch_size,
            omlx_url=omlx_url,
            omlx_model=omlx_model,
            omlx_api_key=omlx_api_key,
        )
        config.embedding.device = device

        # Per-video data directory (resume semantics shared with `ingest`/`run`)
        slug, reused = _prepare_video_run(
            config, video, fresh=fresh,
            required="frames/frames_meta.json",
            matches=extraction_meta_matches,
        )
        if reused:
            console.print(f"[dim]  Resuming run {slug}[/dim]")

        pipeline = build_ingest_graph()
        initial_state = {
            "video_path": str(video.resolve()),
            "config": config.model_dump(),
        }

        t0 = time.time()
        try:
            result = pipeline.invoke(initial_state)
            elapsed = time.time() - t0
            num_frames = result.get("num_frames", 0)
            console.print(f"[green]  ✓ {video.name} — {num_frames} frames in {elapsed:.1f}s[/green]")
        except Exception as e:
            elapsed = time.time() - t0
            console.print(f"[red]  ✗ {video.name} — failed after {elapsed:.1f}s: {e}[/red]")

    console.print(f"\n[bold green]Batch complete — processed {len(videos)} videos.[/bold green]")


@app.command()
def reconstruct(
    folder: Optional[str] = typer.Argument(None, help="Specific video folder to reconstruct (e.g. existinginvestment_20260408_223036)"),
    data_dir: str = typer.Option("./data", help="Base data directory containing ingested video folders"),
    backend: str = typer.Option(DEFAULT_INFERENCE_BACKEND, help="Inference backend: vllm or omlx"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_model: Optional[str] = typer.Option(
        None, "--inference-model", "--vllm-model", "--omlx-model",
        help="Served model id",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Reconstruct artifacts from ingested video captions.

    Scans all folders in the data directory, classifies each recording
    (Python code, Markdown doc, PDF, or GUI demo), and uses LangGraph
    deep agents to reconstruct the original artifacts with QA reflection.

    Examples:
        screenlens reconstruct                                    # Reconstruct all videos
        screenlens reconstruct existinginvestment_20260408_223036  # Reconstruct one video
    """
    from .reconstruct import reconstruct_folder

    base = Path(data_dir)
    if not base.is_dir():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)

    # Find folders to process
    if folder:
        # Target specific folder
        specific = base / folder
        if not specific.is_dir():
            # Try to find matching folder (with or without timestamp)
            matches = sorted(
                d for d in base.iterdir()
                if d.is_dir() and folder in d.name and (d / "captions" / "all_captions.json").exists()
            )
            if matches:
                specific = matches[0]
            else:
                console.print(f"[red]Error: Folder not found: {folder}[/red]")
                raise typer.Exit(1)
        folders = [specific]
    else:
        # Find all folders with captions
        folders = sorted(
            d for d in base.iterdir()
            if d.is_dir() and (d / "captions" / "all_captions.json").exists()
        )

    if not folders:
        console.print(f"[yellow]No ingested video data found in {data_dir}[/yellow]")
        console.print("[dim]Run 'screenlens ingest' first to process videos.[/dim]")
        raise typer.Exit(1)

    config = _load_config(config_file)
    _apply_captioning_options(
        config,
        backend=backend,
        omlx_url=omlx_url,
        omlx_model=omlx_model,
        omlx_api_key=omlx_api_key,
    )

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Artifact Reconstruction[/bold green]\n"
        f"Data dir: {base.resolve()}\n"
        f"Folders: {len(folders)}\n"
        f"Model: {_caption_model_display(config)}",
        title="Reconstruction Pipeline",
    ))

    t0_total = time.time()
    results_summary = []

    for i, folder in enumerate(folders, 1):
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]  Folder {i}/{len(folders)}: {folder.name}[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")

        # Skip folders already successfully reconstructed (meta.json is only
        # written on success). Delete output/ to force a re-run.
        if (folder / "output" / "reconstruction_meta.json").exists():
            console.print(f"[dim]  Skipped — already reconstructed (delete output/ to redo)[/dim]")
            results_summary.append((folder.name, "skipped", 0.0))
            continue

        t0 = time.time()
        try:
            result = reconstruct_folder(str(folder), config)
            elapsed = time.time() - t0

            if result.get("error"):
                console.print(f"[red]  Error: {result['error']}[/red]")
                results_summary.append((folder.name, "error", elapsed))
                continue

            content_type = result.get("content_type", "unknown")
            saved = result.get("saved_paths", [])
            qa_scores = result.get("qa_scores", {})
            overall_qa = qa_scores.get("completeness", 0)

            console.print(
                f"[green]  Reconstructed: {content_type} | "
                f"{len(saved)} files | QA: {json.dumps(qa_scores)} | "
                f"{elapsed:.1f}s[/green]"
            )
            for path in saved:
                console.print(f"[dim]    {path}[/dim]")

            results_summary.append((folder.name, content_type, elapsed))

        except Exception as e:
            elapsed = time.time() - t0
            console.print(f"[red]  Failed after {elapsed:.1f}s: {e}[/red]")
            results_summary.append((folder.name, "failed", elapsed))

    total_time = time.time() - t0_total

    # Summary table
    console.print(f"\n")
    table = Table(title="Reconstruction Summary")
    table.add_column("Folder", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Time (s)", justify="right")
    for name, ctype, elapsed in results_summary:
        style = "red" if ctype in ("error", "failed") else ""
        table.add_row(name, ctype, f"{elapsed:.1f}", style=style)
    table.add_row("Total", "", f"{total_time:.1f}", style="bold")
    console.print(table)


@app.command()
def assemble(
    data_dir: str = typer.Option("./data", help="Directory containing data/*/output/ artifacts"),
    output_dir: str = typer.Option("./OUTPUT", help="Where to write the assembled tree"),
    backend: str = typer.Option(DEFAULT_INFERENCE_BACKEND, help="Inference backend: vllm or omlx"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_model: Optional[str] = typer.Option(
        None, "--inference-model", "--vllm-model", "--omlx-model",
        help="Served model id",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
    mapping: Optional[str] = typer.Option(None, help="Path to a hand-edited MANIFEST.json — skips LLM inference"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Stop after corpus classification, write nothing"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Assemble per-folder reconstructions into a single project tree.

    Detects whether the corpus represents a coding project, infers the original
    source-tree path of every artifact via batched LLM sub-agents, validates the
    assembled tree, and writes to OUTPUT/<timestamp>/.

    Examples:
        screenlens assemble                                # full pipeline
        screenlens assemble --dry-run                      # gate + classify only
        screenlens assemble --mapping path/to/manifest.json  # skip inference
    """
    from .assemble import assemble_corpus

    config = _load_config(config_file)
    _apply_captioning_options(
        config,
        backend=backend,
        omlx_url=omlx_url,
        omlx_model=omlx_model,
        omlx_api_key=omlx_api_key,
    )

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Corpus Assembly[/bold green]\n"
        f"Data dir:   {Path(data_dir).resolve()}\n"
        f"Output dir: {Path(output_dir).resolve()}\n"
        f"Model:      {_caption_model_display(config)}\n"
        f"Mode:       {'DRY RUN' if dry_run else 'FULL'}"
        + (f"\nMapping:    {mapping}" if mapping else ""),
        title="Assembly Pipeline",
    ))

    t0 = time.time()
    result = assemble_corpus(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
        mapping_override=mapping,
        dry_run=dry_run,
    )
    elapsed = time.time() - t0

    console.print(f"\n[bold]Pipeline finished in {elapsed:.1f}s[/bold]")
    console.print(f"  Final stage: {result.get('stage', '?')}")
    if result.get("stage") == "gate_failed":
        console.print(f"  [yellow]Gate decided this is not a coding project — nothing to assemble[/yellow]")


@app.command()
def transcribe(
    video_path: str = typer.Argument(..., help="Path to the screen recording (.mov, .mp4, ...)"),
    backend: str = typer.Option(DEFAULT_INFERENCE_BACKEND, help="Inference backend: vllm or omlx"),
    ocr_model: Optional[str] = typer.Option(None, help="Vision OCR model id (defaults to the served model)"),
    llm_model: Optional[str] = typer.Option(None, help="Text model for optional seam cleanup"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
    sample_fps: float = typer.Option(2.0, help="Frames/sec to sample before dedup"),
    cleanup: bool = typer.Option(False, "--cleanup", help="Run the optional LLM seam/indent cleanup pass (default: off — the raw stitched transcript is already verbatim)"),
    deterministic: bool = typer.Option(False, "--deterministic", help="Also run Apple Vision backstop (macOS only)"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore cached OCR from prior runs of this video"),
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Verbatim transcription: faithfully reconstruct the text/code shown in a recording.

    Pipeline: scroll-safe frame selection → vision OCR → text-space stitch →
    LLM seam/indent cleanup. Output is written to ./data/<slug>/output/transcript.md.
    """
    from .transcribe import transcribe_video
    from .omlx_client import normalize_api_base_url

    video = Path(video_path)
    if not video.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    config = _load_config(config_file)
    direct_backend = InferenceBackend(backend)
    base = normalize_api_base_url(omlx_url)
    config.ocr.backend = direct_backend
    config.reconstruction.backend = direct_backend
    config.ocr.base_url = base
    config.reconstruction.base_url = base
    if ocr_model is not None:
        config.ocr.model = ocr_model
    if llm_model is not None:
        config.reconstruction.model = llm_model
    if omlx_api_key is not None:
        config.ocr.api_key = omlx_api_key
        config.reconstruction.api_key = omlx_api_key
    config.frame_selection.sample_fps = sample_fps
    config.reconstruction.enabled = cleanup
    config.ocr.deterministic_backstop = deterministic

    slug, reused = _prepare_video_run(
        config, video, fresh=fresh,
        required="ocr",
        matches=lambda run_dir, vid, _cfg: transcribe_run_matches(run_dir, vid),
    )
    if reused:
        console.print(f"[dim]Resuming run {slug} — cached OCR is reused (--fresh to start over)[/dim]")

    from .omlx_client import resolve_ocr_model, resolve_llm_model
    console.print(Panel.fit(
        f"[bold green]ScreenLens — Verbatim Transcription[/bold green]\n"
        f"Video: {video.name} ({video.stat().st_size / (1024**2):.0f} MB)\n"
        f"Output: {config.data_dir}/output/transcript.md\n"
        f"OCR (vision): {resolve_ocr_model(config.ocr)}\n"
        f"Cleanup (text): {resolve_llm_model(config.reconstruction) if cleanup else 'disabled'}\n"
        f"Sample: {sample_fps} fps | Deterministic backstop: {deterministic}",
        title="Configuration",
    ))

    t0 = time.time()
    try:
        result = transcribe_video(str(video.resolve()), config, config.data_dir)
    except RuntimeError as exc:
        console.print(f"\n[red]Transcription aborted:[/red] {exc}")
        raise typer.Exit(1)
    elapsed = time.time() - t0

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")
    table = Table(title="Transcription Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Frames selected", str(result.get("frames_selected", "?")))
    table.add_row("Frames with text", str(result.get("frames_with_text", "?")))
    table.add_row("OCR model", result.get("ocr_model", "?"))
    table.add_row("Transcript", result.get("transcript_path", "?"))
    console.print(table)


@app.command()
def models(
    backend: str = typer.Option(DEFAULT_INFERENCE_BACKEND, help="Inference backend: vllm or omlx"),
    omlx_url: str = typer.Option(
        DEFAULT_INFERENCE_URL, "--inference-url", "--vllm-url", "--omlx-url",
        help="OpenAI-compatible inference API URL",
    ),
    omlx_api_key: Optional[str] = typer.Option(
        None, "--inference-api-key", "--vllm-api-key", "--omlx-api-key",
        help="Optional inference API key",
    ),
):
    """List served models and flag which can do OCR (vision)."""
    from .omlx_client import (
        list_models,
        is_known_text_only_model,
        is_known_vision_model,
        _env_value,
    )

    if backend == InferenceBackend.vllm.value:
        key = omlx_api_key or _env_value(
            "VLLM_API_KEY", ignore_placeholders=True
        ) or "local"
    else:
        key = omlx_api_key or _env_value(
            "MLX_API_KEY", "OMLX_API_KEY", ignore_placeholders=True
        )
    try:
        ids = list_models(omlx_url, key)
    except Exception as exc:
        console.print(f"[red]Could not reach {backend}: {exc}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"{backend} models ({len(ids)})")
    table.add_column("Model id", style="cyan")
    table.add_column("OCR-capable?", justify="center")
    for mid in sorted(ids):
        if is_known_text_only_model(mid):
            cap = "[red]no (text-only)[/red]"
        elif is_known_vision_model(mid):
            cap = "[green]yes (vision)[/green]"
        else:
            cap = "[yellow]unknown[/yellow]"
        table.add_row(mid, cap)
    console.print(table)
    console.print("[dim]Use a vision model for `transcribe --ocr-model`; a text model is fine for cleanup.[/dim]")


@app.command()
def serve(
    port: int = typer.Option(8760, help="Port for the local command deck"),
    host: str = typer.Option("127.0.0.1", help="Bind address (loopback only)"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser tab"),
    view: Optional[str] = typer.Option(None, help="Open straight to a view: overview, runs, frames, search, pipeline, artifacts"),
    restart: bool = typer.Option(False, "--restart", help="Stop a command deck already on this port, then start a fresh one"),
    force: bool = typer.Option(False, "--force", help="With --restart, stop the running deck even mid-job"),
):
    """Launch the web command deck — a local dashboard over every pipeline.

    Binds loopback-only by design: it starts jobs and reads frames off disk.
    If a deck already holds the port, this opens that one instead of failing.
    """
    from .web.server import serve as serve_dashboard

    console.print(Panel.fit(
        f"[bold green]ScreenLens — Web Command Deck[/bold green]\n"
        f"URL: http://{host}:{port}\n"
        f"Vision model: {resolve_ocr_model(_load_config(None).ocr)}\n"
        f"Text model: {resolve_llm_model(_load_config(None).reconstruction)}",
        title="Configuration",
    ))
    serve_dashboard(
        host=host,
        port=port,
        open_browser=not no_browser,
        query=f"#{view}" if view else "",
        restart=restart,
        force=force,
    )


@app.command()
def info(
    config_file: Optional[str] = typer.Option(None, help="Path to config JSON file"),
):
    """Show info about the current vector store."""
    config = _load_config(config_file)
    from .vector_store import ScreenLensVectorStore

    store = ScreenLensVectorStore(config.vector_db)
    count = store.count()

    console.print(Panel.fit(
        f"Collection: {config.vector_db.collection_name}\n"
        f"Frames stored: {count}\n"
        f"Persist dir: {config.vector_db.persist_directory}",
        title="[bold]Vector Store Info[/bold]",
    ))

    if count > 0:
        frames = store.get_all_frames()
        if frames:
            first_ts = frames[0].get("timestamp_str", "?")
            last_ts = frames[-1].get("timestamp_str", "?")
            console.print(f"  Time range: {first_ts} — {last_ts}")


if __name__ == "__main__":
    app()
