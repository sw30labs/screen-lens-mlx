"""Textual/Rich terminal UI for ScreenLens."""

from __future__ import annotations

import io
import logging
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterable
from .config import CaptionBackend, ScreenLensConfig
from .session import apply_video_slug, load_config, point_config_at_data_dir
from .omlx_client import (
    is_known_text_only_model,
    list_models,
    resolve_inference_api_key,
    resolve_inference_base_url,
    resolve_inference_model,
)


TUI_INSTALL_HINT = "Install TUI support with: pip install -e '.[tui]'"
OMLX_AUTH_HINT = "Set the selected provider's API key, then refresh models."


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


# Shared with the CLI and the web command deck — see src/session.py.
_load_config = load_config
_apply_video_slug = apply_video_slug
_point_config_at_data_dir = point_config_at_data_dir


def _model_label(config: ScreenLensConfig) -> str:
    if config.captioning.backend in (CaptionBackend.vllm, CaptionBackend.omlx):
        return f"{resolve_inference_model(config.captioning)} via {config.captioning.backend.value}"
    return f"{config.captioning.ollama_model} via Ollama"


def _summary_rows(config: ScreenLensConfig, config_path: Path | None) -> list[tuple[str, str]]:
    """Return display rows for the current configuration."""
    config_label = str(config_path.resolve()) if config_path and config_path.exists() else "defaults"
    if config.captioning.backend == CaptionBackend.ollama:
        inference_url = config.captioning.ollama_base_url
        inference_key = "n/a"
    else:
        inference_url = resolve_inference_base_url(config.captioning)
        inference_key = _yes_no(bool(resolve_inference_api_key(config.captioning)))
    return [
        ("Config", config_label),
        ("Data dir", str(config.data_dir)),
        ("Frame strategy", config.frame_extraction.strategy.value),
        ("Max interval", f"{config.frame_extraction.max_interval_seconds}s"),
        ("Caption backend", config.captioning.backend.value),
        ("Model", _model_label(config)),
        ("Batch size", str(config.captioning.batch_size)),
        ("Caption tokens", f"{config.captioning.max_tokens:,}"),
        ("Reconstruct timeout", f"{config.reconstruction.timeout_seconds:g}s"),
        ("Inference URL", inference_url),
        ("Inference key", inference_key),
        ("Embedding", f"{config.embedding.model_name} on {config.embedding.device}"),
    ]


def _omlx_model_options(model_ids: Iterable[str], configured_model: str) -> list[tuple[str, str]]:
    """Build unique Select options, hiding known text-only models."""
    unique: list[str] = []
    for model_id in model_ids:
        if model_id and not is_known_text_only_model(model_id) and model_id not in unique:
            unique.append(model_id)
    if (
        configured_model
        and not is_known_text_only_model(configured_model)
        and configured_model not in unique
    ):
        unique.insert(0, configured_model)
    return [(model_id, model_id) for model_id in unique]


def _is_omlx_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in ("api_key", "api key", "authentication", "unauthorized", "http 401")
    )


def _fetch_omlx_model_ids(config: ScreenLensConfig, timeout: float = 10.0) -> list[str]:
    """Fetch model ids from the configured direct provider."""
    model_ids = list_models(
        resolve_inference_base_url(config.captioning),
        resolve_inference_api_key(config.captioning),
        timeout,
    )
    if not model_ids:
        raise ValueError("Inference server returned no model ids")
    return model_ids


class _TextualLogHandler(logging.Handler):
    """Forward Python logs from worker threads into the Textual log widget."""

    def __init__(self, app: Any) -> None:
        super().__init__(logging.INFO)
        self.app = app
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.app.call_from_thread(self.app.write_log, self.format(record))
        except Exception:
            self.handleError(record)


class _TextualStream(io.TextIOBase):
    """Line-buffer stdout/stderr bridge for pipeline print output."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.app.call_from_thread(self.app.write_log, line)
        return len(s)

    def flush(self) -> None:
        if self._buffer.strip():
            self.app.call_from_thread(self.app.write_log, self._buffer.rstrip())
        self._buffer = ""


def run_tui(config_path: str | Path | None = None) -> int:
    """Launch the Textual app. Returns non-zero if optional dependencies are missing."""
    try:
        from rich.panel import Panel
        from rich.table import Table
        from textual.app import App, ComposeResult
        from textual.containers import Container, Horizontal, Vertical
        from textual.widgets import Button, Footer, Header, Input, Log, Select, Static
    except ImportError as exc:
        print(f"Textual/Rich TUI dependencies are not installed. {TUI_INSTALL_HINT}", file=sys.stderr)
        print(f"Missing import: {exc}", file=sys.stderr)
        return 1

    class ScreenLensTUI(App[None]):
        """Terminal GUI for common ScreenLens workflows."""

        TITLE = "ScreenLens"
        SUB_TITLE = "Local video scene intelligence"
        BINDINGS = [
            ("ctrl+i", "ingest", "Ingest"),
            ("ctrl+r", "ingest_reconstruct", "Reconstruct"),
            ("ctrl+q", "quit", "Quit"),
        ]
        CSS = """
        Screen {
            layout: vertical;
        }

        #top {
            height: auto;
            padding: 1 2;
            border-bottom: solid $primary;
        }

        .row {
            height: auto;
            margin-bottom: 1;
        }

        #config-path, #video-path, #data-dir, #output-dir {
            width: 1fr;
        }

        #backend {
            width: 18;
        }

        #omlx-model {
            width: 1fr;
        }

        Button {
            margin-left: 1;
        }

        #main {
            height: 1fr;
        }

        #summary {
            width: 42%;
            min-width: 44;
            padding: 1 2;
            border-right: solid $primary;
        }

        #activity {
            width: 1fr;
            padding: 1 2;
        }

        #status {
            height: auto;
            margin-bottom: 1;
        }

        #log {
            height: 1fr;
            border: round $surface;
        }
        """

        def __init__(self, initial_config_path: Path | None) -> None:
            super().__init__()
            self.initial_config_path = initial_config_path
            self.running = False
            self.current_config: ScreenLensConfig | None = None
            self.selected_omlx_model: str | None = None
            self.omlx_model_ids: list[str] = []
            self.models_loading = False
            self.models_refresh_attempted = False

        def compose(self) -> ComposeResult:
            yield Header()
            with Container(id="top"):
                with Horizontal(classes="row"):
                    yield Input(
                        value=str(self.initial_config_path or ""),
                        placeholder="Optional JSON config path",
                        id="config-path",
                    )
                    yield Button("Validate", id="validate", variant="default")
                    yield Button("Quit", id="quit", variant="error")
                with Horizontal(classes="row"):
                    yield Select(
                        [("vLLM (DGX Spark)", "vllm"), ("oMLX", "omlx"), ("Ollama", "ollama")],
                        value=ScreenLensConfig().captioning.backend.value,
                        allow_blank=False,
                        id="backend",
                    )
                    yield Select([], prompt="Select inference model", allow_blank=True, id="omlx-model")
                    yield Button("Refresh Models", id="refresh-models", variant="default")
                with Horizontal(classes="row"):
                    yield Input(placeholder="Video path for ingest/reconstruct", id="video-path")
                with Horizontal(classes="row"):
                    yield Input(value="./data", placeholder="./data", id="data-dir")
                    yield Input(value="./OUTPUT", placeholder="./OUTPUT", id="output-dir")
                with Horizontal(classes="row"):
                    yield Button("Ingest + Reconstruct", id="ingest-reconstruct", variant="primary")
                    yield Button("Ingest Only", id="ingest", variant="default")
                    yield Button("Reconstruct Existing", id="reconstruct", variant="default")
                    yield Button("Assemble", id="assemble", variant="default")
            with Horizontal(id="main"):
                yield Static(id="summary")
                with Vertical(id="activity"):
                    yield Static(id="status")
                    yield Log(id="log", auto_scroll=True, highlight=False)
            yield Footer()

        def on_mount(self) -> None:
            self._validate_config(write_success=False)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            button_id = event.button.id
            if button_id == "validate":
                self.action_validate_config()
            elif button_id == "quit":
                self.exit()
            elif button_id == "refresh-models":
                self.action_refresh_models()
            elif button_id == "ingest":
                self.action_ingest()
            elif button_id == "ingest-reconstruct":
                self.action_ingest_reconstruct()
            elif button_id == "reconstruct":
                self.action_reconstruct()
            elif button_id == "assemble":
                self.action_assemble()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "config-path":
                self.action_validate_config()

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id == "backend":
                self.selected_omlx_model = None
                self.omlx_model_ids = []
                self.models_refresh_attempted = False
                self._validate_config(write_success=False)
            elif event.select.id == "omlx-model" and event.value != Select.NULL:
                self.selected_omlx_model = str(event.value)

        def action_validate_config(self) -> None:
            self._validate_config(write_success=True)

        def action_refresh_models(self) -> None:
            config = self._build_config_from_controls()
            if config.captioning.backend == CaptionBackend.ollama:
                self.write_log("Model refresh is available for vLLM and oMLX.")
                return
            self._refresh_omlx_models(config)

        def action_ingest(self) -> None:
            self._start_task("ingest", self._task_ingest)

        def action_ingest_reconstruct(self) -> None:
            self._start_task("ingest+reconstruct", self._task_ingest_reconstruct)

        def action_reconstruct(self) -> None:
            self._start_task("reconstruct", self._task_reconstruct)

        def action_assemble(self) -> None:
            self._start_task("assemble", self._task_assemble)

        def _validate_config(self, *, write_success: bool) -> bool:
            try:
                config = self._build_config_from_controls()
            except Exception as exc:
                self.current_config = None
                self._update_status(f"Config error: {type(exc).__name__}: {exc}", "red")
                self._update_summary([("Status", "invalid")])
                self.write_log(f"Config error: {type(exc).__name__}: {exc}")
                return False

            self.current_config = config
            path = self._config_path()
            self._update_summary(_summary_rows(config, path))
            self._configure_omlx_selector(config)
            self._update_status("Ready", "green")
            if write_success:
                self.write_log("Configuration ready.")
            if (
                config.captioning.backend in (CaptionBackend.vllm, CaptionBackend.omlx)
                and not self.models_refresh_attempted
                and not self.models_loading
            ):
                self._refresh_omlx_models(config)
            return True

        def _config_path(self) -> Path | None:
            value = self.query_one("#config-path", Input).value.strip()
            return Path(value) if value else None

        def _build_config_from_controls(self) -> ScreenLensConfig:
            config = _load_config(self._config_path())

            backend_value = str(
                self.query_one("#backend", Select).value
                or ScreenLensConfig().captioning.backend.value
            )
            config.captioning.backend = CaptionBackend(backend_value)
            if self.selected_omlx_model:
                if config.captioning.backend == CaptionBackend.vllm:
                    config.captioning.vllm_model = self.selected_omlx_model
                elif config.captioning.backend == CaptionBackend.omlx:
                    config.captioning.omlx_model = self.selected_omlx_model

            data_dir = Path(self.query_one("#data-dir", Input).value.strip() or "./data")
            _point_config_at_data_dir(config, data_dir)
            return config

        def _configure_omlx_selector(self, config: ScreenLensConfig) -> None:
            select = self.query_one("#omlx-model", Select)
            refresh = self.query_one("#refresh-models", Button)
            if config.captioning.backend == CaptionBackend.ollama:
                select.set_options([])
                select.prompt = "Ollama model is configured separately"
                select.disabled = True
                refresh.disabled = True
                return

            configured_model = resolve_inference_model(config.captioning)
            options = _omlx_model_options(self.omlx_model_ids, configured_model)
            select.set_options(options)
            selected = self.selected_omlx_model or configured_model
            if selected not in {value for _, value in options}:
                selected = options[0][1] if options else configured_model
            if options:
                select.value = selected
            select.prompt = "Loading models..." if self.models_loading else "Select inference model"
            select.disabled = self.running or self.models_loading or not options
            refresh.disabled = self.running or self.models_loading
            self.selected_omlx_model = selected

        def _refresh_omlx_models(self, config: ScreenLensConfig) -> None:
            if self.models_loading:
                return
            self.models_refresh_attempted = True
            self.models_loading = True
            self.query_one("#refresh-models", Button).disabled = True
            self.query_one("#omlx-model", Select).prompt = "Loading models..."
            self.run_worker(
                lambda: self._load_omlx_models(config),
                name="omlx-model-refresh",
                group="omlx-model-refresh",
                exclusive=True,
                thread=True,
                exit_on_error=False,
            )

        def _load_omlx_models(self, config: ScreenLensConfig) -> None:
            try:
                models = _fetch_omlx_model_ids(config)
                self.call_from_thread(self._show_omlx_models, models, None, config)
            except Exception as exc:
                self.call_from_thread(self._show_omlx_models, [], exc, config)

        def _show_omlx_models(
            self,
            models: list[str],
            error: Exception | None,
            config: ScreenLensConfig,
        ) -> None:
            self.models_loading = False
            if config.captioning.backend == CaptionBackend.ollama:
                self._configure_omlx_selector(config)
                return

            configured_model = resolve_inference_model(config.captioning)
            if error is None:
                self.omlx_model_ids = models
            options = _omlx_model_options(self.omlx_model_ids, configured_model)
            select = self.query_one("#omlx-model", Select)
            select.set_options(options)
            model_to_select = self.selected_omlx_model or configured_model
            if model_to_select in {value for _, value in options}:
                select.value = model_to_select
                self.selected_omlx_model = model_to_select
            elif options:
                model_to_select = options[0][1]
                select.value = model_to_select
                self.selected_omlx_model = model_to_select
            if error and not self.omlx_model_ids:
                select.prompt = OMLX_AUTH_HINT if _is_omlx_auth_error(error) else "Model list unavailable"
            elif not options:
                select.prompt = "No vision-capable models found"
            else:
                select.prompt = "Select inference model"
            select.disabled = self.running or not options
            self.query_one("#refresh-models", Button).disabled = self.running

            if error:
                self.write_log(f"Could not load inference models: {type(error).__name__}: {error}")
                hint = f" {OMLX_AUTH_HINT}" if _is_omlx_auth_error(error) else ""
                self._update_status(
                    f"Ready; using {configured_model}. Model list unavailable.{hint}",
                    "yellow",
                )
                return
            filtered = len(models) - len(options)
            suffix = f" ({filtered} text-only hidden)" if filtered else ""
            self.write_log(f"Loaded {len(options)} vision-capable model(s){suffix}.")
            self.current_config = self._build_config_from_controls()
            self._update_summary(_summary_rows(self.current_config, self._config_path()))

        def _start_task(self, label: str, task: Callable[[], dict[str, Any]]) -> None:
            if self.running:
                self.write_log("A run is already in progress.")
                return
            if not self._validate_config(write_success=False):
                return
            self._set_running(True)
            self.write_log(f"Starting {label}...")
            self.run_worker(
                lambda: self._run_task(label, task),
                name=f"screenlens-{label}",
                group="screenlens-task",
                exclusive=True,
                thread=True,
                exit_on_error=False,
            )

        def _run_task(self, label: str, task: Callable[[], dict[str, Any]]) -> None:
            handler = _TextualLogHandler(self)
            stream = _TextualStream(self)
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            t0 = time.time()
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = task()
                stream.flush()
                elapsed = time.time() - t0
                self.call_from_thread(self._show_result, label, result, elapsed)
            except Exception as exc:
                stream.flush()
                self.call_from_thread(
                    self._update_status,
                    f"{label} failed: {type(exc).__name__}: {exc}",
                    "red",
                )
                self.call_from_thread(
                    self.write_log,
                    f"{label} failed: {type(exc).__name__}: {exc}",
                )
            finally:
                root_logger.removeHandler(handler)
                self.call_from_thread(self._set_running, False)

        def _task_ingest(self) -> dict[str, Any]:
            from .pipeline import build_ingest_graph

            video = self._video_path()
            config = self._build_config_from_controls()
            _apply_video_slug(config, video)
            return build_ingest_graph().invoke({
                "video_path": str(video.resolve()),
                "config": config.model_dump(),
            })

        def _task_ingest_reconstruct(self) -> dict[str, Any]:
            from .pipeline import build_ingest_graph
            from .reconstruct import reconstruct_folder

            video = self._video_path()
            config = self._build_config_from_controls()
            _apply_video_slug(config, video)
            folder = config.data_dir

            ingest_result = build_ingest_graph().invoke({
                "video_path": str(video.resolve()),
                "config": config.model_dump(),
            })
            reconstruct_result = reconstruct_folder(str(folder), config)

            if reconstruct_result.get("error"):
                return {
                    "error": reconstruct_result["error"],
                    "stage": "reconstruct_failed",
                    "num_frames": ingest_result.get("num_frames", 0),
                    "data_dir": str(folder),
                }

            return {
                "stage": "reconstructed",
                "num_frames": ingest_result.get("num_frames", 0),
                "content_type": reconstruct_result.get("content_type", "unknown"),
                "saved_paths": reconstruct_result.get("saved_paths", []),
                "qa_scores": reconstruct_result.get("qa_scores", {}),
                "data_dir": str(folder),
                "reconstruction": reconstruct_result,
            }

        def _task_reconstruct(self) -> dict[str, Any]:
            from .reconstruct import reconstruct_folder

            config = self._build_config_from_controls()
            data_dir = Path(self.query_one("#data-dir", Input).value.strip() or "./data")
            folders = self._caption_folders(data_dir)
            results = []
            for folder in folders:
                self.call_from_thread(self.write_log, f"Reconstructing {folder}")
                results.append({"folder": str(folder), "result": reconstruct_folder(str(folder), config)})
            return {"folders": len(folders), "results": results}

        def _task_assemble(self) -> dict[str, Any]:
            from .assemble import assemble_corpus

            config = self._build_config_from_controls()
            data_dir = self.query_one("#data-dir", Input).value.strip() or "./data"
            output_dir = self.query_one("#output-dir", Input).value.strip() or "./OUTPUT"
            return assemble_corpus(data_dir=data_dir, output_dir=output_dir, config=config)

        def _video_path(self) -> Path:
            raw = self.query_one("#video-path", Input).value.strip()
            if not raw:
                raise ValueError("Video path is required.")
            video = Path(raw).expanduser()
            if not video.exists():
                raise FileNotFoundError(video)
            return video

        def _caption_folders(self, data_dir: Path) -> list[Path]:
            if (data_dir / "captions" / "all_captions.json").exists():
                return [data_dir]
            folders = sorted(
                d for d in data_dir.iterdir()
                if d.is_dir() and (d / "captions" / "all_captions.json").exists()
            )
            if not folders:
                raise FileNotFoundError(f"No caption folders found under {data_dir}")
            return folders

        def _show_result(self, label: str, result: dict[str, Any], elapsed: float) -> None:
            if result.get("error"):
                self._update_status(f"{label} error: {result['error']}", "red")
                self.write_log(f"{label} error: {result['error']}")
                return

            details = []
            if "num_frames" in result:
                details.append(f"frames={result['num_frames']}")
            if "summary" in result and result["summary"]:
                details.append("summary=yes")
                self.write_log("")
                self.write_log(str(result["summary"]))
            if "content_type" in result:
                details.append(f"type={result['content_type']}")
            if "folders" in result:
                details.append(f"folders={result['folders']}")
            if "saved_paths" in result:
                details.append(f"saved={len(result['saved_paths'])}")
                for path in result["saved_paths"]:
                    self.write_log(str(path))
            if "data_dir" in result:
                details.append(f"data={result['data_dir']}")
            if "stage" in result:
                details.append(f"stage={result['stage']}")
            suffix = ", ".join(details) if details else "complete"
            self._update_status(f"{label} complete in {elapsed:.1f}s: {suffix}", "green")
            self.write_log(f"{label} complete in {elapsed:.1f}s.")

        def _set_running(self, running: bool) -> None:
            self.running = running
            for button_id in (
                "validate",
                "refresh-models",
                "ingest",
                "ingest-reconstruct",
                "reconstruct",
                "assemble",
            ):
                self.query_one(f"#{button_id}", Button).disabled = running
            self.query_one("#backend", Select).disabled = running
            no_direct_model_list = (
                self.current_config is not None
                and self.current_config.captioning.backend == CaptionBackend.ollama
            )
            self.query_one("#omlx-model", Select).disabled = running or no_direct_model_list
            self.query_one("#refresh-models", Button).disabled = running or no_direct_model_list
            if running:
                self._update_status("Running...", "yellow")

        def _update_status(self, message: str, style: str) -> None:
            self.query_one("#status", Static).update(Panel(message, title="Status", border_style=style))

        def _update_summary(self, rows: Iterable[tuple[str, str]]) -> None:
            table = Table.grid(expand=True)
            table.add_column("Field", style="bold cyan", ratio=1)
            table.add_column("Value", overflow="fold", ratio=3)
            for label, value in rows:
                table.add_row(label, value)
            self.query_one("#summary", Static).update(
                Panel(table, title="Configuration", border_style="cyan")
            )

        def write_log(self, message: str) -> None:
            self.query_one("#log", Log).write_line(message)

    initial_path = Path(config_path) if config_path else None
    ScreenLensTUI(initial_path).run()
    return 0
