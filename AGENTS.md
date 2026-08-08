# Repository Guidelines

## Project Structure & Module Organization

ScreenLens is a Python 3.11+ package for local video scene intelligence on NVIDIA DGX Spark and Apple Silicon. Core code lives in `src/`: `cli.py` exposes the Typer CLI, `pipeline.py` wires LangGraph flows, and modules such as `frame_extractor.py`, `captioner.py`, `embedder.py`, `vector_store.py`, `reconstruct.py`, and `config.py` own individual stages. The verbatim path adds `frame_select.py`, `ocr.py`, `stitch.py`, and `transcribe.py`. vLLM and oMLX share the provider-neutral `InferenceClient` in the legacy-named `omlx_client.py`; Ollama is optional. Platform launchers are `setup_and_run_dgx.sh` and `setup_and_run_macos.sh`; DGX deployment also uses `compose.dgx-spark.yaml` and `docs/DGX_SPARK.md`. Tests live in `tests/`, with end-to-end scenarios in `tests/test_cases.yaml`. Generated outputs, model caches, videos, virtual environments, and databases belong in ignored paths such as `.local-models/`, `data/`, `OUTPUT/`, `ratita/`, and `input-videos/`.

## Build, Test, and Development Commands

Install the package locally:

```bash
pip install -e .
```

Install test dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest tests/ -v
```

Run the CLI directly during development:

```bash
python -m src.cli info
python -m src.cli ingest "video.mov"
python -m src.cli search "What application is shown?"
python -m src.cli transcribe "video.mov"   # verbatim OCR path; cleanup off by default (--cleanup to enable)
```

Use provider-neutral direct-inference flags; provider-specific spellings remain aliases:

```bash
python -m src.cli ingest "video.mov" --backend vllm \
  --inference-model nvidia/Qwen3.6-27B-NVFP4 \
  --device cuda --batch-size 2
```

Run focused transcribe tests:

```bash
pytest tests/test_transcribe.py -v
```

On DGX Spark, do not use `setup_and_run_macos.sh` or a generic PyPI torch wheel. Use:

```bash
./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh smoke
./setup_and_run_dgx.sh run
```

The DGX helper creates `.venv-dgx` with Python 3.12 and pinned CUDA 13 torch/torchvision wheels. It reuses an exact-model service already owned by DigitalTwin and never stops that external stack. `setup_and_run_macos.sh` is the Apple/Conda launcher.

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, clear type hints where useful, and concise docstrings for public classes, functions, and non-obvious helpers. Prefer `Path` for filesystem work and Pydantic config models from `src/config.py` instead of scattered constants. Keep module names lowercase with underscores, test functions named `test_*`, and classes named with `CamelCase`. No formatter or linter is currently configured in `pyproject.toml`; keep imports organized and run tests before submitting.

## Testing Guidelines

Tests use `pytest`, `pytest-asyncio`, and `pyyaml` from the `dev` extra. Add focused tests beside related coverage in `tests/test_pipeline.py` or split into `tests/test_<module>.py` files as coverage grows. Mock ChromaDB, OpenCLIP, vLLM, oMLX, Ollama, and video processing so unit tests stay local and repeatable. Keep CPU explicit in portable embedding tests. `./setup_and_run_dgx.sh smoke` is the intentional live multimodal check: it must read `test.mov` from `assets/ingest-demo.png`. Avoid committing generated frames, captions, embeddings, videos, model caches, or `.venv-dgx`.

## Commit & Pull Request Guidelines

Recent history mostly follows Conventional Commit style, for example `feat(assemble): ...`, `fix(reconstruct): ...`, and `refactor(reconstruct): ...`. Use short, imperative subjects with a scope when helpful. Pull requests should include a summary, test results, linked issues when applicable, and screenshots or terminal output for user-visible changes. Note DGX/GB10, CUDA 13, large-model, or Apple Silicon assumptions.

## Agent-Specific Instructions

Keep edits narrowly scoped and preserve local artifacts, especially `.env`, `.local-models/`, and existing DigitalTwin-owned services. Do not move generated data into version control. Linux/ARM64 defaults are vLLM, CUDA, and concurrency two; Darwin/ARM64 defaults are oMLX, MPS, and concurrency four. Caption output has a 32K requested ceiling. At a matching 32K vLLM context, ScreenLens omits the literal `max_tokens` field so vLLM allocates the context remaining after prompt and image tokens instead of rejecting a zero-input reservation. Caption and reconstruction batching must remain size-budgeted: model repetition can make one caption much larger than the global average. When changing pipeline behavior, update README examples and the relevant platform guide.
