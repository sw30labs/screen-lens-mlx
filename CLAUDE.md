# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScreenLens is a local video scene intelligence pipeline for NVIDIA DGX Spark and Apple Silicon. It ingests screen recordings, extracts keyframes, generates dense captions with a vision-language model, embeds them with OpenCLIP, stores them in ChromaDB, and answers natural-language queries. Linux/ARM64 defaults to vLLM + CUDA with two concurrent image requests; Darwin/ARM64 defaults to oMLX + MPS with four. A second pipeline (`reconstruct`) rebuilds visible Python files, Markdown, PDFs, and GUI walkthroughs. A third pipeline (`transcribe`) OCRs frames character-for-character and stitches scrolling overlap in text space.

## Common Commands

```bash
# Install (editable)
pip install -e .

# Ingest with native platform defaults (DGX: vLLM/CUDA; Apple: oMLX/MPS)
python -m src.cli ingest "video.mov"

# Optional Ollama captioning fallback
python -m src.cli ingest "video.mov" --backend ollama --strategy fixed_fps --fps 1.0

# Provider-neutral direct inference flags; --vllm-* and --omlx-* are aliases
python -m src.cli ingest "video.mov" --backend vllm \
  --inference-model nvidia/Qwen3.6-27B-NVFP4 --device cuda --batch-size 2

# Batch-ingest a folder (each video gets its own ./data/<slug>/ collection)
python -m src.cli batch "/path/to/recordings/"

# Search the most recently configured collection
python -m src.cli search "What application is being demonstrated?"

# Ingest + search in one shot
python -m src.cli run "video.mov" "Summarize what happens"

# Full-video summary from all captions (single-pass or hierarchical chunking)
python -m src.cli summarize

# Reconstruct artifacts from all ingested folders under ./data/
python -m src.cli reconstruct

# Verbatim transcription (frame_select → vision OCR → text-space stitch)
# LLM seam/indent cleanup is OFF by default; opt in with --cleanup.
python -m src.cli transcribe "video.mov"
python -m src.cli transcribe "code.mov" --deterministic   # Apple Vision cross-check (needs: pip install ocrmac)
python -m src.cli transcribe "doc.mov" --cleanup          # run the optional LLM seam/indent pass

# List models from the selected vLLM/oMLX endpoint
python -m src.cli models

# Web command deck (local dashboard over every pipeline)
python -m src.cli serve                      # http://127.0.0.1:8760
python -m src.web --port 9000 --no-browser

# Vector store stats
python -m src.cli info

# Tests
pytest tests/ -v
pytest tests/test_pipeline.py::TestEmbedder -v        # one class
pytest tests/test_pipeline.py::TestEmbedder::test_embed_text -v   # one test

# DGX Spark bootstrap, validation, and launcher
./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh smoke
./setup_and_run_dgx.sh run ingest input-videos/demo.mov
```

`ffmpeg` must be on PATH. On DGX Spark use `setup_and_run_dgx.sh` and `docs/DGX_SPARK.md`; the helper creates `.venv-dgx` with Python 3.12 and the checked CUDA 13 torch/torchvision wheels, then starts or reuses the exact NVIDIA Qwen model. On Apple Silicon use `setup_and_run_macos.sh` plus oMLX, with `MLX_API_KEY`/`OMLX_API_KEY` only when authentication is enabled. Ollama is optional and requires `ollama pull llama3.2-vision` only when selected.

## Architecture

The codebase has two LangGraph `StateGraph` pipelines plus the straight-line transcribe path. They share one `ScreenLensConfig` and the provider-neutral `InferenceClient` in the legacy-named `omlx_client.py` module.

Three front ends drive those pipelines — `cli.py` (Typer), `tui.py` (Textual), and `web/` (the command deck). All three build on `session.py`, which owns config loading, per-video slug allocation, role-aware inference wiring, and read-only run discovery. Put anything a second front end would otherwise copy there; the config/slug helpers were duplicated between the CLI and TUI before it existed.

### Model roles (`src/session.py`)

Two roles resolve separately against one endpoint:

- **vision** — captioning (`captioner.py`) and OCR (`ocr.py`). Must be vision-capable.
- **text** — search/full-video summaries, reconstruction plan/QA, transcript cleanup. Never sees an image.

`text_role_captioning_config()` builds the text-role client from `config.reconstruction`; `pipeline.py` and `reconstruct.py` both go through it, so text work never lands on the vision model. On Spark both roles resolve to the single served vLLM checkpoint, making the split a no-op there. On Apple the oMLX defaults are `Qwen3.6-27B-bf16` for vision and `DeepSeek-V4-Flash-0731-MLX` for text.

### Front end 3 — Web command deck (`src/web/`)

Stdlib `http.server` only — the same ADR-008 pattern as contingency-atlas and book-buddy-2026: hand-rolled JSON handlers, one static page, no framework and no build step, so the GUI adds no dependency.

- `server.py` — loopback-only bind plus a per-request loopback check on every `/api/` route. Run slugs, artifact names and frame names are contained *by name* (no separators, no dotfiles, resolved parent must match) rather than resolved, and frames/artifacts are restricted to known-safe suffixes so an `.env` sitting beside a transcript is not readable.
- `runner.py` — one job at a time behind a lock (the pipelines share one model endpoint and one CLIP device). Pipeline `print()`/`logging` output is captured into a ring buffer the page tails, the same approach `tui.py` uses. `set_pipeline_override` is the test seam: `tests/test_web.py` exercises the whole job lifecycle without a model, GPU, or ffmpeg.
- `static/index.html` — the SPA. If you add an endpoint, `test_index_calls_only_real_endpoints` will catch a page that calls something the server does not serve.

### Pipeline 1 — Ingest / Search (`src/pipeline.py`)

`StateGraph` over a `ScreenLensState` TypedDict. Three graph builders:

- `build_ingest_graph()`  — `ingest → caption → embed`
- `build_search_graph()`  — `search → summarize`
- `build_full_graph()`    — both chained end-to-end

Per-stage modules:

| Node | Module | Purpose |
|---|---|---|
| `ingest_node` | `frame_extractor.py` | Hybrid keyframe detection (SSIM + pHash + HSV histogram, OpenCV) or fixed-FPS fallback. Decision logic in `_extract_keyframes` — emits when any signal trips AND `min_interval_seconds` has elapsed, or unconditionally every `max_interval_seconds`. |
| `caption_node` | `captioner.py` | Backend factory (`_get_captioner`). `OpenAICompatibleCaptioner` serves vLLM and oMLX; `OllamaCaptioner` is the optional fallback. |
| `embed_node` | `embedder.py`, `vector_store.py` | OpenCLIP `ViT-B-32` on CUDA, MPS, or CPU, then writes embeddings + metadata into ChromaDB. |
| `search_node` | `vector_store.py` | Encodes query text via the same CLIP, ChromaDB cosine search. |
| `summarize_node` | `pipeline.py` | Summarizes top-k results through the configured vLLM/oMLX client; uses ChatOllama only when Ollama was explicitly selected. |
| `summarize_all_node` | `pipeline.py` | Full-video summary through the selected direct backend, using its configured context for single-pass vs. hierarchical chunking. |

State flows by returning partial dicts that LangGraph merges; `elapsed_seconds` accumulates per-stage timings.

### Pipeline 2 — Reconstruct (`src/reconstruct.py`)

A more sophisticated graph: `classify → plan → (parallel workers | sequential) → qa_reflect → save`, with a retry edge from `qa_reflect → plan` that can loop up to `MAX_QA_ITERATIONS = 3`.

Key mechanics worth knowing before editing:

- **Parallel fan-out via `Send`.** `route_to_workers` returns either a list of `Send("reconstruct_worker", ...)` payloads (parallel) or the literal string `"reconstruct_sequential"` (single node). Parallel is only chosen when `parallel_safe=True` AND there is more than one task. The planner sets `parallel_safe` per content type — Python files only when the LLM judges them independent (no cross-imports), GUI demos always (walkthrough + reference are independent), Markdown/PDF never.
- **Reducer for collecting sub-agent outputs.** `artifacts: Annotated[list[dict], operator.add]` lets multiple `Send`-dispatched workers append their results without clobbering each other. Each artifact carries an `iteration` field so `qa_reflect_node` and `save_node` can distinguish current-iteration outputs from prior-iteration ones.
- **Client cache.** `_MODEL_CACHE` is keyed by provider, endpoint, and model so the direct inference client is reused across reconstruction nodes.
- **Reflection feedback loop.** When `qa_reflect_node` fails, it stores `qa_feedback` and increments `qa_iteration`; the next pass through `plan_node` injects `PREVIOUS QA FEEDBACK` into each task prompt. After `MAX_QA_ITERATIONS - 1`, QA force-passes to avoid infinite loops.
- **JSON parsing.** LLM JSON responses are parsed via `parse_json_response`, which tries direct → fenced → first `{...}` block. Always use this helper rather than `json.loads` directly on model output.

### Pipeline 3 — Transcribe (`src/transcribe.py`)

The verbatim path. Not a `StateGraph` — a straight function pipeline:
`select_frames → VerbatimOCR → stitch_frames → (optional) LLM cleanup → output/transcript.md`.

| Stage | Module | Purpose |
|---|---|---|
| frame select | `frame_select.py` | Dense sample (default 2 fps), drop ONLY near-exact static duplicates (SSIM > 0.992). On scrolling text, pixel metrics can't tell "new content" from "same content shifted", so the real dedup happens later in text space. |
| verbatim OCR | `ocr.py` | `VerbatimOCR` sends each frame to the selected **vision** model and copies text character-for-character. A hard capability guard and live image probe reject text-only deployments; Apple Vision (`ocrmac`) is an optional macOS-only backstop. |
| stitch | `stitch.py` | Text-space dedup of scroll overlap (fuzzy line canonicalization + difflib matching-blocks), strips page headers/footers. Writes `transcript.raw.md`. |
| cleanup (optional) | `transcribe.py` | LLM seam/indent repair only. **Off by default.** |

Key mechanics worth knowing before editing:

- **Thinking is disabled for OCR and cleanup.** Both pass `chat_template_kwargs={"enable_thinking": false}` (gated by `OCRConfig.disable_thinking` / `ReconstructionConfig.disable_thinking`, default true). A reasoning model used for verbatim OCR otherwise spends the entire `max_tokens` budget on chain-of-thought and never emits the transcription. `omlx_client.strip_thinking` also strips complete/dangling `<think>` blocks, and `_post_chat` warns on `finish_reason == "length"`.
- **Cleanup is OFF by default** (`ReconstructionConfig.enabled = False`; CLI opt-in via `--cleanup`). The raw stitched OCR is already verbatim; an LLM tends to drop content while "repairing".
- **Cleanup coverage guard.** When cleanup runs, chunk input is bounded by the output token cap (not just the context window) so a chunk can't truncate mid-output. After each chunk, `_chunk_coverage` checks the fraction of distinct input lines that survived; below `MIN_CHUNK_COVERAGE` (0.97) the LLM output is discarded and the **raw stitched chunk is kept** — `transcript.md` can never lose content vs. `transcript.raw.md`.
- **Role-specific model resolution.** On Spark, OCR and cleanup default to the single `VLLM_MODEL` (`nvidia/Qwen3.6-27B-NVFP4`). On Apple, OCR and cleanup retain their `OCR_MODEL` and `LLM_MODEL`/`MLX_MODEL` resolution. Each role gets a separately configured `InferenceClient` built via `from_endpoint`.

### Configuration (`src/config.py`)

A single `ScreenLensConfig` Pydantic model composed of `FrameExtractionConfig`, `CaptioningConfig`, `EmbeddingConfig`, `VectorDBConfig`, `SearchConfig`, plus `OCRConfig`, `FrameSelectionConfig`, and `ReconstructionConfig`. `CaptionBackend` is `vllm | omlx | ollama`; direct OCR/reconstruction uses `InferenceBackend` (`vllm | omlx`). The CLI mutates the config from flags before passing `config.model_dump()` into graph state.

Use `--inference-url`, `--inference-model`, and `--inference-api-key` for direct endpoints. `--vllm-*` and `--omlx-*` are aliases. Environment resolution is provider-specific (`VLLM_*` versus `MLX_*`/`OMLX_*`), and `SCREENLENS_BACKEND`, `SCREENLENS_DEVICE`, and `SCREENLENS_BATCH_SIZE` override platform defaults. Caption `max_tokens` defaults to a 32K requested ceiling. When that equals the vLLM context, the client omits the literal field so vLLM uses the exact context remaining after image and prompt tokens. Reconstruction chunks by serialized caption size, not a global average or fixed frame count; keep the outlier-caption tests when changing token planning.

### Data Layout

Each ingested or transcribed video gets a fresh folder under `./data/<stem>_<YYYYMMDD_HHMMSS>/`:

```
data/<slug>/
  frames/                 # extracted keyframe / sampled JPGs (PNGs for transcribe)
  captions/
    caption_NNNNNN.json   # one per frame (ingest path)
    all_captions.json     # combined — what reconstruct/summarize read
  ocr/
    ocr_NNNNNN.json       # one per frame (transcribe path)
    all_ocr.json          # combined verbatim OCR
  chromadb/               # per-video persistent collection
  output/                 # reconstruct.py and transcribe.py write here
    reconstruction_meta.json
    transcript.raw.md     # stitched verbatim OCR (faithful, no LLM)
    transcript.md         # == raw unless --cleanup ran
    transcribe_meta.json
```

`ingest`, `run`, `batch`, and `transcribe` all derive timestamped slugs, so repeated runs do not overwrite earlier data. `search --data-dir ./data` can query every discovered collection.

## Things to Watch For

- **CLIP device** defaults to CUDA on Linux/ARM64, MPS on Darwin/ARM64, and CPU elsewhere. Tests pin it to CPU so they remain portable.
- **`data/` is gitignored** along with `*.mov`/`*.mp4` and other video formats — don't try to commit fixtures.
- **Context planning is provider-specific.** Spark uses `vllm_model_context`; Apple uses `omlx_model_context`. Keep it aligned with the serving limit.
- **Transcribe OCR must use a vision model.** Spark defaults to the checked NVIDIA Qwen checkpoint; Apple retains its recommended oMLX OCR model. A text-only choice aborts via the capability guard + live probe.
- **`disable_thinking` stays on for OCR/cleanup.** Turning it off with a reasoning model regresses to truncated, all-reasoning output (see `tests/test_transcribe.py::test_strip_thinking_handles_truncated_open_tag`).
- **Don't size cleanup chunks past the output token cap.** Chunk input is intentionally bounded by `max_tokens`, not just `model_context` — otherwise output truncates mid-chunk and silently loses content.
- **Caption token budget is not free.** `CaptioningConfig.max_tokens` defaults to the full context. On a dense frame the model keeps generating past the real content and degenerates into a repetition loop (observed: `'E0000000…'` for thousands of tokens, tens of minutes per frame on oMLX). The repetition guards are honoured by both servers but do not bound runtime — a bounded budget does. The command deck exposes it per run and defaults to 1500.
- **Text work must not land on the vision model.** Route it through `session.text_role_captioning_config()`, never `InferenceClient(config.captioning)` directly.
- **Vet a text model against a real caption.** Captions are markdown with indented list items. `DeepSeek-V4-Flash-0731-MLX` under oMLX degenerates into its BOS token on exactly that shape (`\n   - ` is the minimal reproducer; length, backticks and the system turn are all irrelevant), so it cannot serve the text role even though it is a capable reasoner. `omlx_client.degenerate_repetition` catches the shape, the client warns, and summaries raise `InferenceDegenerateError` rather than saving it — never weaken that into a silent pass.
- **DGX concurrency stays at two.** The bundled vLLM service admits two sequences and shares one 128 GB unified-memory pool with OpenCLIP and the OS. Do not raise it casually. oMLX may serialize requests for a loaded model; tune Apple concurrency to the host.
- **Do not start two port-8000 stacks.** The Spark helper reuses DigitalTwin's exact-model endpoint and only manages containers created by this repository. See `docs/DGX_SPARK.md`.

## Coding Guidelines
Behavioral guidelines to reduce common LLM coding mistakes, on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
