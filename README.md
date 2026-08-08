# ScreenLens

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![DGX Spark](https://img.shields.io/badge/NVIDIA_DGX_Spark-CUDA-76B900?logo=nvidia&logoColor=white)](docs/DGX_SPARK.md)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-000000?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Pipeline-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)]()
[![Author](https://img.shields.io/badge/Author-Nicolas_Cravino-orange)](https://github.com/ai-agents-cybersecurity)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

Local video scene intelligence for NVIDIA DGX Spark and Apple Silicon. ScreenLens extracts meaningful frames from screen recordings, captions or transcribes them with a local multimodal model, embeds them with OpenCLIP, indexes them in ChromaDB, and reconstructs visible code, documents, PDFs, and GUI walkthroughs. Linux/ARM64 uses vLLM + CUDA by default; Apple Silicon uses oMLX + MPS. Ollama remains an optional fallback.

## Demo

![ScreenLens ingestion pipeline](assets/ingest-demo.png)

## Architecture

```mermaid
graph TD
    A[Video Input<br/>.mov / .mp4] --> B[Hybrid Keyframe Detection<br/>SSIM + pHash + HSV]
    B --> C{Local vision backend}
    C -->|DGX Spark| D[vLLM<br/>NVIDIA Qwen3.6 NVFP4]
    C -->|Apple Silicon| E[oMLX<br/>configured multimodal model]
    C -->|optional| F[Ollama vision]
    D --> G[Captions / Verbatim OCR]
    E --> G
    F --> G
    G --> H[OpenCLIP Embeddings<br/>CUDA / MPS / CPU]
    H --> I[ChromaDB Search]
    G --> J[LangGraph Reconstruction<br/>plan → workers → QA]
    J --> K[Reconstructed Artifacts<br/>data/*/output]
```

The vLLM and oMLX paths share the same small OpenAI-compatible client, including image data URLs and `chat_template_kwargs.enable_thinking=false` for Qwen-style models. Platform detection changes defaults, not the pipeline itself.

## Component Overview

| Component | Technology | Purpose |
|---|---|---|
| Frame extraction | Hybrid SSIM + pHash + HSV detection | Capture distinct screens while bounding long static intervals |
| DGX vision/text | `nvidia/Qwen3.6-27B-NVFP4` via vLLM | Caption frames, perform OCR, summarize, and reconstruct through one dense local model |
| Apple vision/text | Configured multimodal model via oMLX | Native Apple Silicon OpenAI-compatible inference |
| Optional fallback | Ollama | Alternative captioning and summarization when explicitly selected |
| Visual embeddings | OpenCLIP ViT-B-32 | Semantic image/text vectors on CUDA, MPS, or CPU |
| Vector storage | ChromaDB | Persistent per-video semantic search |
| Reconstruction | LangGraph | Classify recordings and rebuild code/docs/demo artifacts with QA retries |
| Interfaces | Typer + Rich, optional Textual | CLI and terminal GUI |

## Platform Defaults

| Host | Inference | Embeddings | Concurrent image requests |
|---|---|---|---:|
| Linux ARM64 / DGX Spark | vLLM | CUDA | 2 |
| Darwin ARM64 / Apple Silicon | oMLX | MPS | 4 |
| Other hosts | oMLX unless overridden | CPU | 4 |

Caption output is capped at 32,768 tokens by default. The bundled dense Spark service exposes a 262,144-token total context, leaving ample prompt and image-token headroom while reconstruction can use the full served window.

## Installation

### NVIDIA DGX Spark

The checked helper creates an isolated Python 3.12 CUDA environment, validates the host and OpenCLIP, starts or reuses the exact vLLM model, and performs a real vision smoke test:

```bash
(umask 077; touch .env)
chmod 600 .env
${EDITOR:-nano} .env  # add HF_TOKEN=hf_... if ScreenLens must start vLLM

./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh llm-wait
./setup_and_run_dgx.sh smoke
./setup_and_run_dgx.sh run
```

`run` launches the TUI with no arguments or passes a CLI command through:

```bash
./setup_and_run_dgx.sh run ingest input-videos/demo.mov
./setup_and_run_dgx.sh run transcribe input-videos/demo.mov
```

DigitalTwin uses the same model and loopback port. The helper reuses an already-ready exact-model service instead of starting a conflicting container. See [the complete DGX Spark guide](docs/DGX_SPARK.md) for CUDA wheel pins, cache sharing, memory sizing, service ownership, and troubleshooting.

### Apple Silicon / Conda

Start oMLX on `http://127.0.0.1:8000/v1`, configure `MLX_*` or `OMLX_*` values in `.env`, then use the existing Conda launcher:

```bash
./setup_and_run_macos.sh
./setup_and_run_macos.sh ingest "video.mov"
```

The script creates a `screenlens` Conda environment with Python 3.11 and ffmpeg, installs the TUI extra, preserves an existing `.env`, and defaults to the TUI. Set `SCREENLENS_CONDA_ENV` to choose another environment name. On Linux/ARM64 it exits with directions to the checked Spark helper so a generic pip install cannot replace CUDA 13 wheels with CPU wheels.

### Manual development install

```bash
pip install -e ".[dev,tui]"
pytest tests/ -v
```

Install ffmpeg separately and ensure the selected inference server is running. Ollama is not required for the normal vLLM or oMLX paths.

## Usage

### Provider Selection

Commands that contact the direct inference service accept provider-neutral flags:

```bash
python -m src.cli ingest video.mov \
  --backend vllm \
  --inference-url http://127.0.0.1:8000/v1 \
  --inference-model nvidia/Qwen3.6-27B-NVFP4 \
  --inference-api-key local \
  --device cuda \
  --batch-size 2
```

`--vllm-url`, `--vllm-model`, and `--vllm-api-key` are provider-specific aliases; the existing `--omlx-*` spellings remain compatible. `SCREENLENS_BACKEND`, `SCREENLENS_DEVICE`, and `SCREENLENS_BATCH_SIZE` override platform defaults. Provider credentials and models can also come from `VLLM_*`, `MLX_*`, or `OMLX_*` variables as appropriate.

### Model Roles

ScreenLens drives two roles against one OpenAI-compatible endpoint:

| Role | Used by | Requirement | oMLX default |
|---|---|---|---|
| **vision** | captioning, verbatim OCR | must be vision-capable — a text-only model answers "no image provided" for every frame | `Qwen3.6-27B-bf16` (`OCR_MODEL` / `MLX_MODEL`) |
| **text** | summaries, reconstruction plan/QA, transcript cleanup | never sees an image; a reasoning model is the right pick | `DeepSeek-V4-Flash-0731-MLX` (`LLM_MODEL`) |

On DGX Spark both resolve to the single served vLLM checkpoint, so the split
costs nothing there. On Apple Silicon they are normally two different oMLX
models loaded side by side. Check what your endpoint serves and which side of
the line each model falls on:

```bash
python -m src.cli models
```

The capability guard aborts a `transcribe` run whose OCR model is text-only,
and the command deck's vision selector only offers models that can see.

### Ingest and Search

```bash
# Uses vLLM/CUDA on DGX Spark or oMLX/MPS on Apple Silicon.
python -m src.cli ingest "video.mov"

# Optional Ollama captioning fallback.
python -m src.cli ingest "video.mov" --backend ollama --strategy fixed_fps --fps 1

# Search one or all timestamped collections and synthesize an answer locally.
python -m src.cli search "What application is shown?" --data-dir ./data --top-k 5

# Ingest and search in one command.
python -m src.cli run "video.mov" "Summarize the workflow"
```

Search summarization follows the configured platform backend; it does not require Ollama unless Ollama was explicitly selected. Each ingestion creates `data/<stem>_<YYYYMMDD_HHMMSS>/` with independent frames, captions, and ChromaDB data.

### Batch Ingestion and Full-Video Summary

```bash
python -m src.cli batch input-videos/
python -m src.cli summarize
```

Batch ingestion gives every video its own timestamped directory. Full-video summary reads all stored captions and chooses a single-pass or hierarchical strategy based on the selected backend's configured context window.

### Reconstruct and Assemble Artifacts

```bash
python -m src.cli reconstruct
python -m src.cli assemble
```

Reconstruction classifies each recording, plans the artifact work, processes it deterministically, and runs up to three QA iterations before saving under `data/<slug>/output/`. Assembly can combine per-recording outputs into one project tree under `OUTPUT/`.

### Verbatim Transcription

```bash
# Discover served models and their likely vision capability.
python -m src.cli models

# OCR frames, stitch scrolling overlap, and preserve raw text.
python -m src.cli transcribe input-videos/policies.mov

# Optional seam/indent cleanup; guarded against dropped content.
python -m src.cli transcribe input-videos/document.mov --cleanup

# Apple-only deterministic Vision cross-check for code.
python -m src.cli transcribe input-videos/code.mov --deterministic
```

The transcribe path copies visible text rather than describing it. A live image probe rejects blind/text-only deployments before a full run. Cleanup is off by default; when enabled, a per-chunk coverage guard keeps the raw stitched chunk whenever the model drops content. Outputs are `transcript.raw.md`, `transcript.md`, `ocr/all_ocr.json`, and metadata under the timestamped data directory.

### Web command deck

```bash
python -m src.cli serve                 # http://127.0.0.1:8760, opens a browser
python -m src.cli serve --port 9000 --no-browser
python -m src.web                       # same thing without Typer
```

A local dashboard over every pipeline: run register, frame gallery served
straight off disk, semantic search, pipeline control, and an artifact reader.
It is stdlib `http.server` plus one static page — no framework, no build step,
and no dependency beyond what ScreenLens already installs.

It binds loopback-only and refuses non-loopback clients on every `/api/` route,
because it starts jobs and reads frames off disk. One job runs at a time: the
pipelines share a single model endpoint and one CLIP device.

The header shows both model roles at once. If the configured vision model is
not vision-capable the badge turns red before you waste a run on it, and the
vision selector only offers models that can actually see.

### Status and TUI

```bash
python -m src.cli info
python -m src.cli tui
```

The TUI detects the native platform default, lists models from the selected OpenAI-compatible endpoint, and supports ingest, reconstruct, and combined workflows.

## Keyframe Detection

The hybrid change detector uses three complementary signals to decide when the screen has actually changed:

| Signal | What it detects | Threshold |
|---|---|---|
| SSIM | Pixel-level structural changes | < 0.97 |
| pHash | Perceptual content changes via DCT | hamming >= 8 |
| HSV histogram | Color-distribution shifts | correlation <= 0.90 |

A keyframe is emitted when any signal triggers and the minimum 0.5-second interval has elapsed. A forced keyframe every four seconds catches slow scrolling. This typically captures a small fraction of source frames while preserving distinct screens.

## Configuration

All settings live in `src/config.py` as Pydantic models. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `frame_extraction.strategy` | keyframe | Smart change detection or `fixed_fps` |
| `frame_extraction.max_interval_seconds` | 4.0 | Maximum gap between keyframes |
| `captioning.backend` | platform-dependent | `vllm`, `omlx`, or optional `ollama` |
| `captioning.vllm_base_url` | http://127.0.0.1:8000/v1 | DGX Spark OpenAI-compatible API URL |
| `captioning.vllm_model` | null | Falls back to `VLLM_MODEL`, then the checked NVIDIA Qwen model |
| `captioning.omlx_base_url` | http://127.0.0.1:8000/v1 | Apple oMLX URL; root/dashboard URLs are normalized |
| `captioning.omlx_model` | null | Falls back to `MLX_MODEL`/`OMLX_MODEL`/`LLM_MODEL`, then `default` |
| `captioning.batch_size` | 2 DGX / 4 Apple | Concurrent direct-server caption requests |
| `captioning.max_tokens` | 32768 | Requested caption ceiling; vLLM uses the context remaining after image/prompt tokens |
| `captioning.retry_attempts` | 1 | Per-frame retries after a direct caption request fails |
| `captioning.retry_max_tokens` | 2048 | Bounded retry ceiling that prevents malformed generations from consuming another full caption budget |
| `captioning.repetition_penalty` | 1.05 | Discourage pathological long caption loops |
| `captioning.no_repeat_ngram_size` | 12 | Block repeated long caption sequences; `0` disables |
| `captioning.disable_thinking` | true | Spend the output budget on the visible answer |
| `embedding.model_name` | ViT-B-32 | OpenCLIP architecture |
| `embedding.device` | CUDA DGX / MPS Apple | Accelerator for OpenCLIP; CPU elsewhere |
| `ocr.backend` | platform-dependent | Direct `vllm` or `omlx` vision endpoint |
| `ocr.concurrency` | 2 DGX / 4 Apple | Concurrent verbatim OCR requests |
| `reconstruction.backend` | platform-dependent | Direct text inference endpoint |
| `reconstruction.timeout_seconds` | 1800 | Timeout for long-running reconstruction generations |

## Performance Notes

Direct backends receive one OpenAI-compatible image request per frame. Image encoding and prompt prefill usually dominate short captions, so the main levers are frame dimensions, model size, and request concurrency. The bundled Spark service admits two sequences and ScreenLens therefore defaults to two requests there. Apple defaults to four, but large oMLX models may benefit from a lower value. Concurrent results are isolated per frame: one failed request cannot overwrite a successful peer in the same chunk. A failed frame is retried once deterministically with a bounded 2K completion ceiling. A retry that fills that ceiling without terminating is rejected instead of storing a truncated loop; ScreenLens stores an error marker for that frame alone.

DGX Spark has one 128 GB unified-memory pool rather than separate system RAM and VRAM. The checked single-service recipe uses the dense NVFP4 27B model, a `0.45` vLLM allocator target, a 262K context, and concurrency two. Caption requests retain a 32K output ceiling, leaving the rest of the served window for prompt, chat-template, and image tokens. If a service is deliberately reduced to a matching 32K context, ScreenLens omits the literal output limit and lets vLLM allocate the exact context remaining after its input.

Reconstruction groups captions by serialized size rather than frame count, so an unusually long caption cannot overfill a later prompt. Its shared extraction pass requests dense notes under 1,400 tokens but gives the model the full context exposed by the server as completion headroom. Recursive synthesis filters notes to the file or artifact currently being rebuilt and uses that same full ceiling. These long text generations use the independent 1,800-second `reconstruction.timeout_seconds` budget instead of the shorter caption timeout. If a response still ends with `finish_reason=length`, ScreenLens discards the incomplete prefix and retries a smaller input group. The 262K service context is therefore used automatically when `VLLM_MAX_MODEL_LEN` matches it. Native two-token Qwen MTP accelerates decoding without changing answers; DFlash is not loaded because it would add a second draft model and does not increase context or output quality. See [the Spark memory notes](docs/DGX_SPARK.md#memory-and-performance-notes).

`ffprobe` remains optional: when it is absent, video metadata is read through OpenCV without a warning. OpenCLIP's checked checkpoint is public, so ScreenLens suppresses only the Hub's unauthenticated-download advisory; actual authorization, rate-limit, and download failures remain visible. A Hugging Face token is still required when the DGX helper must start the gated vLLM model service.

## Project Structure

```text
compose.dgx-spark.yaml # Bounded ARM64/CUDA vLLM service
docs/
  DGX_SPARK.md         # Setup, sizing, reuse, lifecycle, troubleshooting
setup_and_run_dgx.sh   # Checked Spark setup, validation, and launcher
setup_and_run_macos.sh # Apple Silicon Conda/oMLX launcher
src/
  config.py            # Platform-aware Pydantic configuration
  frame_extractor.py   # Hybrid keyframe detection + fixed-FPS fallback
  captioner.py         # vLLM, oMLX, and optional Ollama captioning
  embedder.py          # OpenCLIP embeddings on CUDA/MPS/CPU
  vector_store.py      # ChromaDB storage and search
  pipeline.py          # LangGraph ingest/search/summarize graphs
  reconstruct.py       # Artifact reconstruction with QA reflection
  frame_select.py      # Scroll-safe dense sampling for transcription
  ocr.py               # Verbatim vision OCR and capability probe
  stitch.py            # Text-space scroll-overlap stitching
  transcribe.py        # Verbatim pipeline and guarded cleanup
  omlx_client.py       # Shared inference client; legacy module name retained
  session.py           # Shared config/slug/run-discovery layer for all front ends
  cli.py               # Typer CLI
  tui.py               # Optional Textual terminal GUI
  web/                 # Web command deck (stdlib HTTP, no build step)
    server.py          # Loopback-only JSON API + static page
    runner.py          # One-job-at-a-time background runner
    static/index.html  # Single-file SPA
tests/
  test_pipeline.py     # Core configuration, inference, embedding, and graph tests
  test_transcribe.py   # Stitching, OCR guards, and cleanup safety
  test_web.py          # Command deck assets, API contracts, and security
  test_cases.yaml      # Dual-platform/DGX end-to-end scenarios
```

Generated frames, captions, databases, videos, outputs, virtual environments, and model caches are ignored by Git.

## How It Compares to NVIDIA VSS

| Feature | NVIDIA VSS | ScreenLens |
|---|---|---|
| Frame extraction | Custom + TensorRT | Hybrid SSIM/pHash/HSV detection |
| Vision model | NVIDIA VILA | NVIDIA Qwen3.6 via vLLM or a configured oMLX model |
| Embeddings | TensorRT visual encoder | OpenCLIP ViT-B-32 |
| Vector DB | Milvus | ChromaDB |
| LLM | Llama 3.1 70B / NIM | Same local vLLM/oMLX endpoint used by the pipeline |
| Hardware | NVIDIA GPU | DGX Spark or Apple Silicon |
| Deployment | Docker + NIM | Bounded vLLM container + Python, or native oMLX + Python |
| Cloud dependency | None when self-hosted | None; local model endpoints only |

## Roadmap

### Duplicate Detection

- [ ] Harden near-duplicate keyframe filtering (perceptual hash + SSIM fusion threshold tuning)
- [ ] Cross-video deduplication for multi-file ingestion
- [ ] Consider leveraging [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to iterate on dedup thresholds against a fixed evaluation set

### Video Profiles

Preconfigured extraction and captioning strategies tailored to content type:

| Profile | Description | Audio | Typical Source |
|---|---|---|---|
| `code` | Silent screen recording of browsing/editing code | No | IDE walkthroughs, code reviews |
| `demo` | Screencast with voice-over demonstrating software | Yes | Product demos, tutorials, onboarding |
| `pdf` | Continuous scroll/browse of a PDF | No | Recorded read-throughs, slide decks |
| `meeting` | Video call or presentation | Yes | Zoom/Teams recordings, webinars |

Each profile will tune frame extraction, captioning prompts, chunking, and audio processing.

### Audio Support

- [ ] Integrate Whisper speech-to-text via ONNX Runtime and/or MLX
- [ ] Support `small`, `medium`, and `large` model sizes
- [ ] Add word-level timestamps aligned to the keyframe timeline
- [ ] Fuse captions and transcripts for richer semantic search

### Output Generators

- [ ] Manual generator with extracted screenshots and navigation flow
- [ ] PDF summary preserving headings, key points, and figures
- [ ] Source-code reconstruction with files, signatures, and project structure
- [ ] Meeting notes with action items, decisions, and speaker attribution
