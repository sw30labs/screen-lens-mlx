# ScreenLens redesign — verbatim transcription path

## TL;DR of why it was failing

The pipeline was sound on paper but **the OCR step was reading every frame
blind.** `.env` had `MLX_MODEL=MiniMax-M2.7`, a **text-only** model, and the
captioner sent it images. MiniMax can't see pixels, so all 173 captions came
back *"No image or video frame has been provided"* — and every downstream stage
(embedding, reconstruction) faithfully processed that emptiness. The hybrid
SSIM/pHash keyframe logic and the difflib dedup were never the bottleneck.

A second, independent confirmation came from measuring pixel metrics on the real
frames: on scrolling dense text, **SSIM/pHash cannot tell "new content" from
"same content shifted up a few rows"** — overlap-SSIM stays ~0.5–0.7 even for a
clean scroll. That is why deterministic frame dedup "failed": it was the wrong
tool for the job. Pixels reliably detect exactly one thing here — *near-exact
static duplicates*.

## The new approach (verbatim, scroll-aware)

```
video.mov
  → frame_select.py   sample densely (2 fps), drop ONLY near-exact dupes
  → ocr.py            verbatim OCR via a VISION model (transcribe, don't describe)
  → stitch.py         text-space dedup: fuzzy-canonicalize lines, difflib
                      matching-blocks find the scroll overlap, splice the new tail,
                      strip page headers/footers, majority-vote OCR flicker
  → transcribe.py     optional text-LLM cleanup (OFF by default, coverage-guarded)
  → data/<slug>/output/transcript.md  (== transcript.raw.md unless --cleanup)
```

Two models, separated by job:

| Job | Model | Why |
|-----|-------|-----|
| **Read frames (OCR)** | a **vision** model — default `Qwen3.6-27B-bf16` | purpose-built for dense document fidelity; never a text-only model again |
| **Clean seams (LLM)** | a **text** model — any capable model that fits the memory ceiling | it never sees images; tidying stitched text is what it's good at |

The dedup that matters happens **in text space, after OCR** — proven robust to
OCR flicker, dropped lines, and static-pause duplicate frames (see
`tests/test_transcribe.py`).

## What changed in the code

New modules:
- `src/video/frame_select.py` — scroll-safe frame selection (dense sample + drop near-exact dupes).
- `src/inference/ocr.py` — `VerbatimOCR`: vision OCR with a hard capability guard + live probe, anti-loop sampler controls, and an optional Apple Vision (ocrmac) deterministic backstop for code.
- `src/video/stitch.py` — text-space stitcher (the core engine).
- `src/workflows/transcribe.py` — orchestrator + constrained LLM cleanup.

Changed:
- `src/config.py` — added `OCRConfig`, `FrameSelectionConfig`, `ReconstructionConfig`; verbatim OCR prompts.
- `src/inference/client.py` — generic `from_endpoint` constructor, `list_models()`, real capability checks, sampler-param passthrough; added MiniMax/Kimi to the known-text-only guard list (VL variants still detected as vision).
- `src/cli.py` — new `transcribe` and `models` commands.
- `.env` / `.env.example` — split `OCR_MODEL` (vision) from `MLX_MODEL`/`LLM_MODEL` (text).

## Run it

```bash
# 1. See which of your oMLX models can actually OCR:
python -m src.cli models

# 2. Transcribe a recording (verbatim):
python -m src.cli transcribe input/policies.mov

# For code recordings, add the deterministic cross-check:
python -m src.cli transcribe input/code.mov --deterministic
#   (requires: pip install ocrmac)

# Output: data/<slug>/output/transcript.md  (+ transcript.raw.md, ocr/all_ocr.json)
```

The first thing `transcribe` does is **probe the OCR model with one real frame**;
if it answers as if no image was sent, it aborts immediately with an actionable
message instead of silently producing 173 empty files.

## Model selection — from YOUR oMLX inventory (June 2026)

You already have strong vision models served locally; no download needed. Their
names hide it (no "VL"), but Qwen3.6 and Gemma-4 are unified multimodal.

**OCR (vision) — pick one:**

| Model | Size | Why |
|-------|------|-----|
| `Qwen3.6-27B-bf16` | 51 GB | **Max fidelity.** Qwen-VL lineage is the OCR gold standard; bf16 = no quant loss. Default. |
| `gemma-4-31b-bf16` | 58 GB | Purpose-built for OCR / document / PDF / screen-UI parsing. Excellent alternative. |
| `Qwen3.6-35B-A3B-8bit-MTPLX-Optimized-Speed` | 37 GB | **Fastest.** MoE (~3B active) + speed-tuned — best for many frames. |

Text-only (do NOT use for OCR; fine for cleanup/analysis): MiniMax-M2/M3,
Kimi-K2.6/2.7, DeepSeek-V4, Nemotron-3, GLM-5.1. The `*-DFlash` / `*-MTP`
entries are speculative-decode draft models, not standalone.

**For code recordings:** keep `--deterministic` on. VLMs hallucinate code tokens
(linguistic-prior drift); Apple Vision never invents a line, so a character-level
disagreement is flagged for review rather than silently wrong.

**LLM cleanup (text):** any capable text model. `Kimi-K2.7-Code` is a great
choice for code recordings; `MiniMax-M2` is fine for prose. It only fixes seams
and indentation — prompted never to re-invent content.

> The default OCR model is just a starting point. `screenlens models` now labels
> each served model vision / text-only / draft, and `transcribe` probes the model
> with one real frame before processing — so a wrong choice fails instantly with
> a clear message instead of producing empty output.

## Update — reasoning-model hardening & cleanup made safe

A later round of fixes after running on a reasoning OCR model (`Qwen3.6-27B`):

- **Thinking is now disabled for OCR and cleanup.** A reasoning model used for
  verbatim OCR emitted chain-of-thought ("The user wants me to transcribe…")
  that consumed the entire `max_tokens` budget *before* reaching the answer — and
  with the opening `<think>` in the chat-template prefix, there were no tags for
  `strip_thinking` to catch. Both passes now send
  `chat_template_kwargs={"enable_thinking": false}` (`OCRConfig.disable_thinking`
  / `ReconstructionConfig.disable_thinking`, default true). `strip_thinking` also
  handles dangling/unclosed `<think>`, and the client warns on
  `finish_reason == "length"`.
- **LLM cleanup is now OFF by default** (`--cleanup` to opt in). On verbatim
  code/text the raw stitched OCR is already correct, and the LLM tends to drop
  content while "repairing".
- **Cleanup can no longer lose content.** Chunk input is bounded by the output
  token cap (a bug previously fed ~55k-char chunks into an 8k-token output, so
  each chunk truncated). And a per-chunk coverage guard (`MIN_CHUNK_COVERAGE =
  0.97`) discards any LLM output that dropped lines, keeping the raw stitched
  chunk — `transcript.md` is guaranteed ≥ the fidelity of `transcript.raw.md`.
- **Model must fit the oMLX memory ceiling.** `MiniMax-M3-4bit` is ~236 GB and
  won't load under a 71 GB ceiling; `Qwen3.6-27B-bf16` (~54 GB) is a safe text
  cleanup model that also fits. Note oMLX serializes requests.
