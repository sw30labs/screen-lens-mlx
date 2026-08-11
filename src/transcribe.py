"""
Verbatim transcription pipeline (the new primary path).

    video.mov
      → select_frames    (dense sample, drop static dupes)        frame_select.py
      → VerbatimOCR       (vision model, char-faithful)            ocr.py
      → stitch_frames     (text-space dedup of scroll overlap)     stitch.py
      → LLM cleanup       (seams + indentation ONLY, optional)     this file
      → output/transcript.md

Everything is local: vision OCR + text cleanup both use the selected
OpenAI-compatible server (vLLM on DGX Spark or oMLX on Apple Silicon).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from .config import ScreenLensConfig
from .frame_select import select_frames
from .ocr import VerbatimOCR
from .omlx_client import (
    InferenceClient,
    degenerate_repetition,
    resolve_llm_model,
    resolve_ocr_model,
    resolve_role_api_key,
    resolve_role_backend,
    resolve_role_base_url,
    resolve_role_context,
)
from .stitch import stitch_frames

logger = logging.getLogger("screenlens.transcribe")

# Cleanup is seam/indent repair ONLY — it must never drop content. An LLM
# (especially a reasoning model) tends to "improve" by condensing, silently
# dropping code blocks or lists. After each chunk we check what fraction of its
# distinct non-blank input lines survived; below this we discard the LLM output
# and keep the raw stitched chunk. The small slack tolerates legitimate edits
# (stray header/footer removal, rejoining a line split across a frame seam).
MIN_CHUNK_COVERAGE = 0.97


def _chunk_coverage(src: str, repaired: str) -> float:
    """Fraction of distinct non-blank input lines (whitespace-normalized) that
    still appear in the repaired output. 1.0 means nothing was dropped."""
    def norm_lines(t: str) -> set[str]:
        return {re.sub(r"\s+", "", l) for l in t.splitlines() if l.strip()}

    src_lines = norm_lines(src)
    if not src_lines:
        return 1.0
    out_lines = norm_lines(repaired)
    return sum(1 for l in src_lines if l in out_lines) / len(src_lines)


CLEANUP_SYSTEM = (
    "You repair a transcript that was OCR'd frame-by-frame from a scrolling "
    "screen recording and then stitched together. Your edits are STRICTLY "
    "limited:\n"
    "1. Fix obvious stitch seams: remove a duplicated line where two frames "
    "overlapped, or rejoin a line that was split across the overlap.\n"
    "2. Restore consistent indentation for code blocks.\n"
    "3. Remove stray page headers/footers that slipped through (e.g. 'Page 3 of "
    "16', running titles).\n\n"
    "You must NOT paraphrase, summarize, translate, complete, or 'improve' any "
    "content. Do not invent text. Do not add commentary. If a word is garbled "
    "and you cannot be certain, leave it exactly as-is. Output ONLY the repaired "
    "transcript."
)


def _llm_client(cfg) -> InferenceClient:
    rc = cfg.reconstruction
    return InferenceClient.from_endpoint(
        base_url=resolve_role_base_url(rc),
        model=resolve_llm_model(rc),
        api_key=resolve_role_api_key(rc, "VLLM_LLM_API_KEY", "LLM_API_KEY"),
        backend=resolve_role_backend(rc),
        timeout=rc.timeout_seconds,
        context_size=resolve_role_context(rc),
        default_max_tokens=rc.max_tokens,
        default_temperature=rc.temperature,
    )


def _cleanup_transcript(text: str, cfg) -> str:
    """LLM seam/indent cleanup, chunked by blank-line boundaries to fit context."""
    client = _llm_client(cfg)
    extra = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if cfg.reconstruction.disable_thinking
        else None
    )
    # Cleanup is near-verbatim, so the repaired output is ~the same size as the
    # input. The binding limit is therefore the OUTPUT cap (max_tokens), not just
    # the context window: a chunk larger than max_tokens can emit guarantees
    # mid-chunk truncation and silent content loss. Bound chunk input by BOTH the
    # output cap and the context window (input+output+prompt must co-fit), with a
    # safety margin. (chars ≈ tokens*4)
    chars_per_token = 4
    max_out_chars = cfg.reconstruction.max_tokens * chars_per_token
    max_ctx_chars = int(cfg.reconstruction.model_context * 0.45) * chars_per_token
    budget_chars = int(min(max_out_chars, max_ctx_chars) * 0.85)
    paras = text.split("\n\n")
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur and cur_len + len(p) > budget_chars:
            chunks.append("\n\n".join(cur)); cur, cur_len = [], 0
        cur.append(p); cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))

    out = []
    for i, ch in enumerate(chunks):
        logger.info("LLM cleanup chunk %d/%d", i + 1, len(chunks))
        repaired = client.chat(
            CLEANUP_SYSTEM,
            "Repair this stitched transcript segment. Output only the repaired text:\n\n" + ch,
            max_tokens=cfg.reconstruction.max_tokens,
            temperature=0.0,
            extra=extra,
        ).strip()
        coverage = _chunk_coverage(ch, repaired)
        if coverage < MIN_CHUNK_COVERAGE:
            logger.warning(
                "Cleanup chunk %d/%d dropped content (line coverage %.0f%% < %.0f%%); "
                "keeping the raw stitched chunk to preserve fidelity.",
                i + 1, len(chunks), coverage * 100, MIN_CHUNK_COVERAGE * 100,
            )
            out.append(ch.strip())
        else:
            out.append(repaired)
    return "\n\n".join(out).strip() + "\n"


def _video_size(video_path: str) -> int | None:
    """Byte size of the video, None when it cannot be stat'ed (tests, removed files)."""
    try:
        return Path(video_path).stat().st_size
    except OSError:
        return None


def _load_cached_ocr(ocr_dir: Path) -> dict[str, str]:
    """Map frame filename → OCR text from the run folder's saved records.

    Prefers the combined ``all_ocr.json``; falls back to the per-frame files
    a run interrupted mid-OCR would have left behind.
    """
    records: list[dict] = []
    combined = ocr_dir / "all_ocr.json"
    try:
        if combined.exists():
            data = json.loads(combined.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records = data
    except Exception:
        records = []
    if not records:
        for path in sorted(ocr_dir.glob("ocr_*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    cached: dict[str, str] = {}
    for rec in records:
        name = Path(str(rec.get("path", ""))).name
        text = rec.get("ocr")
        if name and isinstance(text, str):
            cached[name] = text
    return cached


def transcribe_video(video_path: str, config: ScreenLensConfig, data_dir: Path) -> dict:
    """Run the full verbatim pipeline for one video. Returns a result dict."""
    t0 = time.time()
    data_dir = Path(data_dir)
    frames_dir = data_dir / "frames"
    ocr_dir = data_dir / "ocr"
    out_dir = data_dir / "output"
    for d in (frames_dir, ocr_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Select frames (scroll-safe) ─────────────────────────────────────────
    frames = select_frames(video_path, str(frames_dir), config.frame_selection)
    if not frames:
        return {"error": "No frames extracted", "stage": "select"}
    logger.info("Selected %d frames", len(frames))

    # 2. Verbatim OCR (vision model) ─────────────────────────────────────────
    # Resume from any OCR this run folder already holds: records pair with
    # frames by deterministic filename, so a re-run of the same video only
    # sends the frames the model has not read yet.
    cached = _load_cached_ocr(ocr_dir)
    texts_by_name: dict[str, str] = {}
    missing: list[dict] = []
    for f in frames:
        hit = cached.get(Path(f["path"]).name)
        if hit is None:
            missing.append(f)
        else:
            texts_by_name[Path(f["path"]).name] = hit

    ocr_model = resolve_ocr_model(config.ocr)
    if missing:
        ocr = VerbatimOCR(config.ocr)
        ocr_model = ocr.model
        new_texts = ocr.ocr_frames([f["path"] for f in missing])  # raises loudly if the model is blind
        for f, txt in zip(missing, new_texts):
            texts_by_name[Path(f["path"]).name] = txt
    texts = [texts_by_name[Path(f["path"]).name] for f in frames]
    if cached:
        logger.info(
            "Reused %d cached OCR result(s); %d frame(s) sent to the model",
            len(frames) - len(missing), len(missing),
        )

    ocr_records = []
    for f, txt in zip(frames, texts):
        rec = {**f, "ocr": txt}
        ocr_records.append(rec)
        (ocr_dir / f"ocr_{f['frame_id']:06d}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8")
    (ocr_dir / "all_ocr.json").write_text(json.dumps(ocr_records, indent=2), encoding="utf-8")

    non_empty = sum(1 for t in texts if t.strip())
    logger.info("OCR done: %d/%d frames had text", non_empty, len(texts))

    # The raw transcript stays byte-faithful to what the model read, so a frame
    # where the model got stuck is reported rather than edited — trimming it
    # here could just as easily delete a screen that genuinely repeats.
    degenerate_frames = []
    for f, txt in zip(frames, texts):
        unit = degenerate_repetition(txt)
        if unit is None:
            continue
        degenerate_frames.append(f["frame_id"])
        logger.warning(
            "frame %d OCR ends in a repetition loop (%r repeated); the "
            "transcript keeps it verbatim, but treat that frame as suspect.",
            f["frame_id"],
            unit,
        )

    # 3. Stitch (text-space dedup) ───────────────────────────────────────────
    frames_lines = [t.splitlines() for t in texts]
    stitched = stitch_frames(frames_lines, fuzzy=0.85, strip_boilerplate=True)
    transcript = stitched.text()
    raw_path = out_dir / "transcript.raw.md"
    raw_path.write_text(transcript, encoding="utf-8")
    logger.info("Stitched transcript: %d lines", len(stitched.lines))

    # 4. Optional LLM seam/indent cleanup ────────────────────────────────────
    clean_path = None
    if config.reconstruction.enabled and transcript.strip():
        try:
            cleaned = _cleanup_transcript(transcript, config)
            clean_path = out_dir / "transcript.md"
            clean_path.write_text(cleaned, encoding="utf-8")
        except Exception as exc:
            logger.error("LLM cleanup failed (%s); raw stitched transcript kept", exc)
            clean_path = out_dir / "transcript.md"
            clean_path.write_text(transcript, encoding="utf-8")
    else:
        clean_path = out_dir / "transcript.md"
        clean_path.write_text(transcript, encoding="utf-8")

    meta = {
        "video": str(Path(video_path).resolve()),
        "video_size": _video_size(video_path),
        "frames_selected": len(frames),
        "frames_with_text": non_empty,
        "degenerate_frames": degenerate_frames,
        "ocr_model": ocr_model,
        "llm_model": resolve_llm_model(config.reconstruction) if config.reconstruction.enabled else None,
        "transcript_path": str(clean_path),
        "raw_transcript_path": str(raw_path),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "transcribe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"stage": "done", **meta}
