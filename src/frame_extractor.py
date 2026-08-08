"""
Video Frame Extraction Module.

Two strategies:
  1. **keyframe** (default) — Hybrid change detection ported from QWEN3-VL-Python-OCR-Script-MLX.
     Uses SSIM + perceptual hash (pHash) + HSV histogram correlation to capture only
     distinct screens. Ideal for screen recordings where the display is mostly static.
  2. **fixed_fps** — Simple extraction at N frames per second. Useful for action-heavy video.

Backends: OpenCV (default), ffmpeg (fallback), decord (optional).
"""
import json
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import FrameExtractionConfig, ExtractionStrategy

logger = logging.getLogger("screenlens.frame_extractor")


# ── Video Metadata ──────────────────────────────────────────────────────────

def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffprobe when the optional binary exists."""
    if shutil.which("ffprobe") is None:
        logger.info("ffprobe is unavailable; reading video metadata with OpenCV")
        return {}
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
        video_stream = next(
            (s for s in probe.get("streams", []) if s["codec_type"] == "video"), {}
        )
        duration = float(probe.get("format", {}).get("duration", 0))
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = int(num) / int(den) if int(den) != 0 else 30.0
        return {
            "duration": duration,
            "fps": fps,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
            "total_frames": int(duration * fps),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, StopIteration, ValueError) as e:
        logger.warning(f"ffprobe failed ({e}), will get metadata from OpenCV")
        return {}


# ── Hybrid Keyframe Detection Helpers ───────────────────────────────────────
# Ported from QWEN3-VL-Python-OCR-Script-MLX/src/run_ocr.py

def _to_gray_small(bgr: np.ndarray, long_side: int = 640) -> np.ndarray:
    """Convert to grayscale and shrink to speed up comparisons."""
    h, w = bgr.shape[:2]
    if max(h, w) > long_side:
        if h >= w:
            new_h, new_w = long_side, int(round(w * (long_side / h)))
        else:
            new_w, new_h = long_side, int(round(h * (long_side / w)))
        bgr_small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr
    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduces cursor-blink noise
    return gray


def _phash(gray: np.ndarray) -> np.ndarray:
    """Compute perceptual hash (pHash) via DCT — 8x8 bit array."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    dct_low = dct[:8, :8]
    median = np.median(dct_low[1:, 1:])  # skip DC
    return (dct_low > median).astype(np.uint8)


def _phash_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    """Hamming distance between two 8x8 perceptual hashes."""
    return int(np.sum(bits_a.flatten() ^ bits_b.flatten()))


def _hsv_hist_corr(bgr_a: np.ndarray, bgr_b: np.ndarray, bins=(24, 32)) -> float:
    """Correlation of HSV histograms in [0..1]. Lower → more different."""
    def hist(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten().astype("float32")
    corr = cv2.compareHist(hist(bgr_a), hist(bgr_b), cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))


def _ssim_and_changed_area(gray_a: np.ndarray, gray_b: np.ndarray) -> Tuple[float, float]:
    """Compute SSIM and fraction of significantly changed pixels."""
    from skimage.metrics import structural_similarity as ssim
    s, diff = ssim(gray_a, gray_b, full=True)
    delta = 1.0 - diff
    changed_fraction = float((delta > 0.15).mean())
    return float(s), changed_fraction


# ── Main Extraction Entrypoint ──────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    config: Optional[FrameExtractionConfig] = None,
) -> list[dict]:
    """
    Extract frames from a video file.

    Uses the strategy configured in config:
      - keyframe: hybrid change detection (best for screen recordings)
      - fixed_fps: simple interval-based extraction

    Returns list of frame metadata dicts.
    """
    if config is None:
        config = FrameExtractionConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.strategy == ExtractionStrategy.keyframe:
        return _extract_keyframes(str(video_path), output_dir, config)
    else:
        return _extract_fixed_fps(str(video_path), output_dir, config)


# ── Keyframe Extraction (Hybrid Change Detection) ──────────────────────────

def _extract_keyframes(
    video_path: str,
    output_dir: Path,
    config: FrameExtractionConfig,
) -> list[dict]:
    """
    Smart keyframe extraction using hybrid change detection.

    A new keyframe is emitted when ANY of these triggers fire:
      - SSIM < threshold AND changed area >= min_changed_area
      - pHash hamming distance >= threshold
      - HSV histogram correlation <= threshold
    Plus a forced periodic keyframe every max_interval_seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / max(fps, 1e-6)

    print(f"Video: {duration:.1f}s, {fps:.1f} fps, {total_frames} total frames")
    print(f"Strategy: keyframe (SSIM={config.ssim_threshold}, pHash={config.phash_threshold}, "
          f"hist={config.hist_corr_threshold}, interval={config.min_interval_seconds}-{config.max_interval_seconds}s)")

    frames_meta = []
    last_emit_ts = -1e9
    last_gray: Optional[np.ndarray] = None
    last_bgr_small: Optional[np.ndarray] = None
    last_ph: Optional[np.ndarray] = None

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Scanning for keyframes", unit="fr")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts = frame_idx / max(fps, 1e-6)
        gray = _to_gray_small(frame_bgr)
        should_emit = False

        if last_gray is None:
            # Always emit first frame
            should_emit = True
        else:
            # SSIM + changed area
            try:
                ssim_val, changed_fraction = _ssim_and_changed_area(last_gray, gray)
            except Exception:
                ssim_val, changed_fraction = 1.0, 0.0

            # Perceptual hash
            try:
                ph = _phash(gray)
                ph_dist = _phash_distance(ph, last_ph) if last_ph is not None else 0
            except Exception:
                ph, ph_dist = None, 0

            # HSV histogram correlation
            try:
                bgr_small = _make_small_bgr(frame_bgr)
                corr = _hsv_hist_corr(
                    last_bgr_small if last_bgr_small is not None else bgr_small,
                    bgr_small
                )
            except Exception:
                corr, bgr_small = 1.0, frame_bgr

            # Decision logic
            big_change = (ssim_val < config.ssim_threshold and changed_fraction >= config.min_changed_area)
            phash_change = (ph is not None and last_ph is not None and ph_dist >= config.phash_threshold)
            hist_change = (corr <= config.hist_corr_threshold)

            time_gate = (ts - last_emit_ts) >= config.min_interval_seconds
            force_periodic = (ts - last_emit_ts) >= config.max_interval_seconds

            if (time_gate and (big_change or phash_change or hist_change)) or force_periodic:
                should_emit = True

        if should_emit:
            # Convert and save the frame
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = _resize_frame(img, config.max_dimension)

            extracted_id = len(frames_meta)
            frame_filename = f"frame_{extracted_id:06d}.{config.output_format}"
            frame_path = output_dir / frame_filename

            save_kwargs = {"quality": config.quality} if config.output_format == "jpg" else {}
            img.save(str(frame_path), **save_kwargs)

            frames_meta.append({
                "frame_id": extracted_id,
                "frame_index": frame_idx,
                "timestamp": round(ts, 3),
                "timestamp_str": _format_timestamp(ts),
                "path": str(frame_path),
                "width": img.width,
                "height": img.height,
            })

            # Update reference state
            last_emit_ts = ts
            last_gray = gray
            last_ph = _phash(gray)
            last_bgr_small = _make_small_bgr(frame_bgr)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"Selected {len(frames_meta)} keyframes from {total_frames} total frames "
          f"({100 * len(frames_meta) / max(total_frames, 1):.1f}% of video)")
    return frames_meta


# ── Fixed FPS Extraction ────────────────────────────────────────────────────

def _extract_fixed_fps(
    video_path: str,
    output_dir: Path,
    config: FrameExtractionConfig,
) -> list[dict]:
    """Simple extraction: one frame every 1/fps seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / video_fps

    frame_interval = max(1, int(video_fps / config.fps))
    expected_frames = total_frames // frame_interval

    print(f"Video: {duration:.1f}s, {video_fps:.1f} fps, {total_frames} total frames")
    print(f"Strategy: fixed_fps (extracting ~{expected_frames} frames at {config.fps} fps)")

    frames_meta = []
    frame_count = 0
    extracted = 0
    pbar = tqdm(total=expected_frames, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = _resize_frame(img, config.max_dimension)

            frame_filename = f"frame_{extracted:06d}.{config.output_format}"
            frame_path = output_dir / frame_filename
            save_kwargs = {"quality": config.quality} if config.output_format == "jpg" else {}
            img.save(str(frame_path), **save_kwargs)

            timestamp = frame_count / video_fps
            frames_meta.append({
                "frame_id": extracted,
                "frame_index": frame_count,
                "timestamp": round(timestamp, 3),
                "timestamp_str": _format_timestamp(timestamp),
                "path": str(frame_path),
                "width": img.width,
                "height": img.height,
            })
            extracted += 1
            pbar.update(1)
        frame_count += 1

    pbar.close()
    cap.release()
    print(f"Extracted {extracted} frames")
    return frames_meta


# ── Shared Helpers ──────────────────────────────────────────────────────────

def _make_small_bgr(frame_bgr: np.ndarray, long_side: int = 640) -> np.ndarray:
    """Shrink BGR frame for histogram comparison."""
    h, w = frame_bgr.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame_bgr


def _resize_frame(img: Image.Image, max_dimension: int) -> Image.Image:
    """Resize image so the longest side <= max_dimension."""
    w, h = img.size
    if max(w, h) <= max_dimension:
        return img
    if w > h:
        new_w, new_h = max_dimension, int(h * max_dimension / w)
    else:
        new_h, new_w = max_dimension, int(w * max_dimension / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
