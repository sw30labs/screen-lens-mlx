"""
Scroll-safe frame selection for verbatim transcription.

Deliberately the OPPOSITE philosophy to frame_extractor's keyframe detector.
Empirically (see tests), pixel metrics on scrolling dense text are unreliable
for deciding "is this new content?" — every scrolled row shifts, so genuinely
new frames look ~50% similar and get wrongly merged, while static pauses look
identical. Trying to be clever there is what lost content before.

So here we do the safe thing:
  * Sample densely (default 2 fps) so no scrolled line is skipped.
  * Drop ONLY near-exact duplicates (SSIM ≳ 0.99 vs the last kept frame) —
    i.e. static pauses where the screen truly didn't move. This is the ONE
    decision pixels make reliably.
  * Let the text-space stitcher (stitch.py) do the real dedup after OCR.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from src.config import FrameSelectionConfig

logger = logging.getLogger("screenlens.frame_select")


def _gray_small(bgr: np.ndarray, long_side: int = 512) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _resize_max(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    if w >= h:
        return img.resize((max_dim, int(h * max_dim / w)), Image.LANCZOS)
    return img.resize((int(w * max_dim / h), max_dim), Image.LANCZOS)


def select_frames(
    video_path: str,
    output_dir: str,
    config: FrameSelectionConfig | None = None,
) -> list[dict]:
    """Extract frames at sample_fps, dropping near-exact static duplicates.

    Returns ordered frame metadata dicts (frame_id, timestamp, path, ...).
    """
    config = config or FrameSelectionConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(video_fps / max(config.sample_fps, 0.01))))

    logger.info(
        "Selecting frames: %.1f fps source, sampling every %d frames (~%.1f fps), "
        "drop-dupe SSIM>%.3f",
        video_fps, step, video_fps / step, config.drop_duplicate_ssim,
    )

    frames_meta: list[dict] = []
    last_gray: np.ndarray | None = None
    frame_idx = 0
    kept = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = _gray_small(bgr)
        is_dup = False
        if last_gray is not None and last_gray.shape == gray.shape:
            try:
                if ssim(last_gray, gray) >= config.drop_duplicate_ssim:
                    is_dup = True
            except Exception:
                is_dup = False

        if not is_dup:
            ts = frame_idx / max(video_fps, 1e-6)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = _resize_max(Image.fromarray(rgb), config.max_dimension)
            fname = f"frame_{kept:06d}.{config.output_format}"
            fpath = out / fname
            save_kw = {"quality": config.quality} if config.output_format == "jpg" else {}
            img.save(str(fpath), **save_kw)
            frames_meta.append({
                "frame_id": kept,
                "frame_index": frame_idx,
                "timestamp": round(ts, 3),
                "timestamp_str": _ts(ts),
                "path": str(fpath),
                "width": img.width,
                "height": img.height,
            })
            kept += 1
            last_gray = gray

        frame_idx += 1

    cap.release()
    logger.info("Kept %d frames (sampled %d, dropped %d static dupes)",
                kept, frame_idx // step + 1, (frame_idx // step + 1) - kept)
    return frames_meta


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
