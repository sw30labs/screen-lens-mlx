"""
Text-space stitching for scrolling screen recordings.

This is the core of ScreenLens' *verbatim* reconstruction. Pixel metrics
(SSIM/pHash) are unreliable on scrolling dense text — every row shifts, so
"95% identical" frames score as very different. The robust place to dedup and
reassemble a scrolled document is in TEXT space, after OCR:

  1. Strip repeating page boilerplate (headers/footers like "Page 10 of 16").
  2. Stitch consecutive frame OCR by finding the longest run of (fuzzily)
     matching lines — the scroll overlap — and appending only the new tail.
  3. Resolve OCR flicker (the same line read slightly differently across
     frames) with fuzzy matching + cross-frame majority voting.

No model is required here; this operates purely on the per-frame line lists.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable


# ── Line normalization ───────────────────────────────────────────────────────

_WS = re.compile(r"[ \t]+")


def _norm(line: str) -> str:
    """Normalize a line for *comparison* (not for output)."""
    return _WS.sub(" ", line.strip()).lower()


def line_ratio(a: str, b: str) -> float:
    """Fuzzy similarity of two lines in [0,1], whitespace/case-insensitive."""
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


# ── Boilerplate (header/footer) detection ────────────────────────────────────

# Footer/page-marker patterns that vary per page (so frequency won't catch a
# single form) but are always boilerplate. Matched against normalized lines.
_FOOTER_PATTERNS = [
    re.compile(r"^page\s+\d+\s+of\s+\d+$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^page\s+\d+$"),
    re.compile(r"^-\s*\d+\s*-$"),
]


def detect_boilerplate(
    frames_lines: list[list[str]],
    *,
    edge_lines: int = 4,
    min_fraction: float = 0.4,
) -> set[str]:
    """Find EXACT normalized lines that recur near the top/bottom of many frames.

    Fixed page headers/footers ("UBS MRM Guidelines…", "Published: 30 April
    2026", running titles) sit at the same edge of nearly every scrolled frame.
    A genuine content line only grazes an edge for the frame or two while it
    scrolls past, so frequency cleanly separates the two.

    We deliberately do NOT digit-template here: uniform body text (e.g. numbered
    list items) would collapse to one template and be wrongly stripped. Varying
    page numbers are handled separately by ``_FOOTER_PATTERNS``.
    """
    n = len(frames_lines)
    if n < 4:
        return set()

    counts: Counter[str] = Counter()
    for lines in frames_lines:
        edge = lines[:edge_lines] + lines[-edge_lines:]
        for s in {_norm(x) for x in edge if x.strip()}:
            counts[s] += 1

    threshold = max(3, int(n * min_fraction))
    return {t for t, c in counts.items() if c >= threshold}


def _is_boilerplate(line: str, exact: set[str]) -> bool:
    nl = _norm(line)
    if not nl:
        return False
    if nl in exact:
        return True
    return any(p.match(nl) for p in _FOOTER_PATTERNS)


# ── Overlap stitching ────────────────────────────────────────────────────────

@dataclass
class StitchResult:
    lines: list[str] = field(default_factory=list)
    # For each output line, the variants seen across frames (for majority vote).
    _variants: list[Counter] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def _canon_ids(norm_a: list[str], norm_b: list[str], fuzzy: float) -> tuple[list[int], list[int]]:
    """Map fuzzily-equal lines to a shared integer id.

    OCR reads the same line slightly differently across frames; difflib compares
    by exact equality, so we first collapse near-duplicate lines to one id. This
    lets ``SequenceMatcher`` align the two frames while tolerating both OCR
    flicker (fuzzy) AND inserted/dropped lines (difflib's matching blocks).

    Performance: most lines repeat *exactly* across consecutive frames, so an
    exact-match dict answers those in O(1). A line already in ``canons`` always
    maps to its own index — when it was added, no earlier canon reached
    ``fuzzy``, and ``ratio(s, s) == 1.0`` does — so the dict is precisely
    equivalent to the scan. For the remaining lines, a length gate skips
    candidates that cannot reach the threshold: ``ratio <= 2*min/(la+lb)``, so
    ``|la-lb| * fuzzy > 2*min*(1-fuzzy)`` makes a match impossible. First-match
    order is unchanged, so results are identical to the naive scan.
    """
    canons: list[str] = []
    exact: dict[str, int] = {}

    def get_id(s: str) -> int:
        if not s:
            return -1
        hit = exact.get(s)
        if hit is not None:
            return hit
        ls = len(s)
        for idx, c in enumerate(canons):
            lc = len(c)
            if abs(ls - lc) * fuzzy > 2 * min(ls, lc) * (1 - fuzzy):
                continue
            if SequenceMatcher(None, s, c).ratio() >= fuzzy:
                exact[s] = idx
                return idx
        exact[s] = len(canons)
        canons.append(s)
        return len(canons) - 1

    return [get_id(s) for s in norm_a], [get_id(s) for s in norm_b]


def _best_overlap(tail: list[str], head: list[str], *, fuzzy: float) -> tuple[int, int]:
    """Find where ``head`` (next frame) re-shows the end of ``tail`` (accumulated).

    A vertical scroll means ``head`` begins inside content already at the end of
    the accumulator. We fuzzy-canonicalize both, then use difflib's matching
    blocks (indel-tolerant) to find the overlap. To avoid latching onto a
    coincidental match in the middle of the document, we require the matched
    region to sit at the END of the accumulator window (where a scroll overlap
    must be).

    Returns (tail_start, head_len): accumulator from ``tail_start`` onward
    corresponds to ``head[:head_len]``. (head_len == 0 → no overlap.)
    """
    if not tail or not head:
        return len(tail), 0

    window = min(len(tail), max(len(head) * 3, 60))
    sub = tail[-window:]
    base = len(tail) - window
    a, b = _canon_ids([_norm(x) for x in sub], [_norm(x) for x in head], fuzzy)

    sm = SequenceMatcher(None, a, b, autojunk=False)
    blocks = [bl for bl in sm.get_matching_blocks() if bl.size > 0]
    if not blocks:
        return len(tail), 0

    total = sum(bl.size for bl in blocks)
    if total < 2:
        return len(tail), 0

    tail_match_end = max(bl.a + bl.size for bl in blocks)  # last matched accumulator idx
    head_end = max(bl.b + bl.size for bl in blocks)        # last matched head idx
    first_a = min(bl.a for bl in blocks)

    # The overlap must reach the accumulator's end (a real scroll). If the match
    # is buried mid-window, only accept it if it's substantial.
    reaches_end = tail_match_end >= len(sub) - 2
    strong = total >= min(len(head), len(sub)) * 0.4
    if not (reaches_end or strong):
        return len(tail), 0

    return base + first_a, head_end


def stitch_frames(
    frames_lines: list[list[str]],
    *,
    fuzzy: float = 0.85,
    strip_boilerplate: bool = True,
) -> StitchResult:
    """Stitch ordered per-frame OCR line-lists into one deduplicated document."""
    boiler = detect_boilerplate(frames_lines) if strip_boilerplate else set()

    cleaned: list[list[str]] = []
    for lines in frames_lines:
        kept = [ln for ln in lines if not _is_boilerplate(ln, boiler)]
        # collapse runs of blank lines
        out: list[str] = []
        for ln in kept:
            if not ln.strip() and out and not out[-1].strip():
                continue
            out.append(ln.rstrip())
        cleaned.append(out)

    acc: list[str] = []
    variants: list[Counter] = []

    for lines in cleaned:
        if not lines:
            continue
        if not acc:
            acc = list(lines)
            variants = [Counter([ln]) for ln in lines]
            continue

        tail_start, head_len = _best_overlap(acc, lines, fuzzy=fuzzy)

        # Majority-vote the overlapping region: record this frame's variant for
        # each aligned accumulator line so we can later pick the best reading.
        for k in range(head_len):
            idx = tail_start + k
            if 0 <= idx < len(variants) and k < len(lines):
                variants[idx][lines[k]] += 1

        # Append only the genuinely new tail.
        new_tail = lines[head_len:]
        for ln in new_tail:
            acc.append(ln)
            variants.append(Counter([ln]))

    # Resolve each line to its majority (then longest) variant.
    resolved = []
    for c in variants:
        # most common; tie-break by longer string (more complete OCR)
        best = max(c.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        resolved.append(best)

    return StitchResult(lines=resolved, _variants=variants)


def stitch_text(frame_texts: Iterable[str], **kw) -> str:
    """Convenience wrapper: stitch a sequence of multi-line OCR strings."""
    frames_lines = [t.splitlines() for t in frame_texts]
    return stitch_frames(frames_lines, **kw).text()
