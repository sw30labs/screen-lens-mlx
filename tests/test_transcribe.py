"""
Tests for the verbatim transcription path: text-space stitching, scroll-safe
frame selection, and the capability guard that prevents the original
blind-model regression (text-only model used for vision).

Run:  pytest tests/test_transcribe.py -v
"""
import random
from pathlib import Path

import pytest

from src.stitch import stitch_frames, detect_boilerplate, line_ratio
from src.config import OCRConfig, FrameSelectionConfig
from src.ocr import VerbatimOCR, _NO_IMAGE_RE


# ── Stitching ────────────────────────────────────────────────────────────────

def _make_scroll_frames(doc, view=20, step=3, header=None, footer=None, noise=0.0, seed=1):
    rng = random.Random(seed)
    frames, top = [], 0
    while top < len(doc):
        view_lines = doc[top:top + view]
        if noise:
            view_lines = [_noisy(x, rng, noise) for x in view_lines]
        page = 1 + top // view
        rendered = (header or []) + view_lines + ([f.format(page=page) for f in (footer or [])])
        frames.append(rendered)
        top += step
    return frames


def _noisy(line, rng, p):
    if rng.random() < p and line:
        i = rng.randrange(len(line))
        line = line[:i] + rng.choice("aeior ") + line[i + 1:]
    return line


def test_stitch_recovers_document_in_order():
    doc = [f"line {i:02d} content {i*7 % 13}" for i in range(60)]
    frames = _make_scroll_frames(doc, view=20, step=3)
    out = [l for l in stitch_frames(frames).lines if l.strip()]
    # every doc line present, in order
    j = 0
    for d in doc:
        while j < len(out) and line_ratio(d, out[j]) < 0.9:
            j += 1
        assert j < len(out), f"missing line: {d}"
        j += 1


def test_stitch_no_duplication():
    doc = [f"unique row number {i}" for i in range(40)]
    frames = _make_scroll_frames(doc, view=15, step=2)
    out = [l for l in stitch_frames(frames).lines if l.strip()]
    # length must be ~document length, not frames*view (no overlap leak)
    assert len(out) <= len(doc) + 2


def test_stitch_absorbs_exact_duplicate_frames():
    doc = [f"row {i}" for i in range(30)]
    frames = _make_scroll_frames(doc, view=12, step=3)
    frames.insert(3, list(frames[3]))   # static pause
    frames.insert(7, list(frames[7]))
    out = [l for l in stitch_frames(frames).lines if l.strip()]
    assert len(out) <= len(doc) + 2


def test_stitch_tolerates_ocr_noise():
    doc = [f"the model risk validation step {i} requires approval" for i in range(50)]
    frames = _make_scroll_frames(doc, view=18, step=3, noise=0.25, seed=4)
    out = [l for l in stitch_frames(frames, fuzzy=0.8).lines if l.strip()]
    recovered = sum(1 for d in doc if any(line_ratio(d, o) >= 0.75 for o in out))
    assert recovered / len(doc) >= 0.9


def test_stitch_tolerates_dropped_lines():
    # OCR sometimes drops a line inside the overlap; difflib matching blocks
    # must still align around the indel without scrambling or duplicating.
    rng = random.Random(11)
    doc = [f"section {i}: the validation requires model approval step {i}" for i in range(50)]
    frames = _make_scroll_frames(doc, view=18, step=3, noise=0.15, seed=3)
    for fr in frames:                       # randomly drop one mid line per frame
        if len(fr) > 6 and rng.random() < 0.5:
            del fr[rng.randrange(2, len(fr) - 2)]
    out = [l for l in stitch_frames(frames, fuzzy=0.8).lines if l.strip()]
    recovered = sum(1 for d in doc if any(line_ratio(d, o) >= 0.75 for o in out))
    assert recovered / len(doc) >= 0.9
    assert len(out) <= len(doc) * 1.3       # no duplication blow-up


def test_boilerplate_stripped():
    doc = [f"body line {i}" for i in range(40)]
    header = ["UBS MRM Guidelines", "Internal"]
    footer = ["Page {page} of 16", "Published: 30 April 2026"]
    frames = _make_scroll_frames(doc, header=header, footer=footer, view=15, step=3)
    boiler = detect_boilerplate(frames)
    assert any("mrm guidelines" in b for b in boiler)
    out = stitch_frames(frames).lines
    assert not any("of 16" in l for l in out)
    assert not any("MRM Guidelines" in l for l in out)


# ── Capability guard (prevents the blind-model regression) ───────────────────

def test_text_only_model_is_rejected():
    cfg = OCRConfig(model="MiniMax-M2.7")  # text-only — the original bug
    ocr = VerbatimOCR(cfg)
    with pytest.raises(RuntimeError, match="text-only"):
        ocr.assert_vision_capable()


def test_vision_model_passes_guard():
    cfg = OCRConfig(model="mlx-community/olmOCR-2-7B-1025-8bit")
    ocr = VerbatimOCR(cfg)
    ocr.assert_vision_capable()  # must not raise


def test_no_image_sentinel_regex():
    assert _NO_IMAGE_RE.search("No image or video frame has been provided.")
    assert _NO_IMAGE_RE.search("Please attach the image you'd like me to analyze.")
    assert not _NO_IMAGE_RE.search("def main():\n    return 0")


# ── End-to-end glue (mocked OCR server) ──────────────────────────────────────

def test_transcribe_end_to_end_with_mock_ocr(tmp_path, monkeypatch):
    """Full pipeline glue: select → OCR → stitch → write, with no real server."""
    import src.transcribe as T
    from src.config import ScreenLensConfig

    doc = [f"def step_{i}(x):  # row {i}" for i in range(40)]
    frames = _make_scroll_frames(doc, view=16, step=3)
    fake_meta = [{"frame_id": i, "frame_index": i, "timestamp": float(i),
                  "timestamp_str": f"00:00:{i:02d}.000", "path": f"/tmp/f{i}.png",
                  "width": 100, "height": 100} for i in range(len(frames))]

    monkeypatch.setattr(T, "select_frames", lambda *a, **k: fake_meta)

    class _MockOCR:
        model = "mock-vision"
        def __init__(self, cfg): pass
        def ocr_frames(self, paths): return ["\n".join(f) for f in frames]
    monkeypatch.setattr(T, "VerbatimOCR", _MockOCR)

    cfg = ScreenLensConfig()
    cfg.reconstruction.enabled = False      # skip the LLM cleanup (needs server)

    result = T.transcribe_video("/fake/video.mov", cfg, tmp_path)
    assert result["stage"] == "done"
    transcript = (tmp_path / "output" / "transcript.md").read_text()
    out = [l for l in transcript.splitlines() if l.strip()]
    # def lines reconstructed without duplication blow-up (glue check, not a
    # precision re-test — see the dedicated stitch tests for that)
    assert sum(1 for d in doc if any(line_ratio(d, o) >= 0.85 for o in out)) >= 34
    assert len(out) <= len(doc) + 3


# ── Thinking leak regression ─────────────────────────────────────────────────
#
# A reasoning OCR model (e.g. Qwen3.x) emitted chain-of-thought instead of the
# transcription and exhausted max_tokens before closing </think>, so the whole
# response was untagged/truncated reasoning that leaked into transcript.md.

def test_strip_thinking_handles_truncated_open_tag():
    from src.omlx_client import strip_thinking
    # complete block
    assert strip_thinking("<think>reasoning</think>\n\nANSWER") == "ANSWER"
    # dangling close (opening tag was a prompt prefix) — keep the answer
    assert strip_thinking("reasoning</think>\n\nANSWER") == "ANSWER"
    # dangling open, generation truncated mid-thought — no answer survives
    assert strip_thinking("prefix<think>truncated reasoning forever") == "prefix"
    # clean text untouched
    assert strip_thinking("just an answer") == "just an answer"


def test_ocr_disables_thinking_in_request_payload(monkeypatch, tmp_path):
    """OCR must send chat_template_kwargs.enable_thinking=false so a reasoning
    model produces the transcription instead of burning the budget on CoT."""
    import json
    from PIL import Image
    import src.omlx_client as omlx_client

    img_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color="white").save(img_path)

    captured = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(omlx_client, "_urlopen", fake_urlopen)

    ocr = VerbatimOCR(OCRConfig(model="Qwen3-VL-test", disable_thinking=True))
    assert ocr.ocr_frame(str(img_path)) == "hello"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}

    # When disabled, the knob must NOT be sent.
    captured.clear()
    ocr2 = VerbatimOCR(OCRConfig(model="Qwen3-VL-test", disable_thinking=False))
    ocr2.ocr_frame(str(img_path))
    assert "chat_template_kwargs" not in captured["payload"]


# ── Cleanup never drops content (coverage guard) ─────────────────────────────
#
# An LLM (esp. a reasoning model) silently condenses — dropping code blocks /
# lists despite "never remove content". The guard falls back to the raw stitched
# chunk whenever the repaired output dropped too many input lines.

def test_chunk_coverage_metric():
    from src.transcribe import _chunk_coverage
    src = "line one\n    line two\nline three"
    assert _chunk_coverage(src, "line one\nline two\n  line three") == 1.0  # reindent ok
    assert round(_chunk_coverage(src, "line one\nline three"), 2) == 0.67   # dropped a line
    assert _chunk_coverage(src, "") == 0.0


def test_cleanup_falls_back_to_raw_when_llm_drops_content(monkeypatch):
    import src.omlx_client as omlx_client
    import src.transcribe as T
    from src.config import ScreenLensConfig

    raw = "\n\n".join(f"keep_line_{i} = {i}" for i in range(20))

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            # The "LLM" returns only the first line — a gross content drop.
            import json
            return json.dumps(
                {"choices": [{"message": {"content": "keep_line_0 = 0"}}]}
            ).encode()

    monkeypatch.setattr(omlx_client, "_urlopen", lambda req, timeout: FakeResponse())

    cfg = ScreenLensConfig()
    out = T._cleanup_transcript(raw, cfg)
    # All original lines survive because the guard discarded the lossy LLM output.
    for i in range(20):
        assert f"keep_line_{i} = {i}" in out


# ── Scroll-safe frame selection on a REAL recording ──────────────────────────

REAL_VIDEO = Path(__file__).resolve().parents[1] / "input" / "policies.mov"


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="sample recording not present")
def test_select_frames_on_real_video(tmp_path):
    from src.frame_select import select_frames
    meta = select_frames(str(REAL_VIDEO), str(tmp_path), FrameSelectionConfig(sample_fps=2.0))
    assert len(meta) > 10                       # got real frames
    assert all(Path(m["path"]).exists() for m in meta)
    # timestamps strictly increasing
    ts = [m["timestamp"] for m in meta]
    assert ts == sorted(ts)


def test_transcribe_reports_degenerate_frames_without_editing_them():
    """A frame where the OCR model got stuck must be flagged, not trimmed —
    the raw transcript is defined as byte-faithful to what the model read."""
    from src.omlx_client import degenerate_repetition

    stuck = "INSERT INTO t VALUES ('a', 'b');\n" + "!" * 400
    assert degenerate_repetition(stuck) == "!"

    # A screen that genuinely repeats whole statements is not "stuck".
    faithful = "INSERT INTO t VALUES (1, 'abc', 'def');\n" * 20
    assert degenerate_repetition(faithful) is None
