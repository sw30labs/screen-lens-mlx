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


# ── Stitch canonicalization fast path ────────────────────────────────────────
#
# _canon_ids gained an exact-match dict + length gate; these tests pin parity
# with the original all-pairs scan it replaced.

def _naive_canon_ids(norm_a, norm_b, fuzzy):
    """Reference: the original all-pairs fuzzy scan, kept to prove parity."""
    from difflib import SequenceMatcher
    canons = []

    def get_id(s):
        if not s:
            return -1
        for idx, c in enumerate(canons):
            if SequenceMatcher(None, s, c).ratio() >= fuzzy:
                return idx
        canons.append(s)
        return len(canons) - 1

    return [get_id(s) for s in norm_a], [get_id(s) for s in norm_b]


def test_canon_ids_matches_naive_scan():
    from src.stitch import _canon_ids

    fixtures = [
        (["alpha", "beta", "alpha"], ["beta", "gamma"]),          # exact repeats
        (["def main():", "    return 0"], ["def  main():", "    return 1"]),  # flicker
        (["", "x", ""], ["", "x"]),                               # blanks
        (["ab", "abcdefgh", "ab"], ["abcdefgh", "ab"]),           # length extremes
        (["page 3 of 16", "page 4 of 16"], ["page 4 of 16", "page 5 of 16"]),
        (["abcde", "abcdf"], ["abcde"]),                          # first-match-wins
        ([], ["only-b"]),                                         # empty tail
        ([f"line {i}" for i in range(50)],
         [f"line {i}" for i in range(40, 70)]),                   # scroll overlap
    ]
    for a, b in fixtures:
        for fuzzy in (0.85, 0.7, 0.95):
            assert _canon_ids(a, b, fuzzy) == _naive_canon_ids(a, b, fuzzy)


def _counting_sequence_matcher(monkeypatch):
    import src.stitch as stitch
    from difflib import SequenceMatcher as RealSM

    calls = []

    class CountingSM(RealSM):
        def ratio(self):
            calls.append(1)
            return super().ratio()

    monkeypatch.setattr(stitch, "SequenceMatcher", CountingSM)
    return calls


def test_canon_ids_exact_hits_skip_fuzzy_scan(monkeypatch):
    import src.stitch as stitch

    calls = _counting_sequence_matcher(monkeypatch)
    a = ["alpha", "beta"]
    ids = stitch._canon_ids(a, list(a), 0.85)
    # "beta" vs "alpha" is the only pair that reaches the ratio call; the
    # second frame's lines are exact repeats and never touch SequenceMatcher.
    assert len(calls) == 1
    assert ids == ([0, 1], [0, 1])


def test_canon_ids_length_gate_skips_fuzzy_scan(monkeypatch):
    import src.stitch as stitch

    calls = _counting_sequence_matcher(monkeypatch)
    ids = stitch._canon_ids(["ab", "abcdefgh"], [], 0.85)
    # A 6-char difference cannot reach a 0.85 ratio, so no scan call at all.
    assert calls == []
    assert ids == ([0, 1], [])


# ── OCR resume ───────────────────────────────────────────────────────────────

_RESUME_META = [
    {"frame_id": 0, "frame_index": 0, "timestamp": 0.0,
     "timestamp_str": "00:00:00.000", "path": "/tmp/f0.png", "width": 100, "height": 100},
    {"frame_id": 1, "frame_index": 1, "timestamp": 1.0,
     "timestamp_str": "00:00:01.000", "path": "/tmp/f1.png", "width": 100, "height": 100},
]
_RESUME_TEXTS = {
    "f0.png": "first frame line one\nfirst frame line two",
    "f1.png": "ZZZ different\nQQQ unrelated",
}


def _resume_mocks(monkeypatch, calls):
    import src.transcribe as T

    monkeypatch.setattr(T, "select_frames", lambda *a, **k: list(_RESUME_META))

    class _MockOCR:
        model = "mock-vision"

        def __init__(self, cfg):
            pass

        def ocr_frames(self, paths):
            batch = [p for p in paths]
            calls.append(batch)
            from pathlib import Path as _P
            return [_RESUME_TEXTS[_P(p).name] for p in batch]

    monkeypatch.setattr(T, "VerbatimOCR", _MockOCR)


def test_transcribe_resume_reuses_cached_ocr(tmp_path, monkeypatch):
    """A re-run over a populated ocr/ dir never constructs the OCR client."""
    import src.transcribe as T
    from src.config import ScreenLensConfig

    calls = []
    _resume_mocks(monkeypatch, calls)
    cfg = ScreenLensConfig()
    cfg.reconstruction.enabled = False

    first = T.transcribe_video("/fake/video.mov", cfg, tmp_path)
    assert first["stage"] == "done"
    assert calls == [["/tmp/f0.png", "/tmp/f1.png"]]

    class _BombOCR:
        def __init__(self, cfg):
            raise AssertionError("OCR client must not be constructed on a full cache hit")

    monkeypatch.setattr(T, "VerbatimOCR", _BombOCR)
    second = T.transcribe_video("/fake/video.mov", cfg, tmp_path)
    assert second["stage"] == "done"
    transcript = (tmp_path / "output" / "transcript.md").read_text()
    for line in ("first frame line one", "first frame line two", "ZZZ different", "QQQ unrelated"):
        assert line in transcript


def test_transcribe_resume_ocrs_only_missing_frames(tmp_path, monkeypatch):
    """A partial OCR cache sends only the uncovered frames to the model."""
    import json
    import src.transcribe as T
    from src.config import ScreenLensConfig

    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir(parents=True)
    (ocr_dir / "all_ocr.json").write_text(json.dumps(
        [{**_RESUME_META[0], "ocr": _RESUME_TEXTS["f0.png"]}]))

    calls = []
    _resume_mocks(monkeypatch, calls)
    cfg = ScreenLensConfig()
    cfg.reconstruction.enabled = False

    result = T.transcribe_video("/fake/video.mov", cfg, tmp_path)
    assert result["stage"] == "done"
    assert calls == [["/tmp/f1.png"]]
    combined = json.loads((tmp_path / "ocr" / "all_ocr.json").read_text())
    assert [r["ocr"] for r in combined] == [_RESUME_TEXTS["f0.png"], _RESUME_TEXTS["f1.png"]]


def test_image_data_url_reencodes_png_to_jpeg_on_the_wire(tmp_path):
    """OCR payloads ship as JPEG while the stored frame stays lossless PNG."""
    import base64 as b64
    import io
    from PIL import Image
    from src.omlx_client import _image_data_url

    # Noise compresses poorly as PNG, so the wire saving shows up even tiny.
    png = tmp_path / "frame.png"
    Image.effect_noise((640, 400), 64).convert("RGB").save(png)

    plain = _image_data_url(str(png))
    assert plain.startswith("data:image/png;base64,")

    wired = _image_data_url(str(png), force_jpeg=True)
    assert wired.startswith("data:image/jpeg;base64,")
    decoded = Image.open(io.BytesIO(b64.b64decode(wired.split(",", 1)[1])))
    assert decoded.format == "JPEG"
    assert decoded.size == (640, 400)
    assert len(wired) < len(plain)

    # JPEG inputs pass through untouched even when re-encoding is on.
    jpg = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), color=(10, 200, 10)).save(jpg)
    assert _image_data_url(str(jpg), force_jpeg=True).startswith(
        "data:image/jpeg;base64,")


def test_wire_jpeg_is_opt_in_on_the_client():
    from src.omlx_client import InferenceClient

    wired = InferenceClient.from_endpoint(
        base_url="http://127.0.0.1:8000/v1", model="m", api_key=None,
        wire_jpeg=True,
    )
    assert wired.wire_jpeg is True

    default = InferenceClient.from_endpoint(
        base_url="http://127.0.0.1:8000/v1", model="m", api_key=None,
    )
    assert default.wire_jpeg is False


def test_probe_truncation_is_not_reported_as_a_problem(tmp_path, monkeypatch, caplog):
    """The probe caps itself at 256 tokens, so a cut-off answer is expected —
    warning about it puts a false alarm in the command deck's activity feed."""
    import src.omlx_client as omlx_client
    from PIL import Image
    from src.config import OCRConfig
    from src.ocr import VerbatimOCR

    img_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color="white").save(img_path)

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            import json
            return json.dumps({
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": "def main():"},
                }]
            }).encode()

    monkeypatch.setattr(omlx_client, "_urlopen", lambda req, timeout: FakeResponse())
    ocr = VerbatimOCR(OCRConfig(model="Qwen3-VL-test"))

    with caplog.at_level("WARNING"):
        ocr.probe(str(img_path))
    assert "truncated" not in caplog.text, "the probe's own cap is not a warning"

    # A real frame hitting the cap is still worth shouting about.
    caplog.clear()
    with caplog.at_level("WARNING"):
        ocr.ocr_frame(str(img_path))
    assert "truncated the response at" in caplog.text
