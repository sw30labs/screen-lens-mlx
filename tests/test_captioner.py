"""Focused tests for per-frame caption request isolation and retries."""
from collections import defaultdict


def test_concurrent_batch_preserves_peer_and_retries_only_failed_frame(monkeypatch):
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(
            backend=CaptionBackend.vllm,
            batch_size=2,
            retry_attempts=1,
            retry_max_tokens=1234,
        )
    )
    calls = defaultdict(list)

    def fake_caption(
        image_path,
        *,
        max_tokens=None,
        temperature=None,
        require_complete=False,
    ):
        calls[image_path].append((max_tokens, temperature, require_complete))
        if image_path == "slow.jpg" and max_tokens is None:
            raise RuntimeError("request timed out")
        return f"caption:{image_path}:{max_tokens}"

    monkeypatch.setattr(captioner, "caption", fake_caption)

    assert captioner.caption_batch(["slow.jpg", "good.jpg"]) == [
        "caption:slow.jpg:1234",
        "caption:good.jpg:None",
    ]
    assert calls == {
        "slow.jpg": [(None, None, False), (1234, 0.0, True)],
        "good.jpg": [(None, None, False)],
    }


def test_concurrent_batch_marks_only_frame_that_exhausts_retries(monkeypatch):
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(
            backend=CaptionBackend.vllm,
            batch_size=2,
            retry_attempts=2,
            retry_max_tokens=512,
        )
    )
    calls = defaultdict(list)

    def fake_caption(
        image_path,
        *,
        max_tokens=None,
        temperature=None,
        require_complete=False,
    ):
        calls[image_path].append((max_tokens, temperature, require_complete))
        if image_path == "bad.jpg":
            raise RuntimeError("still broken")
        return "valid peer caption"

    monkeypatch.setattr(captioner, "caption", fake_caption)

    assert captioner.caption_batch(["bad.jpg", "good.jpg"]) == [
        "[Error captioning frame: still broken]",
        "valid peer caption",
    ]
    assert calls == {
        "bad.jpg": [
            (None, None, False),
            (512, 0.0, True),
            (512, 0.0, True),
        ],
        "good.jpg": [(None, None, False)],
    }


def test_retry_ceiling_never_exceeds_normal_caption_ceiling(monkeypatch):
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(
            backend=CaptionBackend.vllm,
            max_tokens=256,
            retry_attempts=1,
            retry_max_tokens=2048,
        )
    )
    calls = []

    def fake_caption(
        image_path,
        *,
        max_tokens=None,
        temperature=None,
        require_complete=False,
    ):
        calls.append((max_tokens, temperature, require_complete))
        if max_tokens is None:
            raise RuntimeError("first attempt failed")
        return "recovered"

    monkeypatch.setattr(captioner, "caption", fake_caption)

    assert captioner.caption_batch(["frame.jpg"]) == ["recovered"]
    assert calls == [(None, None, False), (256, 0.0, True)]


def test_caption_frames_reuses_cached_captions(tmp_path, monkeypatch):
    """Per-frame caption files let an interrupted run resume cheaply."""
    import json
    import src.captioner as cap
    from src.config import CaptionBackend, CaptioningConfig

    frames = [
        {"frame_id": 0, "timestamp": 0.0, "path": "/run/frames/frame_000000.jpg"},
        {"frame_id": 1, "timestamp": 1.0, "path": "/run/frames/frame_000001.jpg"},
    ]
    (tmp_path / "caption_000000.json").write_text(json.dumps(
        {**frames[0], "caption": "cached caption"}))

    sent = []

    class _FakeCaptioner:
        def caption_batch(self, paths):
            sent.extend(paths)
            return [f"fresh caption for {p}" for p in paths]

    monkeypatch.setattr(cap, "_get_captioner", lambda cfg: _FakeCaptioner())

    cfg = CaptioningConfig(backend=CaptionBackend.ollama, batch_size=2)
    results = cap.caption_frames(frames, cfg, output_dir=str(tmp_path))

    assert sent == ["/run/frames/frame_000001.jpg"]
    assert [r["caption"] for r in results] == [
        "cached caption",
        "fresh caption for /run/frames/frame_000001.jpg",
    ]
    combined = json.loads((tmp_path / "all_captions.json").read_text())
    assert [r["frame_id"] for r in combined] == [0, 1]


def test_caption_frames_ignores_cache_from_a_different_frame(tmp_path, monkeypatch):
    """A cached record pairs only with the frame whose path it stored."""
    import json
    import src.captioner as cap
    from src.config import CaptionBackend, CaptioningConfig

    frames = [{"frame_id": 0, "timestamp": 0.0, "path": "/run/frames/frame_000000.jpg"}]
    (tmp_path / "caption_000000.json").write_text(json.dumps(
        {**frames[0], "path": "/elsewhere/other.jpg", "caption": "stale"}))

    sent = []

    class _FakeCaptioner:
        def caption_batch(self, paths):
            sent.extend(paths)
            return ["fresh caption" for _ in paths]

    monkeypatch.setattr(cap, "_get_captioner", lambda cfg: _FakeCaptioner())

    cfg = CaptioningConfig(backend=CaptionBackend.ollama, batch_size=1)
    results = cap.caption_frames(frames, cfg, output_dir=str(tmp_path))

    assert sent == ["/run/frames/frame_000000.jpg"]
    assert [r["caption"] for r in results] == ["fresh caption"]


def test_caption_frames_fast_path_submits_all_pending_in_one_call(tmp_path, monkeypatch):
    """vLLM/oMLX captioning pays one pool over ALL pending frames (no chunk
    barrier) and still persists each frame's JSON as its result lands."""
    import json
    import src.captioner as cap
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(backend=CaptionBackend.vllm, batch_size=2)
    )
    monkeypatch.setattr(
        captioner, "_caption_with_retry", lambda p: f"caption:{p}",
    )
    real_batch = captioner.caption_batch
    batch_calls = []

    def spy_batch(paths, on_result=None):
        batch_calls.append(len(paths))
        return real_batch(paths, on_result=on_result)

    monkeypatch.setattr(captioner, "caption_batch", spy_batch)
    monkeypatch.setattr(cap, "_get_captioner", lambda cfg: captioner)

    frames = [
        {"frame_id": i, "timestamp": float(i),
         "timestamp_str": f"00:00:0{i}.000", "path": f"/frames/frame_{i:06d}.png"}
        for i in range(5)
    ]
    cfg = CaptioningConfig(backend=CaptionBackend.vllm, batch_size=2)
    results = cap.caption_frames(frames, cfg, output_dir=str(tmp_path))

    # One call with everything — chunking would have been [2, 2, 1].
    assert batch_calls == [5]
    assert [r["caption"] for r in results] == [f"caption:{f['path']}" for f in frames]
    for f in frames:
        saved = json.loads(
            (tmp_path / f"caption_{f['frame_id']:06d}.json").read_text())
        assert saved["caption"] == f"caption:{f['path']}"
    assert (tmp_path / "all_captions.json").exists()


def test_caption_frames_fast_path_marks_all_frames_when_batch_raises(tmp_path, monkeypatch):
    """A wholesale batch failure still yields one error marker per frame."""
    import src.captioner as cap
    from src.captioner import OpenAICompatibleCaptioner
    from src.config import CaptionBackend, CaptioningConfig

    captioner = OpenAICompatibleCaptioner(
        CaptioningConfig(backend=CaptionBackend.vllm, batch_size=2)
    )

    def boom(paths, on_result=None):
        raise RuntimeError("server unreachable")

    monkeypatch.setattr(captioner, "caption_batch", boom)
    monkeypatch.setattr(cap, "_get_captioner", lambda cfg: captioner)

    frames = [
        {"frame_id": i, "timestamp": float(i),
         "timestamp_str": f"00:00:0{i}.000", "path": f"/frames/f{i}.png"}
        for i in range(3)
    ]
    cfg = CaptioningConfig(backend=CaptionBackend.vllm, batch_size=2)
    results = cap.caption_frames(frames, cfg, output_dir=str(tmp_path))

    assert len(results) == 3
    assert all(r["caption"].startswith("[Error captioning frame:") for r in results)
