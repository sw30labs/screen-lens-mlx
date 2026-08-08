"""Tests for the ScreenLens web command deck.

Three concerns, mirroring book-buddy-2026/contingency-atlas:

* **assets**   — the static page ships and wires itself to the real API
* **api**      — JSON contracts the SPA polls, and the job lifecycle
* **security** — loopback-only enforcement, POST validation, path containment

The job tests run through ``runner.set_pipeline_override`` so nothing here
needs a model server, a GPU, or ffmpeg.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from src.web import runner
from src.web.server import STATIC_DIR, DashboardHandler, _is_loopback_bind_host


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def server():
    """A dashboard bound to an ephemeral loopback port."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(autouse=True)
def clean_runner():
    runner.set_pipeline_override(None)
    yield
    runner.set_pipeline_override(None)
    _wait_idle()


def _wait_idle(timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while runner.busy() and time.time() < deadline:
        time.sleep(0.02)


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, json.loads(r.read())


def post(base: str, path: str, payload, content_type: str = "application/json"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw.decode(errors="replace")}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A ./data/ tree with one finished-looking run, as the CWD."""
    root = tmp_path / "data"
    run = root / "demo_20260101_101010"
    (run / "frames").mkdir(parents=True)
    (run / "captions").mkdir()
    (run / "output").mkdir()

    # A 1x1 PNG is enough to prove the frame route serves real bytes.
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000a49444154789c6300010000050001"
        "0d0a2db40000000049454e44ae426082"
    )
    (run / "frames" / "frame_000000.png").write_bytes(png)
    (run / "captions" / "all_captions.json").write_text(json.dumps([
        {"frame_id": 0, "timestamp_str": "00:00:00.000",
         "path": str(run / "frames" / "frame_000000.png"),
         "caption": "A terminal window shows ScreenLens running."}
    ]))
    (run / "captions" / "caption_000000.json").write_text("{}")
    (run / "output" / "transcript.md").write_text("# Transcript\nhello world\n")
    (run / "output" / "secret.env").write_text("TOKEN=nope\n")

    monkeypatch.chdir(tmp_path)
    return run


# ── assets ──────────────────────────────────────────────────────────────────

class TestWebAssets:
    def test_static_files_ship(self):
        assert (STATIC_DIR / "index.html").is_file()
        assert (STATIC_DIR / "favicon.svg").is_file()

    def test_index_is_self_contained(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        # No framework, no build step, no CDN — ADR-008 in contingency-atlas.
        assert "<script" in html
        assert not re.search(r'<script[^>]+src=["\']https?://', html)
        assert not re.search(r'<link[^>]+href=["\']https?://', html)

    def test_index_calls_only_real_endpoints(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        called = set(re.findall(r'["`/]api/([a-z]+)', html))
        served = {"health", "roles", "backend", "runs", "run", "artifact",
                  "frame", "videos", "events", "jobs", "search"}
        assert called <= served, f"SPA calls endpoints the server does not serve: {called - served}"

    def test_index_serves_over_http(self, server):
        with urllib.request.urlopen(server + "/", timeout=10) as r:
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/html")
            assert r.headers["X-Content-Type-Options"] == "nosniff"
            assert r.headers["X-Frame-Options"] == "DENY"
            assert b"SCREENLENS" in r.read()

    def test_favicon_serves(self, server):
        with urllib.request.urlopen(server + "/favicon.svg", timeout=10) as r:
            assert r.status == 200
            assert r.headers["Content-Type"] == "image/svg+xml"


# ── API ─────────────────────────────────────────────────────────────────────

class TestWebAPI:
    def test_health_reports_pipelines(self, server):
        status, body = get(server, "/api/health")
        assert status == 200
        assert body["ok"] is True
        assert body["busy"] is False
        assert set(body["pipelines"]) == set(runner.PIPELINES)

    def test_roles_expose_vision_and_text(self, server):
        _, body = get(server, "/api/roles")
        assert set(body) == {"caption", "ocr", "text"}
        assert body["caption"]["role"] == "vision"
        assert body["ocr"]["role"] == "vision"
        assert body["text"]["role"] == "text"
        # The text role must not be silently reused for vision work.
        assert body["text"]["model"]

    def test_runs_and_snapshot(self, server, data_root):
        _, body = get(server, "/api/runs")
        slugs = [r["slug"] for r in body["runs"]]
        assert "demo_20260101_101010" in slugs

        _, snap = get(server, "/api/run?slug=demo_20260101_101010")
        assert snap["frames"] == 1
        assert snap["captions"] == 1
        assert snap["frames_list"][0]["name"] == "frame_000000.png"
        assert snap["captions_preview"][0]["preview"].startswith("A terminal window")
        assert "transcript.md" in snap["outputs"]

    def test_empty_chromadb_dir_is_not_an_embedded_run(self, server, data_root):
        """ensure_dirs() makes chromadb/ up front, so directory existence would
        report a vector DB — and label a still-captioning run "embedded"."""
        (data_root / "chromadb").mkdir()
        _, snap = get(server, "/api/run?slug=demo_20260101_101010")
        assert snap["has_chromadb"] is False
        assert snap["stage"] == "captioned"

        (data_root / "chromadb" / "chroma.sqlite3").write_bytes(b"SQLite format 3\x00")
        _, snap = get(server, "/api/run?slug=demo_20260101_101010")
        assert snap["has_chromadb"] is True
        assert snap["stage"] == "embedded"

    def test_unknown_run_is_404(self, server, data_root):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, "/api/run?slug=nope_20200101_000000")
        assert exc.value.code == 404

    def test_frame_route_serves_image_bytes(self, server, data_root):
        with urllib.request.urlopen(
            server + "/api/frame/demo_20260101_101010/frame_000000.png", timeout=10
        ) as r:
            assert r.status == 200
            assert r.headers["Content-Type"] == "image/png"
            assert r.read().startswith(b"\x89PNG")

    def test_artifact_reads_output_file(self, server, data_root):
        _, body = get(server, "/api/artifact?slug=demo_20260101_101010&name=transcript.md")
        assert body["name"] == "transcript.md"
        assert "hello world" in body["text"]

    def test_artifact_refuses_unlisted_suffix(self, server, data_root):
        """An artifact reader that serves any suffix leaks .env files."""
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, "/api/artifact?slug=demo_20260101_101010&name=secret.env")
        assert exc.value.code == 404

    def test_videos_lists_only_video_files(self, server, tmp_path, monkeypatch):
        folder = tmp_path / "clips"
        folder.mkdir()
        (folder / "a.mov").write_bytes(b"x")
        (folder / "notes.txt").write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        _, body = get(server, "/api/videos?folder=./clips")
        assert [v["name"] for v in body["videos"]] == ["a.mov"]

    def test_job_lifecycle(self, server, data_root):
        runner.set_pipeline_override(
            "reconstruct", lambda params, config: {"folders": 1, "ok": True}
        )
        status, body = post(server, "/api/run", {"pipeline": "reconstruct"})
        assert status == 202
        job_id = body["job_id"]

        deadline = time.time() + 15
        job = None
        while time.time() < deadline:
            _, job = get(server, f"/api/jobs/{job_id}")
            if job["status"] != "running":
                break
            time.sleep(0.05)

        assert job["status"] == "done", job
        assert job["result"] == {"folders": 1, "ok": True}
        assert job["elapsed_seconds"] is not None

        _, jobs = get(server, "/api/jobs")
        assert jobs["jobs"][0]["id"] == job_id
        assert jobs["busy"] is False

    def test_job_failure_is_reported_not_raised(self, server, data_root):
        def boom(params, config):
            raise RuntimeError("model went away")

        runner.set_pipeline_override("reconstruct", boom)
        _, body = post(server, "/api/run", {"pipeline": "reconstruct"})
        deadline = time.time() + 15
        job = None
        while time.time() < deadline:
            _, job = get(server, f"/api/jobs/{body['job_id']}")
            if job["status"] != "running":
                break
            time.sleep(0.05)
        assert job["status"] == "error"
        assert "model went away" in job["error"]

    def test_only_one_job_at_a_time(self, server, data_root):
        release = threading.Event()
        runner.set_pipeline_override(
            "reconstruct", lambda params, config: (release.wait(10), {"ok": True})[1]
        )
        try:
            status, first = post(server, "/api/run", {"pipeline": "reconstruct"})
            assert status == 202
            status, second = post(server, "/api/run", {"pipeline": "reconstruct"})
            assert status == 409
            assert "still running" in second["error"]
        finally:
            release.set()
        _wait_idle()

    def test_job_params_never_echo_the_api_key(self, server, data_root):
        runner.set_pipeline_override("reconstruct", lambda params, config: {"ok": True})
        _, body = post(
            server, "/api/run", {"pipeline": "reconstruct", "api_key": "super-secret"}
        )
        _wait_idle()
        _, job = get(server, f"/api/jobs/{body['job_id']}")
        assert "api_key" not in job["params"]
        assert "super-secret" not in json.dumps(job)

    def test_events_capture_pipeline_output(self, server, data_root):
        def chatty(params, config):
            print("[1/3] EXTRACTING FRAMES")
            print("plain progress line")
            return {"ok": True}

        runner.set_pipeline_override("reconstruct", chatty)
        post(server, "/api/run", {"pipeline": "reconstruct"})
        _wait_idle()
        _, body = get(server, "/api/events?limit=50")
        notes = [e["note"] for e in body["events"]]
        assert "EXTRACTING FRAMES" in notes
        assert "plain progress line" in notes
        stage = next(e for e in body["events"] if e["note"] == "EXTRACTING FRAMES")
        assert (stage["step"], stage["steps"]) == (1, 3)

    def test_tqdm_progress_reaches_the_dashboard(self, server, data_root):
        """tqdm redraws with \r, so a naive line splitter shows nothing for
        the whole of a twenty-minute captioning pass."""
        def with_bar(params, config):
            import sys
            sys.stdout.write("Captioning frames:   0%|   | 0/20 [00:00<?, ?it/s]\r")
            sys.stdout.write("Captioning frames:  20%|## | 4/20 [04:05<16:23, 61.48s/it]\r")
            sys.stdout.write("Captioning frames:  20%|## | 4/20 [04:06<16:20, 61.40s/it]\r")
            sys.stdout.write("Captioning frames: 100%|###| 20/20 [20:10<00:00, 60.5s/it]\r")
            return {"ok": True}

        runner.set_pipeline_override("reconstruct", with_bar)
        post(server, "/api/run", {"pipeline": "reconstruct"})
        _wait_idle()
        _, body = get(server, "/api/events?limit=50")
        steps = [(e["step"], e["steps"]) for e in body["events"] if e["kind"] == "progress"]
        # Redraws of the same step must not flood the ring buffer.
        assert steps == [(0, 20), (4, 20), (20, 20)]


# ── security ────────────────────────────────────────────────────────────────

class TestWebSecurity:
    def test_only_loopback_bind_hosts_allowed(self):
        assert _is_loopback_bind_host("127.0.0.1")
        assert _is_loopback_bind_host("localhost")
        assert _is_loopback_bind_host("::1")
        assert not _is_loopback_bind_host("0.0.0.0")
        assert not _is_loopback_bind_host("192.168.1.10")

    def test_serve_refuses_non_loopback_bind(self):
        from src.web.server import serve

        with pytest.raises(SystemExit):
            serve(host="0.0.0.0", open_browser=False)

    def test_non_loopback_client_is_refused(self, server, monkeypatch):
        """A request arriving from off-box must be rejected, not served."""
        monkeypatch.setattr(
            "src.web.server._client_is_loopback", lambda handler: False
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, "/api/runs")
        assert exc.value.code == 403
        status, body = post(server, "/api/run", {"pipeline": "reconstruct"})
        assert status == 403
        assert not runner.busy(), "a refused request must not start a job"

    @pytest.mark.parametrize("name", [
        "../../../etc/passwd",
        "..",
        ".hidden",
        "sub/dir.md",
        "back\\slash.md",
    ])
    def test_artifact_paths_are_contained(self, data_root, name):
        assert runner.artifact("demo_20260101_101010", name) is None

    @pytest.mark.parametrize("slug", ["../..", "..", "a/b", "a\\b", ".hidden", ""])
    def test_run_slugs_are_contained(self, data_root, slug):
        assert runner.resolve_run(slug) is None

    def test_frame_route_rejects_traversal(self, server, data_root):
        for bad in ["../../../etc/passwd", "..%2f..%2fpasswd", "..", ".hidden"]:
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(
                    f"{server}/api/frame/demo_20260101_101010/{bad}", timeout=10
                )
            assert exc.value.code == 404, bad

    def test_frame_route_refuses_non_image_suffix(self, server, data_root):
        (data_root / "frames" / "notes.txt").write_text("secret")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"{server}/api/frame/demo_20260101_101010/notes.txt", timeout=10
            )
        assert exc.value.code == 404

    def test_post_requires_json_content_type(self, server):
        status, _ = post(server, "/api/run", b"{}", content_type="text/plain")
        assert status == 415

    def test_post_rejects_malformed_bodies(self, server):
        assert post(server, "/api/run", b"not json")[0] == 400
        assert post(server, "/api/run", [1, 2, 3])[0] == 400

    def test_post_rejects_oversized_body(self, server):
        from src.web.server import MAX_REQUEST_BYTES

        payload = json.dumps({"pipeline": "reconstruct", "pad": "x" * MAX_REQUEST_BYTES})
        status, body = post(server, "/api/run", payload.encode())
        assert status == 413
        assert "too large" in body["error"]

    def test_unknown_pipeline_is_refused(self, server):
        status, body = post(server, "/api/run", {"pipeline": "rm -rf /"})
        assert status == 400
        assert "must be one of" in body["error"]
        assert not runner.busy()

    def test_missing_video_is_refused_before_starting_a_job(self, server):
        status, body = post(
            server, "/api/run", {"pipeline": "ingest", "video_path": "/nope/missing.mov"}
        )
        assert status == 400
        assert "video not found" in body["error"]
        assert not runner.busy()

    def test_unknown_routes_are_404(self, server):
        for path in ["/api/nope", "/../etc/passwd", "/secrets"]:
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(server + path, timeout=10)
            assert exc.value.code == 404, path
        assert post(server, "/api/nope", {})[0] == 404
