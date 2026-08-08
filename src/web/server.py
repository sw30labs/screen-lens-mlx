"""Web command deck server for ScreenLens.

Stdlib-only HTTP server (no new dependencies). Serves the single-file SPA plus
a JSON API for run state and pipeline control. Binds loopback-only and rejects
non-loopback clients on every ``/api/`` route — this is a single-operator local
dashboard that can start jobs and read frames off disk, so it is never exposed
beyond the machine it runs on.

Run:
    python -m src.web                     # http://127.0.0.1:8760
    python -m src.cli serve --port 9000

Author: Nic Cravino — ScreenLens
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import ipaddress
import json
import logging
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from . import runner

__all__ = ["main", "serve", "DashboardHandler", "STATIC_DIR", "DEFAULT_PORT"]

logger = logging.getLogger("screenlens.web.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REQUEST_BYTES = 256_000
DEFAULT_PORT = 8760

_FRAME_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _is_loopback_bind_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _client_is_loopback(handler: BaseHTTPRequestHandler) -> bool:
    host = handler.client_address[0]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "127.0.0.1", "::1")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ScreenLensDashboard/0.2"
    protocol_version = "HTTP/1.1"

    # ── GET ─────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        query = parse_qs(urlparse(self.path).query)

        if path.startswith("/api/") and not _client_is_loopback(self):
            self._send_json({"error": "dashboard APIs are restricted to loopback"}, status=403)
            return

        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/favicon.svg":
            self._send_file(STATIC_DIR / "favicon.svg", "image/svg+xml")
        elif path == "/api/health":
            self._send_json(self._health())
        elif path == "/api/roles":
            self._send_json(runner.roles(self._one(query, "config_file")))
        elif path == "/api/backend":
            self._send_json(runner.probe_endpoint(self._one(query, "config_file")))
        elif path == "/api/runs":
            self._send_json({
                "runs": runner.list_runs(self._one(query, "data_dir")),
                "busy": runner.busy(),
                "current": runner.current_run(),
            })
        elif path == "/api/run":
            slug = self._one(query, "slug")
            if not slug:
                self._send_json({"error": "slug is required"}, status=400)
                return
            snap = runner.snapshot(slug, self._one(query, "data_dir"))
            if snap is None:
                self._send_json({"error": "unknown run"}, status=404)
            else:
                self._send_json(snap)
        elif path == "/api/artifact":
            slug, name = self._one(query, "slug"), self._one(query, "name")
            if not slug or not name:
                self._send_json({"error": "slug and name are required"}, status=400)
                return
            art = runner.artifact(slug, name, self._one(query, "data_dir"))
            if art is None:
                self._send_json({"error": "artifact not found"}, status=404)
            else:
                self._send_json(art)
        elif path.startswith("/api/frame/"):
            self._send_frame(path)
        elif path == "/api/videos":
            folder = self._one(query, "folder") or "./input"
            self._send_json({"folder": folder, "videos": runner.list_videos(folder)})
        elif path == "/api/events":
            self._send_json({
                "events": runner.live_events(limit=self._int(query, "limit", 120)),
                "busy": runner.busy(),
            })
        elif path == "/api/jobs":
            self._send_json({"jobs": runner.list_jobs(), "busy": runner.busy()})
        elif path.startswith("/api/jobs/"):
            job = runner.get_job(path.rsplit("/", 1)[-1])
            if job:
                self._send_json(job)
            else:
                self._send_json({"error": "unknown job"}, status=404)
        else:
            self.send_error(404, "Not found")

    def _health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "busy": runner.busy(),
            "active_job_id": runner.active_job_id(),
            "current": runner.current_run(),
            "pipelines": list(runner.PIPELINES),
        }

    def _send_frame(self, path: str) -> None:
        """Serve one frame image out of a run's ``frames/`` directory."""
        parts = [unquote(p) for p in path[len("/api/frame/"):].split("/") if p]
        if len(parts) != 2:
            self.send_error(404, "Not found")
            return
        slug, name = parts
        if "\\" in name or name.startswith(".") or "/" in name:
            self.send_error(404, "Not found")
            return
        folder = runner.resolve_run(slug)
        if folder is None:
            self.send_error(404, "Not found")
            return
        frames_dir = (folder / "frames").resolve()
        target = (frames_dir / name).resolve()
        if target.parent != frames_dir or not target.is_file():
            self.send_error(404, "Not found")
            return
        content_type = _FRAME_CONTENT_TYPES.get(target.suffix.lower())
        if content_type is None:
            self.send_error(404, "Not found")
            return
        self._send_file(target, content_type, cache="private, max-age=300")

    # ── POST ────────────────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/api/run", "/api/search"):
            self.send_error(404, "Not found")
            return

        if not _client_is_loopback(self):
            self._send_json(
                {"error": "mutation requests are restricted to the local dashboard"},
                status=403,
            )
            return

        body = self._read_json_body()
        if body is None:
            return  # _read_json_body already answered

        if path == "/api/search":
            result = runner.search_now(body)
            self._send_json(result, status=400 if result.get("error") else 200)
            return

        job_id, error = runner.start_job(body)
        if error:
            self._send_json({"error": error}, status=409 if "running" in error else 400)
        else:
            self._send_json({"job_id": job_id}, status=202)

    def _read_json_body(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type != "application/json":
            self._send_json({"error": "Content-Type must be application/json"}, status=415)
            return None

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_json({"error": "Content-Length is required"}, status=411)
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._send_json({"error": "invalid Content-Length"}, status=400)
            return None
        if length < 0:
            self._send_json({"error": "invalid Content-Length"}, status=400)
            return None
        if length > MAX_REQUEST_BYTES:
            self._send_json({"error": "request body is too large"}, status=413)
            return None

        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "invalid JSON body"}, status=400)
            return None
        if not isinstance(body, dict):
            self._send_json({"error": "JSON body must be an object"}, status=400)
            return None
        return body

    # ── plumbing ────────────────────────────────────────────────────────────

    @staticmethod
    def _one(query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key)
        return values[0] if values else None

    @staticmethod
    def _int(query: dict[str, list[str]], key: str, default: int) -> int:
        values = query.get(key)
        if not values:
            return default
        try:
            return int(values[0])
        except ValueError:
            return default

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str, cache: str = "no-store") -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    query: str = "",
) -> None:
    """Start the dashboard server (blocks until Ctrl-C)."""
    if not _is_loopback_bind_host(host):
        raise SystemExit(
            "--host must be a loopback address or localhost; the command deck "
            "starts jobs and reads frames off disk, so it is local-only by design"
        )

    try:
        server = ThreadingHTTPServer((host, port), DashboardHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"Port {port} is already in use — a command deck is most likely")
            print(f"already running at http://{host}:{port}")
            print(f"(or pick another port: screenlens serve --port {port + 1})")
            raise SystemExit(1) from None
        raise

    url = f"http://{host}:{port}"
    if query:
        url = f"{url}/?{query.lstrip('?')}"
    print(f"ScreenLens command deck → {url}  (Ctrl+C to stop)")
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ScreenLens web command deck")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser tab on start."
    )
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
