"""HTTP server.

Security posture, stated up front because the defaults matter more than the
features:

* Binds ``127.0.0.1`` only. Never all interfaces.
* Every ``/api/*`` request needs a session token. The token is minted at
  startup, handed to the browser once through the launch URL, and thereafter
  lives in a ``SameSite=Strict``, ``HttpOnly`` cookie.
* The ``Host`` header is validated against a loopback allowlist, which is what
  actually stops DNS rebinding — an attacker page that resolves a domain they
  control to 127.0.0.1 still sends *their* hostname.
* No CORS headers and no ``OPTIONS`` handler, so a cross-origin ``fetch`` from
  any other page cannot read a response.
* A restrictive CSP, because the UI is entirely self-hosted and needs no CDN.

Together these mean a random website you visit while Atlas is running cannot
read your library, trigger a job, or reach your API keys.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import socket
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import api, config, db, events, query, tasks, thumbnails
from .jobs import QUEUE

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_TOKEN = secrets.token_urlsafe(32)
COOKIE_NAME = "atlas_session"

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}

CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "form-action 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("image/webp", ".webp")


class AtlasHandler(BaseHTTPRequestHandler):
    server_version = f"LoRALibraryManager/{config.VERSION}"
    protocol_version = "HTTP/1.1"

    # -- logging ------------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:      # noqa: A003
        if config.get("verbose_http"):
            print(f"[http] {self.address_string()} {fmt % args}")

    def log_error(self, fmt: str, *args) -> None:
        pass          # client disconnects are normal for SSE; not worth the noise

    def handle(self) -> None:
        # A browser dropping a keep-alive or SSE socket surfaces as an aborted
        # connection while the server waits for the next request line. That is
        # normal, not an error — swallow it so it never prints a scary
        # traceback that looks like the last action failed.
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, TimeoutError):
            pass

    # -- entry points -------------------------------------------------------

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_HEAD(self) -> None:
        self._dispatch("GET", head_only=True)

    # Deliberately no do_OPTIONS: refusing preflight is what keeps other
    # origins from reading these responses.

    # -- dispatch -----------------------------------------------------------

    def _dispatch(self, method: str, head_only: bool = False) -> None:
        try:
            if not self._host_ok():
                self._send_json(HTTPStatus.MISDIRECTED_REQUEST,
                                {"error": "Unrecognised Host header"})
                return

            parsed = urlparse(self.path)
            raw_segments = [unquote(s) for s in parsed.path.strip("/").split("/") if s]
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            if not raw_segments:
                self._serve_app(params, head_only)
                return

            head = raw_segments[0]
            if head == "api":
                self._serve_api(method, raw_segments, params)
            elif head == "events":
                self._serve_events(params)
            elif head == "media":
                self._serve_media(raw_segments[1:], params, head_only)
            elif head == "static":
                self._serve_static("/".join(raw_segments[1:]), head_only)
            elif head == "health":
                self._send_json(HTTPStatus.OK, {"ok": True, "version": config.VERSION})
            else:
                # Any other path is a client-side route; serve the shell.
                self._serve_app(params, head_only)

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                          # noqa: BLE001
            try:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                {"error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        if not host:
            return False
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        return name.strip("[]").lower() in {h.strip("[]") for h in ALLOWED_HOSTS}

    # -- auth ---------------------------------------------------------------

    def _token_from_request(self, params: dict) -> str:
        cookie_header = self.headers.get("Cookie")
        if cookie_header:
            jar = SimpleCookie()
            try:
                jar.load(cookie_header)
            except Exception:                             # noqa: BLE001
                jar = SimpleCookie()
            morsel = jar.get(COOKIE_NAME)
            if morsel and morsel.value:
                return morsel.value
        header = self.headers.get("X-Atlas-Token")
        if header:
            return header.strip()
        return str(params.get("k", ""))

    def _authorised(self, params: dict) -> bool:
        return secrets.compare_digest(self._token_from_request(params), SESSION_TOKEN)

    # -- responses ----------------------------------------------------------

    def _base_headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)

    def _send_json(self, status: HTTPStatus | int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(int(status))
        self._base_headers("application/json; charset=utf-8", len(body))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str,
                    cache: str = "no-store", extra: dict | None = None,
                    head_only: bool = False) -> None:
        self.send_response(status)
        self._base_headers(content_type, len(body))
        self.send_header("Cache-Control", cache)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    # -- API ----------------------------------------------------------------

    def _serve_api(self, method: str, segments: list[str], params: dict) -> None:
        if not self._authorised(params):
            self._send_json(HTTPStatus.UNAUTHORIZED, {
                "error": "Missing or invalid session token.",
                "hint": "Open the URL printed in the console; it carries the token.",
            })
            return

        body: dict = {}
        if method in ("POST", "DELETE"):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > 0:
                if length > 32 * 1024 * 1024:
                    self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                    {"error": "Body too large"})
                    return
                raw = self.rfile.read(length)
                try:
                    parsed_body = json.loads(raw.decode("utf-8"))
                    body = parsed_body if isinstance(parsed_body, dict) else {"value": parsed_body}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Body must be JSON"})
                    return

        handler = api.resolve(method, segments)
        if handler is None:
            allowed = api.allowed_methods(segments)
            if allowed:
                self._send_json(HTTPStatus.METHOD_NOT_ALLOWED,
                                {"error": f"Use {', '.join(allowed)} for this path"})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"No route for /{'/'.join(segments)}"})
            return

        request = api.Request(method, self.path, params, body, segments)
        try:
            self._send_json(HTTPStatus.OK, handler(request))
        except api.ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except config.ConfigError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc), "needs_setup": True})
        except FileNotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})

    # -- SSE ----------------------------------------------------------------

    def _serve_events(self, params: dict) -> None:
        if not self._authorised(params):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid session token"})
            return

        subscription = events.BROKER.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            # Seed the connection so a reload paints live state immediately.
            events.BROKER.publish("hello", {"jobs": QUEUE.state()})
            for frame in events.sse_stream(subscription):
                self.wfile.write(frame)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            events.BROKER.unsubscribe(subscription)

    # -- static & media -----------------------------------------------------

    def _serve_app(self, params: dict, head_only: bool) -> None:
        """Serve the shell, minting the session cookie when the token is in the URL."""
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            self._send_bytes(404, b"index.html missing", "text/plain; charset=utf-8")
            return
        body = index.read_text(encoding="utf-8").replace(
            "__ASSET_VERSION__", _asset_version()).encode("utf-8")
        extra = {}
        if secrets.compare_digest(str(params.get("k", "")), SESSION_TOKEN):
            extra["Set-Cookie"] = (
                f"{COOKIE_NAME}={SESSION_TOKEN}; Path=/; HttpOnly; SameSite=Strict; Max-Age=86400"
            )
        self._send_bytes(200, body, "text/html; charset=utf-8",
                         cache="no-store", extra=extra, head_only=head_only)

    def _serve_static(self, relative: str, head_only: bool) -> None:
        target = _safe_join(STATIC_DIR, relative)
        if target is None or not target.is_file():
            self._send_bytes(404, b"not found", "text/plain; charset=utf-8")
            return
        data = target.read_bytes()
        etag = f'W/"{len(data)}-{int(target.stat().st_mtime)}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        # `no-cache` still allows the 304 above; it just forces revalidation.
        # Over loopback that costs nothing, and it means editing app.css or
        # app.js never leaves the user staring at a stale UI.
        self._send_bytes(200, data, ctype, cache="no-cache",
                         extra={"ETag": etag}, head_only=head_only)

    def _serve_media(self, segments: list[str], params: dict, head_only: bool) -> None:
        """Serve a preview image from the library, thumbnailed by default.

        Requires the session token: preview filenames leak library contents, and
        the images themselves are user data.
        """
        if not self._authorised(params):
            self._send_bytes(401, b"unauthorised", "text/plain; charset=utf-8")
            return
        try:
            root = config.library_root()
        except config.ConfigError:
            self._send_bytes(404, b"no library", "text/plain; charset=utf-8")
            return

        source = _safe_join(root, "/".join(segments))
        if source is None or not source.is_file():
            self._send_bytes(404, b"not found", "text/plain; charset=utf-8")
            return

        served = source
        if params.get("full") not in ("1", "true"):
            thumb = thumbnails.get_or_create(source, _int_or_none(params.get("size")))
            if thumb is not None:
                served = thumb

        try:
            stat = served.stat()
            etag = f'W/"{stat.st_size}-{int(stat.st_mtime)}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            data = served.read_bytes()
        except OSError:
            self._send_bytes(404, b"unreadable", "text/plain; charset=utf-8")
            return

        ctype = mimetypes.guess_type(served.name)[0] or "application/octet-stream"
        self._send_bytes(200, data, ctype, cache="private, max-age=86400",
                         extra={"ETag": etag}, head_only=head_only)


def _asset_version() -> str:
    """Cache key for the CSS/JS bundle, derived from their mtimes.

    A stamped URL is the only cache strategy that cannot go stale: edit an
    asset and the URL changes, so no browser or intermediary can serve the old
    bytes. Validators alone are not enough — a client is free to reuse a fresh
    response without revalidating.
    """
    stamp = 0.0
    for name in ("app.css", "app.js"):
        try:
            stamp = max(stamp, (STATIC_DIR / name).stat().st_mtime)
        except OSError:
            pass
    return f"{config.VERSION}-{int(stamp)}"


def _safe_join(base: Path, relative: str) -> Path | None:
    """Join and confirm containment. Rejects traversal, absolute paths and symlink escapes."""
    if not relative:
        return None
    candidate = (base / relative).resolve()
    try:
        base_resolved = base.resolve()
    except OSError:
        return None
    if candidate == base_resolved or base_resolved in candidate.parents:
        return candidate
    return None


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class AtlasServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False        # surface a port clash instead of masking it
    request_queue_size = 64


def find_port(preferred: int, attempts: int = 40) -> int:
    for candidate in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve(port: int | None = None, open_browser: bool | None = None) -> None:
    port = int(port or config.get("port") or 8077)
    port = find_port(port)

    conn = db.connect()                      # create/migrate before serving
    QUEUE.start()

    httpd = AtlasServer(("127.0.0.1", port), AtlasHandler)
    url = f"http://127.0.0.1:{port}/?k={SESSION_TOKEN}"

    print("=" * 62)
    print(f"  {config.APP_NAME} {config.VERSION}")
    print(f"  Library : {config.library_root()}")
    print(f"  Database: {config.db_path()}")
    print(f"  URL     : {url}")
    print("  Bound to 127.0.0.1 only. The token in the URL is required.")
    if not thumbnails.available():
        print("  NOTE: Pillow not installed — serving full-size previews.")
    print("=" * 62)

    stats = query.stats(conn)
    print(f"  {stats['total']} model(s) indexed, {stats['missing_metadata']} without metadata")

    if open_browser if open_browser is not None else config.get("open_browser"):
        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)),
                         daemon=True).start()

    if config.get("auto_scan_on_start"):
        tasks.submit("quick-refresh")

    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        QUEUE.stop()
        events.BROKER.close()
        httpd.server_close()
        db.close_thread_connections()
        print("Stopped.")
