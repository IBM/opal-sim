# SPDX-License-Identifier: Apache-2.0
"""
OPAL config builder — minimal local web UI.

Form-driven generator for configs/defaults.json-shaped JSON. Submits the JSON
to /validate, which runs OpalConfig().initialize(tmpfile) to confirm the
generated file loads cleanly. No external deps; stdlib http.server only.

Run from project root:
    python -m opal.webserver.server          # default port 9290
    python -m opal.webserver.server --port 8123
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from opal.opal_config import OpalConfig  # noqa: E402

INDEX_HTML_PATH = HERE / "index.html"

log = logging.getLogger("opal.webserver")


def validate_config_dict(cfg: dict) -> tuple[bool, str]:
    """Round-trip the generated dict through OpalConfig.initialize().

    Returns (ok, message). Message is empty on success, the exception text
    otherwise. We write to a tempfile because OpalConfig.initialize() reads
    from a path; we don't want to refactor that for the sake of this UI.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="opal-cfg-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        cfg_obj = OpalConfig()
        cfg_obj.initialize(tmp_path)
        return True, ""
    except Exception as exc:  # noqa: BLE001 — surface anything OpalConfig raises
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class ConfigBuilderHandler(BaseHTTPRequestHandler):
    server_version = "OpalConfigBuilder/0.1"

    def log_message(self, format, *args):  # noqa: A002 — match base signature
        log.info("%s - %s", self.address_string(), format % args)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(INDEX_HTML_PATH, "text/html; charset=utf-8")
            return
        self._send_json(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        if self.path != "/validate":
            self._send_json(404, {"error": "not found", "path": self.path})
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_json(400, {"ok": False, "error": "empty body"})
            return

        body = self.rfile.read(length)
        try:
            cfg = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"ok": False, "error": f"invalid JSON: {exc}"})
            return

        ok, msg = validate_config_dict(cfg)
        self._send_json(200, {"ok": ok, "error": msg})

    def _serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send_json(500, {"error": f"missing template: {path.name}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _GracefulServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that doesn't block shutdown on in-flight handlers.

    daemon_threads=True lets request worker threads be killed at process exit
    instead of pinning the main thread inside server_close().
    allow_reuse_address avoids "Address already in use" on quick restart.
    """

    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="OPAL config builder UI")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9290, help="bind port (default: 9290)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    addr = (args.host, args.port)
    httpd = _GracefulServer(addr, ConfigBuilderHandler)
    log.info("OPAL config builder listening on http://%s:%d", *addr)
    log.info("Open the URL in a browser. Ctrl-C to stop.")

    # Single-shot signal handler. The first SIGINT/SIGTERM unblocks serve_forever()
    # via httpd.shutdown() running on a worker thread (shutdown() must NOT be called
    # from the same thread that's inside serve_forever, hence the threading.Thread).
    # A second signal escalates to immediate exit so a wedged shutdown can still
    # be killed with Ctrl-C without resorting to kill -9.
    shutting_down = threading.Event()

    def _on_signal(signum, _frame):
        if shutting_down.is_set():
            log.warning("second signal received — exiting immediately")
            sys.stderr.flush()
            os._exit(130)
        shutting_down.set()
        sig_name = signal.Signals(signum).name if signum in signal.Signals.__members__.values() else str(signum)
        log.info("received %s — shutting down (Ctrl-C again to force)", sig_name)
        sys.stderr.flush()
        threading.Thread(target=httpd.shutdown, name="opalweb-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, _on_signal)
    # SIGTERM lets `kill <pid>` (without -9) also drain cleanly.
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (AttributeError, ValueError):
        # SIGTERM not available on Windows; ignore.
        pass

    try:
        # poll_interval is how often serve_forever checks the shutdown flag.
        # 0.5 s keeps Ctrl-C feeling instant without polling overhead.
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        log.info("server closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
