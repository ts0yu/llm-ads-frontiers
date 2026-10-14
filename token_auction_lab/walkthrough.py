"""Loopback-only experiment walkthrough and editable live inference lab.

Run from the checkout with: python -m token_auction_lab.walkthrough
The walkthrough reads recordings. Live Lab loads the cached local model on demand.
"""

from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from token_auction_lab.live import BusyError, LiveService, StaleRunError

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "walkthrough"
ATLAS_DATA = WEB / "atlas_data"
ATLAS_TYPES = {
    ".json": "application/json; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
}
ARCHIVE = Path("smoke_results/mps-2026-10-13")
SOURCE_FILES = {
    "SMOKE_RESULTS.md", "TEST_RESULTS.txt", "MECHANISM.md", "CODEX_BRIEF.md",
    "auction.py", "test_auction.py", "test_decoder.py", "uv.lock",
    "token_auction_lab/decoder.py", "token_auction_lab/runner.py",
    "token_auction_lab/prompts.py", "token_auction_lab/fact_sheets.json",
    "scripts/audit_smoke.py",
}


def evidence_path(name: str) -> Path | None:
    """Only archived evidence and explicitly named source files are public."""
    path = (ROOT / name).resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return None
    in_archive = relative.startswith(ARCHIVE.as_posix() + "/")
    if relative not in SOURCE_FILES and not (
        in_archive and path.suffix in {".json", ".jsonl", ".txt", ".log"}
    ):
        return None
    return path if path.is_file() else None


def atlas_path(name: str) -> Path | None:
    """Only atlas JSON and images inside walkthrough/atlas_data are public."""
    try:
        path = (ATLAS_DATA / name).resolve()
        relative = path.relative_to(ATLAS_DATA.resolve())
    except (ValueError, OSError):
        return None
    if any(part.startswith(".") for part in relative.parts):
        return None
    if path.suffix.lower() not in ATLAS_TYPES or not path.is_file():
        return None
    return path


def load_evidence() -> dict:
    runs = []
    labels = {151645: "[end of message]", 151643: "[end of text]"}
    for group in ("validated", "length-cap", "float16-failed", "float32-strict-failed"):
        for manifest_path in sorted((ROOT / ARCHIVE / group).glob("*/manifest.json")):
            directory = manifest_path.parent
            manifest = json.loads(manifest_path.read_text())
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            tokens = [event for event in events if event["type"] == "token"]
            result_path = directory / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            prior = ""
            for event in tokens:
                current = event["decoded_text_so_far"]
                if current.startswith(prior) and current != prior:
                    labels[event["token_id"]] = current[len(prior):]
                for row in event.get("top_tokens", []):
                    labels.setdefault(row["token_id"], row["token"])
                prior = current
            runs.append({
                "id": directory.name,
                "group": group,
                "path": directory.relative_to(ROOT).as_posix(),
                "manifest": manifest,
                "result": result,
                "events": tokens,
                "branches": dict(Counter(event["branch"] for event in tokens)),
            })
    diagnostic_path = ROOT / ARCHIVE / "float32-strict-failed/cache-diagnostic.jsonl"
    return {
        "recorded_date": "2026-10-13 (Pacific)",
        "source_commit": "b8831a33068115a0876d02c16306f2cc75b40eec",
        "runs": runs,
        "token_labels": labels,
        "facts": json.loads((ROOT / "token_auction_lab/fact_sheets.json").read_text()),
        "diagnostics": [json.loads(line) for line in diagnostic_path.read_text().splitlines()],
        "test_results": (ROOT / "TEST_RESULTS.txt").read_text(),
    }


class Handler(BaseHTTPRequestHandler):
    def respond(self, body, content_type="application/json; charset=utf-8", status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (
            origin is None or origin in {f"http://{host}" for host in hosts}
        )

    def do_POST(self):
        route = urlsplit(self.path).path
        if route not in {"/api/live/run", "/api/live/stop"}:
            self.respond({"error": "Unknown route."}, status=404)
            return
        if not self.local_request() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.respond({"error": "Use same-origin JSON requests from the local lab."}, status=403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Input must be a JSON object of at most 64 KiB.")
            payload = json.loads(self.rfile.read(length))
            if route == "/api/live/stop":
                if not isinstance(payload, dict):
                    raise ValueError("Expected an input object.")
                self.respond(self.server.live.stop(payload.get("run_id")))
            else:
                run_id = self.server.live.start(payload)
                self.respond({"run_id": run_id, "status": "loading"}, status=202)
        except (BusyError, StaleRunError) as error:
            self.respond({"error": str(error)}, status=409)
        except (ValueError, TypeError) as error:
            self.respond({"error": str(error)}, status=400)

    def live_get(self, route, query):
        if not self.local_request():
            self.respond({"error": "The live lab is available on loopback only."}, status=403)
            return
        try:
            value = lambda name, default=None: query.get(name, [default])[0]
            if route == "/api/live":
                after = int(value("after", "0"))
                if after < 0:
                    raise ValueError("Invalid event cursor.")
                result = self.server.live.snapshot(value("run_id"), after)
            elif route == "/api/live/inspect":
                result = self.server.live.inspect(
                    value("run_id"), int(value("step", "0")),
                    a=None if value("a") is None else float(value("a")),
                    b=None if value("b") is None else float(value("b")),
                    token_id=None if value("token_id") is None else int(value("token_id")),
                    offset=int(value("offset", "0")), limit=int(value("limit", "20")),
                )
            elif route == "/api/live/file":
                path = self.server.live.file(value("run_id"), value("name"))
                self.respond(path.read_bytes(), "text/plain; charset=utf-8")
                return
            else:
                self.respond({"error": "Unknown live route."}, status=404)
                return
            self.respond(result)
        except StaleRunError as error:
            self.respond({"error": str(error)}, status=409)
        except (ValueError, TypeError) as error:
            self.respond({"error": str(error)}, status=400)

    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        if route.startswith("/api/live"):
            self.live_get(route, parse_qs(urlsplit(self.path).query))
            return
        if route == "/api/evidence":
            body = json.dumps(load_evidence(), ensure_ascii=False).encode()
            content_type = "application/json; charset=utf-8"
        elif route.startswith("/evidence/"):
            file = evidence_path(route[len("/evidence/"):])
            if file is None:
                self.send_error(404)
                return
            body = file.read_bytes()
            content_type = "text/plain; charset=utf-8"
        elif route.startswith("/atlas_data/"):
            file = atlas_path(route[len("/atlas_data/"):])
            if file is None:
                self.send_error(404)
                return
            body = file.read_bytes()
            content_type = ATLAS_TYPES[file.suffix.lower()]
        else:
            name = "index.html" if route == "/" else route.lstrip("/")
            file = (WEB / name).resolve()
            if file.parent != WEB.resolve() or not file.is_file():
                self.send_error(404)
                return
            body = file.read_bytes()
            content_type = (
                "text/javascript" if file.suffix == ".mjs"
                else mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            ) + "; charset=utf-8"
        self.respond(body, content_type)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.live = LiveService()
    print(f"Token Auction Lab walkthrough: http://127.0.0.1:{server.server_port}", flush=True)
    print("Live Lab: /#live — uses the pinned cached model; loads on first run.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
