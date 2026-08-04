"""Live operations view for Switchboard.

Stdlib only — no FastAPI, no uvicorn. The agent's whole pitch is that it installs
in one command, and a dashboard that drags in a web framework undercuts that.
`ThreadingHTTPServer` plus server-sent events is enough for one operator watching
one agent.

    python -m switchboard.dashboard        # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .store import Store

WEB_ROOT = Path(__file__).resolve().parent / "web"


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, fmt, *args):  # noqa: A003 - silence per-request noise
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, (WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/state"):
            self._send(200, json.dumps(self._state()).encode(), "application/json")
        elif self.path.startswith("/api/stream"):
            self._stream()
        else:
            self._send(404, b"not found", "text/plain")

    def _state(self) -> dict:
        experts = [
            {
                "id": e.id, "name": e.name, "channel": e.channel,
                "skills": e.skills, "paged": e.paged_count,
                "answered": e.answered_count, "reliability": round(e.reliability, 2),
            }
            for e in self.store.experts()
        ]
        knowledge = [
            {"question": k["question_text"], "answer": k["answer_text"],
             "taught_by": k["taught_by"], "reuse": k["reuse_count"]}
            for k in self.store.knowledge()
        ]
        return {
            "stats": self.store.stats(),
            "experts": experts,
            "knowledge": knowledge[-12:],
            "open": self.store.open_questions(),
        }

    def _stream(self) -> None:
        """SSE: replay recent history, then tail the event log."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        cursor = max(0, self._latest_seq() - 40)
        last_state = 0.0
        try:
            while True:
                for event in self.store.events_after(cursor):
                    cursor = event["seq"]
                    self._emit("event", event)

                now = time.time()
                if now - last_state > 1.0:
                    self._emit("state", self._state())
                    last_state = now

                time.sleep(0.4)
        except (BrokenPipeError, ConnectionResetError):
            pass  # operator closed the tab

    def _latest_seq(self) -> int:
        events = self.store.events_after(0, limit=100000)
        return events[-1]["seq"] if events else 0

    def _emit(self, name: str, payload: dict) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()


def serve(port: int = 8765, db: str | None = None) -> None:
    Handler.store = Store(db) if db else Store()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n  Switchboard dashboard → http://127.0.0.1:{port}\n  Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("  stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=None)
    serve(**vars(parser.parse_args()))
