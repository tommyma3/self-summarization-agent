from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any

from self_summarization_agent.bcplus_backend import build_direct_backend
from self_summarization_agent.config import load_train_config, parse_cli_overrides


class RetrievalWorkerServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, backend):
        super().__init__(server_address, request_handler_class)
        self.backend = backend


class RetrievalWorkerHandler(BaseHTTPRequestHandler):
    server: RetrievalWorkerServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, exc: Exception, status: int = 500) -> None:
        self._write_json(
            {
                "error": type(exc).__name__,
                "message": str(exc),
            },
            status=status,
        )

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json({"ok": True})
            return
        self._write_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        try:
            if self.path == "/search_many":
                payload = self._read_json()
                queries = payload.get("queries")
                if not isinstance(queries, list) or not all(isinstance(query, str) for query in queries):
                    raise ValueError("search_many.queries must be a list of strings")
                self._write_json({"results": self.server.backend.search_many(queries)})
                return
            if self.path == "/get_document":
                payload = self._read_json()
                doc_id = payload.get("doc_id")
                if not isinstance(doc_id, str):
                    raise ValueError("get_document.doc_id must be a string")
                self._write_json({"document": self.server.backend.get_document(doc_id)})
                return
            if self.path == "/shutdown":
                self._write_json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._write_json({"error": "not_found"}, status=404)
        except ValueError as exc:
            self._write_error(exc, status=400)
        except Exception as exc:
            self._write_error(exc, status=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local persistent retrieval worker.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", required=True, help="Path where the worker writes its bound URL as JSON.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    backend = build_direct_backend(config.experiment.bc_plus_root, config.retrieval)
    server = RetrievalWorkerServer((args.host, args.port), RetrievalWorkerHandler, backend)
    host, port = server.server_address[:2]
    ready_path = Path(args.ready_file)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(json.dumps({"url": f"http://{host}:{port}"}) + "\n", encoding="utf-8")
    print(f"[retrieval_worker] listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
