from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import RewardJudge


class JudgeService:
    def __init__(self, config: Any) -> None:
        self.judge = RewardJudge(build_generator(config.model, judge_config=config.judge))

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        query_id = str(payload.get("query_id") or "")
        query = str(payload.get("query") or "")
        answer = payload.get("answer")
        status = str(payload.get("status") or "")
        response = str(payload.get("response") or "")
        example = QueryExample(
            query_id=query_id,
            query=query,
            answer=str(answer) if answer is not None else None,
        )
        return asdict(self.judge.evaluate(example, status, response))


def make_handler(service: JudgeService):
    class JudgeHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path not in {"/judge", "/"}:
                self.send_error(404, "Unknown endpoint")
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                response = service.evaluate(payload)
            except Exception as exc:  # pragma: no cover - defensive server path
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            return

    return JudgeHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the existing LLM judge over HTTP for OpenRLHF agents.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    service = JudgeService(config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"serving judge on http://{args.host}:{args.port}/judge", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
