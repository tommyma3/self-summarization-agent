from __future__ import annotations

import time
import traceback
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any

from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.judge_step import build_judge, judge_rollout_rows


SHUTDOWN = "__shutdown__"


def _example_from_payload(payload: dict[str, Any]) -> QueryExample:
    return QueryExample(
        query_id=str(payload["query_id"]),
        query=str(payload["query"]),
        answer=str(payload["answer"]) if payload.get("answer") is not None else None,
    )


def run_judge_worker(
    *,
    config_path: str,
    overrides: list[str],
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    config = load_train_config(config_path, parse_cli_overrides(overrides))
    judge = build_judge(config)
    batch_size = max(1, config.judge.batch_size)
    batch_wait_seconds = max(0, config.judge.batch_wait_ms) / 1000.0
    while True:
        first_message = request_queue.get()
        if first_message == SHUTDOWN:
            return
        messages = [first_message]
        row_count = len(first_message.get("rows") or [])
        shutdown_after_batch = False
        deadline = time.monotonic() + batch_wait_seconds
        while row_count < batch_size and batch_wait_seconds > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                message = request_queue.get(timeout=remaining)
            except Empty:
                break
            if message == SHUTDOWN:
                shutdown_after_batch = True
                break
            messages.append(message)
            row_count += len(message.get("rows") or [])

        try:
            rows: list[dict[str, Any]] = []
            row_counts: list[int] = []
            examples_payloads: dict[str, dict[str, Any]] = {}
            expected_checkpoint_ids = {
                message.get("expected_checkpoint_id") for message in messages
            }
            if len(expected_checkpoint_ids) != 1:
                raise ValueError(
                    "Judge micro-batch contains multiple expected checkpoint IDs"
                )
            for message in messages:
                message_rows = message["rows"]
                rows.extend(message_rows)
                row_counts.append(len(message_rows))
                examples_payloads.update(message["examples_by_query_id"])
            examples = {
                query_id: _example_from_payload(example_payload)
                for query_id, example_payload in examples_payloads.items()
            }
            expected_checkpoint_id = next(iter(expected_checkpoint_ids))
            judged_rows: list[dict[str, Any]] = []
            for batch_start in range(0, len(rows), batch_size):
                judged_rows.extend(
                    judge_rollout_rows(
                        rows[batch_start : batch_start + batch_size],
                        judge=judge,
                        examples_by_query_id=examples,
                        expected_checkpoint_id=expected_checkpoint_id,
                    )
                )
            offset = 0
            for message, message_row_count in zip(messages, row_counts):
                response_queue.put(
                    {
                        "batch_id": message["batch_id"],
                        "rows": judged_rows[offset : offset + message_row_count],
                    }
                )
                offset += message_row_count
        except BaseException as exc:
            traceback_text = traceback.format_exc()
            for message in messages:
                response_queue.put(
                    {
                        "batch_id": message["batch_id"],
                        "error": str(exc),
                        "traceback": traceback_text,
                    }
                )
        if shutdown_after_batch:
            return
