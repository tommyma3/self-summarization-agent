from __future__ import annotations

import traceback
from multiprocessing.queues import Queue
from typing import Any


SHUTDOWN = "__shutdown__"


def _example_from_payload(payload: dict[str, Any]):
    from self_summarization_agent.dataset import QueryExample

    return QueryExample(
        query_id=str(payload["query_id"]),
        query=str(payload["query"]),
        answer=str(payload["answer"]) if payload.get("answer") is not None else None,
    )


def run_judge_worker(
    *,
    config_path: str,
    overrides: list[str],
    gpu_ids: list[int],
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    import os

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    from self_summarization_agent.config import load_train_config, parse_cli_overrides
    from self_summarization_agent.judge_step import build_judge, judge_rollout_rows

    config = load_train_config(config_path, parse_cli_overrides(overrides))
    judge = build_judge(config)
    while True:
        message = request_queue.get()
        if message == SHUTDOWN:
            return
        batch_id = message["batch_id"]
        try:
            rows = message["rows"]
            examples = {
                query_id: _example_from_payload(example_payload)
                for query_id, example_payload in message["examples_by_query_id"].items()
            }
            judge_batch_size = message.get("judge_batch_size")
            if judge_batch_size and len(rows) > judge_batch_size:
                # Process in chunks so that each vLLM generate call stays
                # within the judge engine's effective KV-cache concurrency.
                # A heartbeat is sent after every chunk so the parent drain
                # loop sees progress and does not fire the stall timeout.
                judged_rows: list[dict[str, Any]] = []
                for _start in range(0, len(rows), judge_batch_size):
                    _end = _start + judge_batch_size
                    chunk_rows = rows[_start:_end]
                    chunk_judged = judge_rollout_rows(
                        chunk_rows,
                        judge=judge,
                        examples_by_query_id=examples,
                        expected_checkpoint_id=message.get("expected_checkpoint_id"),
                    )
                    judged_rows.extend(chunk_judged)
                    response_queue.put({"batch_id": batch_id, "heartbeat": True})
                response_queue.put({"batch_id": batch_id, "rows": judged_rows})
            else:
                judged_rows = judge_rollout_rows(
                    rows,
                    judge=judge,
                    examples_by_query_id=examples,
                    expected_checkpoint_id=message.get("expected_checkpoint_id"),
                )
                response_queue.put({"batch_id": batch_id, "rows": judged_rows})
        except BaseException as exc:
            response_queue.put(
                {
                    "batch_id": batch_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
