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
