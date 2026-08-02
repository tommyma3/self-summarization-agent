from __future__ import annotations

import os
import traceback
from multiprocessing.queues import Queue


SHUTDOWN = "__cache_shutdown__"


def run_cache_worker(
    *,
    config_path: str,
    overrides: list[str],
    checkpoint_path: str,
    gpu_ids: list[int],
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    """Load CUDA-dependent cache code only after selecting worker devices."""
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    from self_summarization_agent.cache_step import _attach_training_caches, build_cache_scorer
    from self_summarization_agent.config import load_train_config, parse_cli_overrides
    from self_summarization_agent.trajectory import extract_trainable_samples

    config = load_train_config(config_path, parse_cli_overrides(overrides))
    scorer = build_cache_scorer(config, checkpoint_path=checkpoint_path)
    while True:
        message = request_queue.get()
        if message == SHUTDOWN:
            return
        batch_id = message["batch_id"]
        try:
            cached_rows = []
            for row in message["rows"]:
                if row.get("policy_checkpoint_id") != message["expected_checkpoint_id"]:
                    raise ValueError(
                        "Cache worker received a rollout from checkpoint "
                        f"{row.get('policy_checkpoint_id')!r}; expected "
                        f"{message['expected_checkpoint_id']!r}"
                    )
                samples = extract_trainable_samples(
                    row["trajectory_records"],
                    row["turn_rewards"],
                    rollout_id=f"{row.get('query_id')}:{row.get('rollout_index')}",
                )
                if not samples:
                    cached_rows.append(dict(row))
                    continue
                cached_rows.append(
                    _attach_training_caches(
                        row,
                        cache_payloads=scorer.cache_samples(samples),
                        checkpoint_id=message["expected_checkpoint_id"],
                    )
                )
            response_queue.put({"batch_id": batch_id, "rows": cached_rows})
        except BaseException as exc:
            response_queue.put(
                {
                    "batch_id": batch_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
