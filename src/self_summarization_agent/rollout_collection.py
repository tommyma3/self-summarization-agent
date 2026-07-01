from __future__ import annotations

import argparse
import multiprocessing as mp
from queue import Empty
from dataclasses import replace
import json
import os
from pathlib import Path
import random
from typing import Any

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.checkpoints import checkpoint_id_from_path, resolve_latest_checkpoint
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample, load_query_examples, split_train_eval_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.judge_step import judge_rollout_rows
from self_summarization_agent.judge_worker import SHUTDOWN, run_judge_worker
from self_summarization_agent.launcher_utils import (
    append_jsonl,
    build_runtime,
    ensure_dir,
    iter_batches,
    serialize_runtime_result,
)
from self_summarization_agent.rewards import (
    apply_malformed_tool_penalty,
    apply_terminal_reward,
    trainable_turn_ids_from_records,
)
from self_summarization_agent.trajectory import extract_trainable_samples


def apply_judged_rewards(result, example: QueryExample, judge: Any) -> dict[str, Any]:
    trainable_turn_ids = trainable_turn_ids_from_records(result.turn_records)
    if result.status == "malformed_tool_call":
        result.turn_rewards = apply_malformed_tool_penalty(trainable_turn_ids)
        return {"outcome": "malformed_tool_call", "judge_prompt": None, "judge_response": None, "parse_error": False}
    decision = judge.evaluate(example, result.status, result.final_answer or "")
    result.turn_rewards = apply_terminal_reward(
        outcome=decision.outcome,
        trainable_turn_ids=trainable_turn_ids,
    )
    return {
        "outcome": decision.outcome,
        "judge_prompt": decision.judge_prompt,
        "judge_response": decision.judge_response,
        "parse_error": decision.parse_error,
    }


def _load_completed_rollout_keys(
    rollout_path: Path,
    *,
    checkpoint_id: str,
    expected_keys: set[tuple[str, int]],
) -> set[tuple[str, int]]:
    if not rollout_path.exists():
        return set()

    completed_keys: set[tuple[str, int]] = set()
    with rollout_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid rollout JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Rollout row {line_number} must be a JSON object")
            row_checkpoint_id = row.get("policy_checkpoint_id")
            if row_checkpoint_id != checkpoint_id:
                raise ValueError(
                    f"Cannot resume {rollout_path}: line {line_number} has checkpoint "
                    f"{row_checkpoint_id!r}, expected {checkpoint_id!r}"
                )
            query_id = row.get("query_id")
            rollout_index = row.get("rollout_index")
            if not isinstance(query_id, str) or not isinstance(rollout_index, int):
                raise ValueError(f"Rollout row {line_number} is missing query_id or rollout_index")
            key = (query_id, rollout_index)
            if key not in expected_keys:
                raise ValueError(
                    f"Cannot resume {rollout_path}: line {line_number} contains unexpected rollout key {key!r}"
                )
            completed_keys.add(key)
    return completed_keys


def _example_payload(example: QueryExample) -> dict[str, Any]:
    return {
        "query_id": example.query_id,
        "query": example.query,
        "answer": example.answer,
    }


def _configured_task_count(config, *, split: str) -> tuple[int | None, str | None]:
    if split == "train":
        if config.collection.train_task_count is not None:
            return config.collection.train_task_count, "collection.train_task_count"
        if config.training.rollout_query_count is not None:
            return config.training.rollout_query_count, "training.rollout_query_count"
        return None, None
    if split == "eval":
        if config.collection.eval_task_count is not None:
            return config.collection.eval_task_count, "collection.eval_task_count"
        return None, None
    raise ValueError(f"Unsupported rollout split: {split}")


def _select_collection_examples(
    examples: list[QueryExample],
    *,
    task_count: int | None,
    task_count_key: str | None,
    split: str,
    seed: int,
) -> list[QueryExample]:
    if task_count is None:
        return examples
    if task_count < 1:
        raise ValueError(f"{task_count_key} must be at least 1, got {task_count}")
    if task_count > len(examples):
        raise ValueError(
            f"{task_count_key} cannot exceed available {split} queries: "
            f"{task_count} > {len(examples)}"
        )
    if split == "train":
        return random.Random(seed).sample(examples, task_count)
    return examples[:task_count]


class _InProcessOverlapJudgeClient:
    def __init__(self, *, judge: Any, checkpoint_id: str) -> None:
        self.judge = judge
        self.checkpoint_id = checkpoint_id
        self.completed_rows: list[dict[str, Any]] = []

    def submit(self, rows: list[dict[str, Any]], examples: list[QueryExample]) -> None:
        examples_by_query_id = {example.query_id: example for example in examples}
        self.completed_rows.extend(
            judge_rollout_rows(
                rows,
                judge=self.judge,
                examples_by_query_id=examples_by_query_id,
                expected_checkpoint_id=self.checkpoint_id,
            )
        )

    def drain_available(self) -> list[dict[str, Any]]:
        rows = self.completed_rows
        self.completed_rows = []
        return rows

    def finish(self) -> list[dict[str, Any]]:
        return self.drain_available()

    def close(self) -> None:
        return


class _SubprocessOverlapJudgeClient:
    def __init__(
        self,
        *,
        config_path: str,
        overrides: list[str],
        checkpoint_id: str,
    ) -> None:
        context = mp.get_context("spawn")
        self.request_queue = context.Queue()
        self.response_queue = context.Queue()
        self.process = context.Process(
            target=run_judge_worker,
            kwargs={
                "config_path": config_path,
                "overrides": overrides,
                "request_queue": self.request_queue,
                "response_queue": self.response_queue,
            },
        )
        self.process.start()
        self.checkpoint_id = checkpoint_id
        self.next_batch_id = 0
        self.pending_batch_count = 0

    def submit(self, rows: list[dict[str, Any]], examples: list[QueryExample]) -> None:
        examples_by_query_id = {example.query_id: _example_payload(example) for example in examples}
        self.request_queue.put(
            {
                "batch_id": self.next_batch_id,
                "rows": rows,
                "examples_by_query_id": examples_by_query_id,
                "expected_checkpoint_id": self.checkpoint_id,
            }
        )
        self.next_batch_id += 1
        self.pending_batch_count += 1

    def _handle_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        self.pending_batch_count -= 1
        if response.get("error"):
            traceback_text = response.get("traceback")
            detail = f"\n{traceback_text}" if traceback_text else ""
            raise RuntimeError(f"Overlap judge worker failed: {response['error']}{detail}")
        rows = response.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError(f"Overlap judge worker returned invalid response: {response!r}")
        return rows

    def drain_available(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while self.pending_batch_count:
            try:
                response = self.response_queue.get_nowait()
            except Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        "Overlap judge worker exited before returning all batches "
                        f"(exit_code={self.process.exitcode})"
                    )
                break
            rows.extend(self._handle_response(response))
        return rows

    def finish(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while self.pending_batch_count:
            try:
                response = self.response_queue.get(timeout=5)
            except Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        "Overlap judge worker exited before returning all batches "
                        f"(exit_code={self.process.exitcode})"
                    )
                continue
            rows.extend(self._handle_response(response))
        return rows

    def close(self) -> None:
        if self.process.is_alive():
            self.request_queue.put(SHUTDOWN)
            self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30)


def _build_overlap_judge_client(
    *,
    judge: Any | None,
    config_path: str | Path | None,
    overrides: list[str],
    checkpoint_id: str,
) -> Any:
    if judge is not None:
        return _InProcessOverlapJudgeClient(judge=judge, checkpoint_id=checkpoint_id)
    if config_path is None:
        raise ValueError("config_path is required when overlap judging without an injected judge")
    return _SubprocessOverlapJudgeClient(
        config_path=str(config_path),
        overrides=overrides,
        checkpoint_id=checkpoint_id,
    )


def collect_rollouts(
    config,
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    examples: list[QueryExample] | None = None,
    backend: Any | None = None,
    generator: Any | None = None,
    judge: Any | None = None,
    resume: bool = False,
    judge_inline: bool = False,
    judged_output_path: str | Path | None = None,
    overlap_judge: bool | None = None,
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
    sample_seed: int | None = None,
    split: str = "train",
    retrieval_worker_url: str | None = None,
) -> Path:
    if judge_inline and judged_output_path is not None:
        raise ValueError("judge_inline and judged_output_path cannot be used together")
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    overrides = list(overrides or [])
    examples = examples or load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    train_examples, eval_examples = split_train_eval_examples(
        examples,
        train_limit=config.dataset.train_limit,
        eval_limit=config.dataset.eval_limit,
    )
    if split == "train":
        selected_examples = train_examples
    elif split == "eval":
        selected_examples = eval_examples
    else:
        raise ValueError(f"Unsupported rollout split: {split}")

    task_count, task_count_key = _configured_task_count(config, split=split)
    seed = config.experiment.seed if sample_seed is None else sample_seed
    selected_examples = _select_collection_examples(
        selected_examples,
        task_count=task_count,
        task_count_key=task_count_key,
        split=split,
        seed=seed,
    )
    if not selected_examples:
        raise ValueError(f"No {split} queries available for rollout collection")

    rollout_path = Path(output_path)
    ensure_dir(rollout_path.parent)
    judged_path = Path(judged_output_path) if judged_output_path is not None else None
    should_overlap_judge = bool(
        judged_path is not None and (config.rollout.overlap_judge if overlap_judge is None else overlap_judge)
    )

    rollout_requests = [
        (example, rollout_index)
        for example in selected_examples
        for rollout_index in range(config.training.group_size)
    ]
    expected_keys = {(example.query_id, rollout_index) for example, rollout_index in rollout_requests}
    if resume:
        completed_keys = _load_completed_rollout_keys(
            rollout_path,
            checkpoint_id=checkpoint_id,
            expected_keys=expected_keys,
        )
        rollout_requests = [
            (example, rollout_index)
            for example, rollout_index in rollout_requests
            if (example.query_id, rollout_index) not in completed_keys
        ]
        if not rollout_requests:
            return rollout_path
    elif rollout_path.exists():
        rollout_path.unlink()
    if should_overlap_judge:
        ensure_dir(judged_path.parent)
        if not resume and judged_path.exists():
            judged_path.unlink()

    # Build retrieval before narrowing CUDA visibility for vLLM. The FAISS backend can
    # load its embedding model on the normal/default device. The overlap judge worker
    # is also started before the parent is restricted to config.rollout.gpu_ids.
    backend = backend or build_backend(
        config.experiment.bc_plus_root,
        config.retrieval,
        worker_url=retrieval_worker_url,
    )
    overlap_judge_client = None
    if should_overlap_judge:
        overlap_judge_client = _build_overlap_judge_client(
            judge=judge,
            config_path=config_path,
            overrides=overrides,
            checkpoint_id=checkpoint_id,
        )
    if generator is None and config.rollout.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in config.rollout.gpu_ids)
    rollout_model_config = replace(
        config.model,
        backend=config.rollout.backend,
        model_path=str(checkpoint),
        max_new_tokens=config.rollout.max_new_tokens
        if config.rollout.max_new_tokens is not None
        else config.model.max_new_tokens,
        temperature=config.rollout.temperature
        if config.rollout.temperature is not None
        else config.model.temperature,
        top_p=config.rollout.top_p if config.rollout.top_p is not None else config.model.top_p,
        do_sample=config.rollout.do_sample
        if config.rollout.do_sample is not None
        else config.model.do_sample,
        tensor_parallel_size=config.rollout.tensor_parallel_size,
        attention_backend=config.rollout.attention_backend,
        max_model_len=config.rollout.max_model_len
        if config.rollout.max_model_len is not None
        else config.model.max_model_len,
        language_model_only=True,
    )
    generator = generator or build_generator(rollout_model_config)
    if judge_inline:
        judge = judge or RewardJudge(build_generator(config.model, judge_config=config.judge))
    runtime = build_runtime(
        generator,
        backend,
        config.runtime,
        cache_policy_checkpoint_id=checkpoint_id,
    )

    try:
        for request_batch in iter_batches(rollout_requests, config.rollout.max_concurrent_episodes):
            results = runtime.run_many((example.query_id, example.query) for example, _ in request_batch)
            overlap_rows: list[dict[str, Any]] = []
            overlap_examples: list[QueryExample] = []
            for (example, rollout_index), result in zip(request_batch, results):
                judge_payload = None
                trainable_sample_count = None
                include_rewards = False
                if judge_inline:
                    judge_payload = apply_judged_rewards(result, example, judge)
                    include_rewards = True
                    trainable_sample_count = len(extract_trainable_samples(result.turn_records, result.turn_rewards))
                row = {
                    "policy_checkpoint_id": checkpoint_id,
                    "policy_checkpoint_path": str(checkpoint),
                    "rollout_index": rollout_index,
                    "trainable_sample_count": trainable_sample_count,
                    **serialize_runtime_result(
                        result,
                        query_text=example.query,
                        judge={**judge_payload, "rollout_index": rollout_index} if judge_payload else None,
                        include_rewards=include_rewards,
                    ),
                }
                append_jsonl(rollout_path, row)
                if overlap_judge_client is not None:
                    overlap_rows.append(row)
                    overlap_examples.append(example)
            if overlap_judge_client is not None and overlap_rows:
                overlap_judge_client.submit(overlap_rows, overlap_examples)
                for judged_row in overlap_judge_client.drain_available():
                    append_jsonl(judged_path, judged_row)
        if overlap_judge_client is not None:
            for judged_row in overlap_judge_client.finish():
                append_jsonl(judged_path, judged_row)
    finally:
        if overlap_judge_client is not None:
            overlap_judge_client.close()
    return rollout_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one training pass of rollout artifacts with offline vLLM.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to latest under train dir.")
    parser.add_argument("--latest-root", default=None, help="Directory containing the latest checkpoint pointer.")
    parser.add_argument("--output", required=True, help="Rollout JSONL output path.")
    parser.add_argument("--resume", action="store_true", help="Append missing rollouts and skip rows already in output.")
    parser.add_argument("--judge-inline", action="store_true", help="Judge rollouts during collection instead of writing raw rows.")
    parser.add_argument("--judged-output", default=None, help="Optional judged JSONL output path for overlap judging.")
    parser.add_argument(
        "--no-overlap-judge",
        action="store_true",
        help="Do not overlap collection with judging even when judged output is requested.",
    )
    parser.add_argument("--sample-seed", type=int, default=None, help="Seed for per-iteration training-query sampling.")
    parser.add_argument("--split", choices=["train", "eval"], default="train", help="Dataset split to collect.")
    parser.add_argument("--retrieval-worker-url", default=None, help="Use a persistent retrieval worker at this URL.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
    else:
        latest_root = args.latest_root or Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name
        checkpoint = resolve_latest_checkpoint(latest_root).path
    rollout_path = collect_rollouts(
        config,
        checkpoint_path=checkpoint,
        output_path=args.output,
        resume=args.resume,
        judge_inline=args.judge_inline,
        judged_output_path=args.judged_output,
        overlap_judge=not args.no_overlap_judge,
        config_path=args.config,
        overrides=args.overrides,
        sample_seed=args.sample_seed,
        split=args.split,
        retrieval_worker_url=args.retrieval_worker_url,
    )
    print(rollout_path)


if __name__ == "__main__":
    main()
