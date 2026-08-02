"""Streaming eval/train collection with overlapped judging and cache scoring."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from typing import Any

from self_summarization_agent.cache_worker import SHUTDOWN as CACHE_SHUTDOWN, run_cache_worker
from self_summarization_agent.checkpoints import checkpoint_id_from_path
from self_summarization_agent.config import (
    load_train_config,
    parse_cli_overrides,
    resolved_rollout_sampling_profile,
    sampling_profile_id,
)
from self_summarization_agent.dataset import load_query_examples, split_train_eval_examples
from self_summarization_agent.eval_metrics import write_eval_metrics
from self_summarization_agent.iteration_artifacts import (
    current_cached_rows_by_key,
    expected_eval_rollout_count,
    expected_train_rollout_count,
    has_complete_cached_rollouts,
    has_complete_judged_rollouts,
    has_complete_raw_rollouts,
    has_eval_metrics,
    have_matching_rollout_keys,
    load_jsonl,
    replace_jsonl,
    rollout_key,
    rows_by_key,
    validated_rows_by_key,
)
from self_summarization_agent.launcher_utils import (
    append_jsonl,
    build_runtime,
    ensure_dir,
    iter_batches,
    serialize_runtime_result,
)
from self_summarization_agent.rollout_collection import (
    _build_overlap_judge_client,
    _build_rollout_generator,
    _configured_task_count,
    _load_completed_rollout_keys,
    _select_collection_examples,
    _temporary_sampling_profile,
)


class _CacheOverlapClient:
    def __init__(
        self,
        *,
        config_path: str,
        overrides: list[str],
        checkpoint_path: str,
        checkpoint_id: str,
        gpu_ids: list[int],
        queue_size: int,
        drain_timeout_seconds: float,
    ) -> None:
        context = mp.get_context("spawn")
        self.request_queue = context.Queue(maxsize=queue_size)
        self.response_queue = context.Queue(maxsize=queue_size)
        self.process = context.Process(
            target=run_cache_worker,
            kwargs={
                "config_path": config_path,
                "overrides": overrides,
                "checkpoint_path": checkpoint_path,
                "gpu_ids": gpu_ids,
                "request_queue": self.request_queue,
                "response_queue": self.response_queue,
            },
            name="ssa-cache-worker",
        )
        self.process.start()
        self.checkpoint_id = checkpoint_id
        self.next_batch_id = 0
        self.pending_count = 0
        self._stall_timeout_seconds = drain_timeout_seconds
        self._last_progress_at = time.monotonic()

    def submit(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.request_queue.put(
            {
                "batch_id": self.next_batch_id,
                "rows": rows,
                "expected_checkpoint_id": self.checkpoint_id,
            }
        )
        self.next_batch_id += 1
        self.pending_count += 1
        self._last_progress_at = time.monotonic()

    def _handle_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        self.pending_count -= 1
        self._last_progress_at = time.monotonic()
        if response.get("error"):
            detail = f"\n{response['traceback']}" if response.get("traceback") else ""
            raise RuntimeError(f"Cache overlap worker failed: {response['error']}{detail}")
        rows = response.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError(f"Cache overlap worker returned invalid response: {response!r}")
        return rows

    def _check_health(self) -> None:
        if not self.process.is_alive():
            raise RuntimeError(
                "Cache overlap worker exited before returning all batches "
                f"(exit_code={self.process.exitcode})"
            )
        if time.monotonic() - self._last_progress_at < self._stall_timeout_seconds:
            return
        self.process.terminate()
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=10)
        raise TimeoutError(
            f"Cache overlap worker made no progress for {self._stall_timeout_seconds:.0f}s"
        )

    def drain_available(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while self.pending_count:
            try:
                response = self.response_queue.get_nowait()
            except Empty:
                self._check_health()
                break
            rows.extend(self._handle_response(response))
        return rows

    def finish(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while self.pending_count:
            try:
                response = self.response_queue.get(timeout=5)
            except Empty:
                self._check_health()
                continue
            rows.extend(self._handle_response(response))
        return rows

    def close(self) -> None:
        if self.process.is_alive():
            self.request_queue.put(CACHE_SHUTDOWN)
            self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30)


def _append_unique_rows(
    path: Path,
    rows: list[dict[str, Any]],
    known_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    appended: list[dict[str, Any]] = []
    for row in rows:
        key = rollout_key(row)
        if key in known_keys:
            continue
        append_jsonl(path, row)
        known_keys.add(key)
        appended.append(row)
    return appended


def _drain_cache_rows(
    cache_client: _CacheOverlapClient | None,
    *,
    cached_output_path: Path | None,
    cached_keys: set[tuple[str, int]],
    finish: bool = False,
) -> None:
    if cache_client is None or cached_output_path is None:
        return
    rows = cache_client.finish() if finish else cache_client.drain_available()
    _append_unique_rows(cached_output_path, rows, cached_keys)


def _route_judged_rows(
    rows: list[dict[str, Any]],
    *,
    judged_output_paths: dict[str, Path],
    judged_keys: dict[str, set[tuple[str, int]]],
    cache_client: _CacheOverlapClient | None,
    cached_output_path: Path | None,
    cached_keys: set[tuple[str, int]],
    cache_submitted_keys: set[tuple[str, int]],
) -> None:
    for row in rows:
        split = row.get("rollout_split")
        if split not in judged_output_paths:
            raise ValueError(f"Judged row has unsupported rollout_split={split!r}")
        appended = _append_unique_rows(
            judged_output_paths[split],
            [row],
            judged_keys[split],
        )
        judged_row = appended[0] if appended else row
        key = rollout_key(judged_row)
        if split == "train" and cache_client is not None and key not in cached_keys:
            if key not in cache_submitted_keys:
                cache_client.submit([judged_row])
                cache_submitted_keys.add(key)
            _drain_cache_rows(
                cache_client,
                cached_output_path=cached_output_path,
                cached_keys=cached_keys,
            )


def _collect_split_streaming(
    *,
    config,
    checkpoint_id: str,
    checkpoint_path: Path,
    generator: Any | None,
    backend: Any | None,
    judge_client: Any | None,
    split: str,
    examples: list[Any],
    raw_output_path: Path,
    judged_output_paths: dict[str, Path],
    judged_keys: dict[str, set[tuple[str, int]]],
    sampling_profile: dict[str, Any],
    profile_id: str,
    group_size: int,
    sample_seed: int | None,
    resume: bool,
    cache_client: _CacheOverlapClient | None,
    cached_output_path: Path | None,
    cached_keys: set[tuple[str, int]],
    cache_submitted_keys: set[tuple[str, int]],
) -> None:
    task_count, task_count_key = _configured_task_count(config, split=split)
    seed = config.experiment.seed if sample_seed is None else sample_seed
    selected_examples = _select_collection_examples(
        examples,
        task_count=task_count,
        task_count_key=task_count_key,
        split=split,
        seed=seed,
    )
    if not selected_examples:
        raise ValueError(f"No {split} queries available for rollout collection")

    example_by_query_id = {example.query_id: example for example in selected_examples}
    all_requests = [
        (example, rollout_index)
        for example in selected_examples
        for rollout_index in range(group_size)
    ]
    expected_keys = {(example.query_id, rollout_index) for example, rollout_index in all_requests}
    require_exact_ids = (
        config.rollout.backend.lower().replace("-", "_") in {"openai", "openai_compatible"}
        and config.rollout.require_exact_token_ids
    )
    completed_raw_keys = (
        _load_completed_rollout_keys(
            raw_output_path,
            checkpoint_id=checkpoint_id,
            expected_keys=expected_keys,
            expected_sampling_profile_id=profile_id if split == "eval" else None,
            require_exact_token_ids=require_exact_ids,
        )
        if resume
        else set()
    )
    raw_rows = rows_by_key(raw_output_path) if resume else {}
    existing_judged_rows = (
        validated_rows_by_key(
            judged_output_paths[split],
            checkpoint_id=checkpoint_id,
            expected_sampling_profile_id=profile_id,
        )
        if resume
        else {}
    )

    missing_judged_keys = completed_raw_keys - set(existing_judged_rows)
    if missing_judged_keys and judge_client is None:
        raise RuntimeError(f"{split} judged rows are incomplete but no judge worker was started")
    for key_batch in iter_batches(sorted(missing_judged_keys), config.rollout.max_concurrent_episodes):
        judge_client.submit(
            [raw_rows[key] for key in key_batch],
            [example_by_query_id[key[0]] for key in key_batch],
        )
        _route_judged_rows(
            judge_client.drain_available(),
            judged_output_paths=judged_output_paths,
            judged_keys=judged_keys,
            cache_client=cache_client,
            cached_output_path=cached_output_path,
            cached_keys=cached_keys,
            cache_submitted_keys=cache_submitted_keys,
        )

    if split == "train" and cache_client is not None:
        missing_cache_keys = set(existing_judged_rows) - cached_keys
        for key_batch in iter_batches(sorted(missing_cache_keys), config.rollout.max_concurrent_episodes):
            cache_client.submit([existing_judged_rows[key] for key in key_batch])
            cache_submitted_keys.update(key_batch)
            _drain_cache_rows(
                cache_client,
                cached_output_path=cached_output_path,
                cached_keys=cached_keys,
            )

    pending_requests = [
        request
        for request in all_requests
        if (request[0].query_id, request[1]) not in completed_raw_keys
    ]
    if not pending_requests:
        return
    if generator is None or backend is None or judge_client is None:
        raise RuntimeError(f"{split} generation requires generator, retrieval, and judge workers")

    with _temporary_sampling_profile(generator, sampling_profile):
        runtime = build_runtime(generator, backend, config.runtime)
        for request_batch in iter_batches(pending_requests, config.rollout.max_concurrent_episodes):
            results = runtime.run_many(
                (example.query_id, example.query) for example, _ in request_batch
            )
            new_rows: list[dict[str, Any]] = []
            batch_examples: list[Any] = []
            for (example, rollout_index), result in zip(request_batch, results):
                row = {
                    "policy_checkpoint_id": checkpoint_id,
                    "policy_checkpoint_path": str(checkpoint_path),
                    "rollout_split": split,
                    "rollout_index": rollout_index,
                    "rollout_samples_per_task": group_size,
                    "sampling_profile": sampling_profile,
                    "sampling_profile_id": profile_id,
                    "trainable_sample_count": None,
                    **serialize_runtime_result(
                        result,
                        query_text=example.query,
                        judge=None,
                        include_rewards=False,
                    ),
                }
                append_jsonl(raw_output_path, row)
                new_rows.append(row)
                batch_examples.append(example)
            judge_client.submit(new_rows, batch_examples)
            _route_judged_rows(
                judge_client.drain_available(),
                judged_output_paths=judged_output_paths,
                judged_keys=judged_keys,
                cache_client=cache_client,
                cached_output_path=cached_output_path,
                cached_keys=cached_keys,
                cache_submitted_keys=cache_submitted_keys,
            )


def run_merged_collect(
    config,
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    train_raw_output: str | Path | None = None,
    train_judged_output: str | Path | None = None,
    train_cached_output: str | Path | None = None,
    eval_raw_output: str | Path | None = None,
    eval_judged_output: str | Path | None = None,
    eval_metrics_output: str | Path | None = None,
    eval_iteration: int | None = None,
    sample_seed: int | None = None,
    resume: bool = False,
    overrides: list[str] | None = None,
    retrieval_worker_url: str | None = None,
) -> dict[str, Path]:
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    overrides = list(overrides or [])
    has_train = train_raw_output is not None
    has_eval = config.dataset.eval_limit > 0 and eval_raw_output is not None
    if not has_train and not has_eval:
        raise ValueError("Merged collection requires at least one train or eval output")
    if not config.judge.enabled:
        raise ValueError("The training collection pipeline requires judge.enabled=true")
    if has_train and (train_judged_output is None or train_cached_output is None):
        raise ValueError("Training collection requires judged and cached output paths")
    if has_eval and (eval_judged_output is None or eval_metrics_output is None):
        raise ValueError("Evaluation collection requires judged and metrics output paths")
    if config.collection.worker_queue_size < 1:
        raise ValueError("collection.worker_queue_size must be at least 1")
    if has_train and not config.collection.cache_gpu_ids:
        raise ValueError("collection.cache_gpu_ids must select at least one cache device")

    output_paths = [
        Path(path)
        for path in (
            train_raw_output,
            train_judged_output,
            train_cached_output,
            eval_raw_output,
            eval_judged_output,
        )
        if path is not None
    ]
    for path in output_paths:
        ensure_dir(path.parent)
        if not resume and path.exists():
            path.unlink()
    if not resume and has_eval and Path(eval_metrics_output).exists():
        metric_iteration = eval_iteration if eval_iteration is not None else 0
        retained_metrics = [
            row
            for row in load_jsonl(Path(eval_metrics_output))
            if not (
                row.get("iteration") == metric_iteration
                and row.get("policy_checkpoint_id") == checkpoint_id
            )
        ]
        replace_jsonl(Path(eval_metrics_output), retained_metrics)

    eval_profile = resolved_rollout_sampling_profile(config, split="eval") if has_eval else {}
    eval_profile_id = sampling_profile_id(eval_profile) if has_eval else ""
    train_profile = resolved_rollout_sampling_profile(config, split="train") if has_train else {}
    train_profile_id = sampling_profile_id(train_profile) if has_train else ""
    eval_count = expected_eval_rollout_count(config) if has_eval else 0
    train_count = expected_train_rollout_count(config) if has_train else 0

    eval_raw_done = not has_eval or (
        resume
        and has_complete_raw_rollouts(
            Path(eval_raw_output),
            checkpoint_id=checkpoint_id,
            expected_count=eval_count,
            expected_sampling_profile_id=eval_profile_id,
        )
    )
    eval_judged_done = not has_eval or (
        resume
        and has_complete_judged_rollouts(
            Path(eval_judged_output),
            checkpoint_id=checkpoint_id,
            expected_count=eval_count,
            require_judge=True,
            expected_sampling_profile_id=eval_profile_id,
        )
    )
    eval_metrics_done = not has_eval or (
        resume
        and has_eval_metrics(
            Path(eval_metrics_output),
            iteration=eval_iteration if eval_iteration is not None else 0,
            policy_checkpoint_id=checkpoint_id,
            expected_sampling_profile_id=eval_profile_id,
        )
    )
    train_raw_done = not has_train or (
        resume
        and has_complete_raw_rollouts(
            Path(train_raw_output),
            checkpoint_id=checkpoint_id,
            expected_count=train_count,
            expected_sampling_profile_id=train_profile_id,
        )
    )
    train_judged_done = not has_train or (
        resume
        and has_complete_judged_rollouts(
            Path(train_judged_output),
            checkpoint_id=checkpoint_id,
            expected_count=train_count,
            require_judge=False,
            expected_sampling_profile_id=train_profile_id,
        )
    )
    train_cached_done = not has_train or (
        resume
        and has_complete_cached_rollouts(
            Path(train_cached_output),
            checkpoint_id=checkpoint_id,
            expected_count=train_count,
            expected_sampling_profile_id=train_profile_id,
        )
    )
    if all(
        (
            eval_raw_done,
            eval_judged_done,
            eval_metrics_done,
            train_raw_done,
            train_judged_done,
            train_cached_done,
        )
    ):
        if has_train and not have_matching_rollout_keys(
            Path(train_raw_output),
            Path(train_judged_output),
            Path(train_cached_output),
        ):
            raise ValueError("Complete train artifacts have different rollout keys")
        if has_eval and not have_matching_rollout_keys(
            Path(eval_raw_output),
            Path(eval_judged_output),
        ):
            raise ValueError("Complete eval artifacts have different rollout keys")
        print("[merged_collect] All outputs complete; nothing to do.", flush=True)
        return {}

    examples = load_query_examples(
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
    judged_paths = {
        **({"eval": Path(eval_judged_output)} if has_eval else {}),
        **({"train": Path(train_judged_output)} if has_train else {}),
    }
    judged_keys = {
        split: set(
            validated_rows_by_key(
                path,
                checkpoint_id=checkpoint_id,
                expected_sampling_profile_id=(eval_profile_id if split == "eval" else train_profile_id),
            )
        )
        if resume
        else set()
        for split, path in judged_paths.items()
    }
    cached_path = Path(train_cached_output) if has_train else None
    current_cached_rows = (
        current_cached_rows_by_key(
            cached_path,
            checkpoint_id=checkpoint_id,
            expected_sampling_profile_id=train_profile_id,
        )
        if resume and cached_path is not None
        else {}
    )
    if resume and cached_path is not None and len(rows_by_key(cached_path)) != len(current_cached_rows):
        replace_jsonl(cached_path, list(current_cached_rows.values()))
    cached_keys = set(current_cached_rows)
    cache_submitted_keys: set[tuple[str, int]] = set()

    needs_generation = (has_eval and not eval_raw_done) or (has_train and not train_raw_done)
    needs_judging = (has_eval and not eval_judged_done) or (has_train and not train_judged_done)
    from self_summarization_agent.bcplus_backend import build_backend

    backend = (
        build_backend(
            config.experiment.bc_plus_root,
            config.retrieval,
            worker_url=retrieval_worker_url,
        )
        if needs_generation
        else None
    )
    judge_client = (
        _build_overlap_judge_client(
            judge=None,
            config_path=str(config_path),
            overrides=overrides,
            checkpoint_id=checkpoint_id,
            gpu_ids=list(config.judge.gpu_ids),
            queue_size=config.collection.worker_queue_size,
            drain_timeout_seconds=config.collection.worker_stall_timeout_seconds,
        )
        if needs_generation or needs_judging
        else None
    )
    generator = _build_rollout_generator(config, checkpoint, split="train") if needs_generation else None
    cache_client: _CacheOverlapClient | None = None
    outputs: dict[str, Path] = {}
    try:
        if has_eval and (not eval_raw_done or not eval_judged_done):
            print("[merged_collect] Streaming evaluation collection and judging...", flush=True)
            _collect_split_streaming(
                config=config,
                checkpoint_id=checkpoint_id,
                checkpoint_path=checkpoint,
                generator=generator,
                backend=backend,
                judge_client=judge_client,
                split="eval",
                examples=eval_examples,
                raw_output_path=Path(eval_raw_output),
                judged_output_paths=judged_paths,
                judged_keys=judged_keys,
                sampling_profile=eval_profile,
                profile_id=eval_profile_id,
                group_size=config.evaluation.samples_per_task,
                sample_seed=None,
                resume=resume,
                cache_client=None,
                cached_output_path=None,
                cached_keys=cached_keys,
                cache_submitted_keys=cache_submitted_keys,
            )
            outputs["eval_raw"] = Path(eval_raw_output)

        if has_train and not train_cached_done:
            print(
                f"[merged_collect] Starting cache worker on GPUs {config.collection.cache_gpu_ids}...",
                flush=True,
            )
            cache_client = _CacheOverlapClient(
                config_path=str(config_path),
                overrides=overrides,
                checkpoint_path=str(checkpoint),
                checkpoint_id=checkpoint_id,
                gpu_ids=list(config.collection.cache_gpu_ids),
                queue_size=config.collection.worker_queue_size,
                drain_timeout_seconds=config.collection.worker_stall_timeout_seconds,
            )

        if has_train and (not train_raw_done or not train_judged_done or not train_cached_done):
            print("[merged_collect] Streaming training collection, judging, and caching...", flush=True)
            _collect_split_streaming(
                config=config,
                checkpoint_id=checkpoint_id,
                checkpoint_path=checkpoint,
                generator=generator,
                backend=backend,
                judge_client=judge_client,
                split="train",
                examples=train_examples,
                raw_output_path=Path(train_raw_output),
                judged_output_paths=judged_paths,
                judged_keys=judged_keys,
                sampling_profile=train_profile,
                profile_id=train_profile_id,
                group_size=config.training.group_size,
                sample_seed=sample_seed,
                resume=resume,
                cache_client=cache_client,
                cached_output_path=cached_path,
                cached_keys=cached_keys,
                cache_submitted_keys=cache_submitted_keys,
            )
            outputs["train_raw"] = Path(train_raw_output)

        if judge_client is not None:
            _route_judged_rows(
                judge_client.finish(),
                judged_output_paths=judged_paths,
                judged_keys=judged_keys,
                cache_client=cache_client,
                cached_output_path=cached_path,
                cached_keys=cached_keys,
                cache_submitted_keys=cache_submitted_keys,
            )
        _drain_cache_rows(
            cache_client,
            cached_output_path=cached_path,
            cached_keys=cached_keys,
            finish=True,
        )

        if has_train and not have_matching_rollout_keys(
            Path(train_raw_output),
            Path(train_judged_output),
            Path(train_cached_output),
        ):
            raise RuntimeError("Train raw, judged, and cached artifacts have different rollout keys")
        if has_eval and not have_matching_rollout_keys(
            Path(eval_raw_output),
            Path(eval_judged_output),
        ):
            raise RuntimeError("Eval raw and judged artifacts have different rollout keys")

        if has_eval and not eval_metrics_done:
            write_eval_metrics(
                judged_rollout_path=eval_judged_output,
                metrics_path=eval_metrics_output,
                iteration=eval_iteration if eval_iteration is not None else 0,
                policy_checkpoint_id=checkpoint_id,
            )
            outputs["eval_metrics"] = Path(eval_metrics_output)
        if has_eval:
            outputs["eval_judged"] = Path(eval_judged_output)
        if has_train:
            outputs["train_judged"] = Path(train_judged_output)
            outputs["train_cached"] = Path(train_cached_output)
    finally:
        if cache_client is not None:
            cache_client.close()
        if judge_client is not None:
            judge_client.close()
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream evaluation/training collection with overlapped judging and caching."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-raw-output", default=None)
    parser.add_argument("--train-judged-output", default=None)
    parser.add_argument("--train-cached-output", default=None)
    parser.add_argument("--eval-raw-output", default=None)
    parser.add_argument("--eval-judged-output", default=None)
    parser.add_argument("--eval-metrics-output", default=None)
    parser.add_argument("--eval-iteration", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrieval-worker-url", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    outputs = run_merged_collect(
        config,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        train_raw_output=args.train_raw_output,
        train_judged_output=args.train_judged_output,
        train_cached_output=args.train_cached_output,
        eval_raw_output=args.eval_raw_output,
        eval_judged_output=args.eval_judged_output,
        eval_metrics_output=args.eval_metrics_output,
        eval_iteration=args.eval_iteration,
        sample_seed=args.sample_seed,
        resume=args.resume,
        overrides=args.overrides,
        retrieval_worker_url=args.retrieval_worker_url,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
