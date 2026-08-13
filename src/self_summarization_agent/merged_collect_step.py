"""Merged collection with strict GPU lifecycle boundaries.

The parent process remains CUDA-free and supervises three sequential phases::

    collection workers (policy GPUs): eval, then train + native logprobs
    judge worker (judge GPUs):        eval and train answer judging
    cache fallback (GPU 0):           only rows missing native logprobs

The retrieval worker is owned by the collection phase.  It and every policy
worker must exit before the judge worker is allowed to start.  Process exit,
rather than best-effort object deletion, is the authoritative policy-engine
teardown boundary.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import signal
import sys
import time
import traceback
from typing import Any

from self_summarization_agent.cache_step import (
    _attach_training_caches,
    _completed_cached_rows,
    _load_rollout_rows,
    _materialize_rollout_native_training_caches,
    _row_has_current_training_cache,
    _validate_judged_row,
    build_cache_scorer,
)
from self_summarization_agent.checkpoints import checkpoint_id_from_path
from self_summarization_agent.config import (
    load_train_config,
    parse_cli_overrides,
    resolved_rollout_sampling_profile,
    sampling_profile_id,
)
from self_summarization_agent.dataset import load_query_examples, split_train_eval_examples
from self_summarization_agent.eval_metrics import write_eval_metrics
from self_summarization_agent.iteration_launcher import (
    _expected_eval_rollout_count,
    _expected_train_rollout_count,
    _has_complete_cached_rollouts,
    _has_complete_judged_rollouts,
    _has_complete_raw_rollouts,
    _has_eval_metrics,
    _start_retrieval_worker,
    _stop_retrieval_worker,
)
from self_summarization_agent.launcher_utils import (
    append_jsonl,
    build_runtime,
    ensure_dir,
    iter_batches,
    serialize_runtime_result,
)
from self_summarization_agent.judge_worker import READY
from self_summarization_agent.rollout_collection import (
    _build_overlap_judge_client,
    _build_rollout_generator,
    _configured_task_count,
    _load_completed_rollout_keys,
    _select_collection_examples,
    _temporary_sampling_profile,
)
from self_summarization_agent.trajectory import extract_trainable_samples

# ---------------------------------------------------------------------------
# Cache fallback worker (spawned process on GPU 0 after judging)
# ---------------------------------------------------------------------------

_CACHE_SHUTDOWN = "__cache_shutdown__"


def _run_cache_overlap_worker(
    *,
    config_path: str,
    overrides: list[str],
    checkpoint_path: str,
    request_queue: mp.queues.Queue,
    response_queue: mp.queues.Queue,
) -> None:
    """Spawned process that loads the policy checkpoint on GPU 0 and computes
    reference logprob caches for judged rollout rows."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = load_train_config(config_path, parse_cli_overrides(overrides))
    scorer = build_cache_scorer(config, checkpoint_path=checkpoint_path)
    response_queue.put(READY)
    while True:
        message = request_queue.get()
        if message == _CACHE_SHUTDOWN:
            return
        batch_id = message["batch_id"]
        try:
            rows = message["rows"]
            expected_checkpoint_id = message["expected_checkpoint_id"]
            samples_by_row = [
                extract_trainable_samples(
                    row["trajectory_records"],
                    row["turn_rewards"],
                    rollout_id=f"{row.get('query_id')}:{row.get('rollout_index')}",
                )
                for row in rows
            ]
            all_samples = [sample for row_samples in samples_by_row for sample in row_samples]
            all_cache_payloads: list[dict[str, Any]] = []
            cache_microbatch_size = max(
                1,
                config.training.gradient_accumulation_microbatch_size,
            )
            for sample_batch in iter_batches(all_samples, cache_microbatch_size):
                all_cache_payloads.extend(scorer.cache_samples(sample_batch))
            cached_rows: list[dict[str, Any]] = []
            payload_offset = 0
            for row, row_samples in zip(rows, samples_by_row):
                if not row_samples:
                    cached_rows.append(dict(row))
                    continue
                next_offset = payload_offset + len(row_samples)
                cache_payloads = all_cache_payloads[payload_offset:next_offset]
                payload_offset = next_offset
                cached_row = _attach_training_caches(
                    row,
                    cache_payloads=cache_payloads,
                    checkpoint_id=expected_checkpoint_id,
                )
                cached_rows.append(cached_row)
            response_queue.put({"batch_id": batch_id, "rows": cached_rows})
        except BaseException as exc:
            response_queue.put(
                {
                    "batch_id": batch_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


# ---------------------------------------------------------------------------
# Cache fallback client (manages the spawned cache-worker lifecycle)
# ---------------------------------------------------------------------------

class _CacheOverlapClient:
    """Lazily manages a post-judge cache-fallback process on GPU 0."""

    def __init__(
        self,
        *,
        config_path: str,
        overrides: list[str],
        checkpoint_path: str,
        checkpoint_id: str,
        queue_max_batches: int = 8,
        drain_timeout_seconds: float = 600,
    ) -> None:
        self._context = mp.get_context("spawn")
        self.request_queue = self._context.Queue(maxsize=max(1, queue_max_batches))
        self.response_queue = self._context.Queue()
        self._worker_kwargs = {
            "config_path": config_path,
            "overrides": overrides,
            "checkpoint_path": checkpoint_path,
            "request_queue": self.request_queue,
            "response_queue": self.response_queue,
        }
        self.process: mp.Process | None = None
        self.checkpoint_id = checkpoint_id
        self.next_batch_id = 0
        self.pending_count = 0
        self._drain_timeout_seconds = drain_timeout_seconds
        self._drain_deadline: float | None = None
        self.submitted_row_count = 0
        self.completed_row_count = 0
        self.rollout_native_row_count = 0
        self.queue_block_seconds = 0.0

    def _ensure_started(self) -> None:
        if self.process is not None:
            return
        self.process = self._context.Process(
            target=_run_cache_overlap_worker,
            kwargs=self._worker_kwargs,
        )
        self.process.start()
        # Wait for worker to signal successful initialization (model load).
        try:
            signal = self.response_queue.get(timeout=600)
        except Empty:
            self.process.kill()
            self.process.join(timeout=30)
            raise RuntimeError(
                "Cache overlap worker failed to initialize within 600s startup timeout"
            )
        if not self.process.is_alive():
            raise RuntimeError(
                f"Cache overlap worker exited during startup "
                f"(exit_code={self.process.exitcode})"
            )
        if signal != READY:
            self.process.kill()
            self.process.join(timeout=30)
            raise RuntimeError(
                f"Unexpected startup signal from cache overlap worker: {signal!r}"
            )

    def _put_request(self, message: dict[str, Any]) -> None:
        self._ensure_started()
        assert self.process is not None
        if not self.process.is_alive():
            raise RuntimeError(
                f"Cache overlap worker exited "
                f"(exit_code={self.process.exitcode})"
            )
        started = time.monotonic()
        deadline = started + self._drain_timeout_seconds
        while True:
            try:
                self.request_queue.put(message, timeout=5)
                self.queue_block_seconds += time.monotonic() - started
                return
            except Full:
                if not self.process.is_alive():
                    raise RuntimeError(
                        "Cache overlap worker exited while its request queue was full "
                        f"(exit_code={self.process.exitcode})"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for space in the cache overlap request queue"
                    )

    # ------------------------------------------------------------------
    # Submit / drain / finish (mirrors _SubprocessOverlapJudgeClient)
    # ------------------------------------------------------------------

    def submit(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._put_request(
            {
                "batch_id": self.next_batch_id,
                "rows": rows,
                "expected_checkpoint_id": self.checkpoint_id,
            }
        )
        self.next_batch_id += 1
        self.pending_count += 1
        self.submitted_row_count += len(rows)
        if self._drain_deadline is None:
            self._drain_deadline = time.monotonic() + self._drain_timeout_seconds

    def record_rollout_native_rows(self, count: int) -> None:
        self.rollout_native_row_count += count

    def _handle_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        self.pending_count -= 1
        if response.get("error"):
            traceback_text = response.get("traceback")
            detail = f"\n{traceback_text}" if traceback_text else ""
            raise RuntimeError(f"Cache overlap worker failed: {response['error']}{detail}")
        rows = response.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError(f"Cache overlap worker returned invalid response: {response!r}")
        self.completed_row_count += len(rows)
        self._drain_deadline = (
            time.monotonic() + self._drain_timeout_seconds
            if self.pending_count
            else None
        )
        return rows

    def _ensure_drain_deadline(self) -> None:
        if not self.pending_count:
            self._drain_deadline = None
        elif self._drain_deadline is None:
            self._drain_deadline = time.monotonic() + self._drain_timeout_seconds

    def _check_drain_timeout(self) -> None:
        if self._drain_deadline is None:
            return
        if time.monotonic() < self._drain_deadline:
            return
        assert self.process is not None
        pid = self.process.pid
        print(
            f"[cache_overlap] Drain timeout ({self._drain_timeout_seconds:.0f}s) "
            f"reached with {self.pending_count} batch(es) still pending. "
            f"Terminating cache worker (pid={pid}). "
            f"Missing caches will be re-generated on --resume.",
            flush=True,
        )
        self.process.terminate()
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=10)

    def drain_available(self) -> list[dict[str, Any]]:
        self._ensure_drain_deadline()
        if self.pending_count:
            self._drain_deadline = time.monotonic() + self._drain_timeout_seconds
        rows: list[dict[str, Any]] = []
        while self.pending_count:
            try:
                response = self.response_queue.get_nowait()
            except Empty:
                assert self.process is not None
                if not self.process.is_alive():
                    raise RuntimeError(
                        "Cache overlap worker exited before returning all batches "
                        f"(exit_code={self.process.exitcode})"
                    )
                self._check_drain_timeout()
                if not self.process.is_alive():
                    break
                break
            rows.extend(self._handle_response(response))
        return rows

    def finish(self) -> list[dict[str, Any]]:
        if self.pending_count:
            self._drain_deadline = time.monotonic() + self._drain_timeout_seconds
        rows: list[dict[str, Any]] = []
        while self.pending_count:
            try:
                response = self.response_queue.get(timeout=5)
            except Empty:
                assert self.process is not None
                if not self.process.is_alive():
                    raise RuntimeError(
                        "Cache overlap worker exited before returning all batches "
                        f"(exit_code={self.process.exitcode})"
                    )
                self._check_drain_timeout()
                if not self.process.is_alive():
                    break
                continue
            rows.extend(self._handle_response(response))
        return rows

    def metrics(self) -> dict[str, Any]:
        return {
            "submitted_rows": self.submitted_row_count,
            "completed_rows": self.completed_row_count,
            "rollout_native_rows": self.rollout_native_row_count,
            "pending_batches": self.pending_count,
            "queue_block_seconds": self.queue_block_seconds,
            "fallback_worker_started": self.process is not None,
        }

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            try:
                self.request_queue.put(_CACHE_SHUTDOWN, timeout=5)
            except Full:
                pass
            else:
                self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=10)


# ---------------------------------------------------------------------------
# Per-split collection helper
# ---------------------------------------------------------------------------

def _collect_split(
    *,
    config,
    checkpoint_id: str,
    checkpoint_path: Path,
    generator: Any,
    backend: Any,
    split: str,
    examples: list[Any],
    raw_output_path: Path,
    sampling_profile: dict[str, Any],
    profile_id: str,
    group_size: int,
    sample_seed: int | None,
    resume: bool,
) -> None:
    """Collect one split without constructing or contacting a judge model."""

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

    rollout_requests = [
        (example, rollout_index)
        for example in selected_examples
        for rollout_index in range(group_size)
    ]
    expected_keys = {(example.query_id, rollout_index) for example, rollout_index in rollout_requests}
    completed_raw_keys: set[tuple[str, int]] = set()

    if resume:
        completed_raw_keys = _load_completed_rollout_keys(
            raw_output_path,
            checkpoint_id=checkpoint_id,
            expected_keys=expected_keys,
            expected_sampling_profile_id=profile_id if split == "eval" else None,
            require_exact_token_ids=False,
        )
        rollout_requests = [
            (example, rollout_index)
            for example, rollout_index in rollout_requests
            if (example.query_id, rollout_index) not in completed_raw_keys
        ]
    elif raw_output_path.exists():
        raw_output_path.unlink()

    collection_started = time.monotonic()
    completed_batch_count = 0
    generated_row_count = 0
    with _temporary_sampling_profile(generator, sampling_profile):
        runtime = build_runtime(generator, backend, config.runtime)
        episode_inputs = [
            (example.query_id, example.query) for example, _ in rollout_requests
        ]
        for completed_batch in runtime.run_many_stream(
            episode_inputs,
            max_active_episodes=config.rollout.max_concurrent_episodes,
        ):
            completed_batch_count += 1
            for request_index, result in completed_batch:
                example, rollout_index = rollout_requests[request_index]
                trainable_sample_count = None
                row = {
                    "policy_checkpoint_id": checkpoint_id,
                    "policy_checkpoint_path": str(checkpoint_path),
                    "rollout_split": split,
                    "rollout_index": rollout_index,
                    "rollout_samples_per_task": group_size,
                    "sampling_profile": sampling_profile,
                    "sampling_profile_id": profile_id,
                    "trainable_sample_count": trainable_sample_count,
                    **serialize_runtime_result(
                        result,
                        query_text=example.query,
                        judge=None,
                        include_rewards=False,
                    ),
                }
                if split == "train":
                    # This is a reward-independent transformation of exact
                    # collection IDs and raw sampled-token logprobs.  Persist it
                    # before the policy process exits so normal rows never need
                    # a second policy-model load.
                    row = _materialize_rollout_native_training_caches(
                        row,
                        checkpoint_id=checkpoint_id,
                    )
                append_jsonl(raw_output_path, row)
                generated_row_count += 1

    print(
        "[merged_collect] "
        + json.dumps(
            {
                "event": "streaming_collection_complete",
                "split": split,
                "generated_rows": generated_row_count,
                "completion_batches": completed_batch_count,
                "elapsed_seconds": time.monotonic() - collection_started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _terminate_collection_child(generator: Any | None) -> None:
    """SIGKILL the offline policy engine's subprocess tree from the child.

    sglang's Engine spawns scheduler, tensor-parallel, and detokenizer
    subprocesses and registers an atexit shutdown hook, but the collection
    child then hangs in Python interpreter teardown (zmq context teardown,
    mp finalizers, GC) after a completed split.  Every rollout row is
    already persisted by append_jsonl before the completion event, so the
    parent only needs exit code 0 and the GPUs released; bypassing
    interpreter teardown with os._exit is therefore safe.

    This is the same bounded psutil operation sglang's Engine.shutdown()
    performs.  It is invoked directly rather than via engine.shutdown() to
    avoid any watchdog or zmq involvement in the teardown path.  It is a
    no-op for backends without subprocesses (transformers,
    openai_compatible, vllm offline).
    """
    try:
        from sglang.srt.utils.common import kill_process_tree
    except ImportError:
        return
    try:
        kill_process_tree(os.getpid(), include_parent=False)
    except Exception:
        pass


def _run_split_collection_worker(
    *,
    config_path: str,
    overrides: list[str],
    checkpoint_path: str,
    split: str,
    raw_output_path: str,
    sample_seed: int | None,
    resume: bool,
    retrieval_worker_url: str | None,
) -> None:
    """Child entrypoint that owns one split's policy engine and CUDA state."""

    config = load_train_config(config_path, parse_cli_overrides(overrides))
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_id = checkpoint_id_from_path(checkpoint)
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
    split_examples = eval_examples if split == "eval" else train_examples
    sampling_profile = resolved_rollout_sampling_profile(config, split=split)
    profile_id = sampling_profile_id(sampling_profile)
    group_size = config.evaluation.samples_per_task if split == "eval" else config.training.group_size

    generator = None
    try:
        from self_summarization_agent.bcplus_backend import build_backend

        backend = build_backend(
            config.experiment.bc_plus_root,
            config.retrieval,
            worker_url=retrieval_worker_url,
        )
        generator = _build_rollout_generator(config, checkpoint, split=split)
        _collect_split(
            config=config,
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint,
            generator=generator,
            backend=backend,
            split=split,
            examples=split_examples,
            raw_output_path=Path(raw_output_path),
            sampling_profile=sampling_profile,
            profile_id=profile_id,
            group_size=group_size,
            sample_seed=sample_seed,
            resume=resume,
        )
    except BaseException:
        # All rows are persisted per-row, so a failure needs no further
        # policy-engine work; release GPUs and exit without interpreter
        # teardown so the parent's join() returns promptly.
        traceback.print_exc()
        _terminate_collection_child(generator)
        os._exit(1)
    _terminate_collection_child(generator)
    os._exit(0)


def _run_split_collection_process(
    *,
    config_path: str | Path,
    overrides: list[str],
    checkpoint_path: Path,
    split: str,
    raw_output_path: Path,
    sample_seed: int | None,
    resume: bool,
    retrieval_worker_url: str | None,
    per_split_timeout_seconds: float | None = None,
) -> None:
    """Run and join a policy child; successful return is the teardown barrier."""

    context = mp.get_context("spawn")
    process = context.Process(
        target=_run_split_collection_worker,
        kwargs={
            "config_path": str(config_path),
            "overrides": overrides,
            "checkpoint_path": str(checkpoint_path),
            "split": split,
            "raw_output_path": str(raw_output_path),
            "sample_seed": sample_seed,
            "resume": resume,
            "retrieval_worker_url": retrieval_worker_url,
        },
    )
    process.start()
    try:
        process.join(timeout=per_split_timeout_seconds)
        if process.is_alive():
            # timeout expired — child is still running
            print(
                f"[merged_collect] {split.capitalize()} policy collection timed out "
                f"after {per_split_timeout_seconds:.0f}s. Terminating (pid={process.pid})...",
                flush=True,
            )
            process.terminate()
            process.join(timeout=30)
            if process.is_alive():
                print(
                    f"[merged_collect] {split.capitalize()} policy collection "
                    f"did not respond to SIGTERM, killing (pid={process.pid})...",
                    flush=True,
                )
                process.kill()
                process.join(timeout=10)
            raise RuntimeError(
                f"{split.capitalize()} policy collection timed out "
                f"after {per_split_timeout_seconds:.0f}s"
            )
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        raise
    if process.exitcode != 0:
        raise RuntimeError(
            f"{split.capitalize()} policy collection process failed with exit code {process.exitcode}"
        )


def _selected_split_examples(config, *, split: str, examples: list[Any], sample_seed: int | None):
    task_count, task_count_key = _configured_task_count(config, split=split)
    seed = config.experiment.seed if sample_seed is None else sample_seed
    return _select_collection_examples(
        examples,
        task_count=task_count,
        task_count_key=task_count_key,
        split=split,
        seed=seed,
    )


def _judge_split(
    *,
    config,
    checkpoint_id: str,
    judge_client: Any,
    split: str,
    examples: list[Any],
    raw_output_path: Path,
    judged_output_path: Path,
    group_size: int,
    sample_seed: int | None,
    profile_id: str,
    resume: bool,
) -> None:
    """Judge only missing raw rows while preserving resumable JSONL output."""

    selected_examples = _selected_split_examples(
        config,
        split=split,
        examples=examples,
        sample_seed=sample_seed,
    )
    expected_keys = {
        (example.query_id, rollout_index)
        for example in selected_examples
        for rollout_index in range(group_size)
    }
    raw_keys = _load_completed_rollout_keys(
        raw_output_path,
        checkpoint_id=checkpoint_id,
        expected_keys=expected_keys,
        expected_sampling_profile_id=profile_id if split == "eval" else None,
        require_exact_token_ids=False,
    )
    if raw_keys != expected_keys:
        missing = sorted(expected_keys - raw_keys)
        raise ValueError(f"Cannot judge incomplete {split} raw rollouts; missing keys: {missing!r}")

    ensure_dir(judged_output_path.parent)
    completed_judged_keys: set[tuple[str, int]] = set()
    if resume:
        completed_judged_keys = _load_completed_rollout_keys(
            judged_output_path,
            checkpoint_id=checkpoint_id,
            expected_keys=expected_keys,
            expected_sampling_profile_id=profile_id if split == "eval" else None,
            require_exact_token_ids=False,
        )
        if not completed_judged_keys <= raw_keys:
            unexpected = sorted(completed_judged_keys - raw_keys)
            raise ValueError(
                f"Cannot resume {judged_output_path}: judged rows have no raw counterpart: "
                f"{unexpected!r}"
            )
    elif judged_output_path.exists():
        judged_output_path.unlink()

    example_by_query_id = {example.query_id: example for example in selected_examples}
    pending_rows = [
        row
        for row in _load_rollout_rows(raw_output_path)
        if (row.get("query_id"), row.get("rollout_index")) not in completed_judged_keys
    ]

    def append_judged(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            append_jsonl(judged_output_path, row)

    for rows in iter_batches(pending_rows, config.judge.batch_size):
        judge_client.submit(
            rows,
            [example_by_query_id[str(row["query_id"])] for row in rows],
        )
        append_judged(judge_client.drain_available())
    append_judged(judge_client.finish())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_merged_collect(
    config,
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    train_raw_output: str | Path,
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
    """Run merged collection for both eval and train splits.

    Returns a dict mapping output kind to its file path.
    """
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    overrides = list(overrides or [])

    # ------------------------------------------------------------------
    # Load & split examples
    # ------------------------------------------------------------------
    examples = load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    train_examples_all, eval_examples_all = split_train_eval_examples(
        examples,
        train_limit=config.dataset.train_limit,
        eval_limit=config.dataset.eval_limit,
    )

    # ------------------------------------------------------------------
    # Determine what needs to be done
    # ------------------------------------------------------------------
    has_eval = config.dataset.eval_limit > 0 and eval_raw_output is not None
    eval_sampling_profile = resolved_rollout_sampling_profile(config, split="eval") if has_eval else {}
    eval_profile_id = sampling_profile_id(eval_sampling_profile) if has_eval else ""

    eval_expected_count = _expected_eval_rollout_count(config) if has_eval else 0

    eval_raw_done = (
        not has_eval
        or (
            resume
            and _has_complete_raw_rollouts(
                Path(eval_raw_output),
                checkpoint_id=checkpoint_id,
                expected_count=eval_expected_count,
                expected_sampling_profile_id=eval_profile_id,
            )
        )
    )
    eval_judged_done = (
        not has_eval
        or eval_judged_output is None
        or (
            resume
            and _has_complete_judged_rollouts(
                Path(eval_judged_output),
                checkpoint_id=checkpoint_id,
                expected_count=eval_expected_count,
                require_judge=True,
                expected_sampling_profile_id=eval_profile_id,
            )
        )
    )
    eval_metrics_done = (
        not has_eval
        or eval_metrics_output is None
        or (
            resume
            and _has_eval_metrics(
                Path(eval_metrics_output),
                iteration=eval_iteration if eval_iteration is not None else 0,
                policy_checkpoint_id=checkpoint_id,
                expected_sampling_profile_id=eval_profile_id,
            )
        )
    )

    train_expected_count = _expected_train_rollout_count(config)
    train_raw_done = resume and _has_complete_raw_rollouts(
        Path(train_raw_output),
        checkpoint_id=checkpoint_id,
        expected_count=train_expected_count,
    )
    train_judged_done = (
        train_judged_output is None
        or (
            resume
            and _has_complete_judged_rollouts(
                Path(train_judged_output),
                checkpoint_id=checkpoint_id,
                expected_count=train_expected_count,
                require_judge=False,
            )
        )
    )
    train_cached_done = (
        train_cached_output is None
        or (
            resume
            and _has_complete_cached_rollouts(
                Path(train_cached_output),
                checkpoint_id=checkpoint_id,
                expected_count=train_expected_count,
            )
        )
    )

    eval_raw_needed = not eval_raw_done
    train_raw_needed = not train_raw_done
    eval_judge_needed = not eval_judged_done
    train_judge_needed = not train_judged_done
    train_cache_needed = not train_cached_done

    if not any(
        (
            eval_raw_needed,
            train_raw_needed,
            eval_judge_needed,
            train_judge_needed,
            not eval_metrics_done,
            train_cache_needed,
        )
    ):
        print("[merged_collect] All outputs complete — nothing to do.", flush=True)
        return {}

    outputs: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Phase 1: policy collection. Retrieval is scoped to this phase only.
    # ------------------------------------------------------------------
    needs_collection = eval_raw_needed or train_raw_needed
    owned_retrieval_process = None
    active_retrieval_url = retrieval_worker_url
    per_split_timeout = getattr(config.rollout, "per_split_collection_timeout_seconds", None)
    if needs_collection:
        try:
            if config.retrieval.persistent_worker and active_retrieval_url is None:
                print("[merged_collect] Starting collection-scoped retrieval worker...", flush=True)
                owned_retrieval_process, active_retrieval_url = _start_retrieval_worker(
                    config_path=config_path,
                    train_dir=ensure_dir(Path(train_raw_output).parent),
                    python_executable=sys.executable,
                    overrides=overrides,
                    startup_timeout_seconds=config.retrieval.worker_startup_timeout_seconds,
                )

            if eval_raw_needed and has_eval:
                print("[merged_collect] Starting isolated eval policy collection...", flush=True)
                _run_split_collection_process(
                    config_path=config_path,
                    overrides=overrides,
                    checkpoint_path=checkpoint,
                    split="eval",
                    raw_output_path=Path(eval_raw_output),
                    sample_seed=None,
                    resume=resume,
                    retrieval_worker_url=active_retrieval_url,
                    per_split_timeout_seconds=per_split_timeout,
                )
                outputs["eval_raw"] = Path(eval_raw_output)

            if train_raw_needed:
                print("[merged_collect] Starting isolated train policy collection...", flush=True)
                _run_split_collection_process(
                    config_path=config_path,
                    overrides=overrides,
                    checkpoint_path=checkpoint,
                    split="train",
                    raw_output_path=Path(train_raw_output),
                    sample_seed=sample_seed,
                    resume=resume,
                    retrieval_worker_url=active_retrieval_url,
                    per_split_timeout_seconds=per_split_timeout,
                )
                outputs["train_raw"] = Path(train_raw_output)
        finally:
            if owned_retrieval_process is not None:
                print("[merged_collect] Stopping collection-scoped retrieval worker...", flush=True)
                _stop_retrieval_worker(owned_retrieval_process, active_retrieval_url)
                if owned_retrieval_process.poll() is None:
                    raise RuntimeError(
                        "Retrieval worker is still alive after collection teardown; refusing to start judge"
                    )

    # Revalidate the durable boundary before allocating any judge GPU.
    if has_eval and not _has_complete_raw_rollouts(
        Path(eval_raw_output),
        checkpoint_id=checkpoint_id,
        expected_count=eval_expected_count,
        expected_sampling_profile_id=eval_profile_id,
    ):
        raise RuntimeError("Eval raw rollout artifact is incomplete after policy collection")
    if not _has_complete_raw_rollouts(
        Path(train_raw_output),
        checkpoint_id=checkpoint_id,
        expected_count=train_expected_count,
    ):
        raise RuntimeError("Train raw rollout artifact is incomplete after policy collection")

    # ------------------------------------------------------------------
    # Phase 2: one fresh judge process handles both complete raw artifacts.
    # ------------------------------------------------------------------
    if eval_judge_needed or train_judge_needed:
        if not config.judge.enabled:
            raise ValueError("judge.enabled must be true for merged collection")
        print("[merged_collect] Starting post-collection judge...", flush=True)
        judge_client = _build_overlap_judge_client(
            judge=None,
            config_path=str(config_path),
            overrides=overrides,
            checkpoint_id=checkpoint_id,
            queue_max_batches=config.rollout.overlap_queue_max_batches,
        )
        try:
            if eval_judge_needed and has_eval and eval_judged_output is not None:
                _judge_split(
                    config=config,
                    checkpoint_id=checkpoint_id,
                    judge_client=judge_client,
                    split="eval",
                    examples=eval_examples_all,
                    raw_output_path=Path(eval_raw_output),
                    judged_output_path=Path(eval_judged_output),
                    group_size=config.evaluation.samples_per_task,
                    sample_seed=None,
                    profile_id=eval_profile_id,
                    resume=resume,
                )
                outputs["eval_judged"] = Path(eval_judged_output)
            if train_judge_needed and train_judged_output is not None:
                train_profile_id = sampling_profile_id(
                    resolved_rollout_sampling_profile(config, split="train")
                )
                _judge_split(
                    config=config,
                    checkpoint_id=checkpoint_id,
                    judge_client=judge_client,
                    split="train",
                    examples=train_examples_all,
                    raw_output_path=Path(train_raw_output),
                    judged_output_path=Path(train_judged_output),
                    group_size=config.training.group_size,
                    sample_seed=sample_seed,
                    profile_id=train_profile_id,
                    resume=resume,
                )
                outputs["train_judged"] = Path(train_judged_output)
        finally:
            print(
                "[merged_collect] "
                + json.dumps(
                    {"event": "post_collection_judge_metrics", **judge_client.metrics()},
                    sort_keys=True,
                ),
                flush=True,
            )
            judge_client.close()
            judge_process = getattr(judge_client, "process", None)
            if judge_process is not None and judge_process.is_alive():
                raise RuntimeError(
                    "Judge worker is still alive after teardown; refusing to start cache fallback"
                )

    # ------------------------------------------------------------------
    # Phase 3: CPU metrics and native-cache finalization. Any policy rescore
    # fallback starts only after the judge worker above has exited.
    # ------------------------------------------------------------------
    if has_eval and not eval_metrics_done and eval_judged_output is not None:
        print("[merged_collect] Computing eval metrics...", flush=True)
        write_eval_metrics(
            judged_rollout_path=eval_judged_output,
            metrics_path=eval_metrics_output,
            iteration=eval_iteration if eval_iteration is not None else 0,
            policy_checkpoint_id=checkpoint_id,
        )
        outputs["eval_metrics"] = Path(eval_metrics_output)

    if train_cache_needed and train_judged_output is not None:
        print("[merged_collect] Finalizing training caches...", flush=True)
        _run_cache_inline(
            config=config,
            config_path=str(config_path),
            overrides=overrides,
            checkpoint_path=checkpoint,
            judged_rollout_path=Path(train_judged_output),
            cached_output_path=Path(train_cached_output) if train_cached_output else None,
            resume=resume,
        )
        if train_cached_output:
            outputs["train_cached"] = Path(train_cached_output)

    return outputs


def _run_cache_inline(
    *,
    config,
    config_path: str,
    overrides: list[str],
    checkpoint_path: Path,
    judged_rollout_path: Path,
    cached_output_path: Path | None,
    resume: bool,
) -> None:
    """Finalize native caches, then lazily rescore misses in an isolated child."""
    if cached_output_path is None:
        return

    checkpoint_id = checkpoint_id_from_path(checkpoint_path)
    rows = _load_rollout_rows(judged_rollout_path)
    for index, row in enumerate(rows, start=1):
        _validate_judged_row(row, index=index, expected_checkpoint_id=checkpoint_id)

    ensure_dir(cached_output_path.parent)
    completed_keys: set[tuple[str, int]] = set()
    if resume:
        completed_rows = _completed_cached_rows(cached_output_path, expected_checkpoint_id=checkpoint_id)
        completed_keys = set(completed_rows)
        # Write back completed rows to preserve resume ordering
        ordered = [
            completed_rows[(row.get("query_id"), row.get("rollout_index"))]
            for row in rows
            if (
                isinstance(row.get("query_id"), str)
                and isinstance(row.get("rollout_index"), int)
                and (row["query_id"], row["rollout_index"]) in completed_rows
            )
        ]
        if ordered:
            cached_output_path.unlink(missing_ok=True)
            for r in ordered:
                append_jsonl(cached_output_path, r)
    elif cached_output_path.exists():
        cached_output_path.unlink()

    pending_rows = [
        row
        for row in rows
        if (
            isinstance(row.get("query_id"), str)
            and isinstance(row.get("rollout_index"), int)
            and (row["query_id"], row["rollout_index"]) not in completed_keys
        )
    ]
    if not pending_rows:
        return

    fallback_rows: list[dict[str, Any]] = []
    for row in pending_rows:
        cache_candidate = _materialize_rollout_native_training_caches(
            row,
            checkpoint_id=checkpoint_id,
        )
        if _row_has_current_training_cache(cache_candidate):
            append_jsonl(cached_output_path, cache_candidate)
            continue
        fallback_rows.append(cache_candidate)

    if not fallback_rows:
        return

    print(
        f"[merged_collect] Launching post-judge GPU 0 cache fallback for "
        f"{len(fallback_rows)} row(s)...",
        flush=True,
    )
    cache_client = _CacheOverlapClient(
        config_path=config_path,
        overrides=overrides,
        checkpoint_path=str(checkpoint_path),
        checkpoint_id=checkpoint_id,
        queue_max_batches=config.rollout.overlap_queue_max_batches,
    )
    try:
        for rows in iter_batches(fallback_rows, config.judge.batch_size):
            cache_client.submit(rows)
            for cached_row in cache_client.drain_available():
                append_jsonl(cached_output_path, cached_row)
        for cached_row in cache_client.finish():
            append_jsonl(cached_output_path, cached_row)
    finally:
        print(
            "[merged_collect] "
            + json.dumps(
                {"event": "post_judge_cache_fallback_metrics", **cache_client.metrics()},
                sort_keys=True,
            ),
            flush=True,
        )
        cache_client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential collection, teardown, judging, metrics, and cache step."
    )
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint path.")
    parser.add_argument(
        "--train-raw-output", required=True, help="Train raw rollout JSONL output path."
    )
    parser.add_argument(
        "--train-judged-output", default=None, help="Train judged rollout JSONL output path."
    )
    parser.add_argument(
        "--train-cached-output", default=None, help="Train cached rollout JSONL output path."
    )
    parser.add_argument(
        "--eval-raw-output", default=None, help="Eval raw rollout JSONL output path."
    )
    parser.add_argument(
        "--eval-judged-output", default=None, help="Eval judged rollout JSONL output path."
    )
    parser.add_argument(
        "--eval-metrics-output", default=None, help="Eval metrics JSONL output path."
    )
    parser.add_argument(
        "--eval-iteration", type=int, default=None, help="Eval iteration number for metrics."
    )
    parser.add_argument(
        "--sample-seed", type=int, default=None, help="Seed for training-query sampling."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip completed outputs and resume partial work."
    )
    parser.add_argument(
        "--retrieval-worker-url", default=None, help="Use a persistent retrieval worker at this URL."
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    # Convert launcher SIGTERM into a Python unwind so active policy/judge
    # children and the collection-scoped retrieval worker run their finally
    # teardown before this supervisor exits.
    def handle_termination(signum, _frame) -> None:
        raise SystemExit(128 + signum)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_termination)
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
    print(json.dumps({k: str(v) for k, v in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
