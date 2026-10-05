"""Merged collection with strict GPU lifecycle boundaries.

The parent process remains CUDA-free and supervises three sequential phases::

    collection workers (policy GPUs): eval, then train + native logprobs
    judge worker (judge GPUs):        eval and train answer judging
    cache fallback (GPU 0):           only rows missing native logprobs

The retrieval worker is owned by the collection phase.  It and every policy
worker must exit before the judge worker is allowed to start.  Process exit,
rather than best-effort object deletion, is the authoritative vLLM teardown
boundary.

When ``rollout.overlap_judge`` is enabled (and judging is required), the judge
worker starts *before* collection instead, and completed raw rows stream to it
through a queue as collection proceeds.  The post-collection judge phase then
only catches rows the stream missed (for example after a mid-stream judge
failure, which falls back to a fresh sequential judge worker).  Overlap mode
relaxes the "judge starts after policy teardown" boundary and therefore
requires the judge and policy engines to fit in GPU memory simultaneously:
disjoint ``judge.gpu_ids`` / ``rollout.gpu_ids`` or reduced
``gpu_memory_utilization``.
"""

from __future__ import annotations
from self_summarization_agent.collection_contract import collection_profile_id, validate_artifact_lineage
from self_summarization_agent.token_stream import TITO_CONTRACT

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import signal
import sys
import threading
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


def _critic_only_cache_kwargs(config: Any) -> dict[str, bool]:
    value_config = getattr(getattr(config, "training", None), "value", None)
    if bool(getattr(value_config, "enabled", False)):
        return {"retain_critic_only_states": True}
    return {}


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
                    train_compaction_tokens=config.training.train_compaction_tokens,
                    **_critic_only_cache_kwargs(config),
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
# Overlap judge feed (streamed raw rows judged during collection)
# ---------------------------------------------------------------------------

_FEED_SENTINEL = "__overlap_feed_shutdown__"


def _take_chunk_rows(
    rows: list[dict[str, Any]],
    pending_chunks: list[tuple[str, int]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Route drained judge rows back to their split using submission order.

    The judge worker answers every submitted message in order, and each
    ``drain_available``/``finish`` returns whole responses in that same order,
    so rows can be sliced deterministically by the recorded per-submit counts.
    """

    distributed: list[tuple[str, list[dict[str, Any]]]] = []
    offset = 0
    while offset < len(rows) and pending_chunks:
        split, count = pending_chunks[0]
        if offset + count > len(rows):
            raise RuntimeError(
                "Overlap judge returned a partial response for a submitted batch "
                f"({len(rows) - offset} of {count} rows)"
            )
        distributed.append((split, rows[offset : offset + count]))
        pending_chunks.pop(0)
        offset += count
    if offset != len(rows):
        raise RuntimeError(
            "Overlap judge returned rows without a matching submission record"
        )
    return distributed


def _overlap_judge_feeder(
    *,
    feed_queue: Any,
    judge_client: Any,
    judged_output_paths: dict[str, Path],
    examples_by_query_id: dict[str, Any],
    batch_size: int,
    stats: dict[str, Any],
) -> None:
    """Consume streamed raw rows from collection children and judge them.

    Runs in a daemon thread of the merged-collect parent.  ``stats`` receives
    ``fed_rows``/``judged_rows`` counts and, on any failure, a ``failure``
    entry; after a failure the feeder keeps draining the queue so collection
    children never block on a full queue, and the parent re-judges the missed
    rows with a fresh post-collection judge worker.  The feeder never raises:
    any unexpected error is reported through ``stats`` instead, so a stalled
    feeder cannot deadlock the children feeding it.
    """

    stats.setdefault("fed_rows", 0)
    stats.setdefault("judged_rows", 0)
    stats.setdefault("failure", None)
    saw_sentinel = False
    batch_size = max(1, batch_size)
    buffers: dict[str, list[tuple[dict[str, Any], Any]]] = {
        split: [] for split in judged_output_paths
    }
    pending_chunks: list[tuple[str, int]] = []

    def flush_split(split: str) -> None:
        buffer = buffers[split]
        if not buffer:
            return
        chunk, rest = buffer[:batch_size], buffer[batch_size:]
        buffers[split] = rest
        rows = [row for row, _example in chunk]
        examples = [example for _row, example in chunk]
        judge_client.submit(rows, examples)
        pending_chunks.append((split, len(rows)))
        drained = judge_client.drain_available()
        if drained:
            for chunk_split, chunk_rows in _take_chunk_rows(drained, pending_chunks):
                for judged_row in chunk_rows:
                    append_jsonl(judged_output_paths[chunk_split], judged_row)
                stats["judged_rows"] += len(chunk_rows)

    try:
        while True:
            item = feed_queue.get()
            if item == _FEED_SENTINEL:
                saw_sentinel = True
                break
            split, row = item
            if split not in buffers:
                continue
            example = examples_by_query_id.get(str(row.get("query_id")))
            if example is None:
                raise ValueError(
                    f"Overlap feed row references unknown query_id: {row.get('query_id')!r}"
                )
            buffers[split].append((row, example))
            stats["fed_rows"] += 1
            if len(buffers[split]) >= batch_size:
                flush_split(split)

        # Flush stragglers, then reap any rows still in flight.
        for split in list(buffers):
            while buffers[split]:
                flush_split(split)
        finished = judge_client.finish()
        if finished:
            for chunk_split, chunk_rows in _take_chunk_rows(finished, pending_chunks):
                for judged_row in chunk_rows:
                    append_jsonl(judged_output_paths[chunk_split], judged_row)
                stats["judged_rows"] += len(chunk_rows)
    except Exception as exc:  # judged rows stay unwritten; the
        stats["failure"] = exc  # post-collection phase re-judges them

    if stats["failure"] is not None and not saw_sentinel:
        # The failure happened mid-stream: keep consuming so collection
        # children never block on a full queue, until the parent's shutdown
        # sentinel arrives.  After the sentinel the queue is already empty —
        # the parent only sends it once every collection child has exited.
        while True:
            item = feed_queue.get()
            if item == _FEED_SENTINEL:
                break


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
    row_feed_queue: Any | None = None,
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
    tito = hasattr(generator, "create_token_renderer")
    collection_identity = collection_profile_id(config, checkpoint_path) if tito else None

    if resume:
        completed_raw_keys = _load_completed_rollout_keys(
            raw_output_path,
            checkpoint_id=checkpoint_id,
            expected_keys=expected_keys,
            expected_sampling_profile_id=profile_id if split == "eval" else None,
            require_exact_token_ids=tito,
            expected_collection_profile_id=collection_identity,
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
                    "collection_profile_id": collection_identity,
                    "collection_contract": TITO_CONTRACT if tito else None,
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
                        train_compaction_tokens=config.training.train_compaction_tokens,
                        **_critic_only_cache_kwargs(config),
                    )
                append_jsonl(raw_output_path, row)
                if row_feed_queue is not None:
                    row_feed_queue.put((split, row))
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
    row_feed_queue: Any | None = None,
) -> None:
    """Child entrypoint that owns one split's policy engine and CUDA state."""

    config = load_train_config(config_path, parse_cli_overrides(overrides))
    checkpoint = Path(checkpoint_path).resolve()
    if getattr(getattr(config, "benchmark", None), "name", "browsecomp") == "terminal-bench":
        from self_summarization_agent.benchmarks.terminal_bench.collection import collect_terminal_rollouts
        generator = _build_rollout_generator(config, checkpoint, split=split)
        try:
            collect_terminal_rollouts(config, checkpoint_path=checkpoint, output_path=raw_output_path,
                generator=generator, split=split, sample_seed=sample_seed, resume=resume)
        finally:
            core = getattr(getattr(getattr(generator, "llm", None), "llm_engine", None), "engine_core", None)
            if core is not None:
                core.shutdown()
        return
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

    from self_summarization_agent.bcplus_backend import build_backend

    generator = None
    try:
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
            row_feed_queue=row_feed_queue,
        )
    except BaseException:
        # multiprocessing's _bootstrap runs util._exit_function() — which joins
        # child processes with no timeout — BEFORE its exception handler prints
        # the traceback.  With the policy engine child still alive, that join
        # blocks forever and the real error is never logged.  Log first, then
        # stop the engine via vLLM's bounded shutdown, then hard-exit.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        engine_core_client = getattr(
            getattr(getattr(generator, "llm", None), "llm_engine", None),
            "engine_core",
            None,
        )
        if engine_core_client is not None:
            try:
                engine_core_client.shutdown()
            except Exception:
                pass
        os._exit(1)


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
    row_feed_queue: Any | None = None,
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
            "row_feed_queue": row_feed_queue,
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
        raise
    if process.exitcode != 0:
        raise RuntimeError(
            f"{split.capitalize()} policy collection process failed with exit code {process.exitcode}"
        )


def _require_live_retrieval_worker(process: Any, *, after_split: str) -> None:
    returncode = process.poll()
    if returncode is not None:
        raise RuntimeError(
            "Retrieval worker exited unexpectedly during "
            f"{after_split} policy collection with code {returncode}"
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


def _warn_if_judge_shares_gpus(config) -> None:
    """Warn when overlap mode co-locates the judge engine with other engines."""

    judge_devices = set(getattr(config.judge, "gpu_ids", ()) or ())
    rollout_devices = set(getattr(config.rollout, "gpu_ids", ()) or ())
    retrieval_devices = set(getattr(config.retrieval, "gpu_ids", ()) or ())
    shared: list[str] = []
    # An empty gpu_ids list means "all visible devices", which always overlaps.
    if not judge_devices or not rollout_devices:
        shared.append("policy")
    else:
        shared.extend(f"policy GPU {device}" for device in sorted(judge_devices & rollout_devices))
    if not judge_devices or not retrieval_devices:
        shared.append("retrieval")
    else:
        shared.extend(
            f"retrieval GPU {device}" for device in sorted(judge_devices & retrieval_devices)
        )
    if shared:
        print(
            "[merged_collect] WARNING: overlap judging requires the judge engine to share "
            f"device memory with {', '.join(shared)}. Both engines must fit simultaneously; "
            "assign disjoint judge.gpu_ids/rollout.gpu_ids or lower "
            "gpu_memory_utilization, or an engine may fail to start.",
            flush=True,
        )


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
    if getattr(getattr(config, "benchmark", None), "name", "browsecomp") == "terminal-bench":
        from self_summarization_agent.benchmarks.terminal_bench.collection import run_terminal_merged
        return run_terminal_merged(config, config_path=config_path, checkpoint_path=checkpoint_path,
            train_raw_output=train_raw_output, train_judged_output=train_judged_output,
            train_cached_output=train_cached_output, eval_raw_output=eval_raw_output,
            eval_judged_output=eval_judged_output, eval_metrics_output=eval_metrics_output,
            eval_iteration=eval_iteration, sample_seed=sample_seed, resume=resume, overrides=overrides)
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
    if resume:
        validate_artifact_lineage(
            [train_raw_output, train_judged_output, train_cached_output, eval_raw_output, eval_judged_output],
            config=config, checkpoint=checkpoint)
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
                train_compaction_tokens=config.training.train_compaction_tokens,
                **_critic_only_cache_kwargs(config),
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

    # Optional overlap: when rollout.overlap_judge is enabled and judging is
    # required, the judge worker starts *before* collection and streamed raw
    # rows are judged as they complete.  Rows the stream misses (feed failure,
    # splits whose raw artifact already existed) are judged by the normal
    # post-collection phase below, so judged output is identical either way.
    overlap_feed_splits: set[str] = set()
    overlap_judge_client: Any | None = None
    overlap_feed_queue: Any | None = None
    overlap_feed_thread: threading.Thread | None = None
    overlap_feed_stats: dict[str, Any] = {}
    overlap_requested = bool(getattr(config.rollout, "overlap_judge", False))
    overlap_eligible = bool(
        overlap_requested
        and (eval_judge_needed or train_judge_needed)
        and needs_collection
    )
    feed_paths: dict[str, Path] = {}
    if overlap_eligible:
        if not config.judge.enabled:
            raise ValueError("judge.enabled must be true for merged overlap judging")
        if (
            eval_judge_needed
            and has_eval
            and eval_judged_output is not None
            and eval_raw_needed
        ):
            feed_paths["eval"] = Path(eval_judged_output)
        if train_judge_needed and train_judged_output is not None and train_raw_needed:
            feed_paths["train"] = Path(train_judged_output)
        overlap_eligible = bool(feed_paths)

    # The retrieval worker claims its device memory first.  This matters in
    # overlap mode, where the judge engine initializes while the retrieval
    # worker is still resident: vLLM sizes its GPU budget from the memory
    # actually free at init, so retrieval must be resident *before* the judge
    # engine profiles the shared device (otherwise the judge's default
    # utilization starves the retrieval worker at startup).
    owned_retrieval_process = None
    active_retrieval_url = retrieval_worker_url
    per_split_timeout = getattr(config.rollout, "per_split_collection_timeout_seconds", None)
    if (
        needs_collection
        and config.retrieval.persistent_worker
        and active_retrieval_url is None
    ):
        print("[merged_collect] Starting collection-scoped retrieval worker...", flush=True)
        owned_retrieval_process, active_retrieval_url = _start_retrieval_worker(
            config_path=config_path,
            train_dir=ensure_dir(Path(train_raw_output).parent),
            python_executable=sys.executable,
            overrides=overrides,
            startup_timeout_seconds=config.retrieval.worker_startup_timeout_seconds,
            gpu_ids=getattr(config.retrieval, "gpu_ids", ()),
        )

    if overlap_eligible:
        _warn_if_judge_shares_gpus(config)
        try:
            overlap_judge_client = _build_overlap_judge_client(
                judge=None,
                config_path=str(config_path),
                overrides=overrides,
                checkpoint_id=checkpoint_id,
                queue_max_batches=getattr(config.rollout, "overlap_queue_max_batches", 8),
            )
        except Exception as exc:
            print(
                "[merged_collect] Overlap judge worker failed to start "
                f"({exc!r}); falling back to post-collection judging.",
                flush=True,
            )
            overlap_judge_client = None
        if overlap_judge_client is not None:
            overlap_feed_splits = set(feed_paths)
            for path in feed_paths.values():
                ensure_dir(path.parent)
                if not resume:
                    # The feeder owns these files during collection; the
                    # post-collection phase resumes from whatever it wrote.
                    path.unlink(missing_ok=True)
            overlap_feed_queue = mp.get_context("spawn").Queue()
            feed_examples = {
                example.query_id: example
                for example in (*train_examples_all, *eval_examples_all)
            }
            overlap_feed_thread = threading.Thread(
                target=_overlap_judge_feeder,
                kwargs={
                    "feed_queue": overlap_feed_queue,
                    "judge_client": overlap_judge_client,
                    "judged_output_paths": feed_paths,
                    "examples_by_query_id": feed_examples,
                    "batch_size": getattr(config.judge, "batch_size", 32),
                    "stats": overlap_feed_stats,
                },
                daemon=True,
                name="overlap-judge-feeder",
            )
            overlap_feed_thread.start()
            print(
                "[merged_collect] Overlap judging enabled; judge worker starts before "
                "collection and streamed raw rows are judged as they complete.",
                flush=True,
            )

    if needs_collection:
        try:
            try:
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
                        row_feed_queue=(
                            overlap_feed_queue if "eval" in overlap_feed_splits else None
                        ),
                    )
                    if owned_retrieval_process is not None:
                        _require_live_retrieval_worker(
                            owned_retrieval_process,
                            after_split="eval",
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
                        row_feed_queue=(
                            overlap_feed_queue if "train" in overlap_feed_splits else None
                        ),
                    )
                    if owned_retrieval_process is not None:
                        _require_live_retrieval_worker(
                            owned_retrieval_process,
                            after_split="train",
                        )
                    outputs["train_raw"] = Path(train_raw_output)
            finally:
                # Stop the feed before the GPU teardown checks so the judge
                # client is only kept alive when collection completed cleanly.
                if overlap_feed_queue is not None:
                    overlap_feed_queue.put(_FEED_SENTINEL)
                if overlap_feed_thread is not None:
                    overlap_feed_thread.join()
                feed_failure = overlap_feed_stats.get("failure")
                if feed_failure is not None or sys.exc_info()[0] is not None:
                    if overlap_judge_client is not None:
                        if feed_failure is not None:
                            print(
                                "[merged_collect] Overlap judge feed failed "
                                f"({feed_failure!r}); a fresh post-collection judge will "
                                "re-judge the missed rows.",
                                flush=True,
                            )
                        overlap_judge_client.close()
                        overlap_judge_client = None
                        overlap_feed_splits = set()
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
    # Phase 2: judge any rows the collection stream missed.  Reuses the
    # still-healthy overlap judge worker when there is one; otherwise starts
    # one fresh judge process for both complete raw artifacts.
    # ------------------------------------------------------------------
    if eval_judge_needed or train_judge_needed:
        if not config.judge.enabled:
            raise ValueError("judge.enabled must be true for merged collection")
        if overlap_judge_client is not None:
            print("[merged_collect] Reusing overlap judge for remaining rows...", flush=True)
            judge_client = overlap_judge_client
        else:
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
                    resume=resume or "eval" in overlap_feed_splits,
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
                    resume=resume or "train" in overlap_feed_splits,
                )
                outputs["train_judged"] = Path(train_judged_output)
        finally:
            judge_metrics = judge_client.metrics()
            if overlap_requested:
                judge_metrics["overlap_feed"] = {
                    "fed_rows": overlap_feed_stats.get("fed_rows", 0),
                    "judged_rows": overlap_feed_stats.get("judged_rows", 0),
                    "failed": overlap_feed_stats.get("failure") is not None,
                }
            print(
                "[merged_collect] "
                + json.dumps(
                    {"event": "post_collection_judge_metrics", **judge_metrics},
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
        completed_rows = _completed_cached_rows(
            cached_output_path,
            expected_checkpoint_id=checkpoint_id,
            train_compaction_tokens=config.training.train_compaction_tokens,
            **_critic_only_cache_kwargs(config),
        )
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
            train_compaction_tokens=config.training.train_compaction_tokens,
            **_critic_only_cache_kwargs(config),
        )
        if _row_has_current_training_cache(
            cache_candidate,
            train_compaction_tokens=config.training.train_compaction_tokens,
            **_critic_only_cache_kwargs(config),
        ):
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
