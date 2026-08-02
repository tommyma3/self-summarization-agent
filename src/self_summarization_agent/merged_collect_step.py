"""Single-process merged collection step with overlapped cache scoring.

Replaces six separate subprocess phases (eval_rollout, train_rollout,
eval_judge, eval_metrics, train_judge, train_cache) with one process
that builds a single vLLM engine and runs all collection, judging,
metrics, and caching work.

Architecture
------------
Three concurrent workers across 4 GPUs::

    Main process (GPUs 2,3):  vLLM engine → generates rollouts
    Judge worker (GPU 1):     spawned process → scores batches
    Cache worker (GPU 0):     spawned process → reference logprobs

Cache scoring overlaps with train rollout collection so that all
outputs are ready before the downstream ``train_update`` phase starts.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time
import traceback
from typing import Any

from self_summarization_agent.cache_step import (
    _attach_training_caches,
    _completed_cached_rows,
    _load_rollout_rows,
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
)
from self_summarization_agent.judge_step import build_judge, judge_rollout_rows
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
    is_penalized_runtime_status,
    trainable_turn_ids_from_records,
)
from self_summarization_agent.rollout_collection import (
    _build_overlap_judge_client,
    _build_rollout_generator,
    _configured_task_count,
    _example_payload,
    _load_completed_rollout_keys,
    _select_collection_examples,
    _temporary_sampling_profile,
    apply_judged_rewards,
)
from self_summarization_agent.trajectory import extract_trainable_samples

# ---------------------------------------------------------------------------
# Cache overlap worker (spawned process on GPU 0)
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
    while True:
        message = request_queue.get()
        if message == _CACHE_SHUTDOWN:
            return
        batch_id = message["batch_id"]
        try:
            rows = message["rows"]
            expected_checkpoint_id = message["expected_checkpoint_id"]
            cached_rows: list[dict[str, Any]] = []
            for row in rows:
                samples = extract_trainable_samples(
                    row["trajectory_records"],
                    row["turn_rewards"],
                    rollout_id=f"{row.get('query_id')}:{row.get('rollout_index')}",
                )
                if not samples:
                    cached_rows.append(dict(row))
                    continue
                cache_payloads = scorer.cache_samples(samples)
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
# Cache overlap client (manages the spawned cache-worker lifecycle)
# ---------------------------------------------------------------------------

class _CacheOverlapClient:
    """Manages a spawned cache-worker process on GPU 0.

    Follows the same pattern as ``_SubprocessOverlapJudgeClient`` but for
    reference-logprob caching instead of judging.
    """

    def __init__(
        self,
        *,
        config_path: str,
        overrides: list[str],
        checkpoint_path: str,
        checkpoint_id: str,
        drain_timeout_seconds: float = 600,
    ) -> None:
        context = mp.get_context("spawn")
        self.request_queue = context.Queue()
        self.response_queue = context.Queue()
        self.process = context.Process(
            target=_run_cache_overlap_worker,
            kwargs={
                "config_path": config_path,
                "overrides": overrides,
                "checkpoint_path": checkpoint_path,
                "request_queue": self.request_queue,
                "response_queue": self.response_queue,
            },
        )
        self.process.start()
        self.checkpoint_id = checkpoint_id
        self.next_batch_id = 0
        self.pending_count = 0
        self._drain_timeout_seconds = drain_timeout_seconds
        self._drain_deadline: float | None = None

    # ------------------------------------------------------------------
    # Submit / drain / finish (mirrors _SubprocessOverlapJudgeClient)
    # ------------------------------------------------------------------

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

    def _handle_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        self.pending_count -= 1
        if response.get("error"):
            traceback_text = response.get("traceback")
            detail = f"\n{traceback_text}" if traceback_text else ""
            raise RuntimeError(f"Cache overlap worker failed: {response['error']}{detail}")
        rows = response.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError(f"Cache overlap worker returned invalid response: {response!r}")
        return rows

    def _ensure_drain_deadline(self) -> None:
        if self._drain_deadline is None:
            self._drain_deadline = time.monotonic() + self._drain_timeout_seconds

    def _check_drain_timeout(self) -> None:
        if self._drain_deadline is None:
            return
        if time.monotonic() < self._drain_deadline:
            return
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
        rows: list[dict[str, Any]] = []
        while self.pending_count:
            try:
                response = self.response_queue.get_nowait()
            except Empty:
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
        self._ensure_drain_deadline()
        rows: list[dict[str, Any]] = []
        while self.pending_count:
            try:
                response = self.response_queue.get(timeout=5)
            except Empty:
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

    def close(self) -> None:
        if self.process.is_alive():
            self.request_queue.put(_CACHE_SHUTDOWN)
            self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30)


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
    overlap_judge_client: Any | None,
    split: str,
    examples: list[Any],
    raw_output_path: Path,
    judged_output_path: Path | None,
    sampling_profile: dict[str, Any],
    profile_id: str,
    group_size: int,
    sample_seed: int | None,
    resume: bool,
    # Optional cache overlap (train split only)
    cache_overlap_client: _CacheOverlapClient | None = None,
    cached_output_path: Path | None = None,
) -> None:
    """Collect rollouts for a single split with optional overlapped cache scoring."""

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

    if resume:
        completed_keys = _load_completed_rollout_keys(
            raw_output_path,
            checkpoint_id=checkpoint_id,
            expected_keys=expected_keys,
            expected_sampling_profile_id=profile_id if split == "eval" else None,
            require_exact_token_ids=False,
        )
        rollout_requests = [
            (example, rollout_index)
            for example, rollout_index in rollout_requests
            if (example.query_id, rollout_index) not in completed_keys
        ]
        if not rollout_requests:
            return
    elif raw_output_path.exists():
        raw_output_path.unlink()

    overlap_judge = overlap_judge_client is not None
    if overlap_judge:
        ensure_dir(judged_output_path.parent)
        if not resume and judged_output_path.exists():
            judged_output_path.unlink()

    if cached_output_path is not None:
        ensure_dir(cached_output_path.parent)
        if not resume and cached_output_path.exists():
            cached_output_path.unlink()

    with _temporary_sampling_profile(generator, sampling_profile):
        runtime = build_runtime(generator, backend, config.runtime)
        for request_batch in iter_batches(
            rollout_requests,
            config.rollout.max_concurrent_episodes,
        ):
            results = runtime.run_many(
                (example.query_id, example.query) for example, _ in request_batch
            )
            overlap_rows: list[dict[str, Any]] = []
            overlap_examples: list[Any] = []
            for (example, rollout_index), result in zip(request_batch, results):
                trainable_sample_count = None
                judge_payload = None
                # We don't do inline judging in the merged step — overlap is
                # always preferred when available.
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
                append_jsonl(raw_output_path, row)
                if overlap_judge:
                    overlap_rows.append(row)
                    overlap_examples.append(example)

            if overlap_judge and overlap_rows:
                overlap_judge_client.submit(overlap_rows, overlap_examples)
                for judged_row in overlap_judge_client.drain_available():
                    append_jsonl(judged_output_path, judged_row)
                    # Forward to cache worker for train split
                    if cache_overlap_client is not None:
                        cache_overlap_client.submit([judged_row])
                        for cached_row in cache_overlap_client.drain_available():
                            append_jsonl(cached_output_path, cached_row)

        # Drain remaining judge + cache responses
        if overlap_judge:
            for judged_row in overlap_judge_client.finish():
                append_jsonl(judged_output_path, judged_row)
                if cache_overlap_client is not None:
                    cache_overlap_client.submit([judged_row])

        if cache_overlap_client is not None:
            for cached_row in cache_overlap_client.finish():
                append_jsonl(cached_output_path, cached_row)


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

    eval_collection_needed = not (eval_raw_done and eval_judged_done and eval_metrics_done)
    train_collection_needed = not (train_raw_done and train_judged_done)
    train_cache_needed = not train_cached_done

    if not eval_collection_needed and not train_collection_needed and not train_cache_needed:
        print("[merged_collect] All outputs complete — nothing to do.", flush=True)
        return {}

    # ------------------------------------------------------------------
    # Build shared resources
    # ------------------------------------------------------------------
    from self_summarization_agent.bcplus_backend import build_backend

    backend = build_backend(
        config.experiment.bc_plus_root,
        config.retrieval,
        worker_url=retrieval_worker_url,
    )

    needs_collection = eval_collection_needed or train_collection_needed
    overlap_judge = bool(config.rollout.overlap_judge and config.judge.enabled)
    overlap_judge_client = None

    generator = None
    if needs_collection:
        # Build vLLM engine with train defaults (eval sampling is applied
        # temporarily via _temporary_sampling_profile during eval collection).
        generator = _build_rollout_generator(config, checkpoint, split="train")
        if overlap_judge:
            overlap_judge_client = _build_overlap_judge_client(
                judge=None,
                config_path=str(config_path),
                overrides=overrides,
                checkpoint_id=checkpoint_id,
            )

    outputs: dict[str, Path] = {}

    try:
        # --------------------------------------------------------------
        # Stage 2: Eval collection + judging + metrics
        # --------------------------------------------------------------
        if eval_collection_needed and has_eval:
            print("[merged_collect] Starting eval collection...", flush=True)
            _collect_split(
                config=config,
                checkpoint_id=checkpoint_id,
                checkpoint_path=checkpoint,
                generator=generator,
                backend=backend,
                overlap_judge_client=overlap_judge_client,
                split="eval",
                examples=eval_examples_all,
                raw_output_path=Path(eval_raw_output),
                judged_output_path=Path(eval_judged_output) if eval_judged_output else None,
                sampling_profile=eval_sampling_profile,
                profile_id=eval_profile_id,
                group_size=config.evaluation.samples_per_task,
                sample_seed=None,
                resume=resume,
                cache_overlap_client=None,
            )
            outputs["eval_raw"] = Path(eval_raw_output)
            if eval_judged_output:
                outputs["eval_judged"] = Path(eval_judged_output)

        # Eval metrics (cheap — run after collection if not already done)
        if has_eval and not eval_metrics_done and eval_judged_output is not None:
            print("[merged_collect] Computing eval metrics...", flush=True)
            write_eval_metrics(
                judged_rollout_path=eval_judged_output,
                metrics_path=eval_metrics_output,
                iteration=eval_iteration if eval_iteration is not None else 0,
                policy_checkpoint_id=checkpoint_id,
            )
            outputs["eval_metrics"] = Path(eval_metrics_output)

        # --------------------------------------------------------------
        # Stage 3: Train collection + overlapped cache
        # --------------------------------------------------------------
        if train_collection_needed or train_cache_needed:
            # When overlap judging is ON *and* we need fresh collection, run
            # the cache worker on GPU 0 in parallel with rollout generation.
            # Otherwise (resume where collection is already done, or overlap
            # judging disabled) compute caches inline from the judged output.
            cache_overlap_client = None
            if train_collection_needed and train_cache_needed and overlap_judge:
                print("[merged_collect] Starting cache overlap worker on GPU 0...", flush=True)
                cache_overlap_client = _CacheOverlapClient(
                    config_path=str(config_path),
                    overrides=overrides,
                    checkpoint_path=str(checkpoint),
                    checkpoint_id=checkpoint_id,
                )

            try:
                if train_collection_needed:
                    print("[merged_collect] Starting train collection...", flush=True)
                    train_sampling_profile = resolved_rollout_sampling_profile(config, split="train")
                    train_profile_id = sampling_profile_id(train_sampling_profile)
                    _collect_split(
                        config=config,
                        checkpoint_id=checkpoint_id,
                        checkpoint_path=checkpoint,
                        generator=generator,
                        backend=backend,
                        overlap_judge_client=overlap_judge_client,
                        split="train",
                        examples=train_examples_all,
                        raw_output_path=Path(train_raw_output),
                        judged_output_path=(
                            Path(train_judged_output) if train_judged_output else None
                        ),
                        sampling_profile=train_sampling_profile,
                        profile_id=train_profile_id,
                        group_size=config.training.group_size,
                        sample_seed=sample_seed,
                        resume=resume,
                        cache_overlap_client=cache_overlap_client,
                        cached_output_path=(
                            Path(train_cached_output) if train_cached_output else None
                        ),
                    )
                    outputs["train_raw"] = Path(train_raw_output)
                    if train_judged_output:
                        outputs["train_judged"] = Path(train_judged_output)

                # Compute caches inline when the cache worker wasn't used:
                #  - Resume: collection already done, judged output exists
                #  - No overlap judging: launcher handles cache separately
                if train_cache_needed and cache_overlap_client is None:
                    judged_path = (
                        Path(train_judged_output)
                        if train_judged_output
                        else None
                    )
                    if judged_path is not None and judged_path.exists():
                        print("[merged_collect] Computing training caches inline...", flush=True)
                        _run_cache_inline(
                            config=config,
                            checkpoint_path=checkpoint,
                            judged_rollout_path=judged_path,
                            cached_output_path=(
                                Path(train_cached_output) if train_cached_output else None
                            ),
                            resume=resume,
                        )
                if train_cached_output:
                    outputs["train_cached"] = Path(train_cached_output)
            finally:
                if cache_overlap_client is not None:
                    cache_overlap_client.close()

    finally:
        if overlap_judge_client is not None:
            overlap_judge_client.close()

    return outputs


def _run_cache_inline(
    *,
    config,
    checkpoint_path: Path,
    judged_rollout_path: Path,
    cached_output_path: Path | None,
    resume: bool,
) -> None:
    """Fallback: compute training caches inline (no overlap judge, or after collection)."""
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

    scorer = build_cache_scorer(config, checkpoint_path=str(checkpoint_path))
    for row in pending_rows:
        samples = extract_trainable_samples(
            row["trajectory_records"],
            row["turn_rewards"],
            rollout_id=f"{row.get('query_id')}:{row.get('rollout_index')}",
        )
        if not samples:
            append_jsonl(cached_output_path, dict(row))
            continue
        cache_payloads = scorer.cache_samples(samples)
        cached_row = _attach_training_caches(
            row,
            cache_payloads=cache_payloads,
            checkpoint_id=checkpoint_id,
        )
        append_jsonl(cached_output_path, cached_row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-process merged collection, judging, metrics, and cache step."
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
