from __future__ import annotations

import argparse
from contextlib import suppress
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib import request

from self_summarization_agent.checkpoints import (
    advance_latest_checkpoint,
    checkpoint_id_from_path,
    is_vllm_loadable_checkpoint,
    resolve_latest_checkpoint,
)
from self_summarization_agent.config import (
    load_train_config,
    parse_cli_overrides,
    resolved_rollout_sampling_profile,
    sampling_profile_id,
)
from self_summarization_agent.launcher_utils import append_jsonl, ensure_dir, utc_timestamp
from self_summarization_agent.trajectory import (
    TOKEN_CACHE_FIELD,
    is_training_cache_current,
    record_has_training_tokens,
    validate_trajectory_schema,
)


CommandRunner = Callable[[Sequence[str]], int]


def default_command_runner(command: Sequence[str]) -> int:
    completed = subprocess.run(list(command), check=False)
    return completed.returncode


def _wait_for_retrieval_worker(
    process: subprocess.Popen,
    ready_file: Path,
    timeout_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready_file.exists():
            payload = json.loads(ready_file.read_text(encoding="utf-8"))
            url = payload.get("url")
            if not isinstance(url, str):
                raise RuntimeError(f"Retrieval worker ready file is missing url: {ready_file}")
            return url
        if process.poll() is not None:
            raise RuntimeError(f"Retrieval worker exited before becoming ready with code {process.returncode}")
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for retrieval worker readiness at {ready_file}")


def _start_retrieval_worker(
    *,
    config_path: str | Path,
    train_dir: Path,
    python_executable: str,
    overrides: Sequence[str],
    startup_timeout_seconds: int,
) -> tuple[subprocess.Popen, str]:
    ready_file = train_dir / "retrieval_worker.json"
    with suppress(FileNotFoundError):
        ready_file.unlink()
    command = [
        python_executable,
        "-m",
        "self_summarization_agent.retrieval_worker",
        "--config",
        str(config_path),
        "--ready-file",
        str(ready_file),
    ]
    _append_cli_overrides(command, overrides)
    process = subprocess.Popen(command)
    try:
        url = _wait_for_retrieval_worker(process, ready_file, startup_timeout_seconds)
    except Exception:
        _stop_retrieval_worker(process, None)
        raise
    return process, url


def _stop_retrieval_worker(process: subprocess.Popen | None, url: str | None) -> None:
    if process is None:
        return
    if url and process.poll() is None:
        with suppress(Exception):
            req = request.Request(f"{url.rstrip('/')}/shutdown", data=b"{}", method="POST")
            request.urlopen(req, timeout=5).close()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=15)
    if process.poll() is None:
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=15)
    if process.poll() is None:
        process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=15)


def _train_dir(config) -> Path:
    return Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name


def _train_step_command_prefix(
    config,
    python_executable: str,
    *,
    module_name: str = "self_summarization_agent.train_step",
) -> list[str]:
    if config.training.backend == "fsdp2_context_parallel":
        command = [
            "accelerate",
            "launch",
            "--num_processes",
            str(len(config.training.gpu_ids) or config.training.context_parallel_size),
            "--num_machines",
            "1",
            "--machine_rank",
            "0",
            "--main_process_port",
            "0",
            "--use-fsdp",
            "--fsdp_version",
            str(config.training.fsdp_version or 2),
            "--fsdp_auto_wrap_policy",
            "transformer_based_wrap",
            "--parallelism-config-cp-size",
            str(config.training.context_parallel_size),
        ]
        if config.training.activation_checkpointing:
            command.append("--fsdp_activation_checkpointing=true")
        command.extend(["-m", module_name])
        return command
    return [python_executable, "-m", module_name]


def _append_cli_overrides(command: list[str], overrides: Sequence[str]) -> None:
    for override in overrides:
        command.extend(["--set", override])


def _run_timed_phase(
    *,
    phase: str,
    iteration: int,
    command: Sequence[str],
    command_runner: CommandRunner,
    timings_path: Path,
    timeout_seconds: float | None = None,
) -> int:
    print(f"[iteration_launcher] starting {phase}", flush=True)
    started = time.perf_counter()
    if timeout_seconds is None:
        status = command_runner(command)
    else:
        try:
            proc = subprocess.Popen(list(command))
            try:
                proc.wait(timeout=timeout_seconds)
                status = proc.returncode
            except subprocess.TimeoutExpired:
                print(
                    f"[iteration_launcher] {phase} timed out after {timeout_seconds:.0f}s. "
                    f"Terminating (pid={proc.pid})...",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(
                        f"[iteration_launcher] {phase} did not respond to SIGTERM, "
                        f"killing (pid={proc.pid})...",
                        flush=True,
                    )
                    proc.kill()
                    proc.wait(timeout=10)
                raise
        except subprocess.TimeoutExpired:
            raise  # re-raise so the launcher can handle it
    elapsed_seconds = time.perf_counter() - started
    print(
        f"[iteration_launcher] finished {phase}: "
        f"exit_code={status}, elapsed_seconds={elapsed_seconds:.3f}",
        flush=True,
    )
    append_jsonl(
        timings_path,
        {
            "iteration": iteration,
            "timestamp_utc": utc_timestamp(),
            "phase": phase,
            "elapsed_seconds": elapsed_seconds,
            "exit_code": status,
        },
    )
    return status


def _record_skipped_phase(*, phase: str, iteration: int, timings_path: Path) -> None:
    print(f"[iteration_launcher] skipping {phase}: completed artifact exists", flush=True)
    append_jsonl(
        timings_path,
        {
            "iteration": iteration,
            "timestamp_utc": utc_timestamp(),
            "phase": phase,
            "elapsed_seconds": 0.0,
            "exit_code": 0,
            "skipped": True,
        },
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row in {path} on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row in {path} on line {line_number} must be an object")
            rows.append(row)
    return rows


def _expected_train_rollout_count(config) -> int | None:
    if config.collection.train_task_count is not None:
        return config.collection.train_task_count * config.training.group_size
    if config.training.rollout_query_count is not None:
        return config.training.rollout_query_count * config.training.group_size
    if config.dataset.train_limit is not None:
        query_count = config.dataset.train_limit
        if config.dataset.limit is not None:
            query_count = min(query_count, config.dataset.limit)
        return query_count * config.training.group_size
    if config.dataset.limit is not None:
        return config.dataset.limit * config.training.group_size
    return None


def _expected_eval_rollout_count(config) -> int:
    if config.collection.eval_task_count is not None:
        return config.collection.eval_task_count * config.evaluation.samples_per_task
    if config.dataset.train_limit is None:
        return 0
    if config.dataset.limit is None:
        available_after_train = config.dataset.eval_limit
    else:
        available_after_train = max(0, config.dataset.limit - config.dataset.train_limit)
    return min(config.dataset.eval_limit, available_after_train) * config.evaluation.samples_per_task


def _has_complete_raw_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    expected_sampling_profile_id: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = _load_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        return False
    if expected_count is None and not rows:
        return False
    for index, row in enumerate(rows, start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        if (
            expected_sampling_profile_id is not None
            and row.get("sampling_profile_id") != expected_sampling_profile_id
        ):
            return False
        if "turn_rewards" in row:
            raise ValueError(f"Cannot resume from {path}: row {index} is already judged")
        if not isinstance(row.get("query_id"), str):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing query_id")
        if not isinstance(row.get("rollout_index"), int):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing rollout_index")
        if not isinstance(row.get("turn_records"), list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
        if not isinstance(row.get("trajectory_records"), list):
            raise ValueError(
                f"Cannot resume from {path}: row {index} is missing trajectory_records; recollect it"
            )
        validate_trajectory_schema(
            row["trajectory_records"],
            context=f"Cannot resume from {path}: row {index}",
        )
    return True


def _has_complete_judged_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    require_judge: bool,
    expected_sampling_profile_id: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = _load_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        return False
    if expected_count is None and not rows:
        return False
    for index, row in enumerate(rows, start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        if (
            expected_sampling_profile_id is not None
            and row.get("sampling_profile_id") != expected_sampling_profile_id
        ):
            return False
        if not isinstance(row.get("turn_records"), list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
        if not isinstance(row.get("trajectory_records"), list):
            raise ValueError(
                f"Cannot resume from {path}: row {index} is missing trajectory_records; recollect it"
            )
        validate_trajectory_schema(
            row["trajectory_records"],
            context=f"Cannot resume from {path}: row {index}",
        )
        if not isinstance(row.get("turn_rewards"), dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_rewards")
        if require_judge and not isinstance(row.get("judge"), dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing judge payload")
    return True


def _has_complete_cached_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    train_compaction_tokens: bool = True,
) -> bool:
    if not path.exists():
        return False
    rows = _load_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        return False
    if expected_count is None and not rows:
        return False
    for index, row in enumerate(rows, start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        turn_records = row.get("turn_records")
        trajectory_records = row.get("trajectory_records")
        turn_rewards = row.get("turn_rewards")
        if not isinstance(turn_records, list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
        if not isinstance(trajectory_records, list):
            raise ValueError(
                f"Cannot resume from {path}: row {index} is missing trajectory_records; recollect it"
            )
        validate_trajectory_schema(
            trajectory_records,
            context=f"Cannot resume from {path}: row {index}",
        )
        if not isinstance(turn_rewards, dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_rewards")
        if row.get("trainable_sample_count") == 0:
            continue
        trainable_turn_ids = {
            record.get("turn_id")
            for record in trajectory_records
            if isinstance(record, dict)
            and isinstance(record.get("turn_id"), str)
            and record.get("turn_id") in turn_rewards
            and record_has_training_tokens(
                record,
                train_compaction_tokens=train_compaction_tokens,
            )
        }
        if not trainable_turn_ids:
            continue
        current_cache_turn_ids = {
            record.get("turn_id")
            for record in trajectory_records
            if isinstance(record, dict)
            and isinstance(record.get("turn_id"), str)
            and record.get("turn_id") in trainable_turn_ids
            and is_training_cache_current(
                record.get(TOKEN_CACHE_FIELD),
                train_compaction_tokens=train_compaction_tokens,
            )
        }
        if not trainable_turn_ids <= current_cache_turn_ids:
            return False
    return True


def _has_inline_cached_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    train_compaction_tokens: bool = True,
) -> bool:
    if not path.exists():
        return False
    rows = _load_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        return False
    if expected_count is None and not rows:
        return False
    for index, row in enumerate(rows, start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot use inline cache from {path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        turn_records = row.get("turn_records")
        trajectory_records = row.get("trajectory_records")
        turn_rewards = row.get("turn_rewards")
        if (
            not isinstance(turn_records, list)
            or not isinstance(trajectory_records, list)
            or not isinstance(turn_rewards, dict)
        ):
            return False
        validate_trajectory_schema(
            trajectory_records,
            context=f"Cannot use inline cache from {path}: row {index}",
        )
        if row.get("trainable_sample_count") == 0:
            continue
        trainable_turn_ids = {
            record.get("turn_id")
            for record in trajectory_records
            if isinstance(record, dict)
            and isinstance(record.get("turn_id"), str)
            and record.get("turn_id") in turn_rewards
            and record_has_training_tokens(
                record,
                train_compaction_tokens=train_compaction_tokens,
            )
        }
        if not trainable_turn_ids:
            return False
        current_cache_turn_ids = {
            record.get("turn_id")
            for record in trajectory_records
            if isinstance(record, dict)
            and isinstance(record.get("turn_id"), str)
            and record.get("turn_id") in trainable_turn_ids
            and is_training_cache_current(
                record.get(TOKEN_CACHE_FIELD),
                train_compaction_tokens=train_compaction_tokens,
            )
        }
        if not trainable_turn_ids <= current_cache_turn_ids:
            return False
    return True


def _has_eval_metrics(
    metrics_path: Path,
    *,
    iteration: int,
    policy_checkpoint_id: str,
    expected_sampling_profile_id: str,
) -> bool:
    if not metrics_path.exists():
        return False
    for row in _load_jsonl(metrics_path):
        if (
            row.get("iteration") == iteration
            and row.get("policy_checkpoint_id") == policy_checkpoint_id
            and row.get("eval_sampling_profile_id") == expected_sampling_profile_id
        ):
            return True
    return False


def _run_or_skip_phase(
    *,
    phase: str,
    iteration: int,
    command: Sequence[str],
    command_runner: CommandRunner,
    timings_path: Path,
    completed: bool,
    error_message: str,
    timeout_seconds: float | None = None,
) -> None:
    if completed:
        _record_skipped_phase(phase=phase, iteration=iteration, timings_path=timings_path)
        return
    try:
        status = _run_timed_phase(
            phase=phase,
            iteration=iteration,
            command=command,
            command_runner=command_runner,
            timings_path=timings_path,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{error_message} timed out after {timeout_seconds:.0f}s "
            f"(phase={phase}, iteration={iteration})"
        ) from None
    if status != 0:
        raise RuntimeError(f"{error_message} failed with exit code {status}")


def run_checkpoint_evaluation(
    config,
    *,
    config_path: str | Path,
    iteration: int,
    latest_root: str | Path | None = None,
    command_runner: CommandRunner = default_command_runner,
    python_executable: str = sys.executable,
    resume: bool = False,
    overrides: Sequence[str] = (),
) -> Path:
    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}")
    if config.dataset.eval_limit <= 0:
        raise ValueError("dataset.eval_limit must be positive for checkpoint evaluation")

    phase_timeout = config.runtime.phase_timeout_seconds

    train_dir = ensure_dir(latest_root or _train_dir(config))
    current = resolve_latest_checkpoint(train_dir)
    rollouts_dir = ensure_dir(train_dir / "rollouts")
    eval_metrics_path = train_dir / "eval_metrics.jsonl"
    phase_timings_path = train_dir / "phase_timings.jsonl"
    eval_raw_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.eval.raw.jsonl"
    eval_judged_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.eval.jsonl"
    eval_sampling_profile_id = sampling_profile_id(
        resolved_rollout_sampling_profile(config, split="eval")
    )

    if resume and _has_eval_metrics(
        eval_metrics_path,
        iteration=iteration,
        policy_checkpoint_id=current.checkpoint_id,
        expected_sampling_profile_id=eval_sampling_profile_id,
    ):
        return current.path

    eval_rollout_command = [
        python_executable,
        "-m",
        "self_summarization_agent.rollout_collection",
        "--config",
        str(config_path),
        "--checkpoint",
        str(current.path),
        "--output",
        str(eval_raw_rollout_path),
        "--split",
        "eval",
    ]
    if config.rollout.overlap_judge and config.judge.enabled:
        eval_rollout_command.extend(["--judged-output", str(eval_judged_rollout_path)])
    _append_cli_overrides(eval_rollout_command, overrides)
    if resume:
        eval_rollout_command.append("--resume")

    eval_judge_command = [
        python_executable,
        "-m",
        "self_summarization_agent.judge_step",
        "--config",
        str(config_path),
        "--checkpoint",
        str(current.path),
        "--rollouts",
        str(eval_raw_rollout_path),
        "--output",
        str(eval_judged_rollout_path),
        "--split",
        "eval",
    ]
    _append_cli_overrides(eval_judge_command, overrides)
    eval_metrics_command = [
        python_executable,
        "-m",
        "self_summarization_agent.eval_metrics",
        "--rollouts",
        str(eval_judged_rollout_path),
        "--metrics",
        str(eval_metrics_path),
        "--iteration",
        str(iteration),
        "--policy-checkpoint-id",
        current.checkpoint_id,
    ]

    expected_count = _expected_eval_rollout_count(config)
    raw_complete = resume and _has_complete_raw_rollouts(
        eval_raw_rollout_path,
        checkpoint_id=current.checkpoint_id,
        expected_count=expected_count,
        expected_sampling_profile_id=eval_sampling_profile_id,
    )
    retrieval_worker_process = None
    retrieval_worker_url = None
    if config.retrieval.persistent_worker and not raw_complete:
        retrieval_worker_process, retrieval_worker_url = _start_retrieval_worker(
            config_path=config_path,
            train_dir=train_dir,
            python_executable=python_executable,
            overrides=overrides,
            startup_timeout_seconds=config.retrieval.worker_startup_timeout_seconds,
        )
        eval_rollout_command.extend(["--retrieval-worker-url", retrieval_worker_url])

    try:
        _run_or_skip_phase(
            phase="eval_rollout",
            iteration=iteration,
            command=eval_rollout_command,
            command_runner=command_runner,
            timings_path=phase_timings_path,
            completed=raw_complete,
            error_message="Eval rollout subprocess",
            timeout_seconds=phase_timeout,
        )
    finally:
        _stop_retrieval_worker(retrieval_worker_process, retrieval_worker_url)

    judged_complete = resume and _has_complete_judged_rollouts(
        eval_judged_rollout_path,
        checkpoint_id=current.checkpoint_id,
        expected_count=expected_count,
        require_judge=True,
        expected_sampling_profile_id=eval_sampling_profile_id,
    )
    if not judged_complete and config.rollout.overlap_judge and config.judge.enabled:
        judged_complete = _has_complete_judged_rollouts(
            eval_judged_rollout_path,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_count,
            require_judge=True,
            expected_sampling_profile_id=eval_sampling_profile_id,
        )
    _run_or_skip_phase(
        phase="eval_judge",
        iteration=iteration,
        command=eval_judge_command,
        command_runner=command_runner,
        timings_path=phase_timings_path,
        completed=judged_complete,
        error_message="Eval judge subprocess",
        timeout_seconds=phase_timeout,
    )
    metrics_complete = resume and _has_eval_metrics(
        eval_metrics_path,
        iteration=iteration,
        policy_checkpoint_id=current.checkpoint_id,
        expected_sampling_profile_id=eval_sampling_profile_id,
    )
    _run_or_skip_phase(
        phase="eval_metrics",
        iteration=iteration,
        command=eval_metrics_command,
        command_runner=command_runner,
        timings_path=phase_timings_path,
        completed=metrics_complete,
        error_message="Eval metrics subprocess",
        timeout_seconds=phase_timeout,
    )
    return current.path


def run_training_iteration(
    config,
    *,
    config_path: str | Path,
    iteration: int,
    latest_root: str | Path | None = None,
    command_runner: CommandRunner = default_command_runner,
    python_executable: str = sys.executable,
    resume: bool = False,
    resume_rollouts: bool = False,
    overrides: Sequence[str] = (),
) -> Path:
    if iteration < 1:
        raise ValueError(f"iteration must be at least 1, got {iteration}")
    phase_timeout = config.runtime.phase_timeout_seconds
    train_dir = ensure_dir(latest_root or _train_dir(config))
    current = resolve_latest_checkpoint(train_dir)
    should_resume = resume or resume_rollouts
    rollouts_dir = ensure_dir(train_dir / "rollouts")
    checkpoints_dir = ensure_dir(train_dir / "checkpoints")
    metrics_path = train_dir / "step_metrics.jsonl"
    eval_metrics_path = train_dir / "eval_metrics.jsonl"
    phase_timings_path = train_dir / "phase_timings.jsonl"
    raw_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.raw.jsonl"
    judged_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.judged.jsonl"
    cached_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.jsonl"
    eval_iteration = iteration - 1
    eval_raw_rollout_path = rollouts_dir / f"iteration-{eval_iteration:05d}.eval.raw.jsonl"
    eval_judged_rollout_path = rollouts_dir / f"iteration-{eval_iteration:05d}.eval.jsonl"
    eval_sampling_profile_id = sampling_profile_id(
        resolved_rollout_sampling_profile(config, split="eval")
    )
    next_checkpoint = checkpoints_dir / f"iteration-{iteration:05d}"
    training_already_advanced = should_resume and current.checkpoint_id == checkpoint_id_from_path(next_checkpoint)
    eval_checkpoint = current.path
    eval_checkpoint_id = current.checkpoint_id

    if training_already_advanced:
        return current.path

    # The primary merged path always judges after collection teardown.  The
    # legacy overlap_judge setting remains parseable but no longer controls the
    # lifecycle.
    use_merged_judge = config.judge.enabled

    # ------------------------------------------------------------------
    # Merged collect command (replaces eval_rollout, train_rollout,
    # eval_judge, train_judge, eval_metrics, and train_cache).
    # ------------------------------------------------------------------
    merged_collect_command = [
        python_executable,
        "-m",
        "self_summarization_agent.merged_collect_step",
        "--config",
        str(config_path),
        "--checkpoint",
        str(current.path),
        "--train-raw-output",
        str(raw_rollout_path),
        "--train-cached-output",
        str(cached_rollout_path),
        "--sample-seed",
        str(config.experiment.seed + iteration),
    ]
    if use_merged_judge:
        merged_collect_command.extend(["--train-judged-output", str(judged_rollout_path)])
    if config.dataset.eval_limit > 0:
        merged_collect_command.extend(
            [
                "--eval-raw-output",
                str(eval_raw_rollout_path),
                "--eval-metrics-output",
                str(eval_metrics_path),
                "--eval-iteration",
                str(eval_iteration),
            ]
        )
        if use_merged_judge:
            merged_collect_command.extend(
                ["--eval-judged-output", str(eval_judged_rollout_path)]
            )
    _append_cli_overrides(merged_collect_command, overrides)
    if should_resume:
        merged_collect_command.append("--resume")

    # Completion checks for the merged phase
    expected_eval_count = _expected_eval_rollout_count(config)
    expected_train_count = _expected_train_rollout_count(config)
    train_raw_complete = should_resume and _has_complete_raw_rollouts(
        raw_rollout_path,
        checkpoint_id=current.checkpoint_id,
        expected_count=expected_train_count,
    )
    eval_raw_complete = config.dataset.eval_limit <= 0 or (
        should_resume
        and _has_complete_raw_rollouts(
            eval_raw_rollout_path,
            checkpoint_id=eval_checkpoint_id,
            expected_count=expected_eval_count,
            expected_sampling_profile_id=eval_sampling_profile_id,
        )
    )
    train_judged_complete = (
        not use_merged_judge
        or should_resume
        and _has_complete_judged_rollouts(
            judged_rollout_path,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train_count,
            require_judge=False,
        )
    )
    eval_judged_complete = (
        config.dataset.eval_limit <= 0
        or not use_merged_judge
        or (
            should_resume
            and _has_complete_judged_rollouts(
                eval_judged_rollout_path,
                checkpoint_id=eval_checkpoint_id,
                expected_count=expected_eval_count,
                require_judge=True,
                expected_sampling_profile_id=eval_sampling_profile_id,
            )
        )
    )
    train_cached_complete = should_resume and _has_complete_cached_rollouts(
        cached_rollout_path,
        checkpoint_id=current.checkpoint_id,
        expected_count=expected_train_count,
        train_compaction_tokens=config.training.train_compaction_tokens,
    )
    # Rollout-native caches survive the sequential judge transform, so a
    # complete judged artifact can still serve as the cached training input.
    if not train_cached_complete and use_merged_judge:
        train_cached_complete = _has_inline_cached_rollouts(
            judged_rollout_path,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train_count,
            train_compaction_tokens=config.training.train_compaction_tokens,
        )
    eval_metrics_complete = config.dataset.eval_limit <= 0 or (
        should_resume
        and _has_eval_metrics(
            eval_metrics_path,
            iteration=eval_iteration,
            policy_checkpoint_id=eval_checkpoint_id,
            expected_sampling_profile_id=eval_sampling_profile_id,
        )
    )
    merged_collect_done = (
        train_raw_complete
        and eval_raw_complete
        and train_judged_complete
        and eval_judged_complete
        and train_cached_complete
        and eval_metrics_complete
    )

    # merged_collect owns retrieval only while policy collection is active and
    # tears it down before allocating the judge on all four GPUs.
    _run_or_skip_phase(
        phase="merged_collect",
        iteration=iteration,
        command=merged_collect_command,
        command_runner=command_runner,
        timings_path=phase_timings_path,
        completed=merged_collect_done,
        error_message="Merged collect subprocess",
        timeout_seconds=phase_timeout,
    )

    # ------------------------------------------------------------------
    # Diagnostic fallback retained only for configurations with judging
    # disabled in the merged phase.
    # ------------------------------------------------------------------
    if not use_merged_judge:
        if config.dataset.eval_limit > 0:
            eval_judge_command = [
                python_executable,
                "-m",
                "self_summarization_agent.judge_step",
                "--config",
                str(config_path),
                "--checkpoint",
                str(eval_checkpoint),
                "--rollouts",
                str(eval_raw_rollout_path),
                "--output",
                str(eval_judged_rollout_path),
                "--split",
                "eval",
            ]
            _append_cli_overrides(eval_judge_command, overrides)
            eval_judged_complete = should_resume and _has_complete_judged_rollouts(
                eval_judged_rollout_path,
                checkpoint_id=eval_checkpoint_id,
                expected_count=expected_eval_count,
                require_judge=True,
                expected_sampling_profile_id=eval_sampling_profile_id,
            )
            _run_or_skip_phase(
                phase="eval_judge",
                iteration=iteration,
                command=eval_judge_command,
                command_runner=command_runner,
                timings_path=phase_timings_path,
                completed=eval_judged_complete,
                error_message="Eval judge subprocess",
                timeout_seconds=phase_timeout,
            )
            eval_metrics_complete_cmd = should_resume and _has_eval_metrics(
                eval_metrics_path,
                iteration=eval_iteration,
                policy_checkpoint_id=eval_checkpoint_id,
                expected_sampling_profile_id=eval_sampling_profile_id,
            )
            eval_metrics_command = [
                python_executable,
                "-m",
                "self_summarization_agent.eval_metrics",
                "--rollouts",
                str(eval_judged_rollout_path),
                "--metrics",
                str(eval_metrics_path),
                "--iteration",
                str(eval_iteration),
                "--policy-checkpoint-id",
                eval_checkpoint_id,
            ]
            _run_or_skip_phase(
                phase="eval_metrics",
                iteration=iteration,
                command=eval_metrics_command,
                command_runner=command_runner,
                timings_path=phase_timings_path,
                completed=eval_metrics_complete_cmd,
                error_message="Eval metrics subprocess",
                timeout_seconds=phase_timeout,
            )

        train_judge_command = [
            python_executable,
            "-m",
            "self_summarization_agent.judge_step",
            "--config",
            str(config_path),
            "--checkpoint",
            str(current.path),
            "--rollouts",
            str(raw_rollout_path),
            "--output",
            str(judged_rollout_path),
        ]
        _append_cli_overrides(train_judge_command, overrides)
        train_judged_complete = should_resume and _has_complete_judged_rollouts(
            judged_rollout_path,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train_count,
            require_judge=False,
        )
        _run_or_skip_phase(
            phase="train_judge",
            iteration=iteration,
            command=train_judge_command,
            command_runner=command_runner,
            timings_path=phase_timings_path,
            completed=train_judged_complete,
            error_message="Judge subprocess",
            timeout_seconds=phase_timeout,
        )

        cache_command = [
            *_train_step_command_prefix(
                config,
                python_executable,
                module_name="self_summarization_agent.cache_step",
            ),
            "--config",
            str(config_path),
            "--checkpoint",
            str(current.path),
            "--rollouts",
            str(judged_rollout_path),
            "--output",
            str(cached_rollout_path),
        ]
        _append_cli_overrides(cache_command, overrides)
        if should_resume:
            cache_command.append("--resume")
        inline_cached_rollouts = _has_inline_cached_rollouts(
            judged_rollout_path,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train_count,
            train_compaction_tokens=config.training.train_compaction_tokens,
        )
        train_cached_complete = should_resume and (
            _has_complete_cached_rollouts(
                cached_rollout_path,
                checkpoint_id=current.checkpoint_id,
                expected_count=expected_train_count,
                train_compaction_tokens=config.training.train_compaction_tokens,
            )
        )
        train_cached_complete = inline_cached_rollouts or train_cached_complete
        _run_or_skip_phase(
            phase="train_cache",
            iteration=iteration,
            command=cache_command,
            command_runner=command_runner,
            timings_path=phase_timings_path,
            completed=train_cached_complete,
            error_message="Cache subprocess",
            timeout_seconds=phase_timeout,
        )

    train_command = [
        *_train_step_command_prefix(config, python_executable),
        "--config",
        str(config_path),
        "--checkpoint",
        str(current.path),
        "--rollouts",
        str(cached_rollout_path),
        "--output-checkpoint",
        str(next_checkpoint),
        "--metrics",
        str(metrics_path),
    ]
    _append_cli_overrides(train_command, overrides)
    if _has_inline_cached_rollouts(
        judged_rollout_path,
        checkpoint_id=current.checkpoint_id,
        expected_count=expected_train_count,
        train_compaction_tokens=config.training.train_compaction_tokens,
    ):
        train_command[train_command.index("--rollouts") + 1] = str(judged_rollout_path)
    checkpoint_complete = should_resume and (
        is_vllm_loadable_checkpoint(next_checkpoint)
    )
    _run_or_skip_phase(
        phase="train_update",
        iteration=iteration,
        command=train_command,
        command_runner=command_runner,
        timings_path=phase_timings_path,
        completed=checkpoint_complete,
        error_message="Training subprocess",
        timeout_seconds=phase_timeout,
    )
    advanced = advance_latest_checkpoint(train_dir, next_checkpoint)
    return advanced.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one process-isolated rollout/train iteration.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the latest checkpoint without collecting training trajectories or updating weights.",
    )
    parser.add_argument("--latest-root", default=None, help="Directory containing the latest checkpoint pointer.")
    parser.add_argument("--resume", action="store_true", help="Resume from the first incomplete iteration phase.")
    parser.add_argument(
        "--resume-rollouts",
        action="store_true",
        help="Deprecated alias for --resume.",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    if args.eval_only:
        next_checkpoint = run_checkpoint_evaluation(
            config,
            config_path=args.config,
            iteration=args.iteration,
            latest_root=args.latest_root,
            resume=args.resume or args.resume_rollouts,
            overrides=args.overrides,
        )
    else:
        next_checkpoint = run_training_iteration(
            config,
            config_path=args.config,
            iteration=args.iteration,
            latest_root=args.latest_root,
            resume=args.resume,
            resume_rollouts=args.resume_rollouts,
            overrides=args.overrides,
        )
    print(next_checkpoint)


if __name__ == "__main__":
    main()
