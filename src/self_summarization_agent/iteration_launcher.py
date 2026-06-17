from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from self_summarization_agent.checkpoints import (
    advance_latest_checkpoint,
    checkpoint_id_from_path,
    is_vllm_loadable_checkpoint,
    resolve_latest_checkpoint,
)
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.launcher_utils import append_jsonl, ensure_dir, utc_timestamp


CommandRunner = Callable[[Sequence[str]], int]


def default_command_runner(command: Sequence[str]) -> int:
    completed = subprocess.run(list(command), check=False)
    return completed.returncode


def _train_dir(config) -> Path:
    return Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name


def _train_step_command_prefix(config, python_executable: str) -> list[str]:
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
        command.extend(["-m", "self_summarization_agent.train_step"])
        return command
    return [python_executable, "-m", "self_summarization_agent.train_step"]


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
) -> int:
    print(f"[iteration_launcher] starting {phase}", flush=True)
    started = time.perf_counter()
    status = command_runner(command)
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
    if config.dataset.train_limit is None:
        return 0
    if config.dataset.limit is None:
        available_after_train = config.dataset.eval_limit
    else:
        available_after_train = max(0, config.dataset.limit - config.dataset.train_limit)
    return min(config.dataset.eval_limit, available_after_train)


def _has_complete_raw_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
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
        if "turn_rewards" in row:
            raise ValueError(f"Cannot resume from {path}: row {index} is already judged")
        if not isinstance(row.get("query_id"), str):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing query_id")
        if not isinstance(row.get("rollout_index"), int):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing rollout_index")
        if not isinstance(row.get("turn_records"), list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
    return True


def _has_complete_judged_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    require_judge: bool,
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
        if not isinstance(row.get("turn_records"), list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
        if not isinstance(row.get("turn_rewards"), dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_rewards")
        if require_judge and not isinstance(row.get("judge"), dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing judge payload")
    return True


def _has_eval_metrics(metrics_path: Path, *, iteration: int, policy_checkpoint_id: str) -> bool:
    if not metrics_path.exists():
        return False
    for row in _load_jsonl(metrics_path):
        if row.get("iteration") == iteration and row.get("policy_checkpoint_id") == policy_checkpoint_id:
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
) -> None:
    if completed:
        _record_skipped_phase(phase=phase, iteration=iteration, timings_path=timings_path)
        return
    status = _run_timed_phase(
        phase=phase,
        iteration=iteration,
        command=command,
        command_runner=command_runner,
        timings_path=timings_path,
    )
    if status != 0:
        raise RuntimeError(f"{error_message} failed with exit code {status}")


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
    train_dir = ensure_dir(latest_root or _train_dir(config))
    current = resolve_latest_checkpoint(train_dir)
    should_resume = resume or resume_rollouts
    rollouts_dir = ensure_dir(train_dir / "rollouts")
    checkpoints_dir = ensure_dir(train_dir / "checkpoints")
    metrics_path = train_dir / "step_metrics.jsonl"
    eval_metrics_path = train_dir / "eval_metrics.jsonl"
    phase_timings_path = train_dir / "phase_timings.jsonl"
    raw_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.raw.jsonl"
    judged_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.jsonl"
    eval_raw_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.eval.raw.jsonl"
    eval_judged_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.eval.jsonl"
    next_checkpoint = checkpoints_dir / f"iteration-{iteration:05d}"
    training_already_advanced = should_resume and current.checkpoint_id == checkpoint_id_from_path(next_checkpoint)
    eval_checkpoint = current.path if training_already_advanced else next_checkpoint
    eval_checkpoint_id = checkpoint_id_from_path(eval_checkpoint)

    if training_already_advanced:
        if config.dataset.eval_limit <= 0 or _has_eval_metrics(
            eval_metrics_path,
            iteration=iteration,
            policy_checkpoint_id=eval_checkpoint_id,
        ):
            return current.path

    rollout_command = [
        python_executable,
        "-m",
        "self_summarization_agent.rollout_collection",
        "--config",
        str(config_path),
        "--checkpoint",
        str(current.path),
        "--output",
        str(raw_rollout_path),
        "--sample-seed",
        str(config.experiment.seed + iteration),
    ]
    _append_cli_overrides(rollout_command, overrides)
    if should_resume:
        rollout_command.append("--resume")
    judge_command = [
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
    _append_cli_overrides(judge_command, overrides)
    train_command = [
        *_train_step_command_prefix(config, python_executable),
        "--config",
        str(config_path),
        "--checkpoint",
        str(current.path),
        "--rollouts",
        str(judged_rollout_path),
        "--output-checkpoint",
        str(next_checkpoint),
        "--metrics",
        str(metrics_path),
    ]
    _append_cli_overrides(train_command, overrides)
    train_raw_complete = should_resume and (
        training_already_advanced
        or _has_complete_raw_rollouts(
            raw_rollout_path,
            checkpoint_id=current.checkpoint_id,
            expected_count=_expected_train_rollout_count(config),
        )
    )
    _run_or_skip_phase(
        phase="train_rollout",
        iteration=iteration,
        command=rollout_command,
        command_runner=command_runner,
        timings_path=phase_timings_path,
        completed=train_raw_complete,
        error_message="Rollout subprocess",
    )
    train_judged_complete = should_resume and (
        training_already_advanced
        or _has_complete_judged_rollouts(
            judged_rollout_path,
            checkpoint_id=current.checkpoint_id,
            expected_count=_expected_train_rollout_count(config),
            require_judge=False,
        )
    )
    _run_or_skip_phase(
        phase="train_judge",
        iteration=iteration,
        command=judge_command,
        command_runner=command_runner,
        timings_path=phase_timings_path,
        completed=train_judged_complete,
        error_message="Judge subprocess",
    )
    checkpoint_complete = should_resume and (
        training_already_advanced or is_vllm_loadable_checkpoint(next_checkpoint)
    )
    _run_or_skip_phase(
        phase="train_update",
        iteration=iteration,
        command=train_command,
        command_runner=command_runner,
        timings_path=phase_timings_path,
        completed=checkpoint_complete,
        error_message="Training subprocess",
    )
    if config.dataset.eval_limit > 0:
        eval_rollout_command = [
            python_executable,
            "-m",
            "self_summarization_agent.rollout_collection",
            "--config",
            str(config_path),
            "--checkpoint",
            str(eval_checkpoint),
            "--output",
            str(eval_raw_rollout_path),
            "--split",
            "eval",
        ]
        _append_cli_overrides(eval_rollout_command, overrides)
        eval_rollout_command.extend(["--set", "training.group_size=1"])
        if should_resume:
            eval_rollout_command.append("--resume")
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
            eval_checkpoint_id,
        ]
        eval_expected_count = _expected_eval_rollout_count(config)
        eval_raw_complete = should_resume and _has_complete_raw_rollouts(
            eval_raw_rollout_path,
            checkpoint_id=eval_checkpoint_id,
            expected_count=eval_expected_count,
        )
        _run_or_skip_phase(
            phase="eval_rollout",
            iteration=iteration,
            command=eval_rollout_command,
            command_runner=command_runner,
            timings_path=phase_timings_path,
            completed=eval_raw_complete,
            error_message="Eval rollout subprocess",
        )
        eval_judged_complete = should_resume and _has_complete_judged_rollouts(
            eval_judged_rollout_path,
            checkpoint_id=eval_checkpoint_id,
            expected_count=eval_expected_count,
            require_judge=True,
        )
        _run_or_skip_phase(
            phase="eval_judge",
            iteration=iteration,
            command=eval_judge_command,
            command_runner=command_runner,
            timings_path=phase_timings_path,
            completed=eval_judged_complete,
            error_message="Eval judge subprocess",
        )
        eval_metrics_complete = should_resume and _has_eval_metrics(
            eval_metrics_path,
            iteration=iteration,
            policy_checkpoint_id=eval_checkpoint_id,
        )
        _run_or_skip_phase(
            phase="eval_metrics",
            iteration=iteration,
            command=eval_metrics_command,
            command_runner=command_runner,
            timings_path=phase_timings_path,
            completed=eval_metrics_complete,
            error_message="Eval metrics subprocess",
        )
    advanced = advance_latest_checkpoint(train_dir, eval_checkpoint)
    return advanced.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one process-isolated rollout/train iteration.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number.")
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
