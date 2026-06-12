from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from self_summarization_agent.checkpoints import advance_latest_checkpoint, checkpoint_id_from_path, resolve_latest_checkpoint
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.launcher_utils import ensure_dir


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


def run_training_iteration(
    config,
    *,
    config_path: str | Path,
    iteration: int,
    latest_root: str | Path | None = None,
    command_runner: CommandRunner = default_command_runner,
    python_executable: str = sys.executable,
    resume_rollouts: bool = False,
    overrides: Sequence[str] = (),
) -> Path:
    train_dir = ensure_dir(latest_root or _train_dir(config))
    current = resolve_latest_checkpoint(train_dir)
    rollouts_dir = ensure_dir(train_dir / "rollouts")
    checkpoints_dir = ensure_dir(train_dir / "checkpoints")
    metrics_path = train_dir / "step_metrics.jsonl"
    eval_metrics_path = train_dir / "eval_metrics.jsonl"
    raw_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.raw.jsonl"
    judged_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.jsonl"
    eval_raw_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.eval.raw.jsonl"
    eval_judged_rollout_path = rollouts_dir / f"iteration-{iteration:05d}.eval.jsonl"
    next_checkpoint = checkpoints_dir / f"iteration-{iteration:05d}"

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
    if resume_rollouts:
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
    rollout_status = command_runner(rollout_command)
    if rollout_status != 0:
        raise RuntimeError(f"Rollout subprocess failed with exit code {rollout_status}")
    judge_status = command_runner(judge_command)
    if judge_status != 0:
        raise RuntimeError(f"Judge subprocess failed with exit code {judge_status}")
    train_status = command_runner(train_command)
    if train_status != 0:
        raise RuntimeError(f"Training subprocess failed with exit code {train_status}")
    if config.dataset.eval_limit > 0:
        eval_rollout_command = [
            python_executable,
            "-m",
            "self_summarization_agent.rollout_collection",
            "--config",
            str(config_path),
            "--checkpoint",
            str(next_checkpoint),
            "--output",
            str(eval_raw_rollout_path),
            "--split",
            "eval",
        ]
        _append_cli_overrides(eval_rollout_command, overrides)
        eval_rollout_command.extend(["--set", "training.group_size=1"])
        eval_judge_command = [
            python_executable,
            "-m",
            "self_summarization_agent.judge_step",
            "--config",
            str(config_path),
            "--checkpoint",
            str(next_checkpoint),
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
            checkpoint_id_from_path(next_checkpoint),
        ]
        eval_rollout_status = command_runner(eval_rollout_command)
        if eval_rollout_status != 0:
            raise RuntimeError(f"Eval rollout subprocess failed with exit code {eval_rollout_status}")
        eval_judge_status = command_runner(eval_judge_command)
        if eval_judge_status != 0:
            raise RuntimeError(f"Eval judge subprocess failed with exit code {eval_judge_status}")
        eval_metrics_status = command_runner(eval_metrics_command)
        if eval_metrics_status != 0:
            raise RuntimeError(f"Eval metrics subprocess failed with exit code {eval_metrics_status}")
    advanced = advance_latest_checkpoint(train_dir, next_checkpoint)
    return advanced.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one process-isolated rollout/train iteration.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number.")
    parser.add_argument("--latest-root", default=None, help="Directory containing the latest checkpoint pointer.")
    parser.add_argument("--resume-rollouts", action="store_true", help="Resume the iteration rollout JSONL if it exists.")
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
        resume_rollouts=args.resume_rollouts,
        overrides=args.overrides,
    )
    print(next_checkpoint)


if __name__ == "__main__":
    main()
