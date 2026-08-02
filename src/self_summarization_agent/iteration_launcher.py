"""Compatibility CLI for running one iteration of the Python training loop."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from self_summarization_agent.checkpoints import resolve_latest_checkpoint
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.iteration_artifacts import (
    completed_iteration_from_checkpoint_id,
    expected_eval_rollout_count as _expected_eval_rollout_count,
    expected_train_rollout_count as _expected_train_rollout_count,
    has_complete_cached_rollouts as _has_complete_cached_rollouts,
    has_complete_judged_rollouts as _has_complete_judged_rollouts,
    has_complete_raw_rollouts as _has_complete_raw_rollouts,
    has_eval_metrics as _has_eval_metrics,
)
from self_summarization_agent.training_loop import (
    evaluate_latest_checkpoint,
    run_training_iteration as _run_training_iteration,
)


def run_training_iteration(
    config,
    *,
    config_path: str | Path,
    iteration: int,
    latest_root: str | Path | None = None,
    resume: bool = False,
    resume_rollouts: bool = False,
    overrides: Sequence[str] = (),
    **_legacy_options,
) -> Path:
    return _run_training_iteration(
        config,
        config_path=config_path,
        iteration=iteration,
        latest_root=latest_root,
        resume=resume or resume_rollouts,
        overrides=overrides,
    )


def run_checkpoint_evaluation(
    config,
    *,
    config_path: str | Path,
    iteration: int,
    latest_root: str | Path | None = None,
    resume: bool = False,
    overrides: Sequence[str] = (),
    **_legacy_options,
) -> Path:
    train_dir = Path(
        latest_root
        or Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name
    )
    current = resolve_latest_checkpoint(train_dir)
    current_iteration = completed_iteration_from_checkpoint_id(current.checkpoint_id)
    if iteration != current_iteration:
        raise ValueError(
            f"Requested evaluation iteration {iteration}, but latest is iteration {current_iteration}"
        )
    return evaluate_latest_checkpoint(
        config,
        config_path=config_path,
        latest_root=train_dir,
        resume=resume,
        overrides=overrides,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one iteration of the Python training loop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--latest-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-rollouts", action="store_true", help="Deprecated alias for --resume.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    common = {
        "config_path": args.config,
        "iteration": args.iteration,
        "latest_root": args.latest_root,
        "resume": args.resume or args.resume_rollouts,
        "overrides": args.overrides,
    }
    if args.eval_only:
        checkpoint = run_checkpoint_evaluation(config, **common)
    else:
        checkpoint = run_training_iteration(config, **common)
    print(checkpoint)


if __name__ == "__main__":
    main()
