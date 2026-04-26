from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_summarization_agent.checkpoints import checkpoint_id_from_path, mark_checkpoint_complete
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.launcher_utils import append_jsonl, ensure_dir
from self_summarization_agent.train_grpo import group_samples_by_query
from self_summarization_agent.trainer import FSDP2ContextParallelPolicyTrainer, TransformersPolicyTrainer
from self_summarization_agent.trajectory import extract_trainable_samples


def _load_rollout_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid rollout JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Rollout row {line_number} must be a JSON object")
            rows.append(row)
    return rows


def samples_from_rollout_rows(rows: list[dict[str, Any]], *, expected_checkpoint_id: str) -> list[Any]:
    samples = []
    for index, row in enumerate(rows, start=1):
        checkpoint_id = row.get("policy_checkpoint_id")
        if checkpoint_id != expected_checkpoint_id:
            raise ValueError(
                f"Rollout row {index} checkpoint mismatch: expected {expected_checkpoint_id!r}, got {checkpoint_id!r}"
            )
        turn_records = row.get("turn_records")
        turn_rewards = row.get("turn_rewards")
        if not isinstance(turn_records, list) or not isinstance(turn_rewards, dict):
            raise ValueError(f"Rollout row {index} is missing turn_records or turn_rewards")
        if row.get("trainable_sample_count") == 0:
            continue
        samples.extend(extract_trainable_samples(turn_records, turn_rewards))
    return samples


def run_train_step(
    config,
    *,
    checkpoint_path: str | Path,
    rollout_path: str | Path,
    output_checkpoint_path: str | Path,
    metrics_path: str | Path | None = None,
    trainer: Any | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    rows = _load_rollout_rows(rollout_path)
    samples = samples_from_rollout_rows(rows, expected_checkpoint_id=checkpoint_id)
    grouped_samples = group_samples_by_query(samples)

    if trainer is None:
        if config.training.backend == "fsdp2_context_parallel":
            trainer = FSDP2ContextParallelPolicyTrainer(config.model, config.training)
        elif config.training.backend == "transformers":
            trainer = TransformersPolicyTrainer(config.model, config.training)
        else:
            raise NotImplementedError(
                "The local environment cannot execute backend="
                f"{config.training.backend!r}. Supported backends are 'transformers' and 'fsdp2_context_parallel'."
            )

    metrics = trainer.step(grouped_samples)
    output_checkpoint = Path(output_checkpoint_path)
    ensure_dir(output_checkpoint)
    trainer.save_checkpoint(str(output_checkpoint))
    mark_checkpoint_complete(output_checkpoint)

    if metrics_path is not None:
        append_jsonl(
            metrics_path,
            {
                "policy_checkpoint_id": checkpoint_id,
                "next_checkpoint_id": checkpoint_id_from_path(output_checkpoint),
                "sample_count": metrics.sample_count,
                "mean_reward": metrics.mean_reward,
                "mean_advantage": metrics.mean_advantage,
                "loss": metrics.loss,
            },
        )
    return output_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one policy update from checkpoint-tagged rollout artifacts.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Input policy checkpoint path.")
    parser.add_argument("--rollouts", required=True, help="Rollout JSONL path.")
    parser.add_argument("--output-checkpoint", required=True, help="Output checkpoint directory.")
    parser.add_argument("--metrics", default=None, help="Optional metrics JSONL path.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    output_checkpoint = run_train_step(
        config,
        checkpoint_path=args.checkpoint,
        rollout_path=args.rollouts,
        output_checkpoint_path=args.output_checkpoint,
        metrics_path=args.metrics,
    )
    print(output_checkpoint)


if __name__ == "__main__":
    main()
