from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.launcher_utils import (
    append_jsonl,
    build_runtime,
    dataclass_to_jsonable,
    ensure_dir,
    seed_everything,
    serialize_runtime_result,
    utc_timestamp,
    write_json,
)
from self_summarization_agent.rewards import apply_terminal_reward
from self_summarization_agent.train_grpo import group_samples_by_query
from self_summarization_agent.trainer import TransformersPolicyTrainer
from self_summarization_agent.trajectory import extract_trainable_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the self-summarization agent with rollout-driven updates.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset.limit.")
    parser.add_argument("--output-root", default=None, help="Override experiment.output_root.")
    parser.add_argument("--model-path", default=None, help="Override model.model_path.")
    parser.add_argument("--retrieval-backend", default=None, help="Override retrieval.backend.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional dotted config overrides, e.g. training.steps=10",
    )
    return parser.parse_args()


def _merge_launcher_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_cli_overrides(args.overrides)
    if args.limit is not None:
        overrides["dataset.limit"] = args.limit
    if args.output_root is not None:
        overrides["experiment.output_root"] = args.output_root
    if args.model_path is not None:
        overrides["model.model_path"] = args.model_path
    if args.retrieval_backend is not None:
        overrides["retrieval.backend"] = args.retrieval_backend
    return overrides


def _apply_judged_rewards(result, example: QueryExample, judge: RewardJudge) -> dict[str, Any]:
    if result.status == "malformed_tool_call":
        return {"outcome": "malformed_tool_call", "judge_prompt": None, "judge_response": None, "parse_error": False}
    decision = judge.evaluate(example, result.status, result.final_answer or "")
    final_answer_turn_id = "final-answer" if result.final_answer is not None else None
    result.turn_rewards = apply_terminal_reward(
        outcome=decision.outcome,
        summary_turn_ids=result.summary_turns,
        final_answer_turn_id=final_answer_turn_id,
    )
    return {
        "outcome": decision.outcome,
        "judge_prompt": decision.judge_prompt,
        "judge_response": decision.judge_response,
        "parse_error": decision.parse_error,
    }


def train_experiment(
    config,
    *,
    examples: list[QueryExample] | None = None,
    backend: Any | None = None,
    rollout_generator: Any | None = None,
    judge: RewardJudge | None = None,
    trainer: TransformersPolicyTrainer | None = None,
) -> Path:
    seed_everything(config.experiment.seed)
    if config.model.backend.lower() == "vllm":
        raise NotImplementedError("Training with rollout backend='vllm' is reserved for a later swap. Use backend='transformers' for now.")

    examples = examples or load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    if not examples:
        raise ValueError("No training queries available after dataset slicing")
    backend = backend or build_backend(config.experiment.bc_plus_root, config.retrieval)
    rollout_generator = rollout_generator or build_generator(config.model)
    judge = judge or RewardJudge(build_generator(config.model, judge_config=config.judge))
    trainer = trainer or TransformersPolicyTrainer(config.model, config.training)

    runtime = build_runtime(rollout_generator, backend, config.runtime)
    train_dir = ensure_dir(Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name)
    rollouts_dir = ensure_dir(train_dir / "rollouts")
    checkpoints_dir = ensure_dir(train_dir / "checkpoints")
    metrics_path = train_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    rng = random.Random(config.experiment.seed)

    for step in range(1, config.training.steps + 1):
        batch_examples = [rng.choice(examples) for _ in range(config.training.batch_size)]
        step_rollout_path = rollouts_dir / f"step-{step:05d}.jsonl"
        if step_rollout_path.exists():
            step_rollout_path.unlink()
        all_samples = []
        malformed_count = 0
        judged_count = 0
        for example in batch_examples:
            for rollout_index in range(config.training.group_size):
                result = runtime.run(query_id=example.query_id, user_prompt=example.query)
                judge_payload = _apply_judged_rewards(result, example, judge)
                if judge_payload["outcome"] == "malformed_tool_call":
                    malformed_count += 1
                else:
                    judged_count += 1
                    all_samples.extend(extract_trainable_samples(result.turn_records, result.turn_rewards))
                append_jsonl(
                    step_rollout_path,
                    serialize_runtime_result(
                        result,
                        query_text=example.query,
                        judge={**judge_payload, "rollout_index": rollout_index},
                    ),
                )
        grouped_samples = group_samples_by_query(all_samples)
        metrics = trainer.step(grouped_samples)
        append_jsonl(
            metrics_path,
            {
                "step": step,
                "timestamp_utc": utc_timestamp(),
                "sample_count": metrics.sample_count,
                "mean_reward": metrics.mean_reward,
                "mean_advantage": metrics.mean_advantage,
                "loss": metrics.loss,
                "malformed_rollouts": malformed_count,
                "judged_rollouts": judged_count,
            },
        )
        if config.training.checkpoint_interval > 0 and step % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint(str(checkpoints_dir / f"step-{step:05d}"))

    write_json(
        train_dir / "manifest.json",
        {
            "timestamp_utc": utc_timestamp(),
            "config": dataclass_to_jsonable(config),
        },
    )
    return train_dir


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, _merge_launcher_overrides(args))
    train_dir = train_experiment(config)
    print(train_dir)


if __name__ == "__main__":
    main()
