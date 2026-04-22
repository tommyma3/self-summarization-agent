from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(slots=True)
class AccuracyMetrics:
    correct: int
    total: int
    malformed: int
    parse_errors: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


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


def split_train_eval_examples(
    examples: list[QueryExample],
    *,
    train_limit: int | None,
    eval_limit: int,
) -> tuple[list[QueryExample], list[QueryExample]]:
    if train_limit is None:
        train_examples = list(examples)
        eval_examples: list[QueryExample] = []
    else:
        train_examples = list(examples[:train_limit])
        eval_examples = list(examples[train_limit : train_limit + eval_limit])
    return train_examples, eval_examples


def _training_epoch_count(config) -> int:
    return config.training.epochs if config.training.epochs is not None else config.training.steps


def _evaluate_accuracy(
    runtime,
    examples: list[QueryExample],
    judge: RewardJudge,
) -> AccuracyMetrics:
    correct = 0
    malformed = 0
    parse_errors = 0
    for example in examples:
        result = runtime.run(query_id=example.query_id, user_prompt=example.query)
        judge_payload = _apply_judged_rewards(result, example, judge)
        if judge_payload["outcome"] == "correct_answer":
            correct += 1
        if judge_payload["outcome"] == "malformed_tool_call":
            malformed += 1
        if judge_payload["parse_error"]:
            parse_errors += 1
    return AccuracyMetrics(
        correct=correct,
        total=len(examples),
        malformed=malformed,
        parse_errors=parse_errors,
    )


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
    if config.training.backend.lower() != "transformers":
        raise NotImplementedError(
            "train_launcher supports only training.backend='transformers'. "
            "Use self_summarization_agent.iteration_launcher for process-isolated vLLM rollout "
            "and long-context distributed training backends."
        )

    examples = examples or load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    if not examples:
        raise ValueError("No training queries available after dataset slicing")
    train_examples, eval_examples = split_train_eval_examples(
        examples,
        train_limit=config.dataset.train_limit,
        eval_limit=config.dataset.eval_limit,
    )
    if not train_examples:
        raise ValueError("No training queries available after train/eval split")
    backend = backend or build_backend(config.experiment.bc_plus_root, config.retrieval)
    judge = judge or RewardJudge(build_generator(config.model, judge_config=config.judge))
    trainer = trainer or TransformersPolicyTrainer(config.model, config.training)
    rollout_generator = rollout_generator or trainer

    runtime = build_runtime(rollout_generator, backend, config.runtime)
    train_dir = ensure_dir(Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name)
    rollouts_dir = ensure_dir(train_dir / "rollouts")
    checkpoints_dir = ensure_dir(train_dir / "checkpoints")
    metrics_path = train_dir / "metrics.jsonl"
    accuracy_path = train_dir / "accuracy_history.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    if accuracy_path.exists():
        accuracy_path.unlink()

    epoch_count = _training_epoch_count(config)
    for epoch in range(1, epoch_count + 1):
        step_rollout_path = rollouts_dir / f"step-{epoch:05d}.jsonl"
        if step_rollout_path.exists():
            step_rollout_path.unlink()
        all_samples = []
        malformed_count = 0
        judged_count = 0
        for example in train_examples:
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
        train_accuracy = _evaluate_accuracy(runtime, train_examples, judge)
        eval_accuracy = _evaluate_accuracy(runtime, eval_examples, judge)
        accuracy_record = {
            "epoch": epoch,
            "timestamp_utc": utc_timestamp(),
            "train_accuracy": train_accuracy.accuracy,
            "train_correct": train_accuracy.correct,
            "train_total": train_accuracy.total,
            "train_malformed": train_accuracy.malformed,
            "train_parse_errors": train_accuracy.parse_errors,
            "eval_accuracy": eval_accuracy.accuracy,
            "eval_correct": eval_accuracy.correct,
            "eval_total": eval_accuracy.total,
            "eval_malformed": eval_accuracy.malformed,
            "eval_parse_errors": eval_accuracy.parse_errors,
        }
        append_jsonl(accuracy_path, accuracy_record)
        append_jsonl(
            metrics_path,
            {
                "step": epoch,
                "epoch": epoch,
                "timestamp_utc": utc_timestamp(),
                "sample_count": metrics.sample_count,
                "mean_reward": metrics.mean_reward,
                "mean_advantage": metrics.mean_advantage,
                "loss": metrics.loss,
                "malformed_rollouts": malformed_count,
                "judged_rollouts": judged_count,
                "train_accuracy": train_accuracy.accuracy,
                "eval_accuracy": eval_accuracy.accuracy,
            },
        )
        if config.training.checkpoint_interval > 0 and epoch % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint(str(checkpoints_dir / f"step-{epoch:05d}"))

    write_json(
        train_dir / "manifest.json",
        {
            "timestamp_utc": utc_timestamp(),
            "config": dataclass_to_jsonable(config),
            "train_query_count": len(train_examples),
            "eval_query_count": len(eval_examples),
            "accuracy_history": str(accuracy_path),
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
