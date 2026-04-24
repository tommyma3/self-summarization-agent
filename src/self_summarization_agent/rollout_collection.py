from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
from typing import Any

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.checkpoints import checkpoint_id_from_path, resolve_latest_checkpoint
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.launcher_utils import (
    append_jsonl,
    build_runtime,
    ensure_dir,
    iter_batches,
    serialize_runtime_result,
)
from self_summarization_agent.rewards import apply_terminal_reward
from self_summarization_agent.trajectory import extract_trainable_samples


def apply_judged_rewards(result, example: QueryExample, judge: Any) -> dict[str, Any]:
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


def collect_rollouts(
    config,
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    examples: list[QueryExample] | None = None,
    backend: Any | None = None,
    generator: Any | None = None,
    judge: Any | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    examples = examples or load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    train_examples = examples if config.dataset.train_limit is None else examples[: config.dataset.train_limit]
    if not train_examples:
        raise ValueError("No training queries available for rollout collection")

    # Build retrieval before narrowing CUDA visibility for vLLM. The FAISS backend can
    # load its embedding model on the normal/default device, while vLLM is restricted
    # to config.rollout.gpu_ids below.
    backend = backend or build_backend(config.experiment.bc_plus_root, config.retrieval)
    if generator is None and config.rollout.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in config.rollout.gpu_ids)
    rollout_model_config = replace(
        config.model,
        backend=config.rollout.backend,
        max_new_tokens=config.rollout.max_new_tokens
        if config.rollout.max_new_tokens is not None
        else config.model.max_new_tokens,
        temperature=config.rollout.temperature
        if config.rollout.temperature is not None
        else config.model.temperature,
        top_p=config.rollout.top_p if config.rollout.top_p is not None else config.model.top_p,
        do_sample=config.rollout.do_sample
        if config.rollout.do_sample is not None
        else config.model.do_sample,
        tensor_parallel_size=config.rollout.tensor_parallel_size,
        attention_backend=config.rollout.attention_backend,
    )
    generator = generator or build_generator(rollout_model_config)
    judge = judge or RewardJudge(build_generator(config.model, judge_config=config.judge))
    runtime = build_runtime(generator, backend, config.runtime)

    rollout_path = Path(output_path)
    ensure_dir(rollout_path.parent)
    if rollout_path.exists():
        rollout_path.unlink()

    rollout_requests = [
        (example, rollout_index)
        for example in train_examples
        for rollout_index in range(config.training.group_size)
    ]
    for request_batch in iter_batches(rollout_requests, config.rollout.max_concurrent_episodes):
        results = runtime.run_many((example.query_id, example.query) for example, _ in request_batch)
        for (example, rollout_index), result in zip(request_batch, results):
            judge_payload = apply_judged_rewards(result, example, judge)
            trainable_sample_count = 0
            if judge_payload["outcome"] != "malformed_tool_call":
                trainable_sample_count = len(extract_trainable_samples(result.turn_records, result.turn_rewards))
            append_jsonl(
                rollout_path,
                {
                    "policy_checkpoint_id": checkpoint_id,
                    "policy_checkpoint_path": str(checkpoint),
                    "rollout_index": rollout_index,
                    "trainable_sample_count": trainable_sample_count,
                    **serialize_runtime_result(
                        result,
                        query_text=example.query,
                        judge={**judge_payload, "rollout_index": rollout_index},
                    ),
                },
            )
    return rollout_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one training pass of rollout artifacts with offline vLLM.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to latest under train dir.")
    parser.add_argument("--latest-root", default=None, help="Directory containing the latest checkpoint pointer.")
    parser.add_argument("--output", required=True, help="Rollout JSONL output path.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
    else:
        latest_root = args.latest_root or Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name
        checkpoint = resolve_latest_checkpoint(latest_root).path
    rollout_path = collect_rollouts(config, checkpoint_path=checkpoint, output_path=args.output)
    print(rollout_path)


if __name__ == "__main__":
    main()
