from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.launcher_utils import build_runtime, ensure_dir, serialize_runtime_result, utc_timestamp, write_json
from self_summarization_agent.rollout_collection import apply_judged_rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe one training rollout with offline vLLM on 4 GPUs. "
            "Runs one query until finish, malformed output, or tool budget exhaustion."
        )
    )
    parser.add_argument("--config", default="configs/train/default.yaml", help="Path to the train YAML config.")
    parser.add_argument("--sample-index", type=int, default=None, help="Use a fixed index after dataset slicing.")
    parser.add_argument("--query-id", default=None, help="Use a specific query_id after dataset slicing.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random query sampling. Defaults to config seed.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults under experiment.output_root.")
    parser.add_argument("--model-path", default=None, help="Override model.model_path.")
    parser.add_argument("--retrieval-backend", default=None, help="Override retrieval.backend.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--attention-backend",
        default="TORCH_SDPA",
        help="vLLM attention backend. Defaults to TORCH_SDPA to avoid flash-attn.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override rollout max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override rollout temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override rollout top_p.")
    parser.add_argument("--do-sample", action="store_true", default=None, help="Force sampled rollout generation.")
    parser.add_argument("--no-sample", action="store_false", dest="do_sample", help="Force deterministic rollout generation.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional dotted config overrides, e.g. runtime.tool_budget=8",
    )
    return parser.parse_args()


def merge_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_cli_overrides(args.overrides)
    if args.model_path is not None:
        overrides["model.model_path"] = args.model_path
    if args.retrieval_backend is not None:
        overrides["retrieval.backend"] = args.retrieval_backend
    return overrides


def choose_example(
    examples: list[QueryExample],
    *,
    query_id: str | None,
    sample_index: int | None,
    seed: int,
) -> tuple[QueryExample, int]:
    if not examples:
        raise ValueError("No examples available after dataset slicing")
    if query_id is not None:
        for index, example in enumerate(examples):
            if example.query_id == query_id:
                return example, index
        raise ValueError(f"query_id not found after dataset slicing: {query_id}")
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(examples):
            raise IndexError(f"--sample-index must be in [0, {len(examples) - 1}], got {sample_index}")
        return examples[sample_index], sample_index
    selected_index = random.Random(seed).randrange(len(examples))
    return examples[selected_index], selected_index


def default_output_path(config: Any, query_id: str) -> Path:
    output_dir = ensure_dir(Path(config.experiment.output_root) / "artifacts" / "vllm_collection_probe" / config.experiment.name)
    safe_query_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in query_id)
    return output_dir / f"{safe_query_id}.json"


def build_rollout_model_config(config: Any, args: argparse.Namespace):
    return replace(
        config.model,
        backend="vllm",
        tensor_parallel_size=args.tensor_parallel_size,
        attention_backend=args.attention_backend,
        max_new_tokens=args.max_new_tokens
        if args.max_new_tokens is not None
        else (config.rollout.max_new_tokens if config.rollout.max_new_tokens is not None else config.model.max_new_tokens),
        temperature=args.temperature
        if args.temperature is not None
        else (config.rollout.temperature if config.rollout.temperature is not None else config.model.temperature),
        top_p=args.top_p
        if args.top_p is not None
        else (config.rollout.top_p if config.rollout.top_p is not None else config.model.top_p),
        do_sample=args.do_sample
        if args.do_sample is not None
        else (config.rollout.do_sample if config.rollout.do_sample is not None else config.model.do_sample),
    )


class DisabledJudge:
    def evaluate(self, example: QueryExample, status: str, response: str):
        del example, response

        class Decision:
            outcome = "correct_answer" if status == "completed" else "budget_exhausted"
            judge_prompt = None
            judge_response = None
            parse_error = False

        return Decision()


def main() -> None:
    args = parse_args()
    if args.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    config = load_train_config(args.config, merge_overrides(args))
    seed = config.experiment.seed if args.seed is None else args.seed
    examples = load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    example, sample_index = choose_example(
        examples,
        query_id=args.query_id,
        sample_index=args.sample_index,
        seed=seed,
    )

    rollout_model_config = build_rollout_model_config(config, args)
    generator = build_generator(rollout_model_config)
    backend = build_backend(config.experiment.bc_plus_root, config.retrieval)
    runtime = build_runtime(generator, backend, config.runtime)

    result = runtime.run(query_id=example.query_id, user_prompt=example.query)
    judge_payload = apply_judged_rewards(result, example, DisabledJudge())
    output_path = Path(args.output) if args.output else default_output_path(config, example.query_id)
    payload = {
        "timestamp_utc": utc_timestamp(),
        "sample_index": sample_index,
        "model_path": rollout_model_config.model_path,
        "tensor_parallel_size": rollout_model_config.tensor_parallel_size,
        "query": example.query,
        "answer": example.answer,
        "judge": judge_payload,
        **serialize_runtime_result(result, query_text=example.query, judge=judge_payload),
    }
    ensure_dir(output_path.parent)
    write_json(output_path, payload)

    print(json.dumps(
        {
            "output_path": str(output_path),
            "query_id": result.query_id,
            "status": result.status,
            "final_answer": result.final_answer,
            "tool_call_counts": result.tool_call_counts,
            "retrieved_docids": result.retrieved_docids,
            "summary_turns": result.summary_turns,
            "tensor_parallel_size": rollout_model_config.tensor_parallel_size,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
