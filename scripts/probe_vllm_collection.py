from __future__ import annotations

import argparse
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
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.launcher_utils import build_runtime, ensure_dir
from simulate_collection import trace_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe one training rollout with offline vLLM on 4 GPUs. "
            "Runs one query until finish, malformed output, or tool budget exhaustion, "
            "and writes every trainable prompt/completion sequence to the trace."
        )
    )
    parser.add_argument("--config", default="configs/train/default.yaml", help="Path to the train YAML config.")
    parser.add_argument("--sample-index", type=int, default=None, help="Use a fixed index after dataset slicing.")
    parser.add_argument("--query-id", default=None, help="Use a specific query_id after dataset slicing.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random query sampling. Defaults to config seed.")
    parser.add_argument("--output", default=None, help="Output trace text path. Defaults under experiment.output_root.")
    parser.add_argument("--model-path", default=None, help="Override model.model_path.")
    parser.add_argument("--retrieval-backend", default=None, help="Override retrieval.backend.")
    parser.add_argument(
        "--vllm-gpus",
        default="2,3",
        help="Comma-separated physical GPU ids made visible to vLLM. Defaults to 2,3.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="vLLM tensor parallel size.")
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
    parser.add_argument("--include-formatted-prompt", action="store_true", help="Also write tokenizer chat-template prompts.")
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
    return output_dir / f"{safe_query_id}.txt"


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

    backend = build_backend(config.experiment.bc_plus_root, config.retrieval)
    if args.vllm_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.vllm_gpus
    rollout_model_config = build_rollout_model_config(config, args)
    generator = build_generator(rollout_model_config)
    judge = None
    if getattr(config, "judge", None) and config.judge.enabled:
        judge_model_path = config.judge.model_path or config.model.model_path
        rollout_model_path = rollout_model_config.model_path
        if judge_model_path == rollout_model_path:
            # Reuse the launched generator's underlying model/engine.
            # dataclasses.replace calls __post_init__ for slotted dataclasses,
            # which would spawn a second vLLM instance and OOM.  We shallow-copy
            # the fields manually instead.
            judge_generator = object.__new__(type(generator))
            for field_name in generator.__dataclass_fields__:
                setattr(judge_generator, field_name, getattr(generator, field_name))
            judge_generator.max_new_tokens = config.judge.max_new_tokens
            judge_generator.temperature = config.judge.temperature
            judge_generator.top_p = config.judge.top_p
            judge_generator.do_sample = config.judge.do_sample
            judge = RewardJudge(judge_generator)
        else:
            if config.judge.gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in config.judge.gpu_ids)
            judge_model_config = replace(
                config.model,
                backend=config.judge.backend or config.model.backend,
                model_path=judge_model_path,
                tensor_parallel_size=config.judge.tensor_parallel_size
                if config.judge.tensor_parallel_size is not None
                else config.model.tensor_parallel_size,
                attention_backend=config.judge.attention_backend
                if config.judge.attention_backend is not None
                else config.model.attention_backend,
                max_model_len=config.judge.max_model_len
                if config.judge.max_model_len is not None
                else config.model.max_model_len,
            )
            judge_generator = build_generator(judge_model_config, judge_config=config.judge)
            judge = RewardJudge(judge_generator)
    runtime = build_runtime(generator, backend, config.runtime)
    output_path = Path(args.output) if args.output else default_output_path(config, example.query_id)
    trace_collection(
        runtime=runtime,
        generator=generator,
        example=example,
        sample_index=sample_index,
        output_path=output_path,
        include_formatted_prompt=args.include_formatted_prompt,
        judge=judge,
    )
    print(output_path)


if __name__ == "__main__":
    main()
