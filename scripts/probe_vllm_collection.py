from __future__ import annotations

import argparse
from collections.abc import Mapping
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
from simulate_collection import format_exact_model_input_sequences, trace_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe one training rollout with the configured offline rollout backend. "
            "Runs one query until finish, malformed output, or tool budget exhaustion with forced answer, "
            "prints every exact model-input token sequence, and writes the trace."
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
        "--rollout-backend",
        choices=("sglang", "sglang_offline", "vllm", "vllm_offline"),
        default=None,
        help="Offline rollout backend. Defaults to rollout.backend from the train config.",
    )
    parser.add_argument(
        "--rollout-gpus",
        "--vllm-gpus",
        dest="rollout_gpus",
        default=None,
        help=(
            "Comma-separated physical GPU ids made visible to the rollout engine. "
            "Defaults to rollout.gpu_ids. --vllm-gpus remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Rollout-engine tensor parallel size. Defaults to rollout.tensor_parallel_size.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help=(
            "Optional rollout attention backend override, e.g. flashinfer for SGLang or "
            "FLASH_ATTN for vLLM. Defaults to rollout.attention_backend."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override rollout max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override rollout temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override rollout top_p.")
    parser.add_argument("--do-sample", action="store_true", default=None, help="Force sampled rollout generation.")
    parser.add_argument("--no-sample", action="store_false", dest="do_sample", help="Force deterministic rollout generation.")
    parser.add_argument("--include-formatted-prompt", action="store_true", help="Also write tokenizer chat-template prompts.")
    parser.add_argument("--training-max-seq-len", type=int, default=None, help="Override training.max_sequence_length for the fit check.")
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


def write_decoded_tokens(
    trajectory_records: list[dict[str, Any]],
    tokenizer: Any,
    output_dir: Path,
) -> None:
    """Decode per-generation token IDs to text and write them to files."""
    output_dir = ensure_dir(output_dir)
    for trajectory_index, record in enumerate(trajectory_records, start=1):
        trajectory_id = record.get("turn_id", f"trajectory-{trajectory_index}")
        collection_tokens = record.get("collection_tokens")
        if not isinstance(collection_tokens, Mapping):
            continue
        generations = collection_tokens.get("generations")
        if not isinstance(generations, list):
            continue

        decoded_path = output_dir / f"{trajectory_id}_decoded_tokens.txt"
        with open(decoded_path, "w", encoding="utf-8") as handle:
            handle.write(f"Decoded token sequences for {trajectory_id}\n")
            handle.write(f"Query ID: {record.get('query_id', '?')}\n")
            handle.write("=" * 80 + "\n\n")

            for generation in generations:
                if not isinstance(generation, Mapping):
                    continue
                gen_index = generation.get("index", "?")
                finish_reason = generation.get("finish_reason")
                prompt_ids = generation.get("prompt_token_ids")
                completion_ids = generation.get("completion_token_ids")
                full_ids = generation.get("full_token_ids")

                handle.write(f"--- Generation {gen_index}")
                if finish_reason:
                    handle.write(f" (finish_reason={finish_reason!r})")
                handle.write(" ---\n\n")

                if isinstance(prompt_ids, list) and prompt_ids:
                    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                    handle.write(f"[PROMPT] ({len(prompt_ids)} tokens):\n")
                    handle.write(prompt_text)
                    handle.write("\n\n")

                if isinstance(completion_ids, list) and completion_ids:
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
                    handle.write(f"[COMPLETION] ({len(completion_ids)} tokens):\n")
                    handle.write(completion_text)
                    handle.write("\n\n")

                if isinstance(full_ids, list) and full_ids:
                    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
                    handle.write(f"[FULL] ({len(full_ids)} tokens, prompt+completion):\n")
                    handle.write(full_text)
                    handle.write("\n\n")

                handle.write("\n")

            # Append decoded model input sequences for this trajectory.
            handle.write("=" * 80 + "\n")
            handle.write("Model Input Sequences (decoded prompt token IDs per generation)\n")
            handle.write("=" * 80 + "\n\n")
            handle.write(
                format_exact_model_input_sequences([record], tokenizer)
            )
            handle.write("\n")

        print(f"Decoded tokens written to: {decoded_path}")


def build_rollout_model_config(config: Any, args: argparse.Namespace):
    rollout_backend = getattr(args, "rollout_backend", None) or config.rollout.backend
    tensor_parallel_size = (
        args.tensor_parallel_size
        if args.tensor_parallel_size is not None
        else config.rollout.tensor_parallel_size
    )
    attention_backend = (
        args.attention_backend
        if args.attention_backend is not None
        else config.rollout.attention_backend
    )
    return replace(
        config.model,
        backend=rollout_backend,
        language_model_only=True,
        tensor_parallel_size=tensor_parallel_size,
        attention_backend=attention_backend,
        max_model_len=config.rollout.max_model_len
        if config.rollout.max_model_len is not None
        else config.model.max_model_len,
        enable_prefix_caching=config.rollout.enable_prefix_caching,
        api_base_url=config.rollout.api_base_url,
        api_model=config.rollout.api_model,
        api_key_env=config.rollout.api_key_env,
        api_timeout_seconds=config.rollout.api_timeout_seconds,
        api_max_retries=config.rollout.api_max_retries,
        api_max_concurrency=config.rollout.max_concurrent_episodes,
        api_extra_body=dict(config.rollout.api_extra_body),
        require_exact_token_ids=config.rollout.require_exact_token_ids,
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


def normalize_backend_name(backend: str) -> str:
    return backend.lower().replace("-", "_")


def rollout_gpu_visibility(config: Any, args: argparse.Namespace) -> str:
    configured = getattr(args, "rollout_gpus", None)
    if configured is not None:
        gpu_ids = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        gpu_ids = [str(gpu_id) for gpu_id in config.rollout.gpu_ids]
    if not gpu_ids:
        raise ValueError("At least one rollout GPU must be configured")
    if not all(gpu_id.isdigit() for gpu_id in gpu_ids):
        raise ValueError(f"Invalid rollout GPU list: {configured!r}")
    return ",".join(gpu_ids)


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, merge_overrides(args))
    rollout_model_config = build_rollout_model_config(config, args)
    rollout_backend = normalize_backend_name(rollout_model_config.backend)
    if rollout_backend in {"vllm", "vllm_offline"} and rollout_model_config.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = rollout_model_config.attention_backend
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

    # Pin retrieval (FAISS embedding model) to GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    backend = build_backend(config.experiment.bc_plus_root, config.retrieval)
    # Hardwire the offline rollout engine to physical GPU 2 (TP=1).
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    rollout_model_config = replace(rollout_model_config, tensor_parallel_size=1)
    generator = build_generator(rollout_model_config)
    judge = None
    if getattr(config, "judge", None) and config.judge.enabled:
        judge_model_path = config.judge.model_path or config.model.model_path
        rollout_model_path = rollout_model_config.model_path
        judge_backend = normalize_backend_name(config.judge.backend or config.model.backend)
        if judge_model_path == rollout_model_path and judge_backend == rollout_backend:
            # Reuse the launched generator's underlying model/engine.
            # dataclasses.replace calls __post_init__ for slotted dataclasses,
            # which would spawn a second inference engine and OOM.  We shallow-copy
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
            # Hardwire judge to GPUs 1,3 (retrieval on GPU 0, rollout on GPU 2).
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
            judge_model_config = replace(
                config.model,
                backend=config.judge.backend or config.model.backend,
                model_path=judge_model_path,
                language_model_only=True,
                tensor_parallel_size=2,
                attention_backend=config.judge.attention_backend
                if config.judge.attention_backend is not None
                else config.model.attention_backend,
                max_model_len=config.judge.max_model_len
                if config.judge.max_model_len is not None
                else config.model.max_model_len,
            )
            # Force TP=2 for the judge so the 35B model fits across 2 GPUs.
            judge_generator = build_generator(
                judge_model_config,
                judge_config=replace(config.judge, tensor_parallel_size=2),
            )
            judge = RewardJudge(judge_generator)
    runtime = build_runtime(generator, backend, config.runtime)
    output_path = Path(args.output) if args.output else default_output_path(config, example.query_id)
    result = trace_collection(
        runtime=runtime,
        generator=generator,
        example=example,
        sample_index=sample_index,
        output_path=output_path,
        include_formatted_prompt=args.include_formatted_prompt,
        judge=judge,
        training_config=getattr(config, "training", None),
        training_max_sequence_length=args.training_max_seq_len,
    )
    tokenizer = generator.tokenizer if hasattr(generator, "tokenizer") else None
    print(format_exact_model_input_sequences(result.trajectory_records, tokenizer))
    print(output_path)

    # Decode per-generation token IDs back to strings and write to text files.
    if hasattr(generator, "tokenizer"):
        decoded_dir = output_path.parent
        write_decoded_tokens(result.trajectory_records, generator.tokenizer, decoded_dir)


if __name__ == "__main__":
    main()
