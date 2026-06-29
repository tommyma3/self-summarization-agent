from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml


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
            "Probe one query under both compaction conditions and write two text traces: "
            "one for no-compaction and one for compaction. Mirrors probe_vllm_collection.py."
        )
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "eval" / "compaction_comparison.yaml"),
        help="Path to the comparison YAML config.",
    )
    parser.add_argument("--sample-index", type=int, default=None, help="Use a fixed index after dataset slicing.")
    parser.add_argument("--query-id", default=None, help="Use a specific query_id after dataset slicing.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random query sampling. Defaults to base config seed.")
    parser.add_argument("--output-dir", default=None, help="Output directory for the two trace files. Defaults under comparison.output_root.")
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


def build_rollout_model_config(config: Any, args: argparse.Namespace) -> Any:
    return replace(
        config.model,
        backend="vllm",
        tensor_parallel_size=args.tensor_parallel_size,
        attention_backend=args.attention_backend,
        max_model_len=config.rollout.max_model_len
        if config.rollout.max_model_len is not None
        else config.model.max_model_len,
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


def build_judge(config: Any, rollout_model_config: Any, generator: Any) -> RewardJudge | None:
    if not getattr(config, "judge", None) or not config.judge.enabled:
        return None
    judge_model_path = config.judge.model_path or config.model.model_path
    rollout_model_path = rollout_model_config.model_path
    if judge_model_path == rollout_model_path:
        judge_generator = object.__new__(type(generator))
        for field_name in generator.__dataclass_fields__:
            setattr(judge_generator, field_name, getattr(generator, field_name))
        judge_generator.max_new_tokens = config.judge.max_new_tokens
        judge_generator.temperature = config.judge.temperature
        judge_generator.top_p = config.judge.top_p
        judge_generator.do_sample = config.judge.do_sample
        return RewardJudge(judge_generator)
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
        return RewardJudge(judge_generator)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Comparison config must be a mapping: {path}")
    return payload


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _default_output_dir(comparison: dict[str, Any], config: Any) -> Path:
    output_root_value = comparison.get("output_root")
    if output_root_value:
        return ensure_dir(Path(output_root_value) / "probe_traces")
    return ensure_dir(Path(config.experiment.output_root) / "artifacts" / "compaction_comparison_probe" / config.experiment.name)


def _max_optional_int(values: list[int | None]) -> int | None:
    concrete_values = [value for value in values if value is not None]
    return max(concrete_values) if concrete_values else None


def _load_condition_configs(
    *,
    base_config_path: Path,
    conditions: dict[str, Any],
    common_overrides: dict[str, Any],
) -> dict[str, Any]:
    condition_configs: dict[str, Any] = {}
    for condition_name, condition_config in conditions.items():
        if not isinstance(condition_config, dict):
            raise ValueError(f"Condition {condition_name!r} must be a mapping")
        condition_overrides = dict(common_overrides)
        condition_cfg_overrides = condition_config.get("overrides", {})
        if isinstance(condition_cfg_overrides, dict):
            condition_overrides.update(condition_cfg_overrides)
        condition_configs[str(condition_name)] = load_train_config(base_config_path, condition_overrides)
    return condition_configs


def _shared_probe_config(base_train_config: Any, condition_configs: dict[str, Any]) -> Any:
    max_model_len = _max_optional_int(
        [
            config.rollout.max_model_len
            if config.rollout.max_model_len is not None
            else config.model.max_model_len
            for config in condition_configs.values()
        ]
    )
    return replace(base_train_config, rollout=replace(base_train_config.rollout, max_model_len=max_model_len))


def main() -> None:
    args = parse_args()
    if args.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    comparison_yaml_path = Path(args.config).resolve()
    raw_config = _read_yaml(comparison_yaml_path)

    comparison = raw_config.get("comparison", {})
    if not isinstance(comparison, dict):
        raise ValueError("comparison section must be a mapping")
    conditions = raw_config.get("conditions", {})
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("conditions section must be a non-empty mapping")
    common_overrides = raw_config.get("common_overrides", {})
    if not isinstance(common_overrides, dict):
        raise ValueError("common_overrides section must be a mapping")

    base_config_value = raw_config.get("base_config")
    if base_config_value is None:
        raise ValueError("base_config is required in comparison YAML")
    base_config_path = _resolve_path(base_config_value, base_dir=comparison_yaml_path.parent)

    cli_overrides = merge_overrides(args)
    merged_common_overrides = {**common_overrides, **cli_overrides}

    base_train_config = load_train_config(base_config_path, merged_common_overrides)

    seed = base_train_config.experiment.seed if args.seed is None else args.seed
    examples = load_query_examples(
        base_train_config.experiment.bc_plus_root,
        base_train_config.dataset,
        require_answers=True,
        seed=base_train_config.experiment.seed,
    )
    example, sample_index = choose_example(
        examples,
        query_id=args.query_id,
        sample_index=args.sample_index,
        seed=seed,
    )

    condition_train_configs = _load_condition_configs(
        base_config_path=base_config_path,
        conditions=conditions,
        common_overrides=merged_common_overrides,
    )

    backend = build_backend(base_train_config.experiment.bc_plus_root, base_train_config.retrieval)
    if args.vllm_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.vllm_gpus
    rollout_model_config = build_rollout_model_config(_shared_probe_config(base_train_config, condition_train_configs), args)
    generator = build_generator(rollout_model_config)
    judge = build_judge(base_train_config, rollout_model_config, generator)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(comparison, base_train_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_query_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in example.query_id)

    for condition_name in conditions:
        print(f"\n--- Probing condition: {condition_name} ---")
        condition_train_config = condition_train_configs[str(condition_name)]
        runtime = build_runtime(generator, backend, condition_train_config.runtime)

        condition_dir = output_dir / str(condition_name)
        condition_dir.mkdir(parents=True, exist_ok=True)
        output_path = condition_dir / f"{safe_query_id}.txt"

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
