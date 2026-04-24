from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TASK_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    seed: int
    output_root: str
    bc_plus_root: str


@dataclass(slots=True)
class DatasetConfig:
    decrypted_path: str | None = None
    queries_tsv_path: str | None = None
    offset: int = 0
    limit: int | None = None
    shuffle: bool = False
    train_limit: int | None = None
    eval_limit: int = 0


@dataclass(slots=True)
class RetrievalConfig:
    backend: str = "faiss"
    top_k: int = 5
    snippet_max_tokens: int | None = 512
    snippet_tokenizer_path: str | None = None
    index_path: str = ""
    model_name: str | None = None
    normalize: bool = False
    pooling: str = "eos"
    torch_dtype: str = "float16"
    dataset_name: str = "Tevatron/browsecomp-plus-corpus"
    task_prefix: str = DEFAULT_TASK_PREFIX
    max_length: int = 8192


@dataclass(slots=True)
class ModelConfig:
    backend: str = "transformers"
    model_path: str = ""
    judge_model_path: str | None = None
    dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    tensor_parallel_size: int = 1
    attention_backend: str | None = None
    trust_remote_code: bool = False
    enable_thinking: bool = True


@dataclass(slots=True)
class RolloutConfig:
    backend: str = "transformers"
    gpu_ids: list[int] = field(default_factory=lambda: [2, 3])
    tensor_parallel_size: int = 2
    attention_backend: str | None = "TORCH_SDPA"
    max_concurrent_episodes: int = 32
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    do_sample: bool | None = None


@dataclass(slots=True)
class RuntimeConfig:
    context_threshold_tokens: int = 24000
    max_context_tokens: int = 32768
    tool_budget: int = 16


@dataclass(slots=True)
class JudgeConfig:
    enabled: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass(slots=True)
class TrainingConfig:
    backend: str = "transformers"
    gpu_ids: list[int] = field(default_factory=list)
    fsdp_version: int | None = None
    context_parallel_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    activation_checkpointing: bool = False
    epochs: int | None = None
    steps: int = 1
    batch_size: int = 1
    group_size: int = 2
    gradient_accumulation_microbatch_size: int = 1
    learning_rate: float = 1e-6
    checkpoint_interval: int = 100
    eval_interval: int = 0
    max_grad_norm: float = 1.0


@dataclass(slots=True)
class RunConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    runtime: RuntimeConfig


@dataclass(slots=True)
class TrainConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    rollout: RolloutConfig
    runtime: RuntimeConfig
    judge: JudgeConfig
    training: TrainingConfig


def _parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def apply_overrides(raw_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = dict(raw_config)
    for dotted_key, value in overrides.items():
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            next_value = cursor.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[part] = next_value
            cursor = next_value
        cursor[parts[-1]] = value
    return updated


def parse_cli_overrides(override_items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        overrides[key] = _parse_override_value(raw_value)
    return overrides


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level config must be a mapping, got {type(loaded).__name__}")
    return loaded


def _require_section(raw: dict[str, Any], section: str) -> dict[str, Any]:
    value = raw.get(section, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def load_run_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    raw = _load_yaml(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return RunConfig(
        experiment=ExperimentConfig(**_require_section(raw, "experiment")),
        dataset=DatasetConfig(**_require_section(raw, "dataset")),
        retrieval=RetrievalConfig(**_require_section(raw, "retrieval")),
        model=ModelConfig(**_require_section(raw, "model")),
        runtime=RuntimeConfig(**_require_section(raw, "runtime")),
    )


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    raw = _load_yaml(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return TrainConfig(
        experiment=ExperimentConfig(**_require_section(raw, "experiment")),
        dataset=DatasetConfig(**_require_section(raw, "dataset")),
        retrieval=RetrievalConfig(**_require_section(raw, "retrieval")),
        model=ModelConfig(**_require_section(raw, "model")),
        rollout=RolloutConfig(**_require_section(raw, "rollout")),
        runtime=RuntimeConfig(**_require_section(raw, "runtime")),
        judge=JudgeConfig(**_require_section(raw, "judge")),
        training=TrainingConfig(**_require_section(raw, "training")),
    )


def config_to_dict(config: RunConfig | TrainConfig) -> dict[str, Any]:
    return asdict(config)
