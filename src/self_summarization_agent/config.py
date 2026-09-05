from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
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
    document_max_tokens: int | None = 8192
    snippet_tokenizer_path: str | None = None
    persistent_worker: bool = False
    worker_startup_timeout_seconds: int = 600
    search_max_batch_size: int | None = None
    gpu_ids: list[int] = field(default_factory=list)
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
    max_model_len: int | None = None
    trust_remote_code: bool = False
    enable_thinking: bool = True
    chat_template_path: str | None = None
    language_model_only: bool = False
    enable_prefix_caching: bool = False
    api_base_url: str | None = None
    api_model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_timeout_seconds: float = 600.0
    api_max_retries: int = 2
    api_max_concurrency: int = 32
    api_extra_body: dict[str, Any] = field(default_factory=dict)
    require_exact_token_ids: bool = True
    # Optional vLLM memory fraction; None keeps the engine default.
    gpu_memory_utilization: float | None = None


@dataclass(slots=True)
class RolloutConfig:
    backend: str = "transformers"
    gpu_ids: list[int] = field(default_factory=lambda: [2, 3])
    tensor_parallel_size: int = 2
    attention_backend: str | None = None
    max_model_len: int | None = None
    enable_prefix_caching: bool = True
    max_concurrent_episodes: int = 32
    # Compatibility switch for standalone rollout_collection. The primary
    # merged collection path is always sequential and ignores this value.
    overlap_judge: bool = True
    overlap_queue_max_batches: int = 8
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    api_base_url: str | None = None
    api_model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_timeout_seconds: float = 600.0
    api_max_retries: int = 2
    api_extra_body: dict[str, Any] = field(default_factory=dict)
    require_exact_token_ids: bool = True
    # Per-split timeout for the collection subprocess inside merged_collect.
    # None = no limit (relies on the outer phase_timeout_seconds).
    per_split_collection_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_concurrent_episodes < 1:
            raise ValueError("rollout.max_concurrent_episodes must be at least 1")
        if self.overlap_queue_max_batches < 1:
            raise ValueError("rollout.overlap_queue_max_batches must be at least 1")


@dataclass(slots=True)
class RuntimeConfig:
    context_threshold_tokens: int = 24000
    max_context_tokens: int = 32768
    max_summary_tokens: int = 2048
    tool_budget: int = 16
    generated_token_budget: int | None = None
    phase_timeout_seconds: int | None = None
    # Optional training-only wall-clock limit. When unset, the general phase
    # timeout remains authoritative for backward compatibility.
    train_update_timeout_seconds: int | None = None
    tool_execution_timeout_seconds: float = 600


@dataclass(slots=True)
class CollectionConfig:
    train_task_count: int | None = None
    eval_task_count: int | None = None


@dataclass(slots=True)
class EvaluationConfig:
    samples_per_task: int = 1
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    extra_sampling_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JudgeConfig:
    enabled: bool = True
    backend: str | None = None
    model_path: str | None = None
    gpu_ids: list[int] = field(default_factory=list)
    tensor_parallel_size: int | None = None
    attention_backend: str | None = None
    max_model_len: int | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    batch_size: int = 32
    batch_wait_ms: int = 25
    batch_timeout_seconds: float = 600

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("judge.batch_size must be at least 1")
        if self.batch_wait_ms < 0:
            raise ValueError("judge.batch_wait_ms cannot be negative")
        if self.batch_timeout_seconds < 0:
            raise ValueError("judge.batch_timeout_seconds cannot be negative")


@dataclass(slots=True)
class VerlFSDPConfig:
    strategy: str = "fsdp"
    ppo_micro_batch_size_per_gpu: int = 1
    ppo_max_token_len_per_gpu: int | None = None
    log_prob_micro_batch_size_per_gpu: int = 1
    log_prob_max_token_len_per_gpu: int | None = None
    use_dynamic_bsz: bool = False
    use_remove_padding: bool = True
    use_torch_compile: bool = True
    ulysses_sequence_parallel_size: int = 1
    forward_prefetch: bool = False
    param_offload: bool = False
    optimizer_offload: bool = False
    fsdp_size: int = -1
    save_hf_model: bool = True


@dataclass(slots=True)
class VerlRayConfig:
    address: str | None = None
    namespace: str = "self-summarization-agent"
    num_cpus: int | None = None
    num_gpus_per_worker: float | None = None
    runtime_env: dict[str, Any] = field(default_factory=dict)
    worker_backend: str = "transformers"
    fsdp: VerlFSDPConfig = field(default_factory=VerlFSDPConfig)
    ignore_reinit_error: bool = True
    log_to_driver: bool = True
    shutdown_ray: bool = True


@dataclass(slots=True)
class CompactionValueConfig:
    enabled: bool = False
    loss_coefficient: float = 0.5
    zero_initialize_head: bool = True
    state_anchor: str = "first_generation_prompt_end"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("training.value.enabled must be a boolean")
        if not isinstance(self.zero_initialize_head, bool):
            raise ValueError("training.value.zero_initialize_head must be a boolean")
        if (
            not isinstance(self.loss_coefficient, (int, float))
            or isinstance(self.loss_coefficient, bool)
            or not math.isfinite(float(self.loss_coefficient))
            or self.loss_coefficient < 0
        ):
            raise ValueError("training.value.loss_coefficient must be finite and non-negative")
        if self.state_anchor != "first_generation_prompt_end":
            raise ValueError(
                "training.value.state_anchor currently supports only "
                "'first_generation_prompt_end'"
            )


@dataclass(slots=True)
class TrainingConfig:
    backend: str = "transformers"
    gpu_ids: list[int] = field(default_factory=list)
    fsdp_version: int | None = None
    context_parallel_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    activation_checkpointing: bool = False
    max_sequence_length: int | None = None
    epochs: int | None = None
    steps: int = 1
    batch_size: int = 1
    group_size: int = 2
    rollout_query_count: int | None = None
    update_epochs: int = 1
    minibatch_size: int | None = None
    clip_range: float = 0.2
    target_kl: float | None = None
    gradient_accumulation_microbatch_size: int = 1
    learning_rate: float = 1e-6
    checkpoint_interval: int = 100
    eval_interval: int = 0
    max_grad_norm: float = 1.0
    train_compaction_tokens: bool = True
    advantage_estimator: str = "group_relative"
    value: CompactionValueConfig = field(default_factory=CompactionValueConfig)
    verl: VerlRayConfig = field(default_factory=VerlRayConfig)

    def __post_init__(self) -> None:
        supported = {"group_relative", "compaction_mc_value"}
        if self.advantage_estimator not in supported:
            raise ValueError(
                f"training.advantage_estimator must be one of {sorted(supported)}, "
                f"got {self.advantage_estimator!r}"
            )
        if self.value.enabled != (self.advantage_estimator == "compaction_mc_value"):
            raise ValueError(
                "training.value.enabled must be true exactly when "
                "training.advantage_estimator='compaction_mc_value'"
            )


@dataclass(slots=True)
class RunConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    runtime: RuntimeConfig
    rollout: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass(slots=True)
class TrainConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    runtime: RuntimeConfig
    judge: JudgeConfig
    training: TrainingConfig
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


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
        rollout=RolloutConfig(**_require_section(raw, "rollout")),
    )


def _derive_rollout_config(raw: dict[str, Any], training: TrainingConfig) -> RolloutConfig:
    rollout_section = _require_section(raw, "rollout")
    if rollout_section:
        return RolloutConfig(**rollout_section)
    if training.backend == "fsdp2_context_parallel":
        return RolloutConfig(
            backend="vllm_offline",
            gpu_ids=list(training.gpu_ids),
            tensor_parallel_size=training.context_parallel_size,
            max_model_len=65536,
        )
    return RolloutConfig()


def _load_training_config(raw: dict[str, Any]) -> TrainingConfig:
    training_section = dict(_require_section(raw, "training"))
    value_section = training_section.pop("value", {})
    if not isinstance(value_section, dict):
        raise ValueError("Config section 'training.value' must be a mapping")
    verl_section = training_section.pop("verl", {})
    if not isinstance(verl_section, dict):
        raise ValueError("Config section 'training.verl' must be a mapping")
    verl_section = dict(verl_section)
    fsdp_section = verl_section.pop("fsdp", {})
    if not isinstance(fsdp_section, dict):
        raise ValueError("Config section 'training.verl.fsdp' must be a mapping")
    return TrainingConfig(
        **training_section,
        value=CompactionValueConfig(**value_section),
        verl=VerlRayConfig(**verl_section, fsdp=VerlFSDPConfig(**fsdp_section)),
    )


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    raw = _load_yaml(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    training = _load_training_config(raw)
    return TrainConfig(
        experiment=ExperimentConfig(**_require_section(raw, "experiment")),
        dataset=DatasetConfig(**_require_section(raw, "dataset")),
        retrieval=RetrievalConfig(**_require_section(raw, "retrieval")),
        model=ModelConfig(**_require_section(raw, "model")),
        runtime=RuntimeConfig(**_require_section(raw, "runtime")),
        judge=JudgeConfig(**_require_section(raw, "judge")),
        training=training,
        collection=CollectionConfig(**_require_section(raw, "collection")),
        rollout=_derive_rollout_config(raw, training),
        evaluation=EvaluationConfig(**_require_section(raw, "evaluation")),
    )


def config_to_dict(config: RunConfig | TrainConfig) -> dict[str, Any]:
    return asdict(config)


def resolved_rollout_sampling_profile(config: TrainConfig, *, split: str) -> dict[str, Any]:
    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported rollout split: {split}")

    profile = {
        "max_new_tokens": (
            config.rollout.max_new_tokens
            if config.rollout.max_new_tokens is not None
            else config.model.max_new_tokens
        ),
        "temperature": (
            config.rollout.temperature
            if config.rollout.temperature is not None
            else config.model.temperature
        ),
        "top_p": config.rollout.top_p if config.rollout.top_p is not None else config.model.top_p,
        "do_sample": (
            config.rollout.do_sample
            if config.rollout.do_sample is not None
            else config.model.do_sample
        ),
        "api_extra_body": dict(config.rollout.api_extra_body),
        "extra_sampling_params": {},
    }
    if split == "eval":
        evaluation = config.evaluation
        for key in ("max_new_tokens", "temperature", "top_p", "do_sample"):
            value = getattr(evaluation, key)
            if value is not None:
                profile[key] = value
        profile["extra_sampling_params"].update(evaluation.extra_sampling_params)
    return profile


def sampling_profile_id(profile: dict[str, Any]) -> str:
    canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
