from pathlib import Path

from self_summarization_agent.config import (
    CompactionValueConfig,
    TrainingConfig,
    config_to_dict,
    load_run_config,
    load_train_config,
    resolved_rollout_sampling_profile,
)


def test_compaction_mc_value_config_is_opt_in_and_validated(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment: {name: demo, seed: 0, output_root: output, bc_plus_root: bc-plus}
dataset: {}
retrieval: {backend: faiss, index_path: index}
model: {backend: transformers, model_path: model}
runtime: {context_threshold_tokens: 32, max_context_tokens: 64, tool_budget: 4}
judge: {enabled: true}
training:
  advantage_estimator: compaction_mc_value
  value:
    enabled: true
    loss_coefficient: 0.25
    zero_initialize_head: true
    state_anchor: first_generation_prompt_end
""".strip(),
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config.training.advantage_estimator == "compaction_mc_value"
    assert config.training.value.enabled is True
    assert config.training.value.loss_coefficient == 0.25


def test_compaction_mc_value_flag_and_estimator_must_agree() -> None:
    try:
        TrainingConfig(value=CompactionValueConfig(enabled=True))
    except ValueError as exc:
        assert "advantage_estimator" in str(exc)
    else:
        raise AssertionError("mismatched value configuration must fail")


def test_load_run_config_applies_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
experiment:
  name: demo
  seed: 7
  output_root: output
  bc_plus_root: bc-plus
dataset: {}
retrieval:
  backend: faiss
  snippet_tokenizer_path: /models/qwen-tokenizer
  gpu_ids: [6]
  index_path: indexes/corpus.pkl
model:
  backend: transformers
  model_path: model-dir
rollout:
  backend: vllm_offline
  gpu_ids: [0, 1, 2, 3]
  tensor_parallel_size: 4
  max_model_len: 65536
  enable_prefix_caching: true
runtime:
  context_threshold_tokens: 32
  max_context_tokens: 64
  tool_budget: 4
  generated_token_budget: 16
""".strip(),
        encoding="utf-8",
    )

    config = load_run_config(
        config_path,
        {
            "dataset.limit": 3,
            "retrieval.backend": "bm25",
            "runtime.tool_budget": 2,
            "runtime.generated_token_budget": 8,
        },
    )

    assert config.dataset.limit == 3
    assert config.retrieval.backend == "bm25"
    assert config.retrieval.snippet_tokenizer_path == "/models/qwen-tokenizer"
    assert config.retrieval.gpu_ids == [6]
    assert config.runtime.tool_budget == 2
    assert config.runtime.generated_token_budget == 8
    assert config.runtime.max_summary_tokens == 2048


def test_load_train_config_reads_training_section(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment:
  name: demo
  seed: 7
  output_root: output
  bc_plus_root: bc-plus
dataset: {}
retrieval:
  backend: faiss
  snippet_tokenizer_path: /models/qwen-tokenizer
  index_path: indexes/corpus.pkl
model:
  backend: transformers
  model_path: model-dir
runtime:
  context_threshold_tokens: 32
  max_context_tokens: 64
  tool_budget: 4
  generated_token_budget: 16
judge:
  enabled: true
  backend: vllm_offline
  model_path: judge-dir
  gpu_ids: [4, 5]
  tensor_parallel_size: 2
  max_model_len: 8192
collection:
  train_task_count: 25
  eval_task_count: 5
evaluation:
  samples_per_task: 2
  temperature: 1.0
  top_p: 0.95
  do_sample: true
  extra_sampling_params:
    top_k: 20
    presence_penalty: 1.5
training:
  backend: fsdp2_context_parallel
  gpu_ids: [0, 1, 2, 3]
  fsdp_version: 2
  context_parallel_size: 4
  steps: 3
  batch_size: 2
  group_size: 2
  rollout_query_count: 100
  update_epochs: 4
  minibatch_size: 16
  clip_range: 0.3
  target_kl: 0.05
""".strip(),
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config.runtime.generated_token_budget == 16
    assert config.collection.train_task_count == 25
    assert config.collection.eval_task_count == 5
    assert config.evaluation.samples_per_task == 2
    assert resolved_rollout_sampling_profile(config, split="eval") == {
        "max_new_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.95,
        "do_sample": True,
        "api_extra_body": {},
        "extra_sampling_params": {"top_k": 20, "presence_penalty": 1.5},
    }
    assert config.training.steps == 3
    assert config.training.batch_size == 2
    assert config.training.group_size == 2
    assert config.training.rollout_query_count == 100
    assert config.training.update_epochs == 4
    assert config.training.minibatch_size == 16
    assert config.training.clip_range == 0.3
    assert config.training.target_kl == 0.05
    assert config.retrieval.snippet_tokenizer_path == "/models/qwen-tokenizer"
    assert config.rollout.backend == "vllm_offline"
    assert config.rollout.gpu_ids == [0, 1, 2, 3]
    assert config.rollout.tensor_parallel_size == 4
    assert config.rollout.max_model_len == 65536
    assert config.rollout.enable_prefix_caching is True
    assert config.judge.backend == "vllm_offline"
    assert config.judge.model_path == "judge-dir"
    assert config.judge.gpu_ids == [4, 5]
    assert config.judge.tensor_parallel_size == 2
    assert config.judge.max_model_len == 8192
    assert config.training.backend == "fsdp2_context_parallel"
    assert config.training.context_parallel_size == 4


def test_load_train_config_reads_verl_ray_section(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment:
  name: demo
  seed: 7
  output_root: output
  bc_plus_root: bc-plus
dataset: {}
retrieval:
  backend: faiss
  index_path: indexes/corpus.pkl
model:
  backend: transformers
  model_path: model-dir
runtime:
  context_threshold_tokens: 32
  max_context_tokens: 64
  tool_budget: 4
judge:
  enabled: true
training:
  backend: verl_ray
  gpu_ids: [0, 1, 2, 3]
  group_size: 2
  verl:
    address: auto
    namespace: remote-train
    num_cpus: 8
    num_gpus_per_worker: 4
    runtime_env:
      env_vars:
        TOKENIZERS_PARALLELISM: "true"
    worker_backend: verl_fsdp
    fsdp:
      strategy: fsdp2
      ppo_micro_batch_size_per_gpu: 2
      ppo_max_token_len_per_gpu: 32768
      log_prob_micro_batch_size_per_gpu: 2
      log_prob_max_token_len_per_gpu: 32768
      use_dynamic_bsz: false
      use_remove_padding: true
      use_torch_compile: false
      ulysses_sequence_parallel_size: 2
      param_offload: true
      optimizer_offload: true
      fsdp_size: 4
      save_hf_model: true
    ignore_reinit_error: false
    log_to_driver: false
    shutdown_ray: false
""".strip(),
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config.training.backend == "verl_ray"
    assert config.training.verl.address == "auto"
    assert config.training.verl.namespace == "remote-train"
    assert config.training.verl.num_cpus == 8
    assert config.training.verl.num_gpus_per_worker == 4
    assert config.training.verl.runtime_env == {"env_vars": {"TOKENIZERS_PARALLELISM": "true"}}
    assert config.training.verl.worker_backend == "verl_fsdp"
    assert config.training.verl.fsdp.strategy == "fsdp2"
    assert config.training.verl.fsdp.ppo_micro_batch_size_per_gpu == 2
    assert config.training.verl.fsdp.ppo_max_token_len_per_gpu == 32768
    assert config.training.verl.fsdp.log_prob_micro_batch_size_per_gpu == 2
    assert config.training.verl.fsdp.log_prob_max_token_len_per_gpu == 32768
    assert config.training.verl.fsdp.use_torch_compile is False
    assert config.training.verl.fsdp.ulysses_sequence_parallel_size == 2
    assert config.training.verl.fsdp.param_offload is True
    assert config.training.verl.fsdp.optimizer_offload is True
    assert config.training.verl.fsdp.fsdp_size == 4
    assert config.training.verl.fsdp.save_hf_model is True
    assert config.training.verl.ignore_reinit_error is False
    assert config.training.verl.log_to_driver is False
    assert config.training.verl.shutdown_ray is False


def test_load_no_compact_32k_training_preset() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train" / "no_compact_32k.yaml"

    config = load_train_config(config_path)

    # Experiment identity.
    assert config.experiment.name == "qwen-bcplus-no-compact-32k-train"

    # Runtime: compaction disabled, large active context.
    assert config.runtime.context_threshold_tokens == 1_000_000_000
    assert config.runtime.max_context_tokens == 40_960
    assert config.runtime.generated_token_budget == 36_000
    assert config.runtime.phase_timeout_seconds == 9_000
    # max_summary_tokens is present but irrelevant when compaction is disabled.
    assert config.runtime.max_summary_tokens is not None

    # Rollout: long-context vLLM, conservative concurrency.
    assert config.rollout.max_model_len == 49_152
    assert config.rollout.max_concurrent_episodes == 10
    assert config.rollout.overlap_queue_max_batches is not None
    # API compatibility fields — needed when backend is switched to openai_compatible.
    assert config.rollout.api_base_url is not None
    assert config.rollout.require_exact_token_ids is True

    # Evaluation: deterministic (matches default).
    assert config.evaluation.temperature == 0.0
    assert config.evaluation.top_p == 1.0
    assert config.evaluation.extra_sampling_params == {}

    # Training: verl/FSDP with full-sequence preservation.
    assert config.training.backend == "verl_ray"
    assert config.training.verl.worker_backend == "verl_fsdp"
    # max_sequence_length matches rollout.max_model_len so sequences are never
    # left-truncated before the verl update.
    assert config.training.max_sequence_length == 49_152
    assert config.training.gradient_accumulation_microbatch_size == 1
    # With ulysses_sequence_parallel_size=4, each of the four 80 GB A100 ranks
    # handles 12,288 tokens of the 49,152-token sequence.
    assert config.training.verl.fsdp.ulysses_sequence_parallel_size == 4
    assert config.training.verl.fsdp.ppo_max_token_len_per_gpu == 12_288
    assert config.training.verl.fsdp.log_prob_max_token_len_per_gpu == 12_288
    # Efficiency settings inherited from default.
    assert config.training.verl.fsdp.forward_prefetch is True
    assert config.training.verl.fsdp.use_dynamic_bsz is True
    assert config.training.verl.fsdp.use_remove_padding is True

    # Judge: larger context window for long answers.
    assert config.judge.max_model_len == 18_000
    assert config.judge.max_new_tokens == 12_000
    assert config.judge.batch_size is not None


def test_load_compact_6k_training_preset() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train" / "compact_6k.yaml"

    config = load_train_config(config_path)

    assert config.experiment.name == "qwen-bcplus-compact-6k-train"
    assert config.runtime.context_threshold_tokens == 6_000
    assert config.runtime.max_context_tokens == 24_000
    assert config.rollout.max_model_len == 32_768
    assert config.rollout.max_concurrent_episodes == 50
    assert config.training.max_sequence_length == 32_768
    assert config.training.gradient_accumulation_microbatch_size == 2
    assert config.training.verl.fsdp.ulysses_sequence_parallel_size == 4
    assert config.training.verl.fsdp.ppo_max_token_len_per_gpu == 8_192
    assert config.training.verl.fsdp.log_prob_max_token_len_per_gpu == 8_192
    assert config.rollout.max_model_len >= config.runtime.max_context_tokens + config.rollout.max_new_tokens
    assert (
        config.training.verl.fsdp.ppo_max_token_len_per_gpu
        * config.training.verl.fsdp.ulysses_sequence_parallel_size
        >= config.training.max_sequence_length
    )


def test_load_compaction_mc_value_training_preset() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train" / "compact_value_mc.yaml"
    default_path = Path(__file__).resolve().parents[1] / "configs" / "train" / "default.yaml"

    config = load_train_config(config_path)
    default = load_train_config(default_path)

    assert config.experiment.name == "qwen-bcplus-compact-value-mc"
    assert config.training.advantage_estimator == "compaction_mc_value"
    assert config.training.value.enabled is True
    assert config.training.value.zero_initialize_head is True
    assert config.training.value.state_anchor == "first_generation_prompt_end"
    assert config.training.verl.worker_backend == "transformers"
    assert config.runtime.context_threshold_tokens == default.runtime.context_threshold_tokens
    assert config.runtime.max_context_tokens == default.runtime.max_context_tokens
    assert config.runtime.max_summary_tokens == default.runtime.max_summary_tokens
    assert config.runtime.generated_token_budget == default.runtime.generated_token_budget
    assert config.rollout.max_model_len == default.rollout.max_model_len
    assert config.rollout.max_new_tokens == default.rollout.max_new_tokens
    assert config.training.max_sequence_length == default.training.max_sequence_length
    assert config.training.gradient_accumulation_microbatch_size == 1
    assert config.training.minibatch_size == 4
    assert config.training.update_epochs == 1
    assert config.training.verl.num_gpus_per_worker == 4
    assert config.model.device_map == "balanced"


def test_load_compact_24k_training_preset() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train" / "compact_24k.yaml"

    config = load_train_config(config_path)

    assert config.experiment.name == "qwen-bcplus-compact-24k-train"
    assert config.runtime.context_threshold_tokens == 24_000
    assert config.runtime.max_context_tokens == 40_960
    assert config.rollout.max_model_len == 49_152
    assert config.rollout.max_concurrent_episodes == 10
    assert config.training.max_sequence_length == 49_152
    assert config.training.gradient_accumulation_microbatch_size == 1
    assert config.training.verl.fsdp.ulysses_sequence_parallel_size == 4
    assert config.training.verl.fsdp.ppo_max_token_len_per_gpu == 12_288
    assert config.training.verl.fsdp.log_prob_max_token_len_per_gpu == 12_288
    assert config.rollout.max_model_len >= config.runtime.max_context_tokens + config.rollout.max_new_tokens
    assert (
        config.training.verl.fsdp.ppo_max_token_len_per_gpu
        * config.training.verl.fsdp.ulysses_sequence_parallel_size
        >= config.training.max_sequence_length
    )


def test_no_compaction_loss_preset_only_changes_ablation_identity_and_mask() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs" / "train"
    default = config_to_dict(load_train_config(config_root / "default.yaml"))
    ablation = config_to_dict(load_train_config(config_root / "no_compaction_loss.yaml"))

    assert default["training"]["train_compaction_tokens"] is True
    assert ablation["training"]["train_compaction_tokens"] is False
    assert ablation["experiment"]["name"] == "qwen-bcplus-no-compaction-loss"
    assert ablation["experiment"]["output_root"].endswith("/ablations/no-compaction-loss/")

    for config in (default, ablation):
        config["experiment"].pop("name")
        config["experiment"].pop("output_root")
        config["training"].pop("train_compaction_tokens")
    assert ablation == default
