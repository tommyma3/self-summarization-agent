from pathlib import Path

from self_summarization_agent.config import load_run_config, load_train_config


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
  index_path: indexes/corpus.pkl
model:
  backend: transformers
  model_path: model-dir
rollout:
  backend: vllm_offline
  gpu_ids: [0, 1, 2, 3]
  tensor_parallel_size: 4
  max_model_len: 65536
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
    assert config.runtime.tool_budget == 2
    assert config.runtime.generated_token_budget == 8


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

    assert config.experiment.name == "qwen-bcplus-no-compact-32k-train"
    assert config.runtime.context_threshold_tokens == 1_000_000_000
    assert config.runtime.max_context_tokens == 40_960
    assert config.runtime.generated_token_budget == 32_768
    assert config.rollout.max_model_len == 49_152
    assert config.rollout.max_concurrent_episodes == 8
    assert config.training.backend == "verl_ray"
    assert config.training.verl.worker_backend == "verl_fsdp"
    assert config.training.max_sequence_length == 49_152
    assert config.training.verl.fsdp.ppo_max_token_len_per_gpu == 49_152
    assert config.training.verl.fsdp.log_prob_max_token_len_per_gpu == 49_152
    assert config.training.verl.fsdp.ulysses_sequence_parallel_size == 4
