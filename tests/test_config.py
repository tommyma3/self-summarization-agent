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
""".strip(),
        encoding="utf-8",
    )

    config = load_run_config(
        config_path,
        {"dataset.limit": 3, "retrieval.backend": "bm25", "runtime.tool_budget": 2},
    )

    assert config.dataset.limit == 3
    assert config.retrieval.backend == "bm25"
    assert config.retrieval.snippet_tokenizer_path == "/models/qwen-tokenizer"
    assert config.runtime.tool_budget == 2


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
judge:
  enabled: true
training:
  backend: fsdp2_context_parallel
  gpu_ids: [0, 1, 2, 3]
  fsdp_version: 2
  context_parallel_size: 4
  steps: 3
  batch_size: 2
  group_size: 2
""".strip(),
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config.training.steps == 3
    assert config.training.batch_size == 2
    assert config.training.group_size == 2
    assert config.retrieval.snippet_tokenizer_path == "/models/qwen-tokenizer"
    assert config.rollout.backend == "vllm_offline"
    assert config.rollout.gpu_ids == [0, 1, 2, 3]
    assert config.rollout.tensor_parallel_size == 4
    assert config.rollout.max_model_len == 65536
    assert config.training.backend == "fsdp2_context_parallel"
    assert config.training.context_parallel_size == 4
