import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from self_summarization_agent.generation import VLLMGenerator


@pytest.mark.parametrize("use_cli_overrides", [False, True])
def test_probe_uses_training_rollout_settings(tmp_path, monkeypatch, use_cli_overrides):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import probe_vllm_collection as probe

    dataset_path = tmp_path / "queries.jsonl"
    dataset_path.write_text('{"query_id": "q", "query": "question", "answer": "answer"}\n')
    config_path = tmp_path / "train.yaml"
    config_path.write_text(f"""\
experiment: {{name: probe, seed: 0, output_root: {tmp_path}}}
dataset: {{decrypted_path: {dataset_path}}}
model: {{model_path: model, max_model_len: 4096}}
retrieval: {{gpu_ids: [0]}}
rollout:
  backend: vllm_offline
  gpu_ids: [4, 5, 6, 7]
  tensor_parallel_size: 4
  attention_backend: FLASH_ATTN
  max_model_len: 8192
  enable_prefix_caching: false
  browsecomp_native_tools: true
  sampling_seed: 7
  do_sample: true
  temperature: 0.8
  max_new_tokens: 128
  extra_sampling_params: {{presence_penalty: 1.5, repetition_penalty: 1.2}}
evaluation:
  extra_sampling_params: {{presence_penalty: 0.1}}
judge: {{do_sample: true}}
""")
    output_path = tmp_path / "trace.txt"
    argv = [str(probe.__file__), "--config", str(config_path), "--output", str(output_path)]
    if use_cli_overrides:
        argv += ["--vllm-gpus", "3", "--tensor-parallel-size", "1", "--no-sample",
                 "--attention-backend", "TRITON_ATTN", "--max-new-tokens", "64"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(probe, "build_backend", lambda *_: object())
    monkeypatch.setattr(probe, "build_runtime", lambda *_: object())

    def initialize_generator(generator):
        generator.tokenizer = None
        generator.llm = object()
        generator._sampling_params_cls = dict

    monkeypatch.setattr(VLLMGenerator, "__post_init__", initialize_generator)

    def trace(*, generator, judge, **kwargs):
        assert generator.enable_prefix_caching is False
        assert generator.max_model_len == 8192
        assert generator.supports_native_tools
        assert generator.engine_seed == 7
        assert generator.sampling_extra == {"presence_penalty": 1.5, "repetition_penalty": 1.2}
        assert judge.generator.sampling_extra == {}
        sampling_kwargs = generator._sampling_kwargs()
        if use_cli_overrides:
            assert generator.tensor_parallel_size == 1
            assert generator.attention_backend == "TRITON_ATTN"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
            assert sampling_kwargs == {"temperature": 0.0, "max_tokens": 64}
        else:
            assert generator.tensor_parallel_size == 4
            assert generator.attention_backend == "FLASH_ATTN"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
            assert sampling_kwargs["max_tokens"] == 128
            assert sampling_kwargs["temperature"] == 0.8
            assert sampling_kwargs["presence_penalty"] == 1.5
            assert sampling_kwargs["repetition_penalty"] == 1.2
        return SimpleNamespace(trajectory_records=[])

    monkeypatch.setattr(probe, "trace_collection", trace)
    probe.main()
