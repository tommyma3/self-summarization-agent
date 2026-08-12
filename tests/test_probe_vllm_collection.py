import argparse
import importlib.util
import sys
from pathlib import Path

from self_summarization_agent.config import ModelConfig, RolloutConfig


SIMULATE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_collection.py"
SIMULATE_SPEC = importlib.util.spec_from_file_location("simulate_collection", SIMULATE_SCRIPT)
simulate_collection = importlib.util.module_from_spec(SIMULATE_SPEC)
sys.modules["simulate_collection"] = simulate_collection
assert SIMULATE_SPEC.loader is not None
SIMULATE_SPEC.loader.exec_module(simulate_collection)

PROBE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_vllm_collection.py"
PROBE_SPEC = importlib.util.spec_from_file_location("probe_vllm_collection", PROBE_SCRIPT)
probe_collection = importlib.util.module_from_spec(PROBE_SPEC)
sys.modules["probe_vllm_collection"] = probe_collection
assert PROBE_SPEC.loader is not None
PROBE_SPEC.loader.exec_module(probe_collection)


class ConfigForProbe:
    model = ModelConfig(
        backend="transformers",
        model_path="model",
        max_model_len=32768,
        attention_backend="model-default",
    )
    rollout = RolloutConfig(
        backend="sglang",
        gpu_ids=[4, 5],
        tensor_parallel_size=2,
        attention_backend="flashinfer",
        max_model_len=49152,
        max_new_tokens=8192,
        enable_prefix_caching=True,
        require_exact_token_ids=True,
    )


def probe_args(**overrides):
    values = {
        "rollout_backend": None,
        "rollout_gpus": None,
        "tensor_parallel_size": None,
        "attention_backend": None,
        "max_new_tokens": None,
        "temperature": None,
        "top_p": None,
        "do_sample": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_probe_uses_sglang_rollout_configuration() -> None:
    model_config = probe_collection.build_rollout_model_config(ConfigForProbe, probe_args())

    assert model_config.backend == "sglang"
    assert model_config.tensor_parallel_size == 2
    assert model_config.attention_backend == "flashinfer"
    assert model_config.max_model_len == 49152
    assert model_config.max_new_tokens == 8192
    assert model_config.enable_prefix_caching is True
    assert model_config.require_exact_token_ids is True


def test_probe_allows_vllm_backend_and_gpu_overrides() -> None:
    args = probe_args(
        rollout_backend="vllm_offline",
        rollout_gpus="6,7",
        tensor_parallel_size=2,
        attention_backend="FLASH_ATTN",
    )

    model_config = probe_collection.build_rollout_model_config(ConfigForProbe, args)

    assert model_config.backend == "vllm_offline"
    assert model_config.attention_backend == "FLASH_ATTN"
    assert probe_collection.rollout_gpu_visibility(ConfigForProbe, args) == "6,7"


def test_probe_uses_configured_rollout_gpus_by_default() -> None:
    assert probe_collection.rollout_gpu_visibility(ConfigForProbe, probe_args()) == "4,5"
