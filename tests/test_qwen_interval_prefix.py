"""Check causal-prefix execution across four sequence-parallel CPU ranks."""
from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from self_summarization_agent.qwen_interval_attention import interval_linear_attention


def _check_rank(rank, init_file):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        config = Qwen3_5TextConfig(hidden_size=32, linear_num_key_heads=2, linear_num_value_heads=2,
                                  linear_key_head_dim=8, linear_value_head_dim=8)
        torch.manual_seed(9)
        with patch.object(qwen, "FusedRMSNormGated", None):
            mixer = qwen.Qwen3_5GatedDeltaNet(config, 0)
        mixer.causal_conv1d_fn = None
        mixer.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
        reference = deepcopy(mixer)
        lengths = [97, 28, 1, 1, 1]
        full = torch.randn(1, sum(lengths), 32, requires_grad=True)
        weights = torch.randn_like(full)
        positions = torch.cat([torch.arange(n) for n in lengths])[None]
        expected = torch.cat([reference(tokens) for tokens in full.split(lengths, dim=1)], dim=1)
        (expected * weights).sum().backward()

        local = slice(rank * 32, (rank + 1) * 32)
        shard = full[:, local].detach().requires_grad_()
        evaluated = []

        def forward(module, tokens, **kwargs):
            evaluated.append(tokens.size(1))
            return module(tokens, **kwargs)

        actual = interval_linear_attention(forward, mixer, shard, positions[:, local],
                                           group=dist.group.WORLD)
        assert evaluated == [[32], [64], [96], lengths][rank]
        torch.testing.assert_close(actual, expected[:, local])
        (actual * weights[:, local]).sum().backward()
        torch.testing.assert_close(shard.grad, full.grad[:, local])
        for parameter, oracle in zip(mixer.parameters(), reference.parameters(), strict=True):
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(parameter.grad, oracle.grad)
    finally:
        dist.destroy_process_group()


def test_distributed_qwen_prefix_matches_full_interval_gradients(tmp_path):
    mp.spawn(_check_rank, args=(str(tmp_path / "gloo"),), nprocs=4)
