from copy import deepcopy
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet

from self_summarization_agent.qwen_interval_attention import interval_linear_attention

# flash-linear-attention's recurrent kernels require CUDA tensors.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')

DEVICE = 'cuda'


def tiny_config():
    return Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, layer_types=['linear_attention'])


def packed_inputs(device=DEVICE):
    torch.manual_seed(8)
    # Two interval boundaries, a boundary-straddling shard, and SP padding.
    x = torch.randn(1, 64, 32, device=device)
    pos = torch.cat([torch.arange(n, device=device) for n in (19, 12, 30, 3)])[None]
    weights = torch.randn_like(x)
    weights[:, -3:] = 0
    return x, pos, weights


@pytest.mark.parametrize('reverse', [False, True])
def test_packed_mixer_matches_independent_values_and_gradients(reverse):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    layer = Qwen3_5GatedDeltaNet(tiny_config(), 0).cuda()
    reference = deepcopy(layer)
    x, pos, weights = packed_inputs()
    if reverse:
        x = x.flip(1); weights = weights.flip(1)
    a = x.clone().requires_grad_(); b = x.clone().requires_grad_()
    actual = interval_linear_attention(Qwen3_5GatedDeltaNet.forward, layer, a, pos)
    boundaries = [0, 19, 31, 61, 64]
    expected = torch.cat([reference(b[:, start:end]) for start, end in zip(boundaries, boundaries[1:])], dim=1)
    torch.testing.assert_close(actual, expected)
    (actual * weights).sum().backward(); (expected * weights).sum().backward()
    torch.testing.assert_close(a.grad, b.grad)
    for left, right in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad)
    # The uncorrected operation really leaks across the same boundaries.
    with torch.no_grad():
        assert not torch.allclose(reference(x)[:, 19:61], expected[:, 19:61])


def test_packed_mixer_rejects_ambiguous_boundaries_and_caches():
    layer = Qwen3_5GatedDeltaNet(tiny_config(), 0).cuda()
    x, pos, _ = packed_inputs()
    bad = pos.clone(); bad[0, 10] = 100
    with pytest.raises(ValueError, match='contiguous'):
        interval_linear_attention(Qwen3_5GatedDeltaNet.forward, layer, x, bad)
    with pytest.raises(ValueError, match='aligned'):
        interval_linear_attention(Qwen3_5GatedDeltaNet.forward, layer, x, None)
    with pytest.raises(ValueError, match='caches'):
        interval_linear_attention(Qwen3_5GatedDeltaNet.forward, layer, x, pos, cache_params=object())


def test_padded_rows_keep_native_semantics():
    layer = Qwen3_5GatedDeltaNet(tiny_config(), 0).cuda()
    x = torch.randn(2, 12, 32, device=DEVICE)
    mask = torch.ones(2, 12, device=DEVICE); mask[1, 8:] = 0
    actual = interval_linear_attention(Qwen3_5GatedDeltaNet.forward, layer, x, None, attention_mask=mask)
    torch.testing.assert_close(actual, layer(x, attention_mask=mask))


@pytest.mark.parametrize("moe", [False, True])
def test_decoder_patch_preserves_checkpointed_gradients(monkeypatch, moe):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeGatedDeltaNet
    from self_summarization_agent.qwen_interval_attention import _patch_pair

    decoder_class = Qwen3_5MoeDecoderLayer if moe else Qwen3_5DecoderLayer
    mixer_class = Qwen3_5MoeGatedDeltaNet if moe else Qwen3_5GatedDeltaNet
    # Restore process-global classes after the test, including on failure.
    monkeypatch.setattr(decoder_class, "forward", decoder_class.forward)
    monkeypatch.setattr(mixer_class, "forward", mixer_class.forward)
    config = tiny_config()
    if moe:
        config = Qwen3_5MoeTextConfig(**{**config.to_dict(), "num_experts": 4,
            "num_experts_per_tok": 2, "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16})
    torch.manual_seed(3)
    layer = decoder_class(config, 0).cuda()
    if moe:
        # Standalone expert modules allocate empty weights; model.post_init()
        # normally initializes these when loading a complete HF model.
        for parameter in layer.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
    reference = deepcopy(layer)
    x, pos, weights = packed_inputs()
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    boundaries = [0, 19, 31, 61, 64]
    expected = torch.cat([reference(b[:, start:end], position_embeddings=None,
                         position_ids=pos[:, start:end])
                         for start, end in zip(boundaries, boundaries[1:])], dim=1)
    (expected * weights).sum().backward()
    _patch_pair(decoder_class, mixer_class)
    installed = decoder_class.forward
    _patch_pair(decoder_class, mixer_class)
    assert installed is decoder_class.forward
    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    actual = layer(a, position_embeddings=None, position_ids=pos)
    # Another forward must not replace the first graph's checkpoint metadata.
    layer(x[:, :19], position_embeddings=None, position_ids=pos[:, :19])
    torch.testing.assert_close(actual, expected)
    (actual * weights).sum().backward()
    torch.testing.assert_close(a.grad, b.grad)
    for left, right in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad)

