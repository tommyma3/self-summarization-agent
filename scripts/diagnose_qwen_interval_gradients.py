"""Compare Qwen packed gradients to a CPU FP64 recurrence and finite differences.

This is a tiny numerical diagnostic, not a replacement training implementation.
It keeps the source FP32 weight/input values for every comparison. The oracle
avoids fused kernels and the upstream fallbacks' explicit FP32 casts. Run again
with TRITON_F32_DEFAULT=ieee to measure the fused kernel precision contribution.
"""
import argparse
import os
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from copy import deepcopy
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet
from self_summarization_agent.qwen_interval_attention import _patch_pair


def tiny_config():
    return Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, layer_types=['linear_attention'])


def packed_inputs():
    torch.manual_seed(8)
    x = torch.randn(1, 64, 32, device='cuda')
    positions = torch.cat([torch.arange(n, device='cuda') for n in (19, 12, 30, 3)])[None]
    weights = torch.randn_like(x)
    weights[:, -3:] = 0
    return x, positions, weights


def rms(x, weight, eps, offset=False):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (weight + 1 if offset else weight)

def reference_mixer(m, x):
    """Direct delta-rule recurrence; no FLA/chunked custom backward is used."""
    batch, length, _ = x.shape
    mixed = F.linear(x, m.in_proj_qkv.weight).transpose(1, 2)
    mixed = F.silu(F.conv1d(mixed, m.conv1d.weight, padding=m.conv_kernel_size - 1, groups=m.conv_dim)[..., :length]).transpose(1, 2)
    q, k, v = mixed.split([m.key_dim, m.key_dim, m.value_dim], -1)
    q = q.reshape(batch, length, m.num_k_heads, m.head_k_dim)
    k = k.reshape_as(q)
    v = v.reshape(batch, length, m.num_v_heads, m.head_v_dim)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    repeat = m.num_v_heads // m.num_k_heads
    q = q.repeat_interleave(repeat, dim=2) / (m.head_k_dim ** .5)
    k = k.repeat_interleave(repeat, dim=2)
    beta = F.linear(x, m.in_proj_b.weight).sigmoid()
    g = -m.A_log.exp() * F.softplus(F.linear(x, m.in_proj_a.weight) + m.dt_bias)
    state = x.new_zeros((batch, m.num_v_heads, m.head_k_dim, m.head_v_dim))
    pieces = []
    for t in range(length):
        state = state * g[:, t, :, None, None].exp()
        delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        pieces.append((state * q[:, t, :, :, None]).sum(-2))
    out = torch.stack(pieces, 1)
    out = rms(out, m.norm.weight, m.layer_norm_epsilon)
    gate = F.linear(x, m.in_proj_z.weight).reshape_as(out)
    return F.linear((out * F.silu(gate)).reshape(batch, length, -1), m.out_proj.weight)

def reference_feed_forward(m, hidden):
    normalized = rms(hidden, m.post_attention_layernorm.weight,
                     m.post_attention_layernorm.eps, True)
    return hidden + F.linear(F.silu(F.linear(normalized, m.mlp.gate_proj.weight))
                             * F.linear(normalized, m.mlp.up_proj.weight), m.mlp.down_proj.weight)


def reference_decoder(m, x, edges, packed):
    def run(x, edges):
        norm = rms(x, m.input_layernorm.weight, m.input_layernorm.eps, True)
        mix = torch.cat([reference_mixer(m.linear_attn, norm[:, s:e]) for s, e in zip(edges, edges[1:])], 1)
        hidden = x + mix
        return reference_feed_forward(m, hidden)
    return run(x, edges) if packed else torch.cat([run(x[:, s:e], [0, e-s]) for s, e in zip(edges, edges[1:])], 1)

def _run(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    base = Qwen3_5DecoderLayer(tiny_config(), 0).cuda()
    x, pos, w = packed_inputs()
    edges = [0, 19, 31, 61, 64]
    models, outputs, inputs = {}, {}, {}
    for kind in ['separate32', 'packed32', 'separate64', 'packed64']:
        m = deepcopy(base)
        if kind.endswith('64'):
            m = m.cpu().double()
            inp = x.cpu().double().requires_grad_()
            out = reference_decoder(m, inp, edges, kind.startswith('packed'))
            loss = (out * w.cpu().double()).sum()
        else:
            inp = x.clone().requires_grad_()
            if kind == 'separate32':
                out = torch.cat([m(inp[:, s:e], position_embeddings=None, position_ids=pos[:, s:e]) for s, e in zip(edges, edges[1:])], 1)
            else:
                _patch_pair(Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet)
                out = m(inp, position_embeddings=None, position_ids=pos)
            loss = (out * w).sum()
        loss.backward()
        models[kind] = m
        outputs[kind] = out.detach().cpu().double()
        inputs[kind] = inp
    oracle = models['separate64']
    truth = dict(oracle.named_parameters())
    report = {}
    for kind, m in models.items():
        differences, vector, refvec = {}, [], []
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach().cpu().double()
            ref = truth[name].grad
            vector.append(grad.flatten())
            refvec.append(ref.flatten())
            if name in {'linear_attn.conv1d.weight', 'linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_b.weight'}:
                differences[name] = dict(max_abs=float((grad-ref).abs().max()), rms=float((grad-ref).square().mean().sqrt()))
        vector, refvec = torch.cat(vector), torch.cat(refvec)
        report[kind] = dict(output_max_abs=float((outputs[kind]-outputs['separate64']).abs().max()),
            input_grad_max_abs=float((inputs[kind].grad.cpu().double()-inputs['separate64'].grad).abs().max()),
            all_parameter_gradient_rms=float((vector-refvec).square().mean().sqrt()),
            all_parameter_relative_l2=float(torch.linalg.vector_norm(vector-refvec)/torch.linalg.vector_norm(refvec)),
            attention_parameters=differences)
    a = dict(models['packed32'].named_parameters())['linear_attn.conv1d.weight'].grad.cpu().double()
    b = dict(models['separate32'].named_parameters())['linear_attn.conv1d.weight'].grad.cpu().double()
    mismatch = (~torch.isclose(a, b, atol=1e-5, rtol=1.3e-6)).nonzero().tolist()
    # Probe the original failing entries even if the precision control passes.
    mismatch = sorted({tuple(index) for index in mismatch} | {(35, 0, 1), (35, 0, 3)})
    finite = []
    weight = oracle.linear_attn.conv1d.weight
    with torch.no_grad():
        for index in mismatch:
            index = tuple(index)
            original = float(weight[index])
            values = []
            for h in [1e-3, 1e-4, 1e-5, 1e-6]:
                weight[index] = original+h
                plus = float((reference_decoder(oracle, x.cpu().double(), edges, False)*w.cpu().double()).sum())
                weight[index] = original-h
                minus = float((reference_decoder(oracle, x.cpu().double(), edges, False)*w.cpu().double()).sum())
                weight[index] = original
                values.append(dict(step=h, derivative=(plus-minus)/(2*h)))
            finite.append(dict(index=list(index), oracle=float(weight.grad[index]), separate32=float(b[index]), packed32=float(a[index]), finite_differences=values))
    report['selected_entries'] = finite
    old = [p.detach().clone() for p in models['separate32'].parameters()]
    for kind in ('separate32', 'packed32'):
        optimizer = torch.optim.AdamW(models[kind].parameters(), lr=1e-5, weight_decay=0.)
        torch.nn.utils.clip_grad_norm_(models[kind].parameters(), 1.)
        optimizer.step()
    left = torch.cat([p.detach().flatten() for p in models['packed32'].parameters()])
    right = torch.cat([p.detach().flatten() for p in models['separate32'].parameters()])
    initial = torch.cat([p.flatten() for p in old])
    report['adamw_step'] = dict(learning_rate=1e-5, gradient_clip=1.,
        max_weight_difference=float((left-right).abs().max()),
        weight_difference_l2=float(torch.linalg.vector_norm(left-right)),
        separate_update_l2=float(torch.linalg.vector_norm(right-initial)))
    report['environment'] = dict(seed=seed, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), triton_f32_default=os.environ.get('TRITON_F32_DEFAULT'),
        torch_matmul_precision=torch.get_float32_matmul_precision())
    return report


def run(seed=3):
    decoder, mixer = Qwen3_5DecoderLayer.forward, Qwen3_5GatedDeltaNet.forward
    try:
        return _run(seed)
    finally:
        Qwen3_5DecoderLayer.forward, Qwen3_5GatedDeltaNet.forward = decoder, mixer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
