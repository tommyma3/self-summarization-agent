"""Keep Qwen3.5 recurrent conditioning intact in verl packed/Ulysses batches.

Full attention understands position resets and Ulysses, but the upstream
GatedDeltaNet convolution/recurrence does not. Reassemble its input with a
SUM-reducing autograd all-gather and run the original mixer independently on
whole intervals. Only intervals overlapping this rank's output are evaluated.
No tokens, masks, rollout probabilities, or checkpoint parameters are changed.
"""
from contextvars import ContextVar
from functools import wraps

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


_NO_DECODER = object()
_decoder_positions = ContextVar("qwen_interval_positions", default=_NO_DECODER)


def interval_linear_attention(original, module, hidden_states, position_ids, *,
                              cache_params=None, attention_mask=None, group=None):
    """Apply a recurrent mixer to independent intervals, preserving SP gradients."""
    world = dist.get_world_size(group) if group is not None else 1
    if world == 1 and (attention_mask is not None or hidden_states.shape[0] != 1):
        # Ordinary padded rows already have independent recurrent states.
        return original(module, hidden_states, cache_params=cache_params, attention_mask=attention_mask)
    if position_ids is None or position_ids.ndim != 2 or position_ids.shape != hidden_states.shape[:2]:
        raise ValueError("Packed Qwen recurrent attention requires aligned text position IDs")
    if hidden_states.shape[0] != 1 or attention_mask is not None:
        raise ValueError("Qwen recurrent Ulysses requires a single packed sequence")
    if cache_params is not None:
        raise ValueError("Packed Qwen recurrent training does not support generation caches")

    local_length = hidden_states.shape[1]
    rank = dist.get_rank(group) if world > 1 else 0
    if world > 1:
        positions = [torch.empty_like(position_ids) for _ in range(world)]
        dist.all_gather(positions, position_ids.contiguous(), group=group)
        position_ids = torch.cat(positions, dim=1)
        # Unlike verl's final-logit Gather (slice-only backward), inputs here
        # influence OTHER ranks' outputs through recurrent state. Backward must
        # SUM those dependencies, with no additional SP gradient multiplier.
        hidden_states = torch.cat(all_gather(hidden_states.contiguous(), group=group), dim=1)

    positions = position_ids[0]
    valid_steps = (positions[1:] == positions[:-1] + 1) | (positions[1:] == 0)
    if positions.numel() == 0 or positions[0].item() != 0 or not bool(valid_steps.all()):
        raise ValueError("Packed Qwen position IDs must start at zero and be contiguous within each interval")
    boundaries = (positions == 0).nonzero().flatten().tolist() + [positions.numel()]
    local_start, local_end = rank * local_length, (rank + 1) * local_length
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        left, right = max(start, local_start), min(end, local_end)
        if left >= right:
            continue
        # Reuse the model's exact convolution, gates, normalization, and kernels.
        # A fresh invocation resets BOTH convolution and recurrence. Ulysses
        # alignment padding has its own position-zero span, so cannot leak.
        output = original(module, hidden_states[:, start:end], cache_params=None, attention_mask=None)
        pieces.append(output[:, left - start:right - start])
    return torch.cat(pieces, dim=1)


def _patch_pair(decoder_class, mixer_class):
    if getattr(decoder_class.forward, "_interval_safe", False):
        return
    decoder_forward = decoder_class.forward
    mixer_forward = mixer_class.forward

    @wraps(decoder_forward)
    def decoder(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        # Set context inside forward, so activation-checkpoint recomputation
        # restores the correct metadata even after another microbatch ran.
        token = _decoder_positions.set(position_ids)
        try:
            return decoder_forward(self, hidden_states, position_embeddings,
                                   attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_values=past_key_values, **kwargs)
        finally:
            _decoder_positions.reset(token)

    @wraps(mixer_forward)
    def mixer(self, hidden_states, cache_params=None, attention_mask=None):
        positions = _decoder_positions.get()
        if positions is _NO_DECODER:
            return mixer_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)
        from verl.utils.ulysses import get_ulysses_sequence_parallel_group
        return interval_linear_attention(mixer_forward, self, hidden_states, positions,
                                         cache_params=cache_params, attention_mask=attention_mask,
                                         group=get_ulysses_sequence_parallel_group())

    decoder._interval_safe = True
    decoder_class.forward = decoder
    mixer_class.forward = mixer


def install_qwen_interval_attention():
    """Install once per training worker, before FSDP wrapping/checkpointing."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeGatedDeltaNet
    _patch_pair(Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet)
    _patch_pair(Qwen3_5MoeDecoderLayer, Qwen3_5MoeGatedDeltaNet)
