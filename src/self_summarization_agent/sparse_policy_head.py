"""Full-vocabulary policy log probabilities for masked Qwen3.5 targets."""

import torch


def local_prediction_indices(input_offsets, response_mask, shard_length, shard_rank):
    """Map retained response targets to their causal predictors on this SP rank."""
    response_offsets = response_mask.offsets()
    selected = response_mask.values().nonzero().flatten()
    rows = torch.searchsorted(response_offsets[1:], selected, right=True)
    # Each response ends at the interval end. Its first target is predicted
    # by the final prompt token, including when the prompt has only one token.
    positions = selected + (input_offsets[1:] - response_offsets[1:] - 1)[rows]
    local = positions - shard_length * shard_rank
    return local[(local >= 0) & (local < shard_length)]


def install_sparse_policy_head(engine):
    """Adapt this worker's packed, policy-only actor; keep verl's loss and SP backward."""
    from verl.models.transformers.qwen3_5 import Qwen3_5CausalLMOutputForPPO, forward_with_normal_backend
    from verl.utils import tensordict_utils as tu
    from verl.utils.torch_functional import logprobs_from_logits
    from verl.utils.ulysses import gather_outputs_and_unpad, get_ulysses_sequence_parallel_rank

    model = getattr(engine.module, "module", engine.module)
    dense_forward = model.forward
    if getattr(dense_forward, "__func__", None) is not forward_with_normal_backend:
        raise ValueError("Sparse policy head requires verl's Qwen3.5 normal forward")
    dense_inputs = engine.prepare_model_inputs
    dense_outputs = engine.prepare_model_outputs

    def prepare_inputs(micro_batch):
        inputs, args = dense_inputs(micro_batch)
        if "response_mask" not in micro_batch or "advantages" not in micro_batch:
            return inputs, args
        dense_features = (
            "calculate_entropy",
            "calculate_sum_pi_squared",
            "distillation_use_topk",
            "use_fused_kernels",
        )
        if "temperature_rmpad" not in args or any(
            tu.get_non_tensor_data(micro_batch, key, default=False) for key in dense_features
        ):
            return inputs, args
        labels = args["input_ids_rmpad_rolled"]
        rank = get_ulysses_sequence_parallel_rank() if engine.use_ulysses_sp else 0
        indices = local_prediction_indices(
            micro_batch["input_ids"].offsets(), micro_batch["response_mask"], labels.numel(), rank
        )
        inputs["policy_positions"] = args["policy_positions"] = indices
        return inputs, args

    def forward(*args, policy_positions=None, **kwargs):
        if policy_positions is None:
            return dense_forward(*args, **kwargs)
        outputs = model.model(*args, **kwargs)
        # One dummy projection on an empty shard uses the same kernel and
        # output dtype as its peers and keeps the FSDP backward edge alive.
        hidden = (
            outputs[0].index_select(1, policy_positions) if policy_positions.numel() else outputs[0][:, :1]
        )
        return Qwen3_5CausalLMOutputForPPO(logits=model.lm_head(hidden), hidden_states=outputs.hidden_states)

    def prepare_outputs(output, output_args, micro_batch, logits_processor_func):
        if "policy_positions" not in output_args:
            return dense_outputs(output, output_args, micro_batch, logits_processor_func)
        indices = output_args["policy_positions"]
        labels = output_args["input_ids_rmpad_rolled"]
        logits = output.logits.squeeze(0)
        projected = indices if indices.numel() else slice(0, 1)
        temperatures = output_args["temperature_rmpad"][projected].clamp(min=1e-8)
        logits.div_(temperatures.unsqueeze(-1).to(logits.dtype))
        selected = logprobs_from_logits(logits, labels[projected])[: indices.numel()]
        log_probs = selected.new_zeros(labels.numel()).index_copy(0, indices, selected)
        if engine.use_ulysses_sp:
            log_probs = gather_outputs_and_unpad(
                log_probs, gather_dim=0, unpad_dim=0, padding_size=output_args["pad_size"]
            )
        packed = torch.nested.nested_tensor_from_jagged(log_probs, micro_batch["input_ids"].offsets())
        return {"log_probs": packed}

    engine.prepare_model_inputs = prepare_inputs
    model.forward = forward
    engine.prepare_model_outputs = prepare_outputs
