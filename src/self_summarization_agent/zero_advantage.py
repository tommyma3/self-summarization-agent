"""Evaluate zero-advantage GRPO intervals without building a backward graph."""

import torch


def install_zero_advantage_forward_only(engine):
    from verl.utils import tensordict_utils as tu
    from verl.utils.device import get_device_id
    from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches

    # SP ranks see identical complete intervals. Different DP ranks could make
    # different partitions and enter different FSDP collectives.
    if engine.get_data_parallel_size() != 1:
        return
    original = engine.forward_backward_batch

    def forward_backward_batch(data, loss_function, forward_only=False):
        if forward_only or not tu.get_non_tensor_data(data, "use_dynamic_bsz", default=True):
            return original(data, loss_function, forward_only=forward_only)
        zero_indices = []
        active_indices = []
        for index, advantage in enumerate(data["advantages"].unbind()):
            if advantage.any().item():
                active_indices.append(index)
            else:
                zero_indices.append(index)
        if not zero_indices or not active_indices:
            # In particular, Adam must see materialized zero gradients on a
            # fully zero optimizer minibatch and advance its moments normally.
            return original(data, loss_function, forward_only=False)

        denominator = int(data["loss_mask"].sum().item())

        def actor_loss(model_output, data, dp_group):
            tu.assign_non_tensor(data, batch_num_tokens=denominator)
            return loss_function(model_output=model_output, data=data, dp_group=dp_group)

        # Repack complete intervals within this optimizer minibatch. Both
        # passes use the same weights, followed by one ordinary optimizer step.
        rows = [None] * len(data)
        for indices, only_forward in ((zero_indices, True), (active_indices, False)):
            subset = tu.index_select_tensor_dict(data, indices)
            result = original(subset, None if only_forward else actor_loss, forward_only=only_forward)
            for index, scores in zip(indices, result["model_output"]["log_probs"].unbind(), strict=True):
                rows[index] = scores

        # verl averages KL and clip fractions per physical microbatch. Reuse
        # the original packing for that reduction; reuse detached log probs so
        # reporting never requires a second model forward.
        group = engine.get_data_parallel_group()
        tu.assign_non_tensor(
            data, batch_num_tokens=denominator, dp_size=1, sp_size=engine.ulysses_sequence_parallel_size
        )
        batches, partitions = prepare_micro_batches(data, dp_group=group)
        outputs = []
        with torch.no_grad():
            for batch, indices in zip(batches, partitions, strict=True):
                log_probs = torch.nested.as_nested_tensor([rows[i] for i in indices], layout=torch.jagged)
                model_output = {"log_probs": log_probs}
                loss, metrics = loss_function(
                    model_output=model_output, data=batch.to(get_device_id()), dp_group=group
                )
                outputs.append({"model_output": model_output, "loss": loss.item(), "metrics": metrics})
        return postprocess_batch_func(outputs, partitions, data)

    engine.forward_backward_batch = forward_backward_batch
