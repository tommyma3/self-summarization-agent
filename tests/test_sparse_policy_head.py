from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("shards", [1, 4])
def test_sparse_policy_head_matches_packed_dense_gradient(shards, monkeypatch):
    from tensordict import TensorDict
    from transformers.modeling_outputs import BaseModelOutput
    from verl.models.transformers.qwen3_5 import forward_with_normal_backend
    from verl.utils import torch_functional, ulysses
    from self_summarization_agent.sparse_policy_head import install_sparse_policy_head

    # Mixed prompt lengths, a target crossing the shard boundary, an entirely
    # masked row, a masked forced completion suffix and a padded last shard.
    ids = torch.nested.as_nested_tensor([torch.arange(n) for n in [4, 5, 5]], layout=torch.jagged)
    mask = torch.nested.as_nested_tensor(
        [
            torch.tensor([False, True, False]),
            torch.tensor([True, False, True]),
            torch.tensor([False, False, False, False]),
        ],
        layout=torch.jagged,
    )
    batch = TensorDict({"input_ids": ids, "response_mask": mask, "advantages": mask}, batch_size=[3])
    predictors = torch.tensor([1, 5, 7])
    length = (14 + shards - 1) // shards
    torch.manual_seed(13)
    hidden = torch.randn(length * shards, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(7, (length * shards,))
    temperatures = torch.rand(length * shards, dtype=torch.float64) + 0.5

    def logprobs(logits, labels):
        return logits.log_softmax(-1).gather(1, labels[:, None]).flatten()

    dense = logprobs((hidden @ weight.T) / temperatures[:, None], targets)
    expected = torch.autograd.grad(dense[predictors].sum(), (hidden, weight))
    monkeypatch.setattr(torch_functional, "logprobs_from_logits", logprobs)
    monkeypatch.setattr(ulysses, "gather_outputs_and_unpad", lambda x, **kwargs: x)
    packed = []
    for rank in range(shards):
        monkeypatch.setattr(ulysses, "get_ulysses_sequence_parallel_rank", lambda: rank)
        local = slice(rank * length, (rank + 1) * length)
        model = SimpleNamespace()
        model.forward = forward_with_normal_backend.__get__(model)
        model.model = lambda **kwargs: BaseModelOutput(last_hidden_state=hidden[local].unsqueeze(0))
        model.lm_head = lambda x: x @ weight.T
        args = {
            "input_ids_rmpad_rolled": targets[local],
            "temperature_rmpad": temperatures[local],
            "pad_size": 2,
        }
        engine = SimpleNamespace(
            module=model,
            use_ulysses_sp=shards > 1,
            prepare_model_inputs=lambda batch: ({}, args),
            prepare_model_outputs=lambda *args: None,
        )
        install_sparse_policy_head(engine)
        inputs, args = engine.prepare_model_inputs(batch)
        output = model.forward(**inputs)
        # Inspect the local tensor before verl's tested SP gather.
        local_offsets = torch.tensor([0, length])
        local_batch = {"input_ids": torch.nested.nested_tensor_from_jagged(targets[local], local_offsets)}
        packed.append(engine.prepare_model_outputs(output, args, local_batch, None)["log_probs"].values())
    actual = torch.cat(packed)[predictors]
    torch.testing.assert_close(actual, dense[predictors])
    grads = torch.autograd.grad(actual.sum(), (hidden, weight))
    for grad, oracle in zip(grads, expected, strict=True):
        torch.testing.assert_close(grad, oracle)
