from types import SimpleNamespace

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches

from self_summarization_agent.zero_advantage import install_zero_advantage_forward_only


def test_grouped_forward_preserves_adam_steps_and_original_microbatch_metrics(monkeypatch):
    monkeypatch.setattr("verl.utils.device.get_device_id", lambda: torch.device("cpu"))

    def run(optimized):
        weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        optimizer = torch.optim.AdamW([weight], lr=0.01)
        forward_only_counts = []
        history = []

        def loss_fn(model_output, data, dp_group):
            scores = model_output["log_probs"].values()
            loss = (scores * data["advantages"].values()).sum() / data["batch_num_tokens"]
            return loss, {"mean": scores.mean().detach().item()}

        def forward_backward(data, loss_function, forward_only=False):
            tu.assign_non_tensor(data, batch_num_tokens=int(data["loss_mask"].sum()), dp_size=1, sp_size=1)
            batches, indices = prepare_micro_batches(data)
            outputs = []
            if forward_only:
                forward_only_counts.append(len(data))
            with torch.set_grad_enabled(not forward_only):
                for batch in batches:
                    probs = batch["input_ids"] * weight.square()
                    out = {"log_probs": probs}
                    loss, metrics = (
                        loss_function(out, batch, None) if loss_function else (weight.new_zeros(()), {})
                    )
                    if not forward_only:
                        loss.backward()
                    outputs.append(
                        {
                            "model_output": {"log_probs": probs.detach()},
                            "loss": loss.item(),
                            "metrics": metrics,
                        }
                    )
            return postprocess_batch_func(outputs, indices, data)

        engine = SimpleNamespace(
            get_data_parallel_size=lambda: 1,
            get_data_parallel_group=lambda: None,
            ulysses_sequence_parallel_size=1,
            forward_backward_batch=forward_backward,
        )
        if optimized:
            install_zero_advantage_forward_only(engine)
        for advantages in [[1.0, 0.0, 0.0, -0.25], [0.0] * 4, [1.0] * 4]:
            optimizer.zero_grad()
            lens = (7, 5, 4, 2)
            fields = {
                "input_ids": [torch.arange(n, dtype=weight.dtype) + i for i, n in enumerate(lens)],
                "loss_mask": [torch.ones(n, dtype=torch.bool) for n in lens],
                "advantages": [torch.full((n,), a) for n, a in zip(lens, advantages, strict=True)],
            }
            data = TensorDict(
                {
                    key: torch.nested.as_nested_tensor(rows, layout=torch.jagged)
                    for key, rows in fields.items()
                },
                batch_size=[len(lens)],
            )
            tu.assign_non_tensor(data, max_token_len_per_gpu=8)
            result = engine.forward_backward_batch(data, loss_fn)
            history.append(
                (
                    weight.grad.clone(),
                    result["metrics"],
                    result["loss"],
                    result["model_output"]["log_probs"].values(),
                )
            )
            optimizer.step()
        return (weight.detach(), optimizer.state[weight], history), forward_only_counts

    actual, counts = run(True)
    expected, _ = run(False)
    torch.testing.assert_close(actual, expected)
    assert counts == [2]
