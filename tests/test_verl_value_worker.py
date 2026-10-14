import pytest
import torch

from self_summarization_agent.value_model import rollout_value_weights
from self_summarization_agent.verl_value_worker import CompactionValueTrainingWorker


def worker_with_logits():
    worker = CompactionValueTrainingWorker.__new__(CompactionValueTrainingWorker)
    worker._pop_state_logits = lambda data: data["logits"]
    worker.value_loss_coefficient = 0.7
    return worker


def test_probe_returns_stable_cross_entropy_without_policy_loss() -> None:
    worker = worker_with_logits()
    worker.policy_loss_fn = lambda **_: pytest.fail("Probe must not evaluate policy loss")
    output = {"log_probs": torch.zeros(2)}
    loss, metrics = worker._compaction_actor_critic_loss(
        model_output=output,
        data={"logits": torch.tensor([[0.0, 100.0], [0.0, 0.0]]),
              "value_probe_mask": torch.ones(2, dtype=torch.bool),
              "rewards": torch.tensor([-1.0, 1.0]), "sample_indices": torch.tensor([0, -1])},
    )
    assert loss == 0
    assert metrics == {}
    assert "log_probs" not in output
    # Expected reward saturates at +1; CE computed from logits still equals 100.
    assert output["compaction_values"][0].item() == 1.0
    assert output["compaction_value_losses"].flatten().tolist() == pytest.approx([100.0, 0.69314718])
    assert output["sample_indices"].shape == (2, 1)


@pytest.mark.parametrize("raises", [False, True])
def test_frozen_pass_disables_gradients_and_restores_head_mode(raises) -> None:
    worker = worker_with_logits()
    worker.value_head = torch.nn.Linear(2, 2)
    worker._captured_hidden_states = []

    def infer(_):
        assert not torch.is_grad_enabled()
        assert not worker.value_head.training
        if raises:
            raise RuntimeError("inference failed")
        return "predictions"

    worker.infer_batch = infer
    if raises:
        with pytest.raises(RuntimeError, match="inference failed"):
            worker.compute_compaction_values({})
    else:
        assert worker.compute_compaction_values({}) == "predictions"
    assert worker.value_head.training
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("dp_size,microbatch_size", [(1, 4), (1, 1), (2, 1), (2, 2)])
def test_actor_critic_gradients_match_minibatch_reference(dp_size, microbatch_size) -> None:
    # An optimizer minibatch containing only part of a multi-interval collection.
    # Test the actual worker loss against the full minibatch objective, including
    # DP averaging and uneven microbatches. No process group or GPU is needed.
    worker = worker_with_logits()
    from verl.trainer.ppo.core_algos import agg_loss

    def policy_loss_fn(*, model_output, data, dp_group):
        return agg_loss(data["actor_terms"], data["token_mask"], "token-mean",
                        dp_size=data["dp_size"], batch_num_tokens=data["batch_num_tokens"]), {}

    worker.policy_loss_fn = policy_loss_fn
    weights = torch.tensor(rollout_value_weights(["a", "b", "b", "b", "c", "c"])[:3] + [0.0])
    rewards = torch.tensor([-1.0, 1.0, 1.0, 1.0])
    valid = torch.tensor([True, True, True, False])
    token_mask = torch.tensor([[1, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.float32)
    features = torch.tensor([[1.0, 0.3], [-0.2, 1.2], [0.7, -0.5], [9.0, 9.0]])
    parameter = torch.tensor([0.3, -0.2], requires_grad=True)
    reference_parameter = parameter.detach().clone().requires_grad_()
    reference_logits = features * reference_parameter
    reference_ce = torch.nn.functional.cross_entropy(reference_logits, (rewards > 0).long(), reduction="none")
    reference_actor = (features * reference_parameter * token_mask).sum() / token_mask.sum()
    reference_value = (reference_ce * weights).sum()
    (reference_actor + 0.7 * reference_value).backward()

    mass = 0.0
    for rank in range(dp_size):
        rank_indices = list(range(rank, 4, dp_size))
        for start in range(0, len(rank_indices), microbatch_size):
            idx = rank_indices[start:start + microbatch_size]
            data = {"logits": features[idx] * parameter,
                    "actor_terms": features[idx] * parameter, "token_mask": token_mask[idx],
                    "batch_num_tokens": token_mask.sum(), "dp_size": dp_size,
                    "value_probe_mask": torch.zeros(len(idx), dtype=torch.bool),
                    "value_valid_mask": valid[idx], "rewards": rewards[idx],
                    "value_weights": weights[idx]}
            loss, metrics = worker._compaction_actor_critic_loss(model_output={}, data=data)
            (loss / dp_size).backward()  # simulated FSDP/head gradient averaging
            mass += float(metrics["value/weight_mass"].aggregate()) / dp_size
    torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    assert mass == pytest.approx(float(weights.sum()))
    assert mass < 1  # coefficient multiplies this collection mass, not a unit mean
