from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


class _Batch(dict):
    def select(self, *keys):
        return _Batch({key: self[key] for key in keys})

    def to_padded_tensor(self):
        return self


@pytest.fixture
def worker(monkeypatch):
    from self_summarization_agent import verl_separate_actor_worker

    monkeypatch.setattr(verl_separate_actor_worker, "no_padding_2_padding", lambda values, data: values)
    return verl_separate_actor_worker


def _batch(old, advantages, mask):
    return _Batch(old_log_probs=old, advantages=advantages, response_mask=mask,
                  batch_num_tokens=int(mask.sum()), dp_size=1)


@pytest.mark.parametrize("clip_ratio_c", [2.0, 3.0])
@pytest.mark.parametrize("clip_low,clip_high", [(None, None), (0.1, 0.4)])
def test_loss_and_gradients_match_verl(worker, clip_ratio_c, clip_low, clip_high):
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    old = torch.full((3, 7), -100.0)
    delta = torch.tensor([[-50., -1., -.1, .1, .4, 2., 50.]]).expand_as(old)
    advantages = torch.tensor([[1.], [-2.], [0.]]).expand_as(old)
    mask = torch.ones_like(old, dtype=torch.bool)
    mask[0, 0] = False
    config = OmegaConf.create({
        "clip_ratio": 0.2, "clip_ratio_low": clip_low, "clip_ratio_high": clip_high,
        "clip_ratio_c": clip_ratio_c,
        "global_batch_info": {"dp_size": 1, "batch_num_tokens": int(mask.sum())},
    })
    actual_logprobs = (old + delta).clone().requires_grad_()
    reference_logprobs = actual_logprobs.detach().clone().requires_grad_()
    actual, metrics = worker.fixed_advantage_ppo_loss(
        config, {"log_probs": actual_logprobs}, _batch(old, advantages, mask))
    reference, reference_metrics = compute_policy_loss_vanilla(
        old, reference_logprobs, advantages, mask, config=config)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, actual_logprobs)[0],
                               torch.autograd.grad(reference, reference_logprobs)[0])
    for key, expected in reference_metrics.items():
        assert float(metrics[key].aggregate()) == pytest.approx(expected)


@pytest.mark.parametrize("ratio,advantage,loss,clip,dual_clip", [
    (4.0, -1.0, 3.0, 0.0, 1.0),
    (1.3, -1.0, 1.3, 0.0, 0.0),
    (1.3, 1.0, -1.2, 1.0, 0.0),
    (0.5, -1.0, 0.8, 1.0, 0.0),
    (4.0, 0.0, 0.0, 0.0, 0.0),
])
def test_dual_clip_and_surrogate_clip_metrics(worker, ratio, advantage, loss, clip, dual_clip):
    old = torch.tensor([[-10.]])
    logprobs = (old + torch.tensor(ratio).log()).requires_grad_()
    advantages = torch.tensor([[advantage]])
    mask = torch.ones_like(old, dtype=torch.bool)
    actual, metrics = worker.fixed_advantage_ppo_loss(
        SimpleNamespace(clip_ratio=0.2), {"log_probs": logprobs}, _batch(old, advantages, mask))
    assert float(actual.detach()) == pytest.approx(loss)
    assert float(metrics["actor/pg_clipfrac"].aggregate()) == clip
    assert float(metrics["actor/pg_clipfrac_lower"].aggregate()) == dual_clip
    if dual_clip:
        torch.testing.assert_close(torch.autograd.grad(actual, logprobs)[0], torch.zeros_like(old))


def test_masked_nonfinite_padding_does_not_affect_loss_or_gradients(worker):
    logprobs = torch.tensor([[-10., float("nan"), float("inf")]], requires_grad=True)
    old = torch.full_like(logprobs, -11.)
    mask = torch.tensor([[True, False, False]])
    loss, metrics = worker.fixed_advantage_ppo_loss(
        SimpleNamespace(clip_ratio=0.2), {"log_probs": logprobs},
        _batch(old, -torch.ones_like(old), mask))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.as_tensor(metric.aggregate())) for metric in metrics.values())
    gradient = torch.autograd.grad(loss, logprobs)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[~mask], torch.zeros(2))


@pytest.mark.parametrize("clip_ratio_c", [0.0, 1.0, float("nan")])
def test_invalid_dual_clip_bound_is_rejected(worker, clip_ratio_c):
    with pytest.raises(ValueError, match="clip_ratio_c > 1"):
        worker.fixed_advantage_ppo_loss(SimpleNamespace(clip_ratio=0.2, clip_ratio_c=clip_ratio_c), {}, {})


def test_worker_config_pins_dual_clip_bound():
    from self_summarization_agent.config import ModelConfig, TrainingConfig
    from self_summarization_agent.verl_ray_trainer import build_verl_fsdp_worker_config

    config = build_verl_fsdp_worker_config(ModelConfig(model_path="unused"), TrainingConfig())
    assert config["actor"]["clip_ratio_c"] == 3.0
