import copy

import pytest
import torch

from self_summarization_agent.value_model import CompactionValueHead, rollout_normalized_value_loss
from self_summarization_agent.value_optimizer import attach_value_head, value_learning_rate_metrics


def joint_update(head_lr):
    torch.manual_seed(17)
    backbone = torch.nn.Linear(2, 2)
    head = CompactionValueHead(2, zero_initialize=False)
    initial_head = copy.deepcopy(head.state_dict())
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-3, weight_decay=0.01)
    group = attach_value_head(optimizer, head.parameters(), learning_rate=head_lr)
    hidden = backbone(torch.tensor([[1.0, -0.3], [-0.4, 0.8]]))
    value_loss = rollout_normalized_value_loss(head(hidden), torch.tensor([1., -1.]), torch.tensor([0.5, 0.5]))
    loss = hidden.square().mean() + 0.1 * value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([*backbone.parameters(), *head.parameters()], 1.0)
    optimizer.step()
    return backbone, head, initial_head, optimizer, group


def test_equal_lr_group_matches_shared_optimizer_and_higher_lr_isolates_first_update():
    shared = joint_update(None)
    separate = joint_update(1e-3)
    faster = joint_update(1e-2)
    assert len(shared[3].param_groups) == 1
    assert len(separate[3].param_groups) == 2
    for old, new in zip(shared[:2], separate[:2]):
        for p, q in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(p, q)
    for p, q in zip(shared[0].parameters(), faster[0].parameters()):
        torch.testing.assert_close(p, q)
    for name, p in separate[1].state_dict().items():
        delta = p - separate[2][name]
        faster_delta = faster[1].state_dict()[name] - faster[2][name]
        torch.testing.assert_close(faster_delta, delta * 10, atol=1e-7, rtol=1e-4)
    for group in separate[3].param_groups:
        assert group["weight_decay"] == 0.01
    params = [id(p) for group in separate[3].param_groups for p in group["params"]]
    assert len(params) == len(set(params))
    assert value_learning_rate_metrics(faster[3], faster[4]) == {
        "value/backbone_learning_rate": 1e-3, "value/head_learning_rate": 1e-2,
        "value/head_to_backbone_lr_ratio": 10,
    }


@pytest.mark.parametrize("warmup,steps_before", [(0, 0), (0, 3), (4, 0), (4, 2)])
def test_pinned_verl_scheduler_includes_new_group_without_reset(warmup, steps_before):
    from verl.utils.torch_functional import get_constant_schedule_with_warmup

    actor = torch.nn.Parameter(torch.ones(2))
    head = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([actor], lr=1e-6, weight_decay=0)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    for _ in range(steps_before):
        actor.grad = torch.ones_like(actor)
        optimizer.step()
        scheduler.step()
    epoch = scheduler.last_epoch
    actor_state = copy.deepcopy(optimizer.state[actor])
    group = attach_value_head(optimizer, [head], learning_rate=1e-4, scheduler=scheduler)
    assert scheduler.last_epoch == epoch
    assert group == 1
    for key in actor_state:
        torch.testing.assert_close(actor_state[key], optimizer.state[actor][key])
    assert scheduler.base_lrs == [1e-6, 1e-4]
    for _ in range(6):
        factor = min(1.0, scheduler.last_epoch / warmup) if warmup else 1.0
        assert scheduler.get_last_lr() == pytest.approx([1e-6 * factor, 1e-4 * factor])
        assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(scheduler.get_last_lr())
        actor.grad = torch.ones_like(actor)
        head.grad = torch.ones_like(head)
        optimizer.step()
        scheduler.step()
    assert optimizer.state[head]["step"] == 6
    assert optimizer.state[actor]["step"] == steps_before + 6


def test_unsupported_scheduler_fails_before_optimizer_mutation():
    actor, head = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([actor], lr=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with pytest.raises(ValueError, match="LambdaLR"):
        attach_value_head(optimizer, [head], learning_rate=1e-4, scheduler=scheduler)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["params"] == [actor]


def test_duplicate_head_is_rejected():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    with pytest.raises(ValueError, match="absent"):
        attach_value_head(optimizer, [parameter], learning_rate=1e-4)


def test_value_gradients_reach_backbone_and_head_while_policy_does_not_reach_head():
    torch.manual_seed(4)
    backbone = torch.nn.Linear(2, 2)
    head = CompactionValueHead(2, zero_initialize=False)
    features = torch.ones(2, 2)
    loss = rollout_normalized_value_loss(
        head(backbone(features)), torch.tensor([1., 1.]), torch.tensor([0.5, 0.5])
    )
    loss.backward()
    assert backbone.weight.grad.abs().sum() > 0
    assert head.projection.weight.grad.abs().sum() > 0
    backbone.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    backbone(features).square().mean().backward()
    assert all(p.grad is None for p in head.parameters())
