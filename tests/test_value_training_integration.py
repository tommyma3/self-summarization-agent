from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from self_summarization_agent.verl_value_worker import CompactionValueTrainingWorker


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.lm_head = torch.nn.Linear(3, 8)
        self.config = SimpleNamespace(hidden_size=3)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, logits_to_keep=0, use_cache=False):
        hidden = self.embedding(input_ids)
        if isinstance(logits_to_keep, list):
            hidden = hidden[:, logits_to_keep]
        elif logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return SimpleNamespace(logits=self.lm_head(hidden))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / "toy.pt")
        (Path(path) / "config.json").write_text("{}")


def make_worker(monkeypatch, checkpoint, head_lr):
    from verl.utils.torch_functional import get_constant_schedule_with_warmup

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    worker = CompactionValueTrainingWorker.__new__(CompactionValueTrainingWorker)
    worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=3))
    worker.device_name = "cpu"
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
    worker.engine = SimpleNamespace(
        module=model, optimizer=optimizer, lr_scheduler=scheduler,
        optimizer_config=SimpleNamespace(clip_grad=1.0), scaler=None,
        optimizer_step=lambda: optimizer.step(),
    )
    worker.set_loss_fn = lambda fn: setattr(worker, "test_loss_fn", fn)
    worker.configure_compaction_value(
        checkpoint_path=str(checkpoint), loss_coefficient=0.1, zero_initialize_head=False,
        head_learning_rate=head_lr, policy_loss_fn=lambda **kwargs: (kwargs["model_output"]["policy_loss"], {}),
    )
    return worker


@pytest.mark.parametrize("head_lr", [None, 1e-6, 1e-4])
def test_real_worker_configuration_and_joint_loss_metrics(monkeypatch, tmp_path, head_lr):
    worker = make_worker(monkeypatch, tmp_path, head_lr)
    model = worker.engine.module
    assert len(worker.engine.optimizer.param_groups) == (1 if head_lr is None else 2)
    hidden = model.embedding(torch.tensor([1, 2]))
    worker._pop_state_logits = lambda data: worker.value_head(hidden)
    policy_loss = model.lm_head(hidden).square().mean()
    loss, metrics = worker._compaction_actor_critic_loss(
        model_output={"policy_loss": policy_loss},
        data={"value_probe_mask": torch.zeros(2, dtype=torch.bool),
              "rewards": torch.tensor([1., 0.]), "value_weights": torch.tensor([0.5, 0.5]),
              "value_valid_mask": torch.ones(2, dtype=torch.bool), "dp_size": 1},
    )
    loss.backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    assert worker.value_head.projection.weight.grad.abs().sum() > 0
    worker.engine.optimizer_step()
    worker.engine.lr_scheduler.step()
    assert worker.engine.optimizer.param_groups[worker.value_head_optimizer_group]["lr"] == (head_lr or 1e-6)
    assert float(metrics["value/head_learning_rate"].aggregate()) == pytest.approx(head_lr or 1e-6)
    assert worker.engine.lr_scheduler.last_epoch == 1
    worker._value_capture_handle.remove()
