import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from self_summarization_agent.config import CompactionValueConfig, ModelConfig, TrainingConfig
from self_summarization_agent.trainer import TransformersPolicyTrainer
from self_summarization_agent.trajectory import RLSample
from self_summarization_agent.value_model import VALUE_HEAD_FILENAME
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


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, *args, **kwargs):
        pytest.fail("Exact cached training tokens must never be retokenized")

    def save_pretrained(self, path):
        pass


def cached_samples():
    return [
        RLSample(query_id="q", turn_id=str(i), prompt="", completion="", reward=reward,
                 trainable_kind="summary" if i == 1 else "final_answer",
                 rollout_id=rollout, input_ids=[1, 2, 3], labels=[2, 3, 4],
                 completion_mask=[False, True, True], reference_logprob=-2.0,
                 reference_logprobs=[0, -2.0, -2.0], state_prefix_length=1,
                 state_kind="post_compaction" if i == 2 else "initial")
        for i, (rollout, reward) in enumerate([("a", -1.), ("b", 1.), ("b", 1.)])
    ]


def test_local_two_updates_reload_head_and_preserve_exact_samples(monkeypatch, tmp_path):
    from self_summarization_agent import trainer as module

    def load_model(path, **kwargs):
        model = TinyPolicy()
        if (Path(path) / "toy.pt").exists():
            model.load_state_dict(torch.load(Path(path) / "toy.pt", weights_only=True))
        return model

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    monkeypatch.setattr(module.AutoModelForMultimodalLM, "from_pretrained", load_model)
    monkeypatch.setattr(module, "configure_tokenizer_chat_template", lambda *a: None)
    config = TrainingConfig(
        advantage_estimator="compaction_mc_value", learning_rate=1e-6,
        value=CompactionValueConfig(enabled=True, head_learning_rate=1e-4),
        update_epochs=2, minibatch_size=2, gradient_accumulation_microbatch_size=1,
    )
    path = tmp_path
    previous_head = None
    samples = cached_samples()
    # A critic-only summary remains eligible and retains the rollout's shared reward.
    samples[1].completion_mask = [False] * 3
    samples[1].reference_logprob = 0.0
    before = copy.deepcopy(samples)
    for iteration in range(2):
        trainer = TransformersPolicyTrainer(ModelConfig(model_path=str(path), dtype="float32"), config)
        if previous_head is not None:
            assert trainer.value_head_loaded
            for name, value in trainer.value_head.state_dict().items():
                torch.testing.assert_close(value, previous_head[name])
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[1]["lr"] == 1e-4
        assert not trainer.optimizer.state  # preserve iteration-level reset behavior
        metrics = trainer.step({"q": samples})
        assert metrics.optimizer_step_count == 4
        assert metrics.extra_metrics["value/head_learning_rate"] == 1e-4
        assert metrics.extra_metrics["value/eval_weight_sum"] == pytest.approx(1)
        assert metrics.extra_metrics["value/eval_before/initial/state_count"] == 2
        assert metrics.extra_metrics["value/eval_after/post_compaction/state_count"] == 1
        assert samples == before
        path = tmp_path / f"iteration-{iteration}"
        path.mkdir()
        trainer.save_checkpoint(str(path))
        assert (path / VALUE_HEAD_FILENAME).is_file()
        previous_head = copy.deepcopy(trainer.value_head.state_dict())


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
              "rewards": torch.tensor([1., -1.]), "value_weights": torch.tensor([0.5, 0.5]),
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
