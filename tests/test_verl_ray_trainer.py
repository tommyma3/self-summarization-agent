from pathlib import Path

import torch

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.trajectory import RLSample
from self_summarization_agent import verl_ray_trainer
from self_summarization_agent.verl_ray_trainer import (
    VerlRayPolicyTrainer,
    build_verl_actor_dataproto,
    build_verl_fsdp_worker_config,
)


class FakeDataProto:
    def __init__(self, payload, meta_info):
        self.batch = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        self.non_tensor_batch = {key: value for key, value in payload.items() if not isinstance(value, torch.Tensor)}
        self.meta_info = dict(meta_info)

    @classmethod
    def from_single_dict(cls, payload, meta_info=None):
        return cls(payload, meta_info or {})

    def __len__(self):
        if not self.batch:
            return 0
        return next(iter(self.batch.values())).shape[0]


class FakeNumpy:
    @staticmethod
    def array(values, dtype=None):
        return list(values)


def sample(turn_id: str, reward: float, reference_logprob: float) -> RLSample:
    return RLSample(
        query_id="q1",
        turn_id=turn_id,
        prompt="prompt",
        completion="completion",
        reward=reward,
        trainable_kind="tool",
        input_ids=[10, 11, 12],
        labels=[11, 12, 13],
        completion_mask=[False, True, True],
        reference_logprob=reference_logprob,
        reference_logprobs=[0.0, reference_logprob - 0.25, reference_logprob + 0.25],
    )


def patch_fake_dataproto(monkeypatch) -> None:
    monkeypatch.setattr(
        verl_ray_trainer,
        "_require_verl_ray",
        lambda: (FakeNumpy, object(), FakeDataProto),
    )


def test_build_verl_fsdp_worker_config_maps_repo_training_knobs() -> None:
    model_config = ModelConfig(model_path="/models/policy", dtype="bfloat16", trust_remote_code=True)
    training_config = TrainingConfig(
        backend="verl_ray",
        group_size=4,
        update_epochs=3,
        minibatch_size=8,
        gradient_accumulation_microbatch_size=2,
        learning_rate=2e-6,
        max_grad_norm=0.7,
        activation_checkpointing=True,
    )
    training_config.verl.worker_backend = "verl_fsdp"
    training_config.verl.fsdp.strategy = "fsdp2"
    training_config.verl.fsdp.ppo_micro_batch_size_per_gpu = 2
    training_config.verl.fsdp.ppo_max_token_len_per_gpu = 32768
    training_config.verl.fsdp.use_torch_compile = False

    config = build_verl_fsdp_worker_config(model_config, training_config)

    assert config["model"]["path"] == "/models/policy"
    assert config["model"]["trust_remote_code"] is True
    assert config["model"]["enable_gradient_checkpointing"] is True
    assert config["actor"]["strategy"] == "fsdp2"
    assert config["actor"]["ppo_mini_batch_size"] == 8
    assert config["actor"]["ppo_micro_batch_size_per_gpu"] == 2
    assert config["actor"]["ppo_max_token_len_per_gpu"] == 32768
    assert config["actor"]["clip_ratio"] == training_config.clip_range
    assert config["actor"]["optim"]["lr"] == 2e-6
    assert config["actor"]["optim"]["clip_grad"] == 0.7
    assert config["actor"]["checkpoint"]["save_contents"] == ["hf_model"]
    assert config["actor"]["fsdp_config"]["strategy"] == "fsdp2"
    assert config["actor"]["fsdp_config"]["use_torch_compile"] is False


def test_verl_fsdp_step_uses_native_worker_group(monkeypatch, tmp_path: Path) -> None:
    patch_fake_dataproto(monkeypatch)
    training_config = TrainingConfig(backend="verl_ray", group_size=2)
    training_config.verl.worker_backend = "verl_fsdp"

    class FakeWorkerGroup:
        def __init__(self):
            self.batch = None

        def update_actor(self, batch):
            self.batch = batch
            return {"loss": [0.2], "actor/ppo_kl": [0.03], "actor/pg_clipfrac": [0.4]}

    fake_worker = FakeWorkerGroup()
    trainer = VerlRayPolicyTrainer.__new__(VerlRayPolicyTrainer)
    trainer.model_config = ModelConfig(model_path=str(tmp_path))
    trainer.training_config = training_config
    trainer.checkpoint_id = "step-00001"
    trainer._ray = None
    trainer._actor = None
    trainer._worker_group = fake_worker
    trainer._owns_ray = False
    trainer._last_batch_meta = {}

    metrics = trainer.step({"q1": [sample("low", 0.0, -0.5), sample("high", 2.0, -0.25)]})

    assert fake_worker.batch is not None
    assert metrics.sample_count == 2
    assert metrics.loss == 0.2
    assert metrics.mean_policy_kl == 0.03
    assert metrics.clip_fraction == 0.4
    assert metrics.extra_metrics["verl_ray/worker_backend"] == "verl_fsdp"
    assert metrics.extra_metrics["verl_fsdp/contributing_sample_count"] == 2
