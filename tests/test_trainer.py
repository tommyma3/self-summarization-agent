from types import SimpleNamespace

import pytest
import torch

from self_summarization_agent.config import CompactionValueConfig, ModelConfig, TrainingConfig
from self_summarization_agent.trainer import (
    FSDP2ContextParallelPolicyTrainer,
    TransformersPolicyTrainer,
    _clipped_grpo_token_losses,
    _encode_shifted_sample_from_text,
    compute_group_advantages,
)
from self_summarization_agent.trajectory import RLSample
from self_summarization_agent.value_model import CompactionValueHead


class FakeAccelerator:
    device = torch.device("cpu")


class FakeTokenizer:
    pad_token_id = 99
    eos_token_id = 100

    def __init__(self, prompt_ids: list[int], full_ids: list[int]) -> None:
        self.prompt_ids = prompt_ids
        self.full_ids = full_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "prompt":
            return self.prompt_ids
        return self.full_ids


def make_fsdp_trainer(tokenizer: FakeTokenizer, *, context_parallel_size: int):
    trainer = FSDP2ContextParallelPolicyTrainer.__new__(FSDP2ContextParallelPolicyTrainer)
    trainer.model_config = ModelConfig()
    trainer.training_config = TrainingConfig(context_parallel_size=context_parallel_size)
    trainer.tokenizer = tokenizer
    trainer.accelerator = FakeAccelerator()
    return trainer


def make_sample() -> RLSample:
    return RLSample(
        query_id="q1",
        turn_id="summary-1",
        prompt="prompt",
        completion=" completion",
        reward=1.0,
        trainable_kind="summary",
    )


def make_rewarded_sample(query_id: str, turn_id: str, reward: float) -> RLSample:
    return RLSample(
        query_id=query_id,
        turn_id=turn_id,
        prompt="prompt",
        completion=f" completion {turn_id}",
        reward=reward,
        trainable_kind="final_answer",
    )


def make_cached_rewarded_sample(query_id: str, turn_id: str, reward: float, reference_logprob: float) -> RLSample:
    return RLSample(
        query_id=query_id,
        turn_id=turn_id,
        prompt="prompt",
        completion=f" completion {turn_id}",
        reward=reward,
        trainable_kind="final_answer",
        input_ids=[1, 2],
        labels=[2, 3],
        completion_mask=[False, True],
        reference_logprob=reference_logprob,
    )


def test_group_advantage_is_computed_once_per_rollout_and_broadcast_to_intervals() -> None:
    low_first = make_rewarded_sample("q1", "low-1", 0.0)
    low_second = make_rewarded_sample("q1", "low-2", 0.0)
    high = make_rewarded_sample("q1", "high-1", 2.0)
    low_first.rollout_id = low_second.rollout_id = "q1:0"
    high.rollout_id = "q1:1"

    advantages = compute_group_advantages([low_first, low_second, high])

    assert advantages[0] == advantages[1]
    assert advantages[0] < 0
    assert advantages[2] > 0


def test_fsdp_context_parallel_encoding_pads_to_required_multiple() -> None:
    trainer = make_fsdp_trainer(
        FakeTokenizer(prompt_ids=[1, 2, 3, 4, 5], full_ids=list(range(17))),
        context_parallel_size=6,
    )

    input_ids, labels, completion_mask = trainer._encode_shifted_sample(make_sample())

    assert input_ids.shape == labels.shape == completion_mask.shape == (1, 24)
    assert input_ids.shape[1] % 12 == 0
    assert input_ids[0, 16:].tolist() == [99] * 8
    assert labels[0, 16:].tolist() == [99] * 8
    assert completion_mask[0, 4:16].all()
    assert not completion_mask[0, 16:].any()


def test_fsdp_context_parallel_encoding_leaves_aligned_sequence_unpadded() -> None:
    trainer = make_fsdp_trainer(
        FakeTokenizer(prompt_ids=[1, 2, 3], full_ids=list(range(13))),
        context_parallel_size=6,
    )

    input_ids, labels, completion_mask = trainer._encode_shifted_sample(make_sample())

    assert input_ids.shape == labels.shape == completion_mask.shape == (1, 12)
    assert input_ids[0, -1].item() == 11
    assert labels[0, -1].item() == 12


def test_exact_collection_ids_are_authoritative_and_not_retokenized() -> None:
    class RejectingTokenizer:
        pad_token_id = 99
        eos_token_id = 100

        def encode(self, *args, **kwargs):
            raise AssertionError("exact collection samples must not be retokenized")

    sample = make_sample()
    sample.collection_full_token_ids = [7, 8, 9, 10]
    sample.collection_assistant_token_mask = [False, False, True, True]

    input_ids, labels, completion_mask = _encode_shifted_sample_from_text(
        sample,
        tokenizer=RejectingTokenizer(),
        device=torch.device("cpu"),
        max_sequence_length=16,
    )

    assert input_ids.tolist() == [7, 8, 9]
    assert labels.tolist() == [8, 9, 10]
    assert completion_mask.tolist() == [False, True, True]


def test_transformers_trainer_reuses_batch_for_clipped_grpo_updates() -> None:
    feature_by_turn = {
        "q1-good": 1.0,
        "q1-bad": 2.0,
        "q2-good": 3.0,
        "q2-bad": 4.0,
    }
    batch_sizes = []

    class FakeBatchedTrainer(TransformersPolicyTrainer):
        def _sequence_token_logprobs_and_mask(self, samples: list[RLSample]) -> tuple[torch.Tensor, torch.Tensor]:
            batch_sizes.append(len(samples))
            features = torch.tensor([[feature_by_turn[sample.turn_id]] for sample in samples], dtype=torch.float32)
            return self.model(features), torch.ones((len(samples), 1), dtype=torch.bool)

        def _model_device(self) -> torch.device:
            return torch.device("cpu")

    trainer = FakeBatchedTrainer.__new__(FakeBatchedTrainer)
    trainer.training_config = TrainingConfig(
        update_epochs=3,
        minibatch_size=2,
        gradient_accumulation_microbatch_size=2,
        clip_range=0.2,
    )
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    grouped_samples = {
        "q1": [
            make_rewarded_sample("q1", "q1-good", 1.0),
            make_rewarded_sample("q1", "q1-bad", 0.0),
        ],
        "q2": [
            make_rewarded_sample("q2", "q2-good", 1.0),
            make_rewarded_sample("q2", "q2-bad", 0.0),
        ],
    }

    metrics = trainer.step(grouped_samples)

    assert metrics.sample_count == 4
    assert metrics.optimizer_step_count == 6
    assert metrics.loss != 0.0
    assert 0.0 <= metrics.clip_fraction <= 1.0
    assert batch_sizes == [2, 2, 2, 2, 2, 2, 2, 2]


def test_transformers_trainer_accumulates_microbatches_within_minibatch() -> None:
    feature_by_turn = {
        "q1-good": 1.0,
        "q1-bad": 2.0,
        "q2-good": 3.0,
        "q2-bad": 4.0,
    }
    batch_sizes = []

    class FakeBatchedTrainer(TransformersPolicyTrainer):
        def _sequence_token_logprobs_and_mask(self, samples: list[RLSample]) -> tuple[torch.Tensor, torch.Tensor]:
            batch_sizes.append(len(samples))
            features = torch.tensor([[feature_by_turn[sample.turn_id]] for sample in samples], dtype=torch.float32)
            return self.model(features), torch.ones((len(samples), 1), dtype=torch.bool)

        def _model_device(self) -> torch.device:
            return torch.device("cpu")

    trainer = FakeBatchedTrainer.__new__(FakeBatchedTrainer)
    trainer.training_config = TrainingConfig(
        update_epochs=1,
        minibatch_size=4,
        gradient_accumulation_microbatch_size=1,
        clip_range=0.2,
    )
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    grouped_samples = {
        "q1": [
            make_rewarded_sample("q1", "q1-good", 1.0),
            make_rewarded_sample("q1", "q1-bad", 0.0),
        ],
        "q2": [
            make_rewarded_sample("q2", "q2-good", 1.0),
            make_rewarded_sample("q2", "q2-bad", 0.0),
        ],
    }

    metrics = trainer.step(grouped_samples)

    assert metrics.sample_count == 4
    assert metrics.optimizer_step_count == 1
    assert batch_sizes == [1, 1, 1, 1, 1, 1, 1, 1]


def test_transformers_trainer_uses_cached_reference_logprobs() -> None:
    feature_by_turn = {
        "q1-good": 1.0,
        "q1-bad": 2.0,
        "q2-good": 3.0,
        "q2-bad": 4.0,
    }
    batch_sizes = []

    class FakeBatchedTrainer(TransformersPolicyTrainer):
        def _sequence_token_logprobs_and_mask(self, samples: list[RLSample]) -> tuple[torch.Tensor, torch.Tensor]:
            batch_sizes.append(len(samples))
            features = torch.tensor([[feature_by_turn[sample.turn_id]] for sample in samples], dtype=torch.float32)
            return self.model(features), torch.ones((len(samples), 1), dtype=torch.bool)

        def _model_device(self) -> torch.device:
            return torch.device("cpu")

    trainer = FakeBatchedTrainer.__new__(FakeBatchedTrainer)
    trainer.training_config = TrainingConfig(
        update_epochs=3,
        minibatch_size=2,
        gradient_accumulation_microbatch_size=2,
        clip_range=0.2,
    )
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    grouped_samples = {
        "q1": [
            make_cached_rewarded_sample("q1", "q1-good", 1.0, -0.5),
            make_cached_rewarded_sample("q1", "q1-bad", 0.0, -0.5),
        ],
        "q2": [
            make_cached_rewarded_sample("q2", "q2-good", 1.0, -0.5),
            make_cached_rewarded_sample("q2", "q2-bad", 0.0, -0.5),
        ],
    }

    metrics = trainer.step(grouped_samples)

    assert metrics.sample_count == 4
    assert metrics.optimizer_step_count == 6
    assert batch_sizes == [2, 2, 2, 2, 2, 2]


def test_token_grpo_loss_aligns_reference_logprobs_to_microbatch_length() -> None:
    logprobs = torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32)
    short_reference = torch.tensor([[-0.1, -0.2]], dtype=torch.float32)
    long_reference = torch.tensor([[-0.1, -0.2, -0.3, -9.0]], dtype=torch.float32)
    advantages = torch.tensor([1.0], dtype=torch.float32)
    completion_mask = torch.tensor([[True, True, True]])

    short_losses, _, _, short_mask = _clipped_grpo_token_losses(
        logprobs,
        short_reference,
        advantages,
        completion_mask,
        clip_range=0.2,
    )
    long_losses, _, _, long_mask = _clipped_grpo_token_losses(
        logprobs,
        long_reference,
        advantages,
        completion_mask,
        clip_range=0.2,
    )

    assert short_losses.shape == logprobs.shape
    assert long_losses.shape == logprobs.shape
    assert short_mask.shape == logprobs.shape
    assert long_mask.shape == logprobs.shape


def test_compaction_value_step_freezes_old_values_across_update_epochs() -> None:
    feature_by_turn = {"positive-1": 1.0, "positive-2": 2.0, "negative": 3.0}

    class FakeActorCriticTrainer(TransformersPolicyTrainer):
        old_value_calls = 0

        def _model_device(self) -> torch.device:
            return torch.device("cpu")

        def _actor_critic_outputs(self, samples: list[RLSample]):
            features = torch.tensor(
                [[feature_by_turn[sample.turn_id]] for sample in samples], dtype=torch.float32
            )
            hidden = self.model(features)
            return hidden, torch.ones_like(hidden, dtype=torch.bool), self.value_head(hidden)

        def _value_logits(self, samples: list[RLSample]):
            features = torch.tensor(
                [[feature_by_turn[sample.turn_id]] for sample in samples], dtype=torch.float32
            )
            return self.value_head(self.model(features))

        def _old_values(self, samples: list[RLSample], *, microbatch_size: int):
            self.old_value_calls += 1
            return super()._old_values(samples, microbatch_size=microbatch_size)

    def cached(turn_id: str, reward: float, rollout_id: str) -> RLSample:
        return RLSample(
            query_id="q1",
            turn_id=turn_id,
            prompt="",
            completion="",
            reward=reward,
            trainable_kind="trajectory",
            rollout_id=rollout_id,
            input_ids=[1],
            labels=[2],
            completion_mask=[True],
            reference_logprob=0.0,
            reference_logprobs=[0.0],
            state_prefix_length=1,
        )

    trainer = FakeActorCriticTrainer.__new__(FakeActorCriticTrainer)
    trainer.training_config = TrainingConfig(
        advantage_estimator="compaction_mc_value",
        value=CompactionValueConfig(enabled=True),
        update_epochs=2,
        minibatch_size=2,
        gradient_accumulation_microbatch_size=1,
        target_kl=None,
    )
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(trainer.model.weight)
    trainer.value_head = CompactionValueHead(1, zero_initialize=True)
    trainer.value_head_loaded = False
    trainer.optimizer = torch.optim.SGD(
        [*trainer.model.parameters(), *trainer.value_head.parameters()], lr=0.1
    )
    grouped = {
        "q1": [
            cached("positive-1", 1.0, "q1:0"),
            cached("positive-2", 1.0, "q1:0"),
            cached("negative", -1.0, "q1:1"),
        ]
    }

    metrics = trainer.step(grouped)

    assert trainer.old_value_calls == 1
    assert metrics.optimizer_step_count == 4
    assert metrics.mean_advantage == pytest.approx(1 / 3)
    assert metrics.extra_metrics["value/rollout_count"] == 2
    assert metrics.extra_metrics["value/state_count"] == 3
    assert any(torch.count_nonzero(parameter) for parameter in trainer.value_head.parameters())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_compaction_value_step_handles_split_policy_value_devices() -> None:
    feature_by_turn = {"positive": 1.0, "negative": -1.0}

    class SplitDeviceActorCriticTrainer(TransformersPolicyTrainer):
        def _model_device(self) -> torch.device:
            return torch.device("cuda:0")

        def _actor_critic_outputs(self, samples: list[RLSample]):
            features = torch.tensor(
                [[feature_by_turn[sample.turn_id]] for sample in samples],
                dtype=torch.float32,
                device="cuda:0",
            )
            hidden = self.model(features)
            return (
                hidden,
                torch.ones_like(hidden, dtype=torch.bool, device="cuda:0"),
                self.value_head(hidden.to("cuda:1")),
            )

        def _value_logits(self, samples: list[RLSample]):
            features = torch.tensor(
                [[feature_by_turn[sample.turn_id]] for sample in samples],
                dtype=torch.float32,
                device="cuda:0",
            )
            return self.value_head(self.model(features).to("cuda:1"))

    def cached(turn_id: str, reward: float, rollout_id: str) -> RLSample:
        return RLSample(
            query_id="q1",
            turn_id=turn_id,
            prompt="",
            completion="",
            reward=reward,
            trainable_kind="trajectory",
            rollout_id=rollout_id,
            input_ids=[1],
            labels=[2],
            completion_mask=[True],
            reference_logprob=0.0,
            reference_logprobs=[0.0],
            state_prefix_length=1,
        )

    trainer = SplitDeviceActorCriticTrainer.__new__(SplitDeviceActorCriticTrainer)
    trainer.training_config = TrainingConfig(
        advantage_estimator="compaction_mc_value",
        value=CompactionValueConfig(enabled=True),
        update_epochs=1,
        minibatch_size=2,
        gradient_accumulation_microbatch_size=1,
        target_kl=None,
    )
    trainer.model = torch.nn.Linear(1, 1, bias=False).to("cuda:0")
    trainer.value_head = CompactionValueHead(1, zero_initialize=True).to("cuda:1")
    trainer.value_head_loaded = False
    trainer.optimizer = torch.optim.SGD(
        [*trainer.model.parameters(), *trainer.value_head.parameters()], lr=0.1
    )
    grouped = {
        "q1": [
            cached("positive", 1.0, "q1:0"),
            cached("negative", -1.0, "q1:1"),
        ]
    }

    metrics = trainer.step(grouped)

    assert metrics.optimizer_step_count == 1
    assert "value/loss" in metrics.extra_metrics
    assert any(torch.count_nonzero(parameter) for parameter in trainer.value_head.parameters())


def test_compaction_value_forward_selects_only_anchor_and_trainable_logits() -> None:
    class SelectedLogitModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 16, bias=False)
            self.last_positions = None
            self.last_input_length = None

        @property
        def device(self):
            return self.embedding.weight.device

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, logits_to_keep=0, use_cache=False):
            del use_cache
            self.last_input_length = input_ids.shape[1]
            hidden = self.embedding(input_ids)
            positions = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            self.last_positions = logits_to_keep
            return SimpleNamespace(logits=self.lm_head(hidden[:, positions, :]))

    sample = RLSample(
        query_id="q1",
        turn_id="trajectory-1",
        prompt="",
        completion="",
        reward=1.0,
        trainable_kind="trajectory",
        rollout_id="q1:0",
        input_ids=[1, 2, 3, 4],
        labels=[2, 3, 4, 5],
        completion_mask=[False, True, False, True],
        reference_logprob=0.0,
        reference_logprobs=[0.0, 0.0, 0.0, 0.0],
        state_prefix_length=1,
    )
    trainer = TransformersPolicyTrainer.__new__(TransformersPolicyTrainer)
    trainer.training_config = TrainingConfig(
        advantage_estimator="compaction_mc_value",
        value=CompactionValueConfig(enabled=True),
    )
    trainer.model = SelectedLogitModel()
    trainer.value_head = CompactionValueHead(4, zero_initialize=False)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)

    token_logprobs, completion_mask, value_logits = trainer._actor_critic_outputs([sample])

    assert trainer.model.last_input_length == 4
    assert trainer.model.last_positions == [0, 1, 3]
    assert token_logprobs.shape == completion_mask.shape == (1, 4)
    assert token_logprobs[0, 0].item() == 0.0
    assert token_logprobs[0, 2].item() == 0.0
    assert value_logits.shape == (1, 2)
    full_hidden = trainer.model.embedding(torch.tensor([sample.input_ids]))
    full_logits = trainer.model.lm_head(full_hidden)
    expected_logprobs = -torch.nn.functional.cross_entropy(
        full_logits.reshape(-1, full_logits.shape[-1]),
        torch.tensor(sample.labels),
        reduction="none",
    ).reshape(1, -1) * torch.tensor([sample.completion_mask])
    expected_value_logits = trainer.value_head(full_hidden[:, sample.state_prefix_length - 1])
    assert torch.allclose(token_logprobs, expected_logprobs)
    assert torch.allclose(value_logits, expected_value_logits)

    trainer._value_logits([sample])
    assert trainer.model.last_input_length == 1
    assert trainer.model.last_positions == 1
