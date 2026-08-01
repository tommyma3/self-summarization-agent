import torch

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.trainer import (
    FSDP2ContextParallelPolicyTrainer,
    TransformersPolicyTrainer,
    _clipped_grpo_token_losses,
    _encode_shifted_sample_from_text,
    compute_group_advantages,
)
from self_summarization_agent.trajectory import RLSample


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
