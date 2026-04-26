import torch

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.trainer import FSDP2ContextParallelPolicyTrainer
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
