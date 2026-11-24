import torch

from self_summarization_agent.cache_scorer import _encode_shifted_sample_from_text
from self_summarization_agent.trainer import compute_group_advantages
from self_summarization_agent.trajectory import RLSample


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
