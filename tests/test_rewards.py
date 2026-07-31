import pytest

from self_summarization_agent.rewards import (
    apply_malformed_tool_penalty,
    apply_terminal_reward,
    trainable_turn_ids_from_records,
)
from self_summarization_agent.train_grpo import group_samples_by_query
from self_summarization_agent.trajectory import extract_trainable_samples, tokenize_interval_messages


def trajectory_record(record_id: str, query_id: str = "q1") -> dict:
    return {
        "schema_version": 1,
        "query_id": query_id,
        "turn_id": record_id,
        "kind": "trajectory",
        "termination_kind": "compaction",
        "messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "task state"},
            {"role": "assistant", "content": "reasoning and action"},
            {"role": "user", "content": "tool result"},
            {"role": "assistant", "content": "reasoning and summary"},
        ],
        "prompt": "debug transcript",
    }


def test_terminal_reward_trains_all_interval_records() -> None:
    rewards = apply_terminal_reward(
        outcome="correct_answer",
        trainable_turn_ids=["trajectory-1", "trajectory-2"],
    )

    assert rewards == {"trajectory-1": 1.0, "trajectory-2": 1.0}


def test_malformed_penalty_marks_all_completed_intervals() -> None:
    rewards = apply_malformed_tool_penalty(["trajectory-1", "trajectory-2"])

    assert rewards == {"trajectory-1": -1.0, "trajectory-2": -1.0}


def test_only_trajectory_records_are_reward_targets() -> None:
    records = [
        {"kind": "tool", "turn_id": "tool-1"},
        trajectory_record("trajectory-1"),
        {"kind": "summary", "turn_id": "summary-1"},
    ]

    assert trainable_turn_ids_from_records(records) == ["trajectory-1"]


def test_extract_trainable_samples_returns_one_sample_per_interval() -> None:
    records = [trajectory_record("trajectory-1"), trajectory_record("trajectory-2")]
    samples = extract_trainable_samples(
        records,
        {"trajectory-1": 1.0, "trajectory-2": -1.0},
        rollout_id="q1:0",
    )

    assert [sample.turn_id for sample in samples] == ["trajectory-1", "trajectory-2"]
    assert [sample.reward for sample in samples] == [1.0, -1.0]
    assert all(sample.rollout_id == "q1:0" for sample in samples)
    assert samples[0].completion == "reasoning and action\nreasoning and summary"


def test_extract_trainable_samples_preserves_query_grouping() -> None:
    records = [trajectory_record("trajectory-1", "q1"), trajectory_record("trajectory-2", "q2")]
    rewards = {"trajectory-1": 1.0, "trajectory-2": -1.0}

    grouped = group_samples_by_query(extract_trainable_samples(records, rewards))

    assert sorted(grouped) == ["q1", "q2"]


def test_extract_trainable_samples_rejects_non_system_prefix() -> None:
    record = trajectory_record("trajectory-1")
    record["messages"][0]["role"] = "user"

    with pytest.raises(ValueError, match="must begin with a system message"):
        extract_trainable_samples([record], {"trajectory-1": 1.0})


def test_extract_trainable_samples_rejects_missing_assistant_completion() -> None:
    record = trajectory_record("trajectory-1")
    record["messages"] = record["messages"][:2]

    with pytest.raises(ValueError, match="has no assistant completion"):
        extract_trainable_samples([record], {"trajectory-1": 1.0})


def test_extract_trainable_samples_rejects_old_per_turn_schema() -> None:
    old_record = {
        "query_id": "q1",
        "turn_id": "tool-1",
        "kind": "tool",
        "prompt": "old prompt",
        "completion": "old completion",
    }

    with pytest.raises(ValueError, match="Unknown trajectory record kind"):
        extract_trainable_samples([old_record], {"tool-1": 1.0})


def test_extract_trainable_samples_rejects_leftover_reward_ids() -> None:
    with pytest.raises(ValueError, match="Reward ids do not match any trajectory record"):
        extract_trainable_samples(
            [trajectory_record("trajectory-1")],
            {"trajectory-1": 1.0, "unused": -1.0},
        )


def test_interval_tokenization_masks_only_assistant_content() -> None:
    class WhitespaceTokenizer:
        chat_template = None

        def __init__(self) -> None:
            self.vocabulary: dict[str, int] = {}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            ids = []
            for token in text.split():
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary) + 1
                ids.append(self.vocabulary[token])
            return ids

    messages = trajectory_record("trajectory-1")["messages"]
    _input_ids, _labels, mask = tokenize_interval_messages(
        WhitespaceTokenizer(),
        messages,
        max_sequence_length=128,
        sample_id="trajectory-1",
    )

    assert sum(mask) == len("reasoning and action".split()) + len("reasoning and summary".split())


def test_interval_tokenization_never_left_truncates_system_prefix() -> None:
    class TinyTokenizer:
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(len(text.split())))

    with pytest.raises(ValueError, match="never left-truncated"):
        tokenize_interval_messages(
            TinyTokenizer(),
            trajectory_record("trajectory-1")["messages"],
            max_sequence_length=2,
            sample_id="trajectory-1",
        )
