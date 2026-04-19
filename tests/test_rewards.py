import pytest

from self_summarization_agent.rewards import apply_malformed_tool_penalty, apply_terminal_reward
from self_summarization_agent.train_grpo import group_samples_by_query
from self_summarization_agent.trajectory import extract_trainable_samples


def test_terminal_correct_reward_trains_summary_and_answer_turns() -> None:
    rewards = apply_terminal_reward(outcome="correct_answer", summary_turn_ids=["s1", "s2"], final_answer_turn_id="a1")
    assert rewards == {"s1": 1.0, "s2": 1.0, "a1": 1.0}


def test_malformed_tool_penalty_only_marks_offending_turn() -> None:
    rewards = apply_malformed_tool_penalty(turn_id="tool-3")
    assert rewards == {"tool-3": -1.0}


def test_apply_terminal_reward_raises_on_invalid_outcome() -> None:
    with pytest.raises(ValueError, match="Unknown terminal outcome: invalid"):
        apply_terminal_reward(outcome="invalid", summary_turn_ids=["s1"], final_answer_turn_id="a1")  # type: ignore[arg-type]


def test_extract_trainable_samples_returns_summary_and_final_answer_turns() -> None:
    turns = [
        {"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"},
        {"query_id": "q1", "kind": "final_answer", "turn_id": "a1", "prompt": "p2", "completion": "c2"},
        {"kind": "tool", "turn_id": "t1"},
    ]
    rewards = {"s1": 1.0, "a1": -1.0}

    samples = extract_trainable_samples(turns, rewards)

    assert [sample.turn_id for sample in samples] == ["s1", "a1"]
    assert [sample.query_id for sample in samples] == ["q1", "q1"]
    assert [sample.reward for sample in samples] == [1.0, -1.0]
    assert [sample.trainable_kind for sample in samples] == ["summary", "final_answer"]


def test_extract_trainable_samples_preserves_query_id_for_grouping() -> None:
    turns = [
        {"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"},
        {"query_id": "q1", "kind": "final_answer", "turn_id": "a1", "prompt": "p2", "completion": "c2"},
        {"query_id": "q2", "kind": "summary", "turn_id": "s2", "prompt": "p3", "completion": "c3"},
    ]
    rewards = {"s1": 1.0, "a1": -1.0, "s2": 0.5}

    grouped = group_samples_by_query(extract_trainable_samples(turns, rewards))

    assert sorted(grouped) == ["q1", "q2"]
    assert [sample.turn_id for sample in grouped["q1"]] == ["s1", "a1"]
    assert [sample.turn_id for sample in grouped["q2"]] == ["s2"]


def test_extract_trainable_samples_raises_on_leftover_reward_ids() -> None:
    turns = [{"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"}]
    rewards = {"s1": 1.0, "unused": -1.0}

    with pytest.raises(ValueError, match="Reward ids do not match any turn: unused"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_malformed_turn_record() -> None:
    turns = [{"query_id": "q1", "kind": "summary", "turn_id": "s1", "completion": "c1"}]
    rewards = {"s1": 1.0}

    with pytest.raises(ValueError, match="Trainable turn record is missing required key: prompt"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_when_trainable_turn_has_no_reward() -> None:
    turns = [{"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"}]
    rewards = {}

    with pytest.raises(ValueError, match="Missing reward for trainable turn_id: s1"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_non_string_prompt_or_completion() -> None:
    turns = [
        {"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": 123, "completion": "c1"},
    ]
    rewards = {"s1": 1.0}

    with pytest.raises(ValueError, match="Trainable turn_id s1 has non-string prompt"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_non_string_completion() -> None:
    turns = [{"query_id": "q1", "kind": "final_answer", "turn_id": "a1", "prompt": "p2", "completion": ["bad"]}]
    rewards = {"a1": 1.0}

    with pytest.raises(ValueError, match="Trainable turn_id a1 has non-string completion"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_non_numeric_reward_value() -> None:
    turns = [{"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"}]
    rewards = {"s1": "bad"}

    with pytest.raises(ValueError, match="Trainable turn_id s1 has non-numeric reward"):
        extract_trainable_samples(turns, rewards)  # type: ignore[arg-type]


def test_extract_trainable_samples_raises_on_non_dict_turn_record() -> None:
    turns = ["not-a-dict"]  # type: ignore[list-item]
    rewards = {}

    with pytest.raises(ValueError, match="Turn record must be a mapping, got str"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_non_finite_reward_value() -> None:
    turns = [{"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"}]
    rewards = {"s1": float("nan")}

    with pytest.raises(ValueError, match="Trainable turn_id s1 has non-finite reward"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_ignores_non_trainable_tool_turn_without_payload() -> None:
    turns = [
        {"kind": "tool", "turn_id": "tool-1"},
        {"query_id": "q1", "kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"},
    ]
    rewards = {"tool-1": -1.0, "s1": 1.0}

    samples = extract_trainable_samples(turns, rewards)

    assert [sample.turn_id for sample in samples] == ["s1"]
    assert [sample.reward for sample in samples] == [1.0]


def test_extract_trainable_samples_keeps_known_non_trainable_reward_from_failing() -> None:
    turns = [{"kind": "tool", "turn_id": "tool-1"}]
    rewards = {"tool-1": -1.0}

    samples = extract_trainable_samples(turns, rewards)

    assert samples == []


def test_extract_trainable_samples_raises_when_reward_matches_no_turn() -> None:
    turns = [{"kind": "tool", "turn_id": "tool-1"}]
    rewards = {"missing": -1.0}

    with pytest.raises(ValueError, match="Reward ids do not match any turn: missing"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_unknown_turn_kind() -> None:
    turns = [{"kind": "assistant", "turn_id": "x1"}]
    rewards = {}

    with pytest.raises(ValueError, match="Unknown turn kind: assistant"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_non_string_turn_id() -> None:
    turns = [{"query_id": "q1", "kind": "summary", "turn_id": ["bad"], "prompt": "p1", "completion": "c1"}]
    rewards = {}

    with pytest.raises(ValueError, match=r"Turn record has non-string turn_id: \['bad'\]"):
        extract_trainable_samples(turns, rewards)  # type: ignore[list-item]


def test_extract_trainable_samples_raises_on_duplicate_trainable_turn_id() -> None:
    turns = [
        {"query_id": "q1", "kind": "summary", "turn_id": "dup", "prompt": "p1", "completion": "c1"},
        {"query_id": "q1", "kind": "final_answer", "turn_id": "dup", "prompt": "p2", "completion": "c2"},
    ]
    rewards = {"dup": 1.0}

    with pytest.raises(ValueError, match="Duplicate turn_id found: dup"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_tool_trainable_turn_id_collision() -> None:
    turns = [
        {"kind": "tool", "turn_id": "dup"},
        {"query_id": "q1", "kind": "summary", "turn_id": "dup", "prompt": "p1", "completion": "c1"},
    ]
    rewards = {"dup": 1.0}

    with pytest.raises(ValueError, match="Duplicate turn_id found: dup"):
        extract_trainable_samples(turns, rewards)


def test_extract_trainable_samples_raises_on_missing_query_id() -> None:
    turns = [{"kind": "summary", "turn_id": "s1", "prompt": "p1", "completion": "c1"}]
    rewards = {"s1": 1.0}

    with pytest.raises(ValueError, match="Trainable turn record is missing required key: query_id"):
        extract_trainable_samples(turns, rewards)
