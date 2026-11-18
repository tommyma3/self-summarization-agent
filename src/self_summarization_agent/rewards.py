from collections.abc import Iterable, Mapping
from typing import Literal


BINARY_REWARD = "binary_zero_one"


def reward_bounds(profile=BINARY_REWARD):
    if profile != BINARY_REWARD:
        raise ValueError(f"Unknown reward profile: {profile}")
    return 0.0, 1.0


def rollout_reward_values(profile=BINARY_REWARD) -> frozenset[float]:
    return frozenset(reward_bounds(profile))


def read_rollout_reward(reward):
    """Map historical failures to zero when reading saved reward labels."""
    return 0.0 if reward == -1.0 else reward


Outcome = Literal["correct_answer", "wrong_answer", "budget_exhausted", "overlong"]
TRAINABLE_TURN_KINDS = {"trajectory"}
PENALIZED_RUNTIME_STATUSES = frozenset(
    {
        "malformed_tool_call",
        "empty_summary",
        "summary_length_exceeded",
        "context_length_exceeded",
        "history_rewrite_detected",
    }
)


def is_penalized_runtime_status(status: object) -> bool:
    return isinstance(status, str) and status in PENALIZED_RUNTIME_STATUSES


def has_runtime_reward(status: object) -> bool:
    return status == "overlong" or is_penalized_runtime_status(status)


def assign_runtime_rewards(
    status: object, trainable_turn_ids: list[str]
) -> dict[str, float] | None:
    if status == "overlong":
        return apply_terminal_reward("overlong", trainable_turn_ids=trainable_turn_ids)
    if is_penalized_runtime_status(status):
        return apply_malformed_tool_penalty(trainable_turn_ids)
    return None


def trainable_turn_ids_from_records(turn_records: Iterable[Mapping[str, object]]) -> list[str]:
    """Return trainable interval ids.

    The historical function name is retained at the artifact boundary, but
    ordinary per-invocation turn records are no longer trainable.
    """
    turn_ids: list[str] = []
    for turn in turn_records:
        if turn.get("kind") not in TRAINABLE_TURN_KINDS:
            continue
        turn_id = turn.get("turn_id")
        if isinstance(turn_id, str):
            turn_ids.append(turn_id)
    return turn_ids


def apply_terminal_reward(
    outcome: Outcome,
    summary_turn_ids: list[str] | None = None,
    final_answer_turn_id: str | None = None,
    trainable_turn_ids: list[str] | None = None,
    reward_profile: str = BINARY_REWARD,
) -> dict[str, float]:
    if outcome not in {"correct_answer", "wrong_answer", "budget_exhausted", "overlong"}:
        raise ValueError(f"Unknown terminal outcome: {outcome}")
    failure, success = reward_bounds(reward_profile)
    reward = success if outcome == "correct_answer" else failure
    if trainable_turn_ids is not None:
        return {turn_id: reward for turn_id in trainable_turn_ids}
    assigned = {turn_id: reward for turn_id in summary_turn_ids or []}
    if final_answer_turn_id is not None:
        assigned[final_answer_turn_id] = reward
    return assigned


def apply_malformed_tool_penalty(
    turn_ids: str | list[str] | None = None,
    *,
    turn_id: str | None = None,
    reward_profile: str = BINARY_REWARD,
) -> dict[str, float]:
    if turn_ids is None:
        if turn_id is None:
            return {}
        turn_ids = turn_id
    failure, _ = reward_bounds(reward_profile)
    if isinstance(turn_ids, str):
        return {turn_ids: failure}
    return {turn_id: failure for turn_id in turn_ids}
