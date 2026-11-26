from collections.abc import Iterable, Mapping
from typing import Literal


BINARY_REWARD = "binary_zero_one"
VOLUNTARY_STOPPING = "voluntary_stopping"
CAP_ANSWER_REWARD = 0.9


def reward_bounds(profile=BINARY_REWARD):
    if profile not in (BINARY_REWARD, VOLUNTARY_STOPPING):
        raise ValueError(f"Unknown reward profile: {profile}")
    return 0.0, 1.0


def rollout_reward_values(profile=BINARY_REWARD) -> frozenset[float]:
    if profile == VOLUNTARY_STOPPING:
        return frozenset((0.0, CAP_ANSWER_REWARD, 1.0))
    return frozenset(reward_bounds(profile))


def read_rollout_reward(reward, profile=BINARY_REWARD):
    """Map historical failures to zero when reading saved reward labels."""
    return 0.0 if profile == BINARY_REWARD and reward == -1.0 else reward


Outcome = Literal["correct_answer", "wrong_answer", "budget_exhausted", "overlong"]
TRAINABLE_TURN_KINDS = {"trajectory"}
MASKED_ROLLOUT_STATUSES = frozenset({"malformed_tool_call", "empty_summary"})
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


def skip_rollout_samples(row: Mapping[str, object], *, for_value: bool) -> bool:
    return row.get("trainable_sample_count") == 0 or (for_value and row.get("status") in MASKED_ROLLOUT_STATUSES)


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


def artifact_reward_profile(row):
    profile = row.get("reward_profile", BINARY_REWARD)
    return BINARY_REWARD if profile == "signed_binary" else profile


def validate_reward_profile(row, expected):
    if artifact_reward_profile(row) != expected:
        raise ValueError("Rollout reward profile differs from the configured reward profile")


def correct_answer_reward(profile: str, forced_answer_reasons: Iterable[str]) -> float:
    _, success = reward_bounds(profile)
    if profile == VOLUNTARY_STOPPING and "max_compactions" in forced_answer_reasons:
        return CAP_ANSWER_REWARD
    return success


def apply_terminal_reward(
    outcome: Outcome,
    summary_turn_ids: list[str] | None = None,
    final_answer_turn_id: str | None = None,
    trainable_turn_ids: list[str] | None = None,
    reward_profile: str = BINARY_REWARD,
    forced_answer_reasons: Iterable[str] = (),
) -> dict[str, float]:
    if outcome not in {"correct_answer", "wrong_answer", "budget_exhausted", "overlong"}:
        raise ValueError(f"Unknown terminal outcome: {outcome}")
    if outcome == "correct_answer":
        reward = correct_answer_reward(reward_profile, forced_answer_reasons)
    else:
        reward, _ = reward_bounds(reward_profile)
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
