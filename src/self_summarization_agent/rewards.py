from collections.abc import Iterable, Mapping
from typing import Literal


Outcome = Literal["correct_answer", "wrong_answer", "budget_exhausted"]
TRAINABLE_TURN_KINDS = {"tool", "summary", "final_answer"}


def trainable_turn_ids_from_records(turn_records: Iterable[Mapping[str, object]]) -> list[str]:
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
) -> dict[str, float]:
    if outcome not in {"correct_answer", "wrong_answer", "budget_exhausted"}:
        raise ValueError(f"Unknown terminal outcome: {outcome}")
    reward = 1.0 if outcome == "correct_answer" else -1.0
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
) -> dict[str, float]:
    if turn_ids is None:
        if turn_id is None:
            return {}
        turn_ids = turn_id
    if isinstance(turn_ids, str):
        return {turn_ids: -1.0}
    return {turn_id: -1.0 for turn_id in turn_ids}
