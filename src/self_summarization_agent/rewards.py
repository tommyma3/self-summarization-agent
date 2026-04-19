from typing import Literal


Outcome = Literal["correct_answer", "wrong_answer", "budget_exhausted"]


def apply_terminal_reward(
    outcome: Outcome,
    summary_turn_ids: list[str],
    final_answer_turn_id: str | None,
) -> dict[str, float]:
    if outcome not in {"correct_answer", "wrong_answer", "budget_exhausted"}:
        raise ValueError(f"Unknown terminal outcome: {outcome}")
    reward = 1.0 if outcome == "correct_answer" else -1.0
    assigned = {turn_id: reward for turn_id in summary_turn_ids}
    if final_answer_turn_id is not None:
        assigned[final_answer_turn_id] = reward
    return assigned


def apply_malformed_tool_penalty(turn_id: str) -> dict[str, float]:
    return {turn_id: -1.0}
