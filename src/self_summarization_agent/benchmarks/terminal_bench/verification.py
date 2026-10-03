"""Verifier rewards are authoritative; protocol failures keep their penalty."""
from copy import deepcopy
import math

from self_summarization_agent.rewards import is_penalized_runtime_status, trainable_turn_ids_from_records
from self_summarization_agent.trajectory import extract_trainable_samples


def apply_verifier_reward(row):
    row = deepcopy(row)
    verifier = row.get("benchmark_verifier") or {}
    raw_reward = (verifier.get("rewards") or {}).get("reward")
    valid_reward = (isinstance(raw_reward, (int, float)) and not isinstance(raw_reward, bool)
                    and math.isfinite(raw_reward) and raw_reward in (0, 1))
    penalized = is_penalized_runtime_status(row.get("status"))
    infra = bool(verifier.get("exception")) or not valid_reward or row.get("status") == "infrastructure_error"
    if penalized and not verifier.get("exception"):
        reward, outcome = -1.0, row["status"]
    elif infra:
        reward, outcome = None, "infrastructure_error"
    else:
        reward = 1.0 if raw_reward == 1 else -1.0
        outcome = "correct_answer" if raw_reward == 1 else "wrong_answer"
    row["benchmark_passed"] = valid_reward and raw_reward == 1 and not infra
    row["training_eligible"] = reward is not None
    row["turn_rewards"] = ({turn_id: reward for turn_id in
                            trainable_turn_ids_from_records(row["trajectory_records"])}
                           if reward is not None else {})
    row["judge"] = dict(outcome=outcome, judge_prompt=None, judge_response=None,
                        parse_error=infra, source="harbor-verifier", raw_reward=raw_reward,
                        rollout_index=row.get("rollout_index"))
    row["trainable_sample_count"] = len(extract_trainable_samples(
        row["trajectory_records"], row["turn_rewards"],
        rollout_id=f"{row['query_id']}:{row['rollout_index']}")) if reward is not None else 0
    return row
