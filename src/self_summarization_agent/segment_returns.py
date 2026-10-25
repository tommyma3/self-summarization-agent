"""Training-only interval links and undiscounted lambda returns.

Collection records and their exact token evidence are never modified here.
"""
from __future__ import annotations

import math

from self_summarization_agent.separate_value import sample_key


TERMINAL_KINDS = {"final_answer", "forced_answer", "malformed", "context_overflow"}


def ordered_rollout_chains(rows, samples):
    """Require complete authoritative interval chains for every selected rollout."""
    by_key = {sample_key(s): s for s in samples}
    if len(by_key) != len(samples):
        raise ValueError("Duplicate interval keys in return construction")
    selected = {key[0] for key in by_key}
    seen_rollouts, seen_keys, chains = set(), set(), []
    for row in rows:
        rollout_id = f"{row.get('query_id')}:{row.get('rollout_index')}"
        if rollout_id in seen_rollouts:
            raise ValueError("Duplicate rollout in return construction")
        seen_rollouts.add(rollout_id)
        if rollout_id not in selected:
            continue
        if row.get("status") == "history_rewrite_detected":
            raise ValueError("Cannot build returns across a history rewrite")
        records = row.get("trajectory_records")
        if not isinstance(records, list) or not records:
            raise ValueError("Missing interval chain")
        chain = []
        for index, record in enumerate(records):
            key = (rollout_id, record.get("turn_id"))
            segment = record.get("segment_index")
            if (key not in by_key or key in seen_keys or isinstance(segment, bool)
                    or not isinstance(segment, int) or segment != index + 1):
                raise ValueError("Missing, duplicate, or out-of-order interval in return chain")
            sample = by_key[key]
            reward = row.get("turn_rewards", {}).get(key[1])
            if (sample.query_id != row.get("query_id") or record.get("query_id") != sample.query_id
                    or isinstance(reward, bool) or reward not in (-1, 1) or reward != sample.reward):
                raise ValueError("Inconsistent rollout identity or reward in return chain")
            if "prefix_summary_turn_id" not in record:
                raise ValueError("Missing compaction linkage metadata")
            if index == 0:
                if record["prefix_summary_turn_id"] is not None:
                    raise ValueError("Initial interval has a predecessor summary")
            else:
                previous_turns = records[index - 1].get("turn_ids")
                if (not isinstance(previous_turns, list) or not previous_turns
                        or not isinstance(record["prefix_summary_turn_id"], str)
                        or not record["prefix_summary_turn_id"]
                        or record["prefix_summary_turn_id"] != previous_turns[-1]):
                    raise ValueError("Broken compaction successor linkage")
            terminal = index == len(records) - 1
            kind = record.get("termination_kind")
            if (terminal and kind not in TERMINAL_KINDS) or (not terminal and kind != "compaction"):
                raise ValueError("Incomplete or inconsistent terminal interval chain")
            chain.append({"key": list(key), "segment_index": segment, "termination_kind": kind,
                          "successor": None if terminal else [rollout_id, records[index + 1].get("turn_id")],
                          "reward": float(reward), "terminal": terminal})
            seen_keys.add(key)
        if len({item["reward"] for item in chain}) != 1:
            raise ValueError("Inconsistent terminal rewards within rollout")
        chains.append(chain)
    if seen_keys != set(by_key):
        raise ValueError("Return chains do not cover exactly the selected interval keys")
    return chains


def segment_gae(chains, old_values, gae_lambda):
    """Return rows keyed by interval; gamma=1 and terminal reward is counted once."""
    if (isinstance(gae_lambda, bool) or not isinstance(gae_lambda, (int, float))
            or not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1):
        raise ValueError("GAE lambda must be finite and in [0, 1]")
    keys = [tuple(item["key"]) for chain in chains for item in chain]
    if len(set(keys)) != len(keys) or set(keys) != set(old_values):
        raise ValueError("Old values must cover exactly the interval chains")
    if any(not math.isfinite(v) or not -1 <= v <= 1 for v in old_values.values()):
        raise ValueError("Old values must be finite and in [-1, 1]")
    result = {}
    for chain in chains:
        next_value, next_advantage = 0.0, 0.0
        for item in reversed(chain):
            key = tuple(item["key"])
            value = old_values[key]
            reward = item["reward"] if item["terminal"] else 0.0
            delta = reward + next_value - value
            advantage = delta + gae_lambda * next_advantage
            target = value + advantage
            if not math.isfinite(target) or abs(target) > 1 + 1e-12:
                raise ValueError("Invalid lambda-return target")
            target = min(1.0, max(-1.0, target))  # cancellation roundoff only
            result[key] = {"key": list(key), "value": value, "target": target,
                           "advantage": target - value, "transition_reward": reward}
            next_value, next_advantage = value, advantage
    return result
