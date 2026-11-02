"""Auditable complete-group selection, preserving all raw interval evidence."""
from copy import deepcopy
from collections import Counter

from .dataset import SELECTION_VERSION, task_manifest
from ...trajectory import extract_trainable_samples, _extract_collection_tokens
from ...segment_returns import ordered_rollout_chains


def select_training_rows(rows, config):
    validate_attempt_keys(rows)
    rows = deepcopy(rows)
    available = {t["name"]: t for t in task_manifest(config.benchmark, split="train")}
    expected_attempts = set(range(config.training.group_size))
    groups, reasons = {}, {}
    seen = set()
    for row in rows:
        key = row["query_id"], row["rollout_index"]
        if key in seen:
            raise ValueError("Duplicate SWE task/attempt")
        seen.add(key)
        if key[0] not in available or type(key[1]) is not int or key[1] not in expected_attempts:
            raise ValueError("Unexpected SWE task/attempt")
        groups.setdefault(key[0], []).append(row)
        reason = None
        if row.get("benchmark") != "swe-dev" or row.get("reward_profile") != config.benchmark.reward_profile:
            raise ValueError("SWE reward/benchmark lineage mismatch")
        if row.get("swe_metadata", {}).get("task_digest") != available[key[0]]["checksum"]:
            raise ValueError("SWE task content changed")
        if row.get("status") == "history_rewrite_detected":
            reason = "history_rewrite_detected"
        elif not row.get("training_eligible"):
            reason = "infrastructure_or_verifier_failure"
        else:
            try:
                records = row["trajectory_records"]
                for record in records:
                    if _extract_collection_tokens(record, turn_id=record["turn_id"]) is None:
                        raise ValueError("Missing exact IDs")
                samples = extract_trainable_samples(records, row["turn_rewards"],
                    rollout_id=f"{key[0]}:{key[1]}")
                if not samples or not all(s.has_exact_collection_tokens for s in samples):
                    raise ValueError("No exact intervals")
                ordered_rollout_chains([row], samples, config.benchmark.reward_profile)
            except (ValueError, KeyError, TypeError) as exc:
                reason = "invalid_interval_chain: " + str(exc)
        reasons[key] = reason
    declared = rows[0]["swe_selection"]["task_ids"] if rows else []
    missing_groups = sorted(set(declared) - set(groups))
    selected = 0
    for query, group in groups.items():
        complete = {r["rollout_index"] for r in group} == expected_attempts
        eligible = complete and all(reasons[query, r["rollout_index"]] is None for r in group)
        selected += eligible
        for row in group:
            row["training_selection"] = dict(version=SELECTION_VERSION, selected=eligible,
                reward_profile=config.benchmark.reward_profile, group_size=config.training.group_size,
                reason=None if eligible else (reasons[query, row["rollout_index"]] or
                    ("incomplete_task_group" if not complete else "ineligible_peer_attempt")))
            if not eligible:
                row["trainable_sample_count"] = 0
    report = dict(version=SELECTION_VERSION, reward_profile=config.benchmark.reward_profile,
        raw_attempts=len(rows), eligible_attempts=sum(v is None for v in reasons.values()),
        task_groups=len(groups), selected_groups=selected, dropped_groups=len(groups)-selected + len(missing_groups),
        missing_task_groups=missing_groups,
        selected_attempts=selected * config.training.group_size,
        exclusions=dict(Counter(r["training_selection"]["reason"] for r in rows
                                if not r["training_selection"]["selected"])))
    return rows, report


def validate_attempt_keys(rows, *, complete=False):
    swe = [r for r in rows if r.get("benchmark") == "swe-dev"]
    if not swe:
        return True
    if len(swe) != len(rows):
        raise ValueError("Mixed benchmark attempt keys")
    declared = swe[0].get("swe_selection")
    if not isinstance(declared, dict) or declared.get("version") != 1:
        raise ValueError("Missing SWE resolved task selection")
    tasks, attempts = declared.get("task_ids"), declared.get("attempts")
    if (not isinstance(tasks, list) or len(tasks) != len(set(tasks))
            or type(attempts) is not int or attempts < 1):
        raise ValueError("Invalid SWE resolved task selection")
    expected = {(task, i) for task in tasks for i in range(attempts)}
    seen = set()
    for row in swe:
        key = row.get("query_id"), row.get("rollout_index")
        if row.get("swe_selection") != declared or key not in expected or key in seen:
            raise ValueError("Duplicate/unexpected SWE task/attempt or changed selection")
        seen.add(key)
    return seen == expected if complete else True
