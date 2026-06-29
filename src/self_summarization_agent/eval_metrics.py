from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_summarization_agent.launcher_utils import append_jsonl, utc_timestamp


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid eval JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Eval row {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _token_usage(row: dict[str, Any]) -> dict[str, Any]:
    usage = row.get("token_usage")
    return usage if isinstance(usage, dict) else {}


def _tool_call_counts(row: dict[str, Any]) -> dict[str, Any]:
    counts = row.get("tool_call_counts")
    return counts if isinstance(counts, dict) else {}


def _numeric_sum(rows: list[dict[str, Any]], field: str) -> float:
    total = 0.0
    for row in rows:
        value = _token_usage(row).get(field, 0)
        if isinstance(value, int | float):
            total += float(value)
    return total


def _numeric_max(rows: list[dict[str, Any]], field: str) -> float:
    maximum = 0.0
    for row in rows:
        value = _token_usage(row).get(field, 0)
        if isinstance(value, int | float):
            maximum = max(maximum, float(value))
    return maximum


def _average(total: float, count: int) -> float:
    return total / count if count else 0.0


def _per_1k(correct: int, tokens: float) -> float:
    return correct / (tokens / 1000.0) if tokens > 0 else 0.0


def write_eval_metrics(
    *,
    judged_rollout_path: str | Path,
    metrics_path: str | Path,
    iteration: int,
    policy_checkpoint_id: str,
) -> dict[str, Any]:
    output = Path(metrics_path)
    if output.exists():
        for row in _load_rows(output):
            if row.get("iteration") == iteration and row.get("policy_checkpoint_id") == policy_checkpoint_id:
                return row

    rows = _load_rows(judged_rollout_path)
    correct = 0
    malformed = 0
    parse_errors = 0
    for index, row in enumerate(rows, start=1):
        judge = row.get("judge")
        if not isinstance(judge, dict):
            raise ValueError(f"Eval row {index} is missing judge payload")
        outcome = judge.get("outcome")
        if outcome == "correct_answer":
            correct += 1
        if outcome == "malformed_tool_call":
            malformed += 1
        if judge.get("parse_error"):
            parse_errors += 1

    total = len(rows)
    reasoning_tokens = _numeric_sum(rows, "reasoning_generated_tokens")
    summary_tokens = _numeric_sum(rows, "summary_generated_tokens")
    forced_answer_tokens = _numeric_sum(rows, "forced_answer_generated_tokens")
    tool_result_tokens = _numeric_sum(rows, "tool_result_tokens")
    total_generated_tokens = _numeric_sum(rows, "total_generated_tokens")
    max_prompt_tokens_sum = _numeric_sum(rows, "max_prompt_tokens_seen")
    summary_count = _numeric_sum(rows, "summary_count")
    search_calls = sum(
        float(_tool_call_counts(row).get("search", 0))
        for row in rows
        if isinstance(_tool_call_counts(row).get("search", 0), int | float)
    )
    document_calls = sum(
        float(_tool_call_counts(row).get("get_document", 0))
        for row in rows
        if isinstance(_tool_call_counts(row).get("get_document", 0), int | float)
    )
    record = {
        "iteration": iteration,
        "timestamp_utc": utc_timestamp(),
        "policy_checkpoint_id": policy_checkpoint_id,
        "eval_accuracy": correct / total if total else 0.0,
        "eval_correct": correct,
        "eval_total": total,
        "eval_malformed": malformed,
        "eval_parse_errors": parse_errors,
        "eval_reasoning_generated_tokens": reasoning_tokens,
        "eval_summary_generated_tokens": summary_tokens,
        "eval_forced_answer_generated_tokens": forced_answer_tokens,
        "eval_tool_result_tokens": tool_result_tokens,
        "eval_total_generated_tokens": total_generated_tokens,
        "eval_avg_reasoning_generated_tokens": _average(reasoning_tokens, total),
        "eval_avg_summary_generated_tokens": _average(summary_tokens, total),
        "eval_avg_forced_answer_generated_tokens": _average(forced_answer_tokens, total),
        "eval_avg_tool_result_tokens": _average(tool_result_tokens, total),
        "eval_avg_total_generated_tokens": _average(total_generated_tokens, total),
        "eval_avg_max_prompt_tokens_seen": _average(max_prompt_tokens_sum, total),
        "eval_max_prompt_tokens_seen": _numeric_max(rows, "max_prompt_tokens_seen"),
        "eval_avg_search_calls": _average(search_calls, total),
        "eval_avg_document_calls": _average(document_calls, total),
        "eval_avg_summary_count": _average(summary_count, total),
        "eval_correct_per_1k_reasoning_tokens": _per_1k(correct, reasoning_tokens),
        "eval_correct_per_1k_total_generated_tokens": _per_1k(correct, total_generated_tokens),
    }
    append_jsonl(output, record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write held-out evaluation metrics from judged rollout JSONL.")
    parser.add_argument("--rollouts", required=True, help="Judged eval rollout JSONL path.")
    parser.add_argument("--metrics", required=True, help="Output eval metrics JSONL path.")
    parser.add_argument("--iteration", type=int, required=True, help="Training iteration number.")
    parser.add_argument("--policy-checkpoint-id", required=True, help="Evaluated checkpoint id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = write_eval_metrics(
        judged_rollout_path=args.rollouts,
        metrics_path=args.metrics,
        iteration=args.iteration,
        policy_checkpoint_id=args.policy_checkpoint_id,
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
