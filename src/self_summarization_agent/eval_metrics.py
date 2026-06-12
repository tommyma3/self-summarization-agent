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


def write_eval_metrics(
    *,
    judged_rollout_path: str | Path,
    metrics_path: str | Path,
    iteration: int,
    policy_checkpoint_id: str,
) -> dict[str, Any]:
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
    record = {
        "iteration": iteration,
        "timestamp_utc": utc_timestamp(),
        "policy_checkpoint_id": policy_checkpoint_id,
        "eval_accuracy": correct / total if total else 0.0,
        "eval_correct": correct,
        "eval_total": total,
        "eval_malformed": malformed,
        "eval_parse_errors": parse_errors,
    }
    append_jsonl(metrics_path, record)
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
