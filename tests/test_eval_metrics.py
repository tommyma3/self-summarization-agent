import json
from pathlib import Path

from self_summarization_agent.eval_metrics import write_eval_metrics


def test_write_eval_metrics_counts_judged_eval_rollouts(tmp_path: Path) -> None:
    judged_rollouts = tmp_path / "eval.jsonl"
    rows = [
        {"judge": {"outcome": "correct_answer", "parse_error": False}},
        {"judge": {"outcome": "wrong_answer", "parse_error": True}},
        {"judge": {"outcome": "malformed_tool_call", "parse_error": False}},
    ]
    judged_rollouts.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "eval_metrics.jsonl"

    record = write_eval_metrics(
        judged_rollout_path=judged_rollouts,
        metrics_path=metrics_path,
        iteration=3,
        policy_checkpoint_id="iteration-00003",
    )

    assert record["iteration"] == 3
    assert record["policy_checkpoint_id"] == "iteration-00003"
    assert record["eval_accuracy"] == 1 / 3
    assert record["eval_correct"] == 1
    assert record["eval_total"] == 3
    assert record["eval_malformed"] == 1
    assert record["eval_parse_errors"] == 1
    written = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert written == [record]


def test_write_eval_metrics_reuses_existing_iteration_record(tmp_path: Path) -> None:
    judged_rollouts = tmp_path / "eval.jsonl"
    judged_rollouts.write_text(
        json.dumps({"judge": {"outcome": "correct_answer", "parse_error": False}}) + "\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "eval_metrics.jsonl"

    first = write_eval_metrics(
        judged_rollout_path=judged_rollouts,
        metrics_path=metrics_path,
        iteration=3,
        policy_checkpoint_id="iteration-00003",
    )
    second = write_eval_metrics(
        judged_rollout_path=judged_rollouts,
        metrics_path=metrics_path,
        iteration=3,
        policy_checkpoint_id="iteration-00003",
    )

    written = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert second == first
    assert written == [first]
