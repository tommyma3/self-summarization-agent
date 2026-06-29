import json
from pathlib import Path

from self_summarization_agent.eval_metrics import write_eval_metrics


def test_write_eval_metrics_counts_judged_eval_rollouts(tmp_path: Path) -> None:
    judged_rollouts = tmp_path / "eval.jsonl"
    rows = [
        {
            "judge": {"outcome": "correct_answer", "parse_error": False},
            "tool_call_counts": {"search": 2, "get_document": 1},
            "token_usage": {
                "reasoning_generated_tokens": 100,
                "summary_generated_tokens": 20,
                "forced_answer_generated_tokens": 5,
                "total_generated_tokens": 125,
                "max_prompt_tokens_seen": 8000,
                "summary_count": 1,
            },
        },
        {
            "judge": {"outcome": "wrong_answer", "parse_error": True},
            "tool_call_counts": {"search": 1, "get_document": 0},
            "token_usage": {
                "reasoning_generated_tokens": 50,
                "summary_generated_tokens": 0,
                "forced_answer_generated_tokens": 0,
                "total_generated_tokens": 50,
                "max_prompt_tokens_seen": 4000,
                "summary_count": 0,
            },
        },
        {
            "judge": {"outcome": "malformed_tool_call", "parse_error": False},
            "tool_call_counts": {"search": 0, "get_document": 0},
            "token_usage": {
                "reasoning_generated_tokens": 25,
                "summary_generated_tokens": 10,
                "forced_answer_generated_tokens": 0,
                "total_generated_tokens": 35,
                "max_prompt_tokens_seen": 6000,
                "summary_count": 1,
            },
        },
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
    assert record["eval_reasoning_generated_tokens"] == 175
    assert record["eval_summary_generated_tokens"] == 30
    assert record["eval_forced_answer_generated_tokens"] == 5
    assert record["eval_total_generated_tokens"] == 210
    assert record["eval_avg_reasoning_generated_tokens"] == 175 / 3
    assert record["eval_avg_summary_generated_tokens"] == 10
    assert record["eval_avg_forced_answer_generated_tokens"] == 5 / 3
    assert record["eval_avg_total_generated_tokens"] == 70
    assert record["eval_avg_max_prompt_tokens_seen"] == 6000
    assert record["eval_max_prompt_tokens_seen"] == 8000
    assert record["eval_avg_search_calls"] == 1
    assert record["eval_avg_document_calls"] == 1 / 3
    assert record["eval_avg_summary_count"] == 2 / 3
    assert record["eval_correct_per_1k_reasoning_tokens"] == 1 / 0.175
    assert record["eval_correct_per_1k_total_generated_tokens"] == 1 / 0.210
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
