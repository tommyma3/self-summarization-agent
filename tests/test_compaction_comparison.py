import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_compaction_comparison.py"
SPEC = importlib.util.spec_from_file_location("run_compaction_comparison", SCRIPT_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(*, outcome: str, reasoning: int, summary: int, prompt: int, answer: str = "answer") -> dict:
    return {
        "query_id": f"q-{reasoning}",
        "query": "question",
        "status": "completed",
        "final_answer": answer,
        "judge": {"outcome": outcome, "parse_error": False},
        "tool_call_counts": {"search": 2, "get_document": 1},
        "summary_turns": ["summary-1"] if summary else [],
        "turn_records": [
            {
                "turn_id": "tool-1",
                "kind": "tool",
                "completion": "<think>look up clue carefully</think>\n<search>clue</search>",
                "completion_tokens": 8,
            },
            {
                "turn_id": "final-answer",
                "kind": "final_answer",
                "completion": "<think>combine evidence</think>\n<answer>answer</answer>",
                "completion_tokens": 5,
            },
        ],
        "token_usage": {
            "reasoning_generated_tokens": reasoning,
            "summary_generated_tokens": summary,
            "forced_answer_generated_tokens": 0,
            "total_generated_tokens": reasoning + summary,
            "max_prompt_tokens_seen": prompt,
            "summary_count": 1 if summary else 0,
            "forced_answer_reasons": [],
        },
    }


def test_summarize_judged_rollouts_reports_behavior_metrics(tmp_path: Path) -> None:
    judged_path = tmp_path / "judged.jsonl"
    _write_jsonl(
        judged_path,
        [
            _row(outcome="correct_answer", reasoning=100, summary=25, prompt=8000),
            _row(outcome="wrong_answer", reasoning=50, summary=0, prompt=4000, answer="wrong answer"),
        ],
    )

    summary = comparison.summarize_judged_rollouts("compact", judged_path)

    assert summary["condition"] == "compact"
    assert summary["rollout_count"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["avg_reasoning_generated_tokens"] == 75
    assert summary["avg_summary_generated_tokens"] == 12.5
    assert summary["summary_overhead_ratio"] == 25 / 175
    assert summary["avg_search_calls"] == 2
    assert summary["avg_document_calls"] == 1
    assert summary["avg_cot_tokens_per_episode"] == 6
    assert summary["avg_action_cot_tokens_per_episode"] == 6
    assert summary["avg_final_answer_tokens"] == 1.5


def test_build_rollout_command_uses_overlap_judged_output() -> None:
    command = comparison.build_rollout_command(
        python_executable="python",
        base_config=Path("configs/train/default.yaml"),
        checkpoint=Path("ckpt"),
        raw_path=Path("raw.jsonl"),
        judged_path=Path("judged.jsonl"),
        split="eval",
        overrides={"training.group_size": 1, "rollout.overlap_judge": True},
        resume=True,
    )

    assert "--judged-output" in command
    assert "judged.jsonl" in command
    assert "--judge-inline" not in command
    assert "--no-overlap-judge" not in command
    assert "--resume" in command
    assert "training.group_size=1" in command
    assert "rollout.overlap_judge=true" in command


def test_build_judge_command_is_available_as_resume_fallback() -> None:
    command = comparison.build_judge_command(
        python_executable="python",
        base_config=Path("configs/train/default.yaml"),
        checkpoint=Path("ckpt"),
        raw_path=Path("raw.jsonl"),
        judged_path=Path("judged.jsonl"),
        split="eval",
        overrides={"training.group_size": 1},
    )

    assert "self_summarization_agent.judge_step" in command
    assert "--rollouts" in command
    assert "raw.jsonl" in command
    assert "--output" in command
    assert "judged.jsonl" in command
    assert "training.group_size=1" in command


def test_run_comparison_writes_manifest_and_summary_with_overlap_commands(tmp_path: Path) -> None:
    base_config = tmp_path / "train.yaml"
    base_config.write_text("experiment: {}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / "iteration-00001"
    checkpoint.mkdir(parents=True)
    config_path = tmp_path / "comparison.yaml"
    output_root = tmp_path / "comparison-output"
    config_path.write_text(
        json.dumps(
            {
                "base_config": str(base_config),
                "comparison": {
                    "output_root": str(output_root),
                    "iteration": 7,
                    "split": "eval",
                    "resume": False,
                },
                "common_overrides": {
                    "training.group_size": 1,
                    "runtime.generated_token_budget": 32,
                    "rollout.overlap_judge": True,
                },
                "conditions": {
                    "no_compact_32k": {
                        "overrides": {
                            "runtime.context_threshold_tokens": 1000000000,
                            "runtime.max_context_tokens": 32768,
                            "rollout.max_model_len": 40960,
                        }
                    },
                    "compact_8k": {
                        "overrides": {
                            "runtime.context_threshold_tokens": 8000,
                            "runtime.max_context_tokens": 18000,
                            "rollout.max_model_len": 27000,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_runner(command):
        command = list(command)
        commands.append(command)
        judged_path = Path(command[command.index("--judged-output") + 1])
        if "no_compact_32k" in str(judged_path):
            _write_jsonl(judged_path, [_row(outcome="correct_answer", reasoning=100, summary=0, prompt=30000)])
        else:
            _write_jsonl(judged_path, [_row(outcome="correct_answer", reasoning=100, summary=20, prompt=9000)])
        return 0

    summary = comparison.run_comparison(
        config_path=config_path,
        checkpoint=checkpoint,
        base_config_override=None,
        output_root_override=None,
        iteration_override=None,
        limit=3,
        resume_override=None,
        python_executable="python",
        command_runner=fake_runner,
    )

    assert len(commands) == 2
    assert all("--judged-output" in command for command in commands)
    assert all("--judge-inline" not in command for command in commands)
    assert all("--set" in command for command in commands)
    assert any("dataset.eval_limit=3" in command for command in commands)
    assert summary["conditions"][0]["condition"] == "no_compact_32k"
    assert summary["conditions"][1]["condition"] == "compact_8k"
    assert summary["deltas"]["avg_summary_generated_tokens"] == 20
    assert (output_root / "manifest.json").exists()
    assert (output_root / "comparison_summary.json").exists()
    assert (output_root / "comparison_summary.md").exists()


def test_condition_overrides_reject_disabling_overlap_judge() -> None:
    try:
        comparison._condition_overrides(
            common_overrides={"rollout.overlap_judge": True},
            condition_config={"overrides": {"rollout.overlap_judge": False}},
            limit=None,
        )
    except ValueError as exc:
        assert "overlap_judge=true" in str(exc)
    else:
        raise AssertionError("Expected disabling overlap judging to be rejected")
