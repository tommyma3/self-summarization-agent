from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_summarization_agent.checkpoints import checkpoint_id_from_path
from self_summarization_agent.eval_metrics import write_eval_metrics


CommandRunner = Callable[[Sequence[str]], int]
_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", flags=re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired no-compaction and compaction BrowseComp-Plus eval rollouts."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "eval" / "compaction_comparison.yaml"),
        help="Comparison YAML path.",
    )
    parser.add_argument("--checkpoint", default=None, help="Policy checkpoint to evaluate.")
    parser.add_argument("--base-config", default=None, help="Override the train config used by rollout_collection.")
    parser.add_argument("--output-root", default=None, help="Override comparison.output_root.")
    parser.add_argument("--iteration", type=int, default=None, help="Override comparison.iteration.")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset.eval_limit for both conditions.")
    parser.add_argument("--resume", action="store_true", help="Resume missing rollout rows for both conditions.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume, even if YAML comparison.resume is true.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for subprocesses.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned rollout commands without running them.")
    return parser.parse_args()


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Comparison config must be a mapping: {path}")
    return payload


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    return repr(value)


def _override_items(overrides: dict[str, Any]) -> list[str]:
    return [f"{key}={_format_override_value(value)}" for key, value in overrides.items()]


def _append_overrides(command: list[str], overrides: dict[str, Any]) -> None:
    for item in _override_items(overrides):
        command.extend(["--set", item])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row in {path} on line {line_number} must be an object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _average(total: float, count: int) -> float:
    return total / count if count else 0.0


def _tokens(text: str | None) -> int:
    return len((text or "").split())


def _think_token_count(completion: str) -> int:
    return sum(_tokens(match.group(1)) for match in _THINK_RE.finditer(completion or ""))


def _record_completion_tokens(record: dict[str, Any]) -> float:
    completion_tokens = record.get("completion_tokens")
    if isinstance(completion_tokens, int | float):
        return float(completion_tokens)
    return float(_tokens(str(record.get("completion", ""))))


def summarize_judged_rollouts(condition: str, judged_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(judged_path)
    total = len(rows)
    outcomes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    forced_reasons: Counter[str] = Counter()
    correct = 0
    malformed = 0
    parse_errors = 0

    reasoning_tokens = 0.0
    summary_tokens = 0.0
    forced_answer_tokens = 0.0
    tool_result_tokens = 0.0
    total_generated_tokens = 0.0
    max_prompt_sum = 0.0
    max_prompt_max = 0.0
    summary_count = 0.0
    search_calls = 0.0
    document_calls = 0.0
    final_answer_tokens = 0.0
    turn_count = 0.0
    completion_token_sum = 0.0
    cot_token_sum = 0.0
    action_cot_token_sum = 0.0
    cot_turn_count = 0
    generated_turn_count = 0
    forced_answer_episode_count = 0

    for row in rows:
        judge = row.get("judge")
        outcome = judge.get("outcome") if isinstance(judge, dict) else None
        if isinstance(outcome, str):
            outcomes[outcome] += 1
        if outcome == "correct_answer":
            correct += 1
        if outcome == "malformed_tool_call":
            malformed += 1
        if isinstance(judge, dict) and judge.get("parse_error"):
            parse_errors += 1

        status = row.get("status")
        if isinstance(status, str):
            statuses[status] += 1

        token_usage = row.get("token_usage") if isinstance(row.get("token_usage"), dict) else {}
        reasoning_tokens += _number(token_usage.get("reasoning_generated_tokens"))
        summary_tokens += _number(token_usage.get("summary_generated_tokens"))
        forced_answer_tokens += _number(token_usage.get("forced_answer_generated_tokens"))
        tool_result_tokens += _number(token_usage.get("tool_result_tokens"))
        total_generated_tokens += _number(token_usage.get("total_generated_tokens"))
        max_prompt = _number(token_usage.get("max_prompt_tokens_seen"))
        max_prompt_sum += max_prompt
        max_prompt_max = max(max_prompt_max, max_prompt)
        summary_count += _number(token_usage.get("summary_count"))
        reasons = token_usage.get("forced_answer_reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                if isinstance(reason, str):
                    forced_reasons[reason] += 1
        if _number(token_usage.get("forced_answer_generated_tokens")) > 0 or reasons:
            forced_answer_episode_count += 1

        tool_call_counts = row.get("tool_call_counts") if isinstance(row.get("tool_call_counts"), dict) else {}
        search_calls += _number(tool_call_counts.get("search"))
        document_calls += _number(tool_call_counts.get("get_document"))
        final_answer_tokens += _tokens(row.get("final_answer") if isinstance(row.get("final_answer"), str) else "")

        turn_records = row.get("turn_records") if isinstance(row.get("turn_records"), list) else []
        turn_count += len(turn_records)
        for record in turn_records:
            if not isinstance(record, dict):
                continue
            completion = str(record.get("completion", ""))
            completion_token_sum += _record_completion_tokens(record)
            generated_turn_count += 1
            cot_tokens = _think_token_count(completion)
            if cot_tokens:
                cot_turn_count += 1
            cot_token_sum += cot_tokens
            if record.get("kind") in {"tool", "final_answer"}:
                action_cot_token_sum += cot_tokens

    return {
        "condition": condition,
        "rollout_count": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "malformed": malformed,
        "parse_errors": parse_errors,
        "outcome_counts": dict(sorted(outcomes.items())),
        "status_counts": dict(sorted(statuses.items())),
        "reasoning_generated_tokens": reasoning_tokens,
        "summary_generated_tokens": summary_tokens,
        "forced_answer_generated_tokens": forced_answer_tokens,
        "tool_result_tokens": tool_result_tokens,
        "total_generated_tokens": total_generated_tokens,
        "avg_reasoning_generated_tokens": _average(reasoning_tokens, total),
        "avg_summary_generated_tokens": _average(summary_tokens, total),
        "avg_forced_answer_generated_tokens": _average(forced_answer_tokens, total),
        "avg_tool_result_tokens": _average(tool_result_tokens, total),
        "avg_total_generated_tokens": _average(total_generated_tokens, total),
        "avg_max_prompt_tokens_seen": _average(max_prompt_sum, total),
        "max_prompt_tokens_seen": max_prompt_max,
        "avg_search_calls": _average(search_calls, total),
        "avg_document_calls": _average(document_calls, total),
        "avg_summary_count": _average(summary_count, total),
        "forced_answer_episode_count": forced_answer_episode_count,
        "forced_answer_reason_counts": dict(sorted(forced_reasons.items())),
        "avg_final_answer_tokens": _average(final_answer_tokens, total),
        "avg_turn_count": _average(turn_count, total),
        "avg_completion_tokens_per_turn": _average(completion_token_sum, generated_turn_count),
        "avg_cot_tokens_per_episode": _average(cot_token_sum, total),
        "avg_cot_tokens_per_cot_turn": _average(cot_token_sum, cot_turn_count),
        "avg_action_cot_tokens_per_episode": _average(action_cot_token_sum, total),
        "summary_overhead_ratio": summary_tokens / total_generated_tokens if total_generated_tokens > 0 else 0.0,
        "correct_per_1k_reasoning_tokens": correct / (reasoning_tokens / 1000.0) if reasoning_tokens > 0 else 0.0,
        "correct_per_1k_total_generated_tokens": correct / (total_generated_tokens / 1000.0)
        if total_generated_tokens > 0
        else 0.0,
    }


def build_rollout_command(
    *,
    python_executable: str,
    base_config: Path,
    checkpoint: Path,
    raw_path: Path,
    judged_path: Path,
    split: str,
    overrides: dict[str, Any],
    resume: bool,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "self_summarization_agent.rollout_collection",
        "--config",
        str(base_config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(raw_path),
        "--judged-output",
        str(judged_path),
        "--split",
        split,
    ]
    _append_overrides(command, overrides)
    if resume:
        command.append("--resume")
    return command


def build_judge_command(
    *,
    python_executable: str,
    base_config: Path,
    checkpoint: Path,
    raw_path: Path,
    judged_path: Path,
    split: str,
    overrides: dict[str, Any],
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "self_summarization_agent.judge_step",
        "--config",
        str(base_config),
        "--checkpoint",
        str(checkpoint),
        "--rollouts",
        str(raw_path),
        "--output",
        str(judged_path),
        "--split",
        split,
    ]
    _append_overrides(command, overrides)
    return command


def _run_command(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode


def _condition_overrides(
    *,
    common_overrides: dict[str, Any],
    condition_config: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    overrides = dict(common_overrides)
    overrides["rollout.overlap_judge"] = True
    if limit is not None:
        overrides["dataset.eval_limit"] = limit
    condition_overrides = condition_config.get("overrides", {})
    if not isinstance(condition_overrides, dict):
        raise ValueError("Condition overrides must be a mapping")
    overrides.update(condition_overrides)
    if overrides.get("rollout.overlap_judge") is not True:
        raise ValueError("Comparison rollouts must keep rollout.overlap_judge=true to hide judge latency")
    return overrides


def _render_markdown(summary: dict[str, Any]) -> str:
    condition_summaries = summary["conditions"]
    rows = [
        "| condition | accuracy | correct/total | avg reasoning tok | avg tool tok | avg summary tok | avg total tok | avg CoT tok/episode | avg prompt max | avg search | avg doc | forced answer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in condition_summaries:
        rows.append(
            "| {condition} | {accuracy:.4f} | {correct}/{rollout_count} | "
            "{avg_reasoning_generated_tokens:.1f} | {avg_tool_result_tokens:.1f} | "
            "{avg_summary_generated_tokens:.1f} | "
            "{avg_total_generated_tokens:.1f} | {avg_cot_tokens_per_episode:.1f} | "
            "{avg_max_prompt_tokens_seen:.1f} | {avg_search_calls:.2f} | "
            "{avg_document_calls:.2f} | {forced_answer_episode_count} |".format(**item)
        )
    if summary.get("deltas"):
        rows.extend(["", "## Deltas", "", "Positive values mean second condition minus first condition."])
        rows.append("")
        rows.append("| metric | delta |")
        rows.append("|---|---:|")
        for key, value in summary["deltas"].items():
            rows.append(f"| {key} | {value:.6g} |")
    return "\n".join(rows) + "\n"


def _numeric_deltas(condition_summaries: list[dict[str, Any]]) -> dict[str, float]:
    if len(condition_summaries) < 2:
        return {}
    first, second = condition_summaries[0], condition_summaries[1]
    keys = [
        "accuracy",
        "avg_reasoning_generated_tokens",
        "avg_tool_result_tokens",
        "avg_summary_generated_tokens",
        "avg_total_generated_tokens",
        "avg_cot_tokens_per_episode",
        "avg_max_prompt_tokens_seen",
        "avg_search_calls",
        "avg_document_calls",
        "summary_overhead_ratio",
        "correct_per_1k_reasoning_tokens",
        "correct_per_1k_total_generated_tokens",
    ]
    return {key: float(second[key]) - float(first[key]) for key in keys}


def run_comparison(
    *,
    config_path: Path,
    checkpoint: Path | None,
    base_config_override: Path | None,
    output_root_override: Path | None,
    iteration_override: int | None,
    limit: int | None,
    resume_override: bool | None,
    python_executable: str,
    dry_run: bool = False,
    command_runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    raw_config = _read_yaml(config_path)
    config_dir = config_path.parent

    comparison = raw_config.get("comparison", {})
    if not isinstance(comparison, dict):
        raise ValueError("comparison section must be a mapping")
    conditions = raw_config.get("conditions", {})
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("conditions section must be a non-empty mapping")
    common_overrides = raw_config.get("common_overrides", {})
    if not isinstance(common_overrides, dict):
        raise ValueError("common_overrides section must be a mapping")

    base_config_value = base_config_override or raw_config.get("base_config")
    if base_config_value is None:
        raise ValueError("base_config is required")
    base_config = _resolve_path(base_config_value, base_dir=config_dir)

    checkpoint_value = checkpoint or comparison.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("A checkpoint is required via --checkpoint or comparison.checkpoint")
    checkpoint_path = _resolve_path(checkpoint_value, base_dir=Path.cwd())
    checkpoint_id = checkpoint_id_from_path(checkpoint_path)

    output_root_value = output_root_override or comparison.get("output_root")
    if output_root_value is None:
        raise ValueError("comparison.output_root is required")
    output_root = _resolve_path(output_root_value, base_dir=Path.cwd())
    output_root.mkdir(parents=True, exist_ok=True)

    iteration = iteration_override if iteration_override is not None else int(comparison.get("iteration", 0))
    split = str(comparison.get("split", "eval"))
    yaml_resume = bool(comparison.get("resume", False))
    resume = yaml_resume if resume_override is None else resume_override

    manifest_conditions: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []

    for condition_name, condition_config in conditions.items():
        if not isinstance(condition_config, dict):
            raise ValueError(f"Condition {condition_name!r} must be a mapping")
        condition_dir = output_root / str(condition_name)
        condition_dir.mkdir(parents=True, exist_ok=True)
        raw_path = condition_dir / "rollouts.raw.jsonl"
        judged_path = condition_dir / "rollouts.judged.jsonl"
        metrics_path = condition_dir / "eval_metrics.jsonl"

        overrides = _condition_overrides(
            common_overrides=common_overrides,
            condition_config=condition_config,
            limit=limit,
        )
        command = build_rollout_command(
            python_executable=python_executable,
            base_config=base_config,
            checkpoint=checkpoint_path,
            raw_path=raw_path,
            judged_path=judged_path,
            split=split,
            overrides=overrides,
            resume=resume,
        )
        started = time.perf_counter()
        exit_code = 0 if dry_run else command_runner(command)
        elapsed_seconds = time.perf_counter() - started
        judge_command: list[str] | None = None
        judge_exit_code: int | None = None
        judge_elapsed_seconds = 0.0
        if not dry_run and exit_code == 0 and not judged_path.exists() and raw_path.exists():
            judge_command = build_judge_command(
                python_executable=python_executable,
                base_config=base_config,
                checkpoint=checkpoint_path,
                raw_path=raw_path,
                judged_path=judged_path,
                split=split,
                overrides=overrides,
            )
            judge_started = time.perf_counter()
            judge_exit_code = command_runner(judge_command)
            judge_elapsed_seconds = time.perf_counter() - judge_started
        manifest_conditions.append(
            {
                "condition": condition_name,
                "description": condition_config.get("description", ""),
                "raw_rollouts": str(raw_path),
                "judged_rollouts": str(judged_path),
                "eval_metrics": str(metrics_path),
                "overrides": overrides,
                "command": command,
                "exit_code": exit_code,
                "elapsed_seconds": elapsed_seconds,
                "fallback_judge_command": judge_command,
                "fallback_judge_exit_code": judge_exit_code,
                "fallback_judge_elapsed_seconds": judge_elapsed_seconds,
            }
        )
        if exit_code != 0:
            raise RuntimeError(f"Rollout command failed for {condition_name} with exit code {exit_code}")
        if judge_exit_code not in {None, 0}:
            raise RuntimeError(f"Fallback judge command failed for {condition_name} with exit code {judge_exit_code}")
        if dry_run:
            continue
        if not judged_path.exists():
            raise FileNotFoundError(
                f"Missing judged rollout artifact for {condition_name}: {judged_path}. "
                "The command must run with overlap judging through --judged-output."
            )
        if not resume and metrics_path.exists():
            metrics_path.unlink()
        eval_record = write_eval_metrics(
            judged_rollout_path=judged_path,
            metrics_path=metrics_path,
            iteration=iteration,
            policy_checkpoint_id=checkpoint_id,
        )
        condition_summary = summarize_judged_rollouts(str(condition_name), judged_path)
        condition_summary["elapsed_seconds"] = elapsed_seconds
        condition_summary["eval_metrics"] = eval_record
        condition_summaries.append(condition_summary)

    summary = {
        "timestamp_utc": _utc_timestamp(),
        "config": str(config_path),
        "base_config": str(base_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_id": checkpoint_id,
        "output_root": str(output_root),
        "iteration": iteration,
        "split": split,
        "dry_run": dry_run,
        "conditions": condition_summaries,
        "deltas": _numeric_deltas(condition_summaries),
    }
    manifest = {
        "timestamp_utc": summary["timestamp_utc"],
        "config": str(config_path),
        "base_config": str(base_config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_id": checkpoint_id,
        "output_root": str(output_root),
        "iteration": iteration,
        "split": split,
        "conditions": manifest_conditions,
    }
    _write_json(output_root / "manifest.json", manifest)
    if not dry_run:
        _write_json(output_root / "comparison_summary.json", summary)
        (output_root / "comparison_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    resume_override: bool | None = None
    if args.resume:
        resume_override = True
    if args.no_resume:
        resume_override = False
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    base_config = Path(args.base_config) if args.base_config else None
    output_root = Path(args.output_root) if args.output_root else None
    summary = run_comparison(
        config_path=Path(args.config),
        checkpoint=checkpoint,
        base_config_override=base_config,
        output_root_override=output_root,
        iteration_override=args.iteration,
        limit=args.limit,
        resume_override=resume_override,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
