from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from self_summarization_agent.checkpoints import checkpoint_id_from_path
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import JudgeDecision, RewardJudge
from self_summarization_agent.launcher_utils import append_jsonl, ensure_dir
from self_summarization_agent.rewards import apply_terminal_reward
from self_summarization_agent.trajectory import extract_trainable_samples


def _load_rollout_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid rollout JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Rollout row {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _load_examples_by_query_id(config) -> dict[str, QueryExample]:
    examples = load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    train_examples = examples if config.dataset.train_limit is None else examples[: config.dataset.train_limit]
    return {example.query_id: example for example in train_examples}


def build_judge(config) -> RewardJudge:
    if not config.judge.enabled:
        raise ValueError("judge.enabled must be true for judge_step")
    if config.judge.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in config.judge.gpu_ids)
    judge_model_config = replace(
        config.model,
        backend=config.judge.backend or config.model.backend,
        model_path=config.judge.model_path or config.model.judge_model_path or config.model.model_path,
        judge_model_path=None,
        tensor_parallel_size=config.judge.tensor_parallel_size
        if config.judge.tensor_parallel_size is not None
        else config.model.tensor_parallel_size,
        attention_backend=config.judge.attention_backend
        if config.judge.attention_backend is not None
        else config.model.attention_backend,
        max_model_len=config.judge.max_model_len
        if config.judge.max_model_len is not None
        else config.model.max_model_len,
    )
    return RewardJudge(build_generator(judge_model_config, judge_config=config.judge))


def _validate_raw_row(row: dict[str, Any], *, index: int, expected_checkpoint_id: str | None) -> None:
    if expected_checkpoint_id is not None and row.get("policy_checkpoint_id") != expected_checkpoint_id:
        raise ValueError(
            f"Rollout row {index} checkpoint mismatch: expected {expected_checkpoint_id!r}, "
            f"got {row.get('policy_checkpoint_id')!r}"
        )
    if "turn_rewards" in row:
        raise ValueError(f"Rollout row {index} is already judged; expected raw rollout rows")
    if not isinstance(row.get("query_id"), str):
        raise ValueError(f"Rollout row {index} is missing query_id")
    if not isinstance(row.get("turn_records"), list):
        raise ValueError(f"Rollout row {index} is missing turn_records")
    if not isinstance(row.get("summary_turns"), list):
        raise ValueError(f"Rollout row {index} is missing summary_turns")


def _apply_decision_to_row(row: dict[str, Any], decision: JudgeDecision) -> dict[str, Any]:
    status = row.get("status")
    judged_row = dict(row)
    if status == "malformed_tool_call":
        judged_row["turn_rewards"] = {}
        judged_row["trainable_sample_count"] = 0
        judged_row["judge"] = {
            "outcome": "malformed_tool_call",
            "judge_prompt": None,
            "judge_response": None,
            "parse_error": False,
            "rollout_index": row.get("rollout_index"),
        }
        return judged_row

    final_answer_turn_id = "final-answer" if row.get("final_answer") is not None else None
    summary_turn_ids = [turn_id for turn_id in row["summary_turns"] if isinstance(turn_id, str)]
    turn_rewards = apply_terminal_reward(
        outcome=decision.outcome,
        summary_turn_ids=summary_turn_ids,
        final_answer_turn_id=final_answer_turn_id,
    )
    judged_row["turn_rewards"] = turn_rewards
    judged_row["trainable_sample_count"] = len(extract_trainable_samples(row["turn_records"], turn_rewards))
    judged_row["judge"] = {
        "outcome": decision.outcome,
        "judge_prompt": decision.judge_prompt,
        "judge_response": decision.judge_response,
        "parse_error": decision.parse_error,
        "rollout_index": row.get("rollout_index"),
    }
    return judged_row


def judge_rollouts(
    config,
    *,
    rollout_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    judge: RewardJudge | None = None,
    examples_by_query_id: dict[str, QueryExample] | None = None,
) -> Path:
    expected_checkpoint_id = checkpoint_id_from_path(checkpoint_path) if checkpoint_path is not None else None
    rows = _load_rollout_rows(rollout_path)
    examples = examples_by_query_id or _load_examples_by_query_id(config)
    for index, row in enumerate(rows, start=1):
        _validate_raw_row(row, index=index, expected_checkpoint_id=expected_checkpoint_id)

    judge = judge or build_judge(config)
    judge_items: list[tuple[QueryExample, str, str]] = []
    judge_row_indices: list[int] = []
    decisions_by_row_index: dict[int, JudgeDecision] = {}
    for index, row in enumerate(rows):
        if row.get("status") == "malformed_tool_call":
            decisions_by_row_index[index] = JudgeDecision(
                outcome="wrong_answer",
                judge_prompt=None,
                judge_response=None,
                parse_error=False,
            )
            continue
        query_id = row["query_id"]
        example = examples.get(query_id)
        if example is None:
            raise ValueError(f"Rollout row {index + 1} references unknown query_id: {query_id}")
        judge_items.append((example, str(row.get("status") or ""), str(row.get("final_answer") or "")))
        judge_row_indices.append(index)

    judge_decisions = judge.evaluate_batch(judge_items)
    if len(judge_decisions) != len(judge_items):
        raise ValueError(f"Judge returned {len(judge_decisions)} decisions for {len(judge_items)} rows")
    for row_index, decision in zip(judge_row_indices, judge_decisions):
        decisions_by_row_index[row_index] = decision

    output = Path(output_path)
    ensure_dir(output.parent)
    if output.exists():
        output.unlink()
    for index, row in enumerate(rows):
        append_jsonl(output, _apply_decision_to_row(row, decisions_by_row_index[index]))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge raw rollout artifacts and assign trainable rewards.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--rollouts", required=True, help="Raw rollout JSONL path.")
    parser.add_argument("--output", required=True, help="Judged rollout JSONL output path.")
    parser.add_argument("--checkpoint", default=None, help="Optional policy checkpoint path for row validation.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    output = judge_rollouts(
        config,
        rollout_path=args.rollouts,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    print(output)


if __name__ == "__main__":
    main()
