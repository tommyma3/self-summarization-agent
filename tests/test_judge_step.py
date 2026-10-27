import json
from pathlib import Path

import pytest

from self_summarization_agent.config import (
    DatasetConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    RetrievalConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainConfig,
    TrainingConfig,
)
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.judge import JudgeDecision, RewardJudge, create_judge_prompt
from self_summarization_agent.judge_step import apply_decision_to_rollout_row, judge_rollouts


class BatchJudgeGenerator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        self.batch_sizes.append(len(prompts))
        return [
            "extracted_final_answer: done\nreasoning: The answers match.\ncorrect: yes\nconfidence: 100"
            for _ in prompts
        ]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_judge_step_excludes_history_rewritten_records_from_trainable_samples() -> None:
    rewritten_record = {
        "schema_version": 3,
        "query_id": "q1",
        "turn_id": "trajectory-1",
        "kind": "trajectory",
        "termination_kind": "malformed",
        "messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "reasoning and action"},
        ],
        "collection_tokens": {
            "version": 2,
            "full_token_ids": [1, 2, 99, 100, 20],
            "assistant_token_mask": [False, False, False, False, True],
            "generations": [
                {
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [10, 11],
                    "full_token_ids": [1, 2, 10, 11],
                },
                {
                    # The provider re-rendered the first assistant turn, so this
                    # prompt no longer extends the first generation's tokens.
                    "prompt_token_ids": [1, 2, 99, 100],
                    "completion_token_ids": [20],
                    "full_token_ids": [1, 2, 99, 100, 20],
                },
            ],
        },
    }
    row = {
        "status": "malformed_tool_call",
        "rollout_index": 0,
        "turn_records": [
            {"query_id": "q1", "turn_id": "tool-1", "kind": "tool", "prompt": "p", "completion": "c"},
        ],
        "trajectory_records": [rewritten_record],
    }

    judged = apply_decision_to_rollout_row(
        row,
        JudgeDecision(outcome="correct_answer", judge_prompt="unused", judge_response="unused", parse_error=False),
    )

    assert judged["turn_rewards"] == {"trajectory-1": -1.0}
    assert judged["trainable_sample_count"] == 0


def test_judge_step_penalizes_summary_length_exceeded_without_judging() -> None:
    row = {
        "status": "summary_length_exceeded",
        "turn_records": [
            {"query_id": "q1", "turn_id": "tool-1", "kind": "tool", "prompt": "p", "completion": "c"},
            {"query_id": "q1", "turn_id": "summary-1", "kind": "summary", "prompt": "p", "completion": "c"},
        ],
        "trajectory_records": [
            {
                "schema_version": 3,
                "query_id": "q1",
                "turn_id": "trajectory-1",
                "kind": "trajectory",
                "termination_kind": "compaction",
                "messages": [
                    {"role": "system", "content": "instructions"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "reasoning and summary"},
                ],
            }
        ],
    }

    judged = apply_decision_to_rollout_row(
        row,
        JudgeDecision(outcome="correct_answer", judge_prompt="unused", judge_response="unused", parse_error=False),
    )

    assert judged["judge"]["outcome"] == "summary_length_exceeded"
    assert judged["turn_rewards"] == {"trajectory-1": -1.0}
    assert judged["trainable_sample_count"] == 1


def test_create_judge_prompt_uses_supplied_browsecomp_system_instructions() -> None:
    prompt = create_judge_prompt(
        question="question",
        response="candidate answer",
        correct_answer="gold answer",
    )

    assert len(prompt.messages) == 2
    assert prompt.messages[0].role == "system"
    assert prompt.messages[1].role == "user"
    content = prompt.messages[0].content
    assert "[question]: question" in content
    assert "[response]: candidate answer" in content
    assert "[correct_answer]: gold answer" in content
    assert "extracted_final_answer:" in content
    assert "reasoning:" in content
    assert any(f"correct: Answer {quoted}" in content for quoted in ("'yes'", "‘yes’"))
    assert "confidence:" in content
    assert "do not\nattempt to solve the problem" in content
    assert "focus only on whether the answers match." in content
    assert "Output only one line:" not in content


def test_build_judge_forwards_judge_gpu_memory_utilization(monkeypatch, tmp_path: Path) -> None:
    from self_summarization_agent import judge_step

    config = train_config(tmp_path)
    config.model.gpu_memory_utilization = 0.7
    config.judge.gpu_memory_utilization = 0.55
    captured: dict = {}
    monkeypatch.setattr(
        judge_step,
        "build_generator",
        lambda model_config, judge_config=None: captured.update(model_config=model_config) or object(),
    )
    monkeypatch.setattr(judge_step, "RewardJudge", lambda generator: generator)

    judge_step.build_judge(config)

    assert captured["model_config"].gpu_memory_utilization == 0.55


def test_build_judge_inherits_model_gpu_memory_utilization(monkeypatch, tmp_path: Path) -> None:
    from self_summarization_agent import judge_step

    config = train_config(tmp_path)
    config.model.gpu_memory_utilization = 0.7
    captured: dict = {}
    monkeypatch.setattr(
        judge_step,
        "build_generator",
        lambda model_config, judge_config=None: captured.update(model_config=model_config) or object(),
    )
    monkeypatch.setattr(judge_step, "RewardJudge", lambda generator: generator)

    judge_step.build_judge(config)

    assert captured["model_config"].gpu_memory_utilization == 0.7


def train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        experiment=ExperimentConfig(name="demo", seed=1, output_root=str(tmp_path), bc_plus_root=str(tmp_path)),
        dataset=DatasetConfig(limit=1, train_limit=1),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(backend="transformers", model_path="unused"),
        rollout=RolloutConfig(backend="vllm_offline"),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=1024, tool_budget=4),
        judge=JudgeConfig(enabled=True),
        training=TrainingConfig(group_size=1),
    )


def write_raw_rollouts(path: Path) -> None:
    rows = [
        {
            "policy_checkpoint_id": "step-00001",
            "policy_checkpoint_path": "step-00001",
            "rollout_index": 0,
            "trainable_sample_count": None,
            "query_id": "q1",
            "query": "question",
            "status": "completed",
            "final_answer": "done",
            "summary_turns": [],
            "turn_records": [
                {
                    "query_id": "q1",
                    "turn_id": "tool-1",
                    "kind": "tool",
                    "prompt": "tool prompt",
                    "completion": '{"tool_name": "search", "arguments": {"query": "question"}}',
                },
                {
                    "query_id": "q1",
                    "turn_id": "final-answer",
                    "kind": "final_answer",
                    "prompt": "prompt",
                    "completion": '{"tool_name": "finish", "arguments": {"answer": "done"}}',
                }
            ],
            "trajectory_records": [
                {
                    "schema_version": 3,
                    "query_id": "q1",
                    "turn_id": "trajectory-1",
                    "kind": "trajectory",
                    "termination_kind": "final_answer",
                    "messages": [
                        {"role": "system", "content": "instructions"},
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "reasoning and search"},
                        {"role": "user", "content": "search result"},
                        {"role": "assistant", "content": "reasoning and answer"},
                    ],
                }
            ],
            "retrieved_docids": [],
            "tool_call_counts": {},
            "judge": None,
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_judge_rollouts_assigns_rewards_and_sample_counts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    raw_path = tmp_path / "raw.jsonl"
    judged_path = tmp_path / "judged.jsonl"
    write_raw_rollouts(raw_path)
    generator = BatchJudgeGenerator()

    judge_rollouts(
        train_config(tmp_path),
        rollout_path=raw_path,
        output_path=judged_path,
        checkpoint_path=checkpoint,
        judge=RewardJudge(generator),
        examples_by_query_id={"q1": QueryExample(query_id="q1", query="question", answer="done")},
    )

    rows = [json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines()]
    assert generator.batch_sizes == [1]
    assert rows[0]["turn_rewards"] == {"trajectory-1": 1.0}
    assert rows[0]["trainable_sample_count"] == 1
    assert rows[0]["judge"]["outcome"] == "correct_answer"


@pytest.mark.parametrize("interval_count", [1, 2])
@pytest.mark.parametrize(
    ("judge_response", "parse_error"),
    [
        ('<think>Output "correct: yes" or "correct: no".</think>\ncorrect: no', False),
        ('Thinking Process:\nOutput "correct: yes" or "correct: no".\nWait,', True),
        (
            'extracted_final_answer: different answer\nreasoning: Despite the example "correct: yes", the answers differ.\ncorrect: no\nconfidence: 100',
            False,
        ),
        ('extracted_final_answer: answer\nreasoning: Matches.\ncorrect: yes', True),
    ],
)
def test_judge_rollouts_does_not_reward_verdicts_in_reasoning(
    tmp_path: Path, interval_count: int, judge_response: str, parse_error: bool
) -> None:
    raw_path = tmp_path / "raw.jsonl"
    judged_path = tmp_path / "judged.jsonl"
    write_raw_rollouts(raw_path)
    raw_row = json.loads(raw_path.read_text(encoding="utf-8"))
    if interval_count == 2:
        first = raw_row["trajectory_records"][0]
        first["termination_kind"] = "compaction"
        first["messages"][-1]["content"] = "<summary>gathered evidence</summary>"
        raw_row["trajectory_records"].append(
            {
                "schema_version": 3,
                "query_id": "q1",
                "turn_id": "trajectory-2",
                "kind": "trajectory",
                "termination_kind": "final_answer",
                "messages": [
                    {"role": "system", "content": "instructions"},
                    {"role": "user", "content": "question"},
                    {"role": "user", "content": "<summary>gathered evidence</summary>"},
                    {"role": "assistant", "content": "reasoning and answer"},
                ],
            }
        )
        raw_path.write_text(json.dumps(raw_row) + "\n", encoding="utf-8")
    original_raw = raw_path.read_bytes()

    class Generator:
        def generate(self, prompt: str) -> str:
            return judge_response

    judge_rollouts(
        train_config(tmp_path),
        rollout_path=raw_path,
        output_path=judged_path,
        judge=RewardJudge(Generator()),
        examples_by_query_id={"q1": QueryExample(query_id="q1", query="question", answer="done")},
    )

    judged = json.loads(judged_path.read_text(encoding="utf-8"))
    assert judged["judge"]["outcome"] == "wrong_answer"
    assert judged["judge"]["parse_error"] is parse_error
    assert judged["judge"]["judge_response"] == judge_response
    assert judged["turn_rewards"] == {f"trajectory-{i}": -1.0 for i in range(1, interval_count + 1)}
    assert judged["trainable_sample_count"] == interval_count
    assert judged["trajectory_records"] == raw_row["trajectory_records"]
    assert raw_path.read_bytes() == original_raw


def test_judge_rollouts_rejects_already_judged_rows(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    write_raw_rollouts(raw_path)
    row = json.loads(raw_path.read_text(encoding="utf-8"))
    row["turn_rewards"] = {"final-answer": 1.0}
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    try:
        judge_rollouts(
            train_config(tmp_path),
            rollout_path=raw_path,
            output_path=tmp_path / "judged.jsonl",
            judge=RewardJudge(BatchJudgeGenerator()),
            examples_by_query_id={"q1": QueryExample(query_id="q1", query="question", answer="done")},
        )
    except ValueError as exc:
        assert "already judged" in str(exc)
    else:
        raise AssertionError("Expected already judged rollout rows to be rejected")
