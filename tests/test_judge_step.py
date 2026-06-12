import json
from pathlib import Path

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
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.judge_step import judge_rollouts


class BatchJudgeGenerator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        self.batch_sizes.append(len(prompts))
        return ["correct: yes" for _ in prompts]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


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
    assert rows[0]["turn_rewards"] == {"tool-1": 1.0, "final-answer": 1.0}
    assert rows[0]["trainable_sample_count"] == 2
    assert rows[0]["judge"]["outcome"] == "correct_answer"


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
