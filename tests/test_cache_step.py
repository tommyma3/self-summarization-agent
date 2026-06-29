import json
from pathlib import Path

from self_summarization_agent.cache_step import build_cache_scorer, run_cache_step
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


class FakeScorer:
    def __init__(self) -> None:
        self.seen_batches: list[list[str]] = []

    def cache_samples(self, samples):
        self.seen_batches.append([sample.turn_id for sample in samples])
        return [
            {
                "version": 1,
                "input_ids": [1, index + 2],
                "labels": [index + 2, index + 3],
                "completion_mask": [False, True],
                "reference_logprob": -0.5 - index,
            }
            for index, _sample in enumerate(samples)
        ]


def train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        experiment=ExperimentConfig(name="demo", seed=1, output_root=str(tmp_path), bc_plus_root=str(tmp_path)),
        dataset=DatasetConfig(limit=1),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(backend="transformers", model_path="unused"),
        rollout=RolloutConfig(backend="vllm_offline"),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=1024, tool_budget=4),
        judge=JudgeConfig(enabled=True),
        training=TrainingConfig(group_size=2),
    )


def judged_row(query_id: str, rollout_index: int, checkpoint_id: str = "step-00001") -> dict:
    return {
        "policy_checkpoint_id": checkpoint_id,
        "policy_checkpoint_path": checkpoint_id,
        "rollout_index": rollout_index,
        "trainable_sample_count": 2,
        "query_id": query_id,
        "query": "question",
        "status": "completed",
        "final_answer": "done",
        "summary_turns": [],
        "turn_records": [
            {
                "query_id": query_id,
                "turn_id": "tool-1",
                "kind": "tool",
                "prompt": "tool prompt",
                "completion": '{"tool_name": "search", "arguments": {"query": "question"}}',
            },
            {
                "query_id": query_id,
                "turn_id": "final-answer",
                "kind": "final_answer",
                "prompt": "prompt",
                "completion": '{"tool_name": "finish", "arguments": {"answer": "done"}}',
            },
        ],
        "turn_rewards": {"tool-1": 1.0, "final-answer": 1.0},
        "judge": {"outcome": "correct_answer", "parse_error": False},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cache_step_writes_training_cache_for_each_trainable_turn(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    judged_path = tmp_path / "judged.jsonl"
    output_path = tmp_path / "cached.jsonl"
    write_jsonl(judged_path, [judged_row("q1", 0)])
    scorer = FakeScorer()

    run_cache_step(
        train_config(tmp_path),
        checkpoint_path=checkpoint,
        rollout_path=judged_path,
        output_path=output_path,
        scorer=scorer,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert scorer.seen_batches == [["tool-1", "final-answer"]]
    assert rows[0]["turn_records"][0]["training_cache"]["input_ids"] == [1, 2]
    assert rows[0]["turn_records"][0]["training_cache"]["reference_logprob"] == -0.5
    assert rows[0]["turn_records"][0]["training_cache"]["policy_checkpoint_id"] == "step-00001"
    assert rows[0]["turn_records"][1]["training_cache"]["completion_mask"] == [False, True]


def test_cache_step_resume_skips_completed_cached_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    judged_path = tmp_path / "judged.jsonl"
    output_path = tmp_path / "cached.jsonl"
    rows = [judged_row("q1", 0), judged_row("q2", 1)]
    write_jsonl(judged_path, rows)
    first_cached = judged_row("q1", 0)
    for turn in first_cached["turn_records"]:
        turn["training_cache"] = {
            "version": 1,
            "input_ids": [1],
            "labels": [2],
            "completion_mask": [True],
            "reference_logprob": -0.1,
        }
    write_jsonl(output_path, [first_cached])
    scorer = FakeScorer()

    run_cache_step(
        train_config(tmp_path),
        checkpoint_path=checkpoint,
        rollout_path=judged_path,
        output_path=output_path,
        scorer=scorer,
        resume=True,
    )

    cached_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert scorer.seen_batches == [["tool-1", "final-answer"]]
    assert [(row["query_id"], row["rollout_index"]) for row in cached_rows] == [("q1", 0), ("q2", 1)]


def test_cache_step_rejects_raw_unjudged_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    raw_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "cached.jsonl"
    row = judged_row("q1", 0)
    del row["turn_rewards"]
    write_jsonl(raw_path, [row])

    try:
        run_cache_step(
            train_config(tmp_path),
            checkpoint_path=checkpoint,
            rollout_path=raw_path,
            output_path=output_path,
            scorer=FakeScorer(),
        )
    except ValueError as exc:
        assert "missing turn_rewards" in str(exc)
    else:
        raise AssertionError("Expected raw rows to be rejected")


def test_build_cache_scorer_accepts_verl_ray_with_transformers_worker(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    config = train_config(tmp_path)
    config.training.backend = "verl_ray"
    config.training.verl.worker_backend = "transformers"
    created = {}

    class FakeTransformersPolicyTrainer:
        def __init__(self, model_config, training_config) -> None:
            created["model_path"] = model_config.model_path
            created["backend"] = training_config.backend

    monkeypatch.setattr(
        "self_summarization_agent.cache_step.TransformersPolicyTrainer",
        FakeTransformersPolicyTrainer,
    )

    scorer = build_cache_scorer(config, checkpoint_path=checkpoint)

    assert isinstance(scorer, FakeTransformersPolicyTrainer)
    assert created == {"model_path": str(checkpoint.resolve()), "backend": "transformers"}


def test_build_cache_scorer_rejects_unsupported_verl_worker_backend(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    config = train_config(tmp_path)
    config.training.backend = "verl_ray"
    config.training.verl.worker_backend = "fsdp2_context_parallel"

    try:
        build_cache_scorer(config, checkpoint_path=checkpoint)
    except NotImplementedError as exc:
        assert "training.verl.worker_backend='transformers'" in str(exc)
    else:
        raise AssertionError("Expected unsupported verl worker backend to be rejected")
