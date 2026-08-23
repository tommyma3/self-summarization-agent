import json
from pathlib import Path

from self_summarization_agent.cache_step import (
    _attach_training_caches,
    _row_has_current_training_cache,
    _validate_cached_row,
    build_cache_scorer,
    run_cache_step,
)
from self_summarization_agent.trajectory import is_training_cache_current
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
                "version": 5,
                "input_ids": [1, index + 2],
                "labels": [index + 2, index + 3],
                "completion_mask": [False, True],
                "reference_logprob": -0.5 - index,
                "reference_logprobs": [0.0, -0.5 - index],
                "reference_logprob_source": "policy_rescore",
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
        "trainable_sample_count": 1,
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
        "trajectory_records": [
            {
                "schema_version": 3,
                "query_id": query_id,
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
                "prompt": "debug transcript",
            }
        ],
        "turn_rewards": {"trajectory-1": 1.0},
        "judge": {"outcome": "correct_answer", "parse_error": False},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def summary_interval_row(query_id: str = "q1") -> dict:
    row = judged_row(query_id, 0)
    row["summary_turns"] = ["summary-1"]
    record = row["trajectory_records"][0]
    record["termination_kind"] = "compaction"
    record["turn_ids"] = ["tool-1", "summary-1"]
    record["collection_tokens"] = {
        "version": 2,
        "full_token_ids": [10, 11, 20, 21],
        "assistant_token_mask": [False, True, False, True],
        "generations": [
            {
                "prompt_token_ids": [10],
                "completion_token_ids": [11],
                "full_token_ids": [10, 11],
                "completion_token_logprobs": [-0.1],
                "logprobs_mode": "raw_logprobs",
            },
            {
                "prompt_token_ids": [10, 11, 20],
                "completion_token_ids": [21],
                "full_token_ids": [10, 11, 20, 21],
                "completion_token_logprobs": [-0.3],
                "logprobs_mode": "raw_logprobs",
            },
        ],
    }
    return row


def test_native_cache_masks_only_compaction_generation_for_ablation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    judged_path = tmp_path / "judged.jsonl"
    output_path = tmp_path / "cached.jsonl"
    write_jsonl(judged_path, [summary_interval_row()])
    config = train_config(tmp_path)
    config.training.train_compaction_tokens = False

    run_cache_step(
        config,
        checkpoint_path=checkpoint,
        rollout_path=judged_path,
        output_path=output_path,
        scorer=FakeScorer(),
    )

    cache = json.loads(output_path.read_text(encoding="utf-8"))["trajectory_records"][0][
        "training_cache"
    ]
    assert cache["input_ids"] == [10, 11, 20]
    assert cache["labels"] == [11, 20, 21]
    assert cache["completion_mask"] == [True, False, False]
    assert cache["reference_logprobs"] == [-0.1, 0.0, -0.3]
    assert cache["reference_logprob"] == -0.1
    assert cache["loss_mask_policy"] == "tool_calls_only"
    assert is_training_cache_current(cache, train_compaction_tokens=False)
    assert not is_training_cache_current(cache, train_compaction_tokens=True)


def test_fallback_cache_uses_same_compaction_mask() -> None:
    row = summary_interval_row()
    payload = {
        "version": 5,
        "input_ids": [10, 11, 20],
        "labels": [11, 20, 21],
        "completion_mask": [True, False, True],
        "reference_logprob": -0.2,
        "reference_logprobs": [-0.1, 0.0, -0.3],
        "reference_logprob_source": "policy_rescore",
    }

    cached = _attach_training_caches(
        row,
        cache_payloads=[payload],
        checkpoint_id="step-00001",
        train_compaction_tokens=False,
    )

    cache = cached["trajectory_records"][0]["training_cache"]
    assert cache["completion_mask"] == [True, False, False]
    assert cache["reference_logprob"] == -0.1
    assert cache["loss_mask_policy"] == "tool_calls_only"


def summary_only_row(query_id: str = "q1") -> dict:
    """A row whose only trajectory record is a pure summary generation."""
    row = judged_row(query_id, 0)
    row["summary_turns"] = ["summary-1"]
    record = row["trajectory_records"][0]
    record["termination_kind"] = "compaction"
    record["turn_ids"] = ["summary-1"]
    record["collection_tokens"] = {
        "version": 2,
        "full_token_ids": [10, 11, 21],
        "assistant_token_mask": [False, False, True],
        "generations": [
            {
                "prompt_token_ids": [10, 11],
                "completion_token_ids": [21],
                "full_token_ids": [10, 11, 21],
                "completion_token_logprobs": [-0.3],
                "logprobs_mode": "raw_logprobs",
            },
        ],
    }
    return row


def test_attach_training_caches_skips_excluded_summary_only_record() -> None:
    row = summary_only_row()
    payload = {
        "version": 5,
        "input_ids": [10, 11],
        "labels": [11, 21],
        "completion_mask": [False, True],
        "reference_logprob": -0.3,
        "reference_logprobs": [0.0, -0.3],
        "reference_logprob_source": "policy_rescore",
    }

    cached = _attach_training_caches(
        row,
        cache_payloads=[payload],
        checkpoint_id="step-00001",
        train_compaction_tokens=False,
    )

    assert "training_cache" not in cached["trajectory_records"][0]


def test_summary_only_row_counts_as_current_cache_under_ablation() -> None:
    row = summary_only_row()

    assert _row_has_current_training_cache(row, train_compaction_tokens=False)
    assert not _row_has_current_training_cache(row, train_compaction_tokens=True)
    _validate_cached_row(
        row,
        index=1,
        expected_checkpoint_id="step-00001",
        train_compaction_tokens=False,
    )


def test_cache_step_excludes_summary_only_record_from_training(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    judged_path = tmp_path / "judged.jsonl"
    output_path = tmp_path / "cached.jsonl"
    write_jsonl(judged_path, [summary_only_row()])
    config = train_config(tmp_path)
    config.training.train_compaction_tokens = False
    scorer = FakeScorer()

    run_cache_step(
        config,
        checkpoint_path=checkpoint,
        rollout_path=judged_path,
        output_path=output_path,
        scorer=scorer,
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))["trajectory_records"][0]
    assert "training_cache" not in record
    assert scorer.seen_batches == []


def test_cache_step_writes_training_cache_for_each_interval(tmp_path: Path) -> None:
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
    assert scorer.seen_batches == [["trajectory-1"]]
    cache = rows[0]["trajectory_records"][0]["training_cache"]
    assert cache["input_ids"] == [1, 2]
    assert cache["version"] == 5
    assert cache["reference_logprob"] == -0.5
    assert cache["reference_logprobs"] == [0.0, -0.5]
    assert cache["policy_checkpoint_id"] == "step-00001"
    assert cache["completion_mask"] == [False, True]


def test_cache_step_reuses_rollout_native_cache_without_rescoring(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    judged_path = tmp_path / "judged.jsonl"
    output_path = tmp_path / "cached.jsonl"
    row = judged_row("q1", 0)
    row["trajectory_records"][0]["collection_tokens"] = {
        "version": 2,
        "full_token_ids": [1, 2],
        "assistant_token_mask": [False, True],
        "generations": [
            {
                "prompt_token_ids": [1],
                "completion_token_ids": [2],
                "full_token_ids": [1, 2],
                "completion_token_logprobs": [-0.25],
                "logprobs_mode": "raw_logprobs",
            }
        ],
    }
    write_jsonl(judged_path, [row])
    scorer = FakeScorer()

    run_cache_step(
        train_config(tmp_path),
        checkpoint_path=checkpoint,
        rollout_path=judged_path,
        output_path=output_path,
        scorer=scorer,
    )

    cached_row = json.loads(output_path.read_text(encoding="utf-8"))
    cache = cached_row["trajectory_records"][0]["training_cache"]
    assert scorer.seen_batches == []
    assert cache["reference_logprob_source"] == "vllm_raw_rollout"
    assert cache["policy_checkpoint_id"] == "step-00001"


def test_cache_step_resume_skips_completed_cached_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    judged_path = tmp_path / "judged.jsonl"
    output_path = tmp_path / "cached.jsonl"
    rows = [judged_row("q1", 0), judged_row("q2", 1)]
    write_jsonl(judged_path, rows)
    first_cached = judged_row("q1", 0)
    for record in first_cached["trajectory_records"]:
        record["training_cache"] = {
            "version": 5,
            "input_ids": [1],
            "labels": [2],
            "completion_mask": [True],
            "reference_logprob": -0.1,
            "reference_logprobs": [-0.1],
            "reference_logprob_source": "policy_rescore",
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
    assert scorer.seen_batches == [["trajectory-1"]]
    assert [(row["query_id"], row["rollout_index"]) for row in cached_rows] == [("q1", 0), ("q2", 1)]
    assert cached_rows[0]["trajectory_records"][0]["training_cache"]["version"] == 5
    assert cached_rows[1]["trajectory_records"][0]["training_cache"]["version"] == 5


def test_cache_step_resume_rewrites_old_cached_rows_to_v5(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    judged_path = tmp_path / "judged.jsonl"
    output_path = tmp_path / "cached.jsonl"
    row = judged_row("q1", 0)
    write_jsonl(judged_path, [row])
    existing_cached = judged_row("q1", 0)
    for record in existing_cached["trajectory_records"]:
        record["training_cache"] = {
            "version": 2,
            "input_ids": [1],
            "labels": [2],
            "completion_mask": [True],
            "reference_logprob": -0.1,
        }
    write_jsonl(output_path, [existing_cached])
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
    assert scorer.seen_batches == [["trajectory-1"]]
    assert len(cached_rows) == 1
    assert cached_rows[0]["trajectory_records"][0]["training_cache"]["version"] == 5
    assert cached_rows[0]["trajectory_records"][0]["training_cache"]["reference_logprobs"] == [0.0, -0.5]


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


def test_build_cache_scorer_accepts_verl_fsdp_worker_with_transformers_cache(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    config = train_config(tmp_path)
    config.training.backend = "verl_ray"
    config.training.verl.worker_backend = "verl_fsdp"
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
        assert "training.verl.worker_backend='transformers' or 'verl_fsdp'" in str(exc)
    else:
        raise AssertionError("Expected unsupported verl worker backend to be rejected")
