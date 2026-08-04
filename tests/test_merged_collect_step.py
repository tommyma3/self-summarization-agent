import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from self_summarization_agent import merged_collect_step
from self_summarization_agent.dataset import QueryExample


class RecordingCacheScorer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def cache_samples(self, samples):
        self.batch_sizes.append(len(samples))
        return [{"sample": sample} for sample in samples]


def test_cache_overlap_worker_microbatches_samples_across_judged_rows(monkeypatch) -> None:
    scorer = RecordingCacheScorer()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test")
    config = SimpleNamespace(
        training=SimpleNamespace(gradient_accumulation_microbatch_size=2)
    )
    monkeypatch.setattr(
        merged_collect_step,
        "load_train_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        merged_collect_step,
        "build_cache_scorer",
        lambda *_args, **_kwargs: scorer,
    )
    monkeypatch.setattr(
        merged_collect_step,
        "extract_trainable_samples",
        lambda records, _rewards, rollout_id: [records[0]["sample"]],
    )
    monkeypatch.setattr(
        merged_collect_step,
        "_attach_training_caches",
        lambda row, *, cache_payloads, checkpoint_id: {
            **row,
            "cache_payloads": cache_payloads,
            "cache_checkpoint_id": checkpoint_id,
        },
    )
    request_queue = Queue()
    response_queue = Queue()
    rows = [
        {
            "query_id": f"q{index}",
            "rollout_index": 0,
            "trajectory_records": [{"sample": index}],
            "turn_rewards": {},
        }
        for index in range(3)
    ]
    request_queue.put(
        {
            "batch_id": 0,
            "rows": rows,
            "expected_checkpoint_id": "iteration-00000",
        }
    )
    request_queue.put(merged_collect_step._CACHE_SHUTDOWN)

    merged_collect_step._run_cache_overlap_worker(
        config_path="train.yaml",
        overrides=[],
        checkpoint_path="checkpoint",
        request_queue=request_queue,
        response_queue=response_queue,
    )

    response = response_queue.get_nowait()
    assert scorer.batch_sizes == [2, 1]
    assert len(response["rows"]) == 3
    assert [row["cache_payloads"][0]["sample"] for row in response["rows"]] == [0, 1, 2]


def test_cache_overlap_progress_resets_watchdog_deadline() -> None:
    client = object.__new__(merged_collect_step._CacheOverlapClient)
    client.pending_count = 2
    client.completed_row_count = 0
    client._drain_timeout_seconds = 10
    client._drain_deadline = 1.0

    assert client._handle_response({"batch_id": 0, "rows": [{"query_id": "q1"}]})
    assert client.pending_count == 1
    assert client._drain_deadline > 1.0

    client._handle_response({"batch_id": 1, "rows": []})
    assert client.pending_count == 0
    assert client._drain_deadline is None


class RecordingResumeJudgeClient:
    def __init__(self) -> None:
        self.submitted_rows: list[dict] = []

    def submit(self, rows, examples) -> None:
        assert len(rows) == len(examples)
        self.submitted_rows.extend(rows)

    def drain_available(self):
        return []

    def finish(self):
        return [
            {
                **row,
                "turn_rewards": {},
                "judge": {"outcome": "correct_answer"},
            }
            for row in self.submitted_rows
        ]


class RecordingFallbackCacheClient:
    def __init__(self) -> None:
        self.submitted_rows: list[dict] = []
        self.rollout_native_row_count = 0

    def submit(self, rows) -> None:
        self.submitted_rows.extend(rows)

    def record_rollout_native_rows(self, count: int) -> None:
        self.rollout_native_row_count += count

    def drain_available(self):
        return []

    def finish(self):
        return []


def test_collect_split_resume_judges_existing_raw_rows_without_regeneration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "train.raw.jsonl"
    judged_path = tmp_path / "train.judged.jsonl"
    raw_row = {
        "policy_checkpoint_id": "iteration-00000",
        "query_id": "q1",
        "rollout_index": 0,
        "trajectory_records": [],
        "turn_records": [],
        "summary_turns": [],
        "status": "completed",
        "final_answer": "answer",
    }
    raw_path.write_text(json.dumps(raw_row) + "\n", encoding="utf-8")
    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=1),
        collection=SimpleNamespace(train_task_count=None, eval_task_count=None),
        training=SimpleNamespace(rollout_query_count=None),
        rollout=SimpleNamespace(max_concurrent_episodes=2),
        judge=SimpleNamespace(batch_size=4),
        runtime=object(),
    )

    class NoGenerationRuntime:
        def run_many_stream(self, episodes, *, max_active_episodes):
            assert list(episodes) == []
            assert max_active_episodes == 2
            return iter(())

    monkeypatch.setattr(
        merged_collect_step,
        "build_runtime",
        lambda *_args, **_kwargs: NoGenerationRuntime(),
    )
    judge_client = RecordingResumeJudgeClient()

    merged_collect_step._collect_split(
        config=config,
        checkpoint_id="iteration-00000",
        checkpoint_path=tmp_path / "checkpoint",
        generator=object(),
        backend=object(),
        overlap_judge_client=judge_client,
        split="train",
        examples=[QueryExample(query_id="q1", query="question", answer="answer")],
        raw_output_path=raw_path,
        judged_output_path=judged_path,
        sampling_profile={"extra_sampling_params": {}},
        profile_id="profile",
        group_size=1,
        sample_seed=1,
        resume=True,
    )

    assert len(judge_client.submitted_rows) == 1
    judged_rows = [json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines()]
    assert judged_rows[0]["query_id"] == "q1"
    assert judged_rows[0]["judge"]["outcome"] == "correct_answer"


def test_collect_split_resume_persists_inline_cache_without_starting_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "train.raw.jsonl"
    judged_path = tmp_path / "train.judged.jsonl"
    cached_path = tmp_path / "train.cached.jsonl"
    raw_row = {
        "policy_checkpoint_id": "iteration-00000",
        "query_id": "q1",
        "rollout_index": 0,
        "trajectory_records": [],
        "turn_records": [],
        "summary_turns": [],
        "status": "completed",
        "final_answer": "answer",
    }
    judged_row = {
        **raw_row,
        "turn_rewards": {},
        "trainable_sample_count": 0,
        "judge": {"outcome": "correct_answer"},
    }
    raw_path.write_text(json.dumps(raw_row) + "\n", encoding="utf-8")
    judged_path.write_text(json.dumps(judged_row) + "\n", encoding="utf-8")
    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=1),
        collection=SimpleNamespace(train_task_count=None, eval_task_count=None),
        training=SimpleNamespace(rollout_query_count=None),
        rollout=SimpleNamespace(max_concurrent_episodes=2),
        judge=SimpleNamespace(batch_size=4),
        runtime=object(),
    )

    class NoGenerationRuntime:
        def run_many_stream(self, episodes, *, max_active_episodes):
            assert list(episodes) == []
            assert max_active_episodes == 2
            return iter(())

    monkeypatch.setattr(
        merged_collect_step,
        "build_runtime",
        lambda *_args, **_kwargs: NoGenerationRuntime(),
    )
    cache_client = RecordingFallbackCacheClient()

    merged_collect_step._collect_split(
        config=config,
        checkpoint_id="iteration-00000",
        checkpoint_path=tmp_path / "checkpoint",
        generator=object(),
        backend=object(),
        overlap_judge_client=RecordingResumeJudgeClient(),
        split="train",
        examples=[QueryExample(query_id="q1", query="question", answer="answer")],
        raw_output_path=raw_path,
        judged_output_path=judged_path,
        sampling_profile={"extra_sampling_params": {}},
        profile_id="profile",
        group_size=1,
        sample_seed=1,
        resume=True,
        cache_overlap_client=cache_client,
        cached_output_path=cached_path,
    )

    assert cache_client.submitted_rows == []
    assert cache_client.rollout_native_row_count == 1
    cached_rows = [json.loads(line) for line in cached_path.read_text(encoding="utf-8").splitlines()]
    assert cached_rows == [judged_row]
