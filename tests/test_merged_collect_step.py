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


def test_collect_split_resume_does_not_judge_existing_raw_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "train.raw.jsonl"
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
    merged_collect_step._collect_split(
        config=config,
        checkpoint_id="iteration-00000",
        checkpoint_path=tmp_path / "checkpoint",
        generator=object(),
        backend=object(),
        split="train",
        examples=[QueryExample(query_id="q1", query="question", answer="answer")],
        raw_output_path=raw_path,
        sampling_profile={"extra_sampling_params": {}},
        profile_id="profile",
        group_size=1,
        sample_seed=1,
        resume=True,
    )

    assert [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()] == [raw_row]


def test_train_collection_persists_native_cache_before_policy_process_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "train.raw.jsonl"
    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=1),
        collection=SimpleNamespace(train_task_count=None, eval_task_count=None),
        training=SimpleNamespace(rollout_query_count=None, train_compaction_tokens=True),
        rollout=SimpleNamespace(max_concurrent_episodes=2),
        runtime=object(),
    )

    class OneResultRuntime:
        def run_many_stream(self, episodes, *, max_active_episodes):
            assert list(episodes) == [("q1", "question")]
            assert max_active_episodes == 2
            yield [(0, object())]

    monkeypatch.setattr(
        merged_collect_step,
        "build_runtime",
        lambda *_args, **_kwargs: OneResultRuntime(),
    )
    monkeypatch.setattr(
        merged_collect_step,
        "serialize_runtime_result",
        lambda *_args, **_kwargs: {
            "query_id": "q1",
            "trajectory_records": [{"turn_id": "interval-0"}],
            "turn_records": [],
            "summary_turns": [],
            "status": "completed",
            "final_answer": "answer",
        },
    )

    def materialize(row, *, checkpoint_id, train_compaction_tokens):
        enriched = dict(row)
        enriched["trajectory_records"] = [
            {
                **row["trajectory_records"][0],
                "training_cache": {"policy_checkpoint_id": checkpoint_id},
            }
        ]
        return enriched

    monkeypatch.setattr(
        merged_collect_step,
        "_materialize_rollout_native_training_caches",
        materialize,
    )

    merged_collect_step._collect_split(
        config=config,
        checkpoint_id="iteration-00000",
        checkpoint_path=tmp_path / "checkpoint",
        generator=object(),
        backend=object(),
        split="train",
        examples=[QueryExample(query_id="q1", query="question", answer="answer")],
        raw_output_path=raw_path,
        sampling_profile={"extra_sampling_params": {}},
        profile_id="profile",
        group_size=1,
        sample_seed=1,
        resume=False,
    )

    raw_row = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw_row["trajectory_records"][0]["training_cache"]["policy_checkpoint_id"] == (
        "iteration-00000"
    )
    assert "turn_rewards" not in raw_row


def test_judge_split_resumes_unjudged_raw_rows_without_policy_generation(
    tmp_path: Path,
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
        judge=SimpleNamespace(batch_size=4),
    )
    judge_client = RecordingResumeJudgeClient()

    merged_collect_step._judge_split(
        config=config,
        checkpoint_id="iteration-00000",
        judge_client=judge_client,
        split="train",
        examples=[QueryExample(query_id="q1", query="question", answer="answer")],
        raw_output_path=raw_path,
        judged_output_path=judged_path,
        group_size=1,
        sample_seed=1,
        profile_id="profile",
        resume=True,
    )

    assert len(judge_client.submitted_rows) == 1
    judged_rows = [json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines()]
    assert judged_rows[0]["query_id"] == "q1"
    assert judged_rows[0]["judge"]["outcome"] == "correct_answer"


def test_merged_collect_stops_retrieval_before_starting_judge(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    collection_complete = False

    class RetrievalProcess:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

    retrieval_process = RetrievalProcess()

    class JudgeClient:
        def metrics(self):
            return {}

        def close(self):
            events.append("judge_stop")

    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=1, bc_plus_root=tmp_path),
        dataset=SimpleNamespace(train_limit=1, eval_limit=1),
        collection=SimpleNamespace(train_task_count=None, eval_task_count=None),
        training=SimpleNamespace(rollout_query_count=None, group_size=1),
        evaluation=SimpleNamespace(samples_per_task=1),
        retrieval=SimpleNamespace(persistent_worker=True, worker_startup_timeout_seconds=10),
        rollout=SimpleNamespace(overlap_queue_max_batches=2),
        judge=SimpleNamespace(enabled=True, batch_size=2),
    )
    examples = [QueryExample(query_id="q1", query="question", answer="answer")]
    monkeypatch.setattr(merged_collect_step, "load_query_examples", lambda *_a, **_k: examples)
    monkeypatch.setattr(merged_collect_step, "split_train_eval_examples", lambda *_a, **_k: (examples, examples))
    monkeypatch.setattr(merged_collect_step, "_expected_eval_rollout_count", lambda _c: 1)
    monkeypatch.setattr(merged_collect_step, "_expected_train_rollout_count", lambda _c: 1)
    monkeypatch.setattr(
        merged_collect_step,
        "_has_complete_raw_rollouts",
        lambda *_a, **_k: collection_complete,
    )
    monkeypatch.setattr(
        merged_collect_step,
        "_start_retrieval_worker",
        lambda **_k: (events.append("retrieval_start") or retrieval_process, "http://worker"),
    )

    def stop_retrieval(process, _url):
        events.append("retrieval_stop")
        process.stopped = True

    monkeypatch.setattr(merged_collect_step, "_stop_retrieval_worker", stop_retrieval)

    def collect_process(**kwargs):
        nonlocal collection_complete
        events.append(f"collect_{kwargs['split']}")
        if kwargs["split"] == "train":
            collection_complete = True

    monkeypatch.setattr(merged_collect_step, "_run_split_collection_process", collect_process)

    def build_judge(**_kwargs):
        assert retrieval_process.stopped
        events.append("judge_start")
        return JudgeClient()

    monkeypatch.setattr(merged_collect_step, "_build_overlap_judge_client", build_judge)
    monkeypatch.setattr(
        merged_collect_step,
        "_judge_split",
        lambda **kwargs: events.append(f"judge_{kwargs['split']}"),
    )
    monkeypatch.setattr(merged_collect_step, "write_eval_metrics", lambda **_k: events.append("metrics"))
    monkeypatch.setattr(merged_collect_step, "_run_cache_inline", lambda **_k: events.append("cache"))
    monkeypatch.setattr(
        merged_collect_step,
        "resolved_rollout_sampling_profile",
        lambda _c, *, split: {"split": split},
    )
    monkeypatch.setattr(merged_collect_step, "sampling_profile_id", lambda profile: profile["split"])

    merged_collect_step.run_merged_collect(
        config,
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "iteration-00000",
        train_raw_output=tmp_path / "train.raw.jsonl",
        train_judged_output=tmp_path / "train.judged.jsonl",
        train_cached_output=tmp_path / "train.cached.jsonl",
        eval_raw_output=tmp_path / "eval.raw.jsonl",
        eval_judged_output=tmp_path / "eval.judged.jsonl",
        eval_metrics_output=tmp_path / "eval.metrics.jsonl",
    )

    assert events == [
        "retrieval_start",
        "collect_eval",
        "collect_train",
        "retrieval_stop",
        "judge_start",
        "judge_eval",
        "judge_train",
        "judge_stop",
        "metrics",
        "cache",
    ]


def test_live_retrieval_worker_check_rejects_unexpected_exit() -> None:
    process = SimpleNamespace(poll=lambda: -9)

    try:
        merged_collect_step._require_live_retrieval_worker(process, after_split="eval")
    except RuntimeError as exc:
        assert "eval policy collection" in str(exc)
        assert "code -9" in str(exc)
    else:
        raise AssertionError("dead retrieval worker must fail collection")
