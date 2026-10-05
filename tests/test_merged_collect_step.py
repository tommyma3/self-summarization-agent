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
        training=SimpleNamespace(
            gradient_accumulation_microbatch_size=2,
            train_compaction_tokens=False,
        )
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
        lambda row, *, cache_payloads, checkpoint_id, **_kwargs: {
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

    assert response_queue.get_nowait() == merged_collect_step.READY
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


def test_merged_collect_overlap_starts_retrieval_before_judge(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    collection_complete = False

    class RetrievalProcess:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

    retrieval_process = RetrievalProcess()

    class JudgeClient:
        def submit(self, rows, examples):
            del rows, examples

        def drain_available(self):
            return []

        def finish(self):
            return []

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
        retrieval=SimpleNamespace(
            persistent_worker=True, worker_startup_timeout_seconds=10, gpu_ids=[0]
        ),
        rollout=SimpleNamespace(overlap_judge=True, overlap_queue_max_batches=2),
        judge=SimpleNamespace(enabled=True, batch_size=2, gpu_ids=[0, 1]),
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

    def start_retrieval(**kwargs):
        assert kwargs["gpu_ids"] == [0]
        events.append("retrieval_start")
        return retrieval_process, "http://worker"

    monkeypatch.setattr(merged_collect_step, "_start_retrieval_worker", start_retrieval)

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
        # Overlap mode: the judge engine initializes while the retrieval
        # worker is still resident, so vLLM profiles the shared GPU around
        # the retrieval footprint instead of starving it.
        assert not retrieval_process.stopped
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
        "judge_start",
        "collect_eval",
        "collect_train",
        "retrieval_stop",
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


# ---------------------------------------------------------------------------
# Overlap judge feed
# ---------------------------------------------------------------------------


class FakeJudgingClient:
    """Deterministic stand-in for the overlap judge clients."""

    def __init__(self, fail_on_submit: int | None = None) -> None:
        self._pending: list[dict] = []
        self.submitted_rows: list[dict] = []
        self.submit_calls = 0
        self.closed = False
        self._fail_on_submit = fail_on_submit

    def submit(self, rows, examples) -> None:
        self.submit_calls += 1
        if self._fail_on_submit is not None and self.submit_calls > self._fail_on_submit:
            raise RuntimeError("judge worker unavailable")
        examples_by_id = {example.query_id: example for example in examples}
        for row in rows:
            example = examples_by_id[row["query_id"]]
            judged = dict(row)
            judged["turn_rewards"] = {}
            judged["judge"] = {
                "outcome": (
                    "correct_answer" if row.get("final_answer") == example.answer else "wrong_answer"
                ),
                "judge_prompt": None,
                "judge_response": None,
                "parse_error": False,
                "rollout_index": row.get("rollout_index"),
            }
            self._pending.append(judged)
            self.submitted_rows.append(row)

    def drain_available(self) -> list[dict]:
        rows, self._pending = self._pending, []
        return rows

    def finish(self) -> list[dict]:
        return self.drain_available()

    def metrics(self) -> dict:
        return {}

    def close(self) -> None:
        self.closed = True


class FinishOnlyJudgeClient(FakeJudgingClient):
    """Holds every judged row until finish(), like a slow real worker."""

    def drain_available(self) -> list[dict]:
        return []

    def finish(self) -> list[dict]:
        rows, self._pending = self._pending, []
        return rows


def _feeder_row(query_id: str) -> dict:
    return {"query_id": query_id, "rollout_index": 0, "final_answer": "answer"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_take_chunk_rows_routes_by_submission_order() -> None:
    chunks = [("eval", 2), ("train", 2), ("eval", 1)]
    rows = [{"i": i} for i in range(5)]

    distributed = merged_collect_step._take_chunk_rows(rows, chunks)

    assert [(split, [row["i"] for row in chunk_rows]) for split, chunk_rows in distributed] == [
        ("eval", [0, 1]),
        ("train", [2, 3]),
        ("eval", [4]),
    ]
    assert chunks == []


def test_take_chunk_rows_rejects_partial_and_orphan_rows() -> None:
    try:
        merged_collect_step._take_chunk_rows([{"i": 0}], [("eval", 2)])
    except RuntimeError as exc:
        assert "partial response" in str(exc)
    else:
        raise AssertionError("partial response must fail")

    try:
        merged_collect_step._take_chunk_rows([{"i": 0}], [])
    except RuntimeError as exc:
        assert "without a matching submission" in str(exc)
    else:
        raise AssertionError("orphan rows must fail")


def test_overlap_judge_feeder_batches_routes_and_appends(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.judged.jsonl"
    train_path = tmp_path / "train.judged.jsonl"
    client = FakeJudgingClient()
    feed_queue = Queue()
    examples = {
        "qe1": QueryExample(query_id="qe1", query="e1", answer="answer"),
        "qt1": QueryExample(query_id="qt1", query="t1", answer="answer"),
        "qt2": QueryExample(query_id="qt2", query="t2", answer="other"),
    }
    stats: dict = {}

    feed_queue.put(("eval", _feeder_row("qe1")))
    feed_queue.put(("bogus", _feeder_row("qe1")))  # split without a judged output
    feed_queue.put(("train", _feeder_row("qt1")))
    feed_queue.put(("train", _feeder_row("qt2")))  # fills the train batch -> flush
    feed_queue.put(("eval", _feeder_row("qe1")))  # fills the eval batch -> flush
    feed_queue.put(merged_collect_step._FEED_SENTINEL)

    merged_collect_step._overlap_judge_feeder(
        feed_queue=feed_queue,
        judge_client=client,
        judged_output_paths={"eval": eval_path, "train": train_path},
        examples_by_query_id=examples,
        batch_size=2,
        stats=stats,
    )

    assert stats == {"fed_rows": 4, "judged_rows": 4, "failure": None}
    assert client.submit_calls == 2
    eval_rows = _read_jsonl(eval_path)
    train_rows = _read_jsonl(train_path)
    assert [row["query_id"] for row in eval_rows] == ["qe1", "qe1"]
    assert [row["query_id"] for row in train_rows] == ["qt1", "qt2"]
    assert all(row["judge"]["outcome"] == "correct_answer" for row in eval_rows)
    assert [row["judge"]["outcome"] for row in train_rows] == [
        "correct_answer",
        "wrong_answer",
    ]


def test_overlap_judge_feeder_routes_finish_only_rows_across_chunks(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.judged.jsonl"
    train_path = tmp_path / "train.judged.jsonl"
    client = FinishOnlyJudgeClient()
    feed_queue = Queue()
    examples = {
        "qe1": QueryExample(query_id="qe1", query="e1", answer="answer"),
        "qt1": QueryExample(query_id="qt1", query="t1", answer="answer"),
        "qt2": QueryExample(query_id="qt2", query="t2", answer="answer"),
    }
    stats: dict = {}

    feed_queue.put(("eval", _feeder_row("qe1")))
    feed_queue.put(("train", _feeder_row("qt1")))
    feed_queue.put(("train", _feeder_row("qt2")))
    feed_queue.put(merged_collect_step._FEED_SENTINEL)

    merged_collect_step._overlap_judge_feeder(
        feed_queue=feed_queue,
        judge_client=client,
        judged_output_paths={"eval": eval_path, "train": train_path},
        examples_by_query_id=examples,
        batch_size=2,
        stats=stats,
    )

    # Both chunks stay pending until finish(); the final drain must still
    # route each row to its own split file.
    assert stats == {"fed_rows": 3, "judged_rows": 3, "failure": None}
    assert [row["query_id"] for row in _read_jsonl(eval_path)] == ["qe1"]
    assert [row["query_id"] for row in _read_jsonl(train_path)] == ["qt1", "qt2"]


def test_overlap_judge_feeder_failure_drains_queue_and_reports(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.judged.jsonl"
    client = FakeJudgingClient(fail_on_submit=0)
    feed_queue = Queue()
    examples = {"qe1": QueryExample(query_id="qe1", query="e1", answer="answer")}
    stats: dict = {}

    feed_queue.put(("eval", _feeder_row("qe1")))
    feed_queue.put(("eval", _feeder_row("qe1")))  # triggers the failing flush
    feed_queue.put(("eval", _feeder_row("qe1")))  # must still be drained
    feed_queue.put(merged_collect_step._FEED_SENTINEL)

    merged_collect_step._overlap_judge_feeder(
        feed_queue=feed_queue,
        judge_client=client,
        judged_output_paths={"eval": eval_path},
        examples_by_query_id=examples,
        batch_size=2,
        stats=stats,
    )

    assert isinstance(stats["failure"], RuntimeError)
    assert feed_queue.empty()
    assert _read_jsonl(eval_path) == []


class FinishFailingJudgeClient(FakeJudgingClient):
    def finish(self) -> list[dict]:
        raise RuntimeError("judge worker died at finish")


def test_overlap_judge_feeder_failure_after_sentinel_returns(tmp_path: Path) -> None:
    # A failure during the post-sentinel flush must not wait for a second
    # sentinel: the parent sends exactly one, already consumed by the main
    # loop, and the queue is empty once collection children have exited.
    eval_path = tmp_path / "eval.judged.jsonl"
    client = FinishFailingJudgeClient()
    feed_queue = Queue()
    examples = {"qe1": QueryExample(query_id="qe1", query="e1", answer="answer")}
    stats: dict = {}

    feed_queue.put(("eval", _feeder_row("qe1")))
    feed_queue.put(merged_collect_step._FEED_SENTINEL)

    merged_collect_step._overlap_judge_feeder(
        feed_queue=feed_queue,
        judge_client=client,
        judged_output_paths={"eval": eval_path},
        examples_by_query_id=examples,
        batch_size=2,
        stats=stats,
    )

    assert isinstance(stats["failure"], RuntimeError)
    # The pre-failure drain already wrote its judged row; only the rows that
    # were still in flight at finish() stay unwritten for the fallback phase.
    assert [row["query_id"] for row in _read_jsonl(eval_path)] == ["qe1"]


# ---------------------------------------------------------------------------
# Merged collect with overlap judging enabled
# ---------------------------------------------------------------------------

# Captured before any monkeypatching so spies never chain across scenarios.
_REAL_JUDGE_SPLIT = merged_collect_step._judge_split


def _run_overlap_scenario(
    tmp_path: Path,
    monkeypatch,
    *,
    overlap: bool,
    judge_factory,
    events: list[str],
) -> tuple[dict, list[str]]:
    """Drive run_merged_collect with scripted collection and judge fakes."""

    train_examples = [
        QueryExample(query_id="qt1", query="question t1", answer="answer"),
        QueryExample(query_id="qt2", query="question t2", answer="answer"),
    ]
    eval_examples = [QueryExample(query_id="qe1", query="question e1", answer="answer")]

    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=1, bc_plus_root=tmp_path),
        dataset=SimpleNamespace(train_limit=2, eval_limit=1),
        collection=SimpleNamespace(train_task_count=None, eval_task_count=None),
        training=SimpleNamespace(
            rollout_query_count=None, group_size=1, train_compaction_tokens=False
        ),
        evaluation=SimpleNamespace(samples_per_task=1),
        retrieval=SimpleNamespace(
            persistent_worker=False, worker_startup_timeout_seconds=10, gpu_ids=[]
        ),
        rollout=SimpleNamespace(overlap_judge=overlap, overlap_queue_max_batches=2),
        judge=SimpleNamespace(enabled=True, batch_size=2, gpu_ids=[]),
    )
    monkeypatch.setattr(
        merged_collect_step,
        "load_query_examples",
        lambda *_a, **_k: [*train_examples, *eval_examples],
    )
    monkeypatch.setattr(
        merged_collect_step,
        "split_train_eval_examples",
        lambda *_a, **_k: (train_examples, eval_examples),
    )
    monkeypatch.setattr(merged_collect_step, "_expected_eval_rollout_count", lambda _c: 1)
    monkeypatch.setattr(merged_collect_step, "_expected_train_rollout_count", lambda _c: 2)

    def has_complete_raw(path, *, checkpoint_id, expected_count, **_kwargs):
        path = Path(path)
        if not path.exists():
            return False
        rows = _read_jsonl(path)
        return len(rows) == expected_count and all(
            row.get("policy_checkpoint_id") == checkpoint_id for row in rows
        )

    monkeypatch.setattr(merged_collect_step, "_has_complete_raw_rollouts", has_complete_raw)
    monkeypatch.setattr(merged_collect_step, "_build_overlap_judge_client", judge_factory)
    monkeypatch.setattr(
        merged_collect_step,
        "resolved_rollout_sampling_profile",
        lambda _c, *, split: {"split": split},
    )
    monkeypatch.setattr(merged_collect_step, "sampling_profile_id", lambda profile: profile["split"])

    judge_split_resumes: list[str] = []
    real_judge_split = _REAL_JUDGE_SPLIT

    def spy_judge_split(**kwargs):
        judge_split_resumes.append(f"{kwargs['split']}:{kwargs['resume']}")
        return real_judge_split(**kwargs)

    monkeypatch.setattr(merged_collect_step, "_judge_split", spy_judge_split)

    def fake_collect_process(**kwargs):
        split = kwargs["split"]
        events.append(f"collect_{split}")
        feed = kwargs.get("row_feed_queue")
        raw_path = Path(kwargs["raw_output_path"])
        for example in (eval_examples if split == "eval" else train_examples):
            row = {
                "policy_checkpoint_id": "iteration-00000",
                "query_id": example.query_id,
                "rollout_index": 0,
                "rollout_split": split,
                "sampling_profile_id": split,
                "trajectory_records": [],
                "turn_records": [],
                "summary_turns": [],
                "status": "completed",
                "final_answer": "answer",
            }
            merged_collect_step.append_jsonl(raw_path, row)
            if feed is not None:
                feed.put((split, row))

    monkeypatch.setattr(merged_collect_step, "_run_split_collection_process", fake_collect_process)

    outputs = merged_collect_step.run_merged_collect(
        config,
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "iteration-00000",
        train_raw_output=tmp_path / "train.raw.jsonl",
        train_judged_output=tmp_path / "train.judged.jsonl",
        train_cached_output=None,
        eval_raw_output=tmp_path / "eval.raw.jsonl",
        eval_judged_output=tmp_path / "eval.judged.jsonl",
        eval_metrics_output=None,
        sample_seed=None,
        resume=False,
        overrides=[],
    )
    return outputs, judge_split_resumes


def test_merged_collect_overlap_judging_matches_sequential_judging(
    tmp_path: Path, monkeypatch
) -> None:
    overlap_dir = tmp_path / "overlap"
    sequential_dir = tmp_path / "sequential"
    overlap_dir.mkdir()
    sequential_dir.mkdir()

    overlap_events: list[str] = []
    overlap_clients: list[FakeJudgingClient] = []
    sequential_events: list[str] = []
    sequential_clients: list[FakeJudgingClient] = []

    def make_factory(events, clients):
        def factory(**_kwargs):
            events.append("judge_start")
            client = FakeJudgingClient()
            clients.append(client)
            return client

        return factory

    (
        overlap_outputs,
        overlap_resumes,
    ) = _run_overlap_scenario(
        overlap_dir,
        monkeypatch,
        overlap=True,
        judge_factory=make_factory(overlap_events, overlap_clients),
        events=overlap_events,
    )
    (
        sequential_outputs,
        sequential_resumes,
    ) = _run_overlap_scenario(
        sequential_dir,
        monkeypatch,
        overlap=False,
        judge_factory=make_factory(sequential_events, sequential_clients),
        events=sequential_events,
    )

    # Identical judged artifacts on both paths.
    for name in ("eval.judged.jsonl", "train.judged.jsonl"):
        overlap_rows = _read_jsonl(overlap_dir / name)
        sequential_rows = _read_jsonl(sequential_dir / name)
        assert overlap_rows == sequential_rows
        assert len(overlap_rows) > 0

    # The overlap run starts its single judge worker before collection and
    # never rebuilds it; the sequential run starts judging after collection.
    assert overlap_events.index("judge_start") < overlap_events.index("collect_eval")
    assert sequential_events.index("judge_start") > sequential_events.index("collect_train")
    assert len(overlap_clients) == 1
    assert len(sequential_clients) == 1
    assert overlap_clients[0].closed
    assert sequential_clients[0].closed

    # Collection streamed every row to the judge feed on the overlap path.
    assert overlap_clients[0].submit_calls > 0
    assert sorted(row["query_id"] for row in overlap_clients[0].submitted_rows) == sorted(
        row["query_id"] for row in sequential_clients[0].submitted_rows
    )

    # The post-collection phase resumes (never rewrites) overlap-fed splits.
    assert overlap_resumes == ["eval:True", "train:True"]
    assert sequential_resumes == ["eval:False", "train:False"]

    assert overlap_outputs["eval_judged"] == overlap_dir / "eval.judged.jsonl"
    assert sequential_outputs["train_judged"] == sequential_dir / "train.judged.jsonl"


def test_merged_collect_overlap_feed_failure_falls_back_to_fresh_judge(
    tmp_path: Path, monkeypatch
) -> None:
    # The overlap client judges the eval row, then dies; the post-collection
    # phase must rebuild a fresh judge and produce complete, duplicate-free
    # judged artifacts.
    first_client = FakeJudgingClient(fail_on_submit=1)
    second_client = FakeJudgingClient()
    clients = [first_client, second_client]
    factory_calls: list[str] = []

    events: list[str] = []

    def factory(**_kwargs):
        factory_calls.append("judge_start")
        events.append("judge_start")
        return clients[len(factory_calls) - 1]

    _outputs, resumes = _run_overlap_scenario(
        tmp_path,
        monkeypatch,
        overlap=True,
        judge_factory=factory,
        events=events,
    )

    assert len(factory_calls) == 2
    assert first_client.closed
    assert second_client.closed
    assert events.count("judge_start") == 2
    assert events.index("judge_start") < events.index("collect_eval")

    # Every raw row judged exactly once in the final artifacts.
    eval_rows = _read_jsonl(tmp_path / "eval.judged.jsonl")
    train_rows = _read_jsonl(tmp_path / "train.judged.jsonl")
    assert [(row["query_id"], row["rollout_index"]) for row in eval_rows] == [("qe1", 0)]
    assert [(row["query_id"], row["rollout_index"]) for row in train_rows] == [
        ("qt1", 0),
        ("qt2", 0),
    ]
    assert all("judge" in row and "turn_rewards" in row for row in eval_rows + train_rows)

    # The fallback re-judges from scratch (resume flag stays False) because the
    # overlap feed only managed a partial, discarded write.
    assert resumes == ["eval:False", "train:False"]
