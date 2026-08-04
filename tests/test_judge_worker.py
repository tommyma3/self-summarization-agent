from queue import Queue
from types import SimpleNamespace

from self_summarization_agent import judge_worker
from self_summarization_agent.judge import JudgeDecision


class RecordingBatchJudge:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_batch(self, items):
        self.batch_sizes.append(len(items))
        return [
            JudgeDecision(
                outcome="correct_answer",
                judge_prompt="prompt",
                judge_response="response",
                parse_error=False,
            )
            for _ in items
        ]


def _raw_row(query_id: str, rollout_index: int) -> dict:
    return {
        "policy_checkpoint_id": "iteration-00000",
        "query_id": query_id,
        "rollout_index": rollout_index,
        "status": "completed",
        "final_answer": "answer",
        "turn_records": [],
        "trajectory_records": [],
        "summary_turns": [],
    }


def _message(batch_id: int, query_id: str) -> dict:
    return {
        "batch_id": batch_id,
        "rows": [_raw_row(query_id, 0)],
        "examples_by_query_id": {
            query_id: {"query_id": query_id, "query": "question", "answer": "answer"}
        },
        "expected_checkpoint_id": "iteration-00000",
    }


def test_judge_worker_microbatches_multiple_streamed_submissions(monkeypatch) -> None:
    judge = RecordingBatchJudge()
    config = SimpleNamespace(judge=SimpleNamespace(batch_size=8, batch_wait_ms=100))
    monkeypatch.setattr(judge_worker, "load_train_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(judge_worker, "build_judge", lambda _config: judge)
    request_queue = Queue()
    response_queue = Queue()
    request_queue.put(_message(0, "q1"))
    request_queue.put(_message(1, "q2"))
    request_queue.put(judge_worker.SHUTDOWN)

    judge_worker.run_judge_worker(
        config_path="train.yaml",
        overrides=[],
        request_queue=request_queue,
        response_queue=response_queue,
    )

    responses = [response_queue.get_nowait(), response_queue.get_nowait()]
    assert judge.batch_sizes == [2]
    assert [response["batch_id"] for response in responses] == [0, 1]
    assert all(response["rows"][0]["judge"]["outcome"] == "correct_answer" for response in responses)


def test_judge_worker_caps_large_submission_at_configured_batch_size(monkeypatch) -> None:
    judge = RecordingBatchJudge()
    config = SimpleNamespace(judge=SimpleNamespace(batch_size=2, batch_wait_ms=0))
    monkeypatch.setattr(judge_worker, "load_train_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(judge_worker, "build_judge", lambda _config: judge)
    request_queue = Queue()
    response_queue = Queue()
    message = _message(0, "q1")
    message["rows"] = [
        _raw_row(f"q{index}", 0)
        for index in range(3)
    ]
    message["examples_by_query_id"] = {
        f"q{index}": {
            "query_id": f"q{index}",
            "query": "question",
            "answer": "answer",
        }
        for index in range(3)
    }
    request_queue.put(message)
    request_queue.put(judge_worker.SHUTDOWN)

    judge_worker.run_judge_worker(
        config_path="train.yaml",
        overrides=[],
        request_queue=request_queue,
        response_queue=response_queue,
    )

    response = response_queue.get_nowait()
    assert judge.batch_sizes == [2, 1]
    assert len(response["rows"]) == 3
