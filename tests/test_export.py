from self_summarization_agent.export import build_run_record
from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.models import RuntimeResult
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel


def test_build_run_record_matches_browsecomp_plus_shape() -> None:
    result = RuntimeResult(
        query_id="q1",
        status="completed",
        final_answer="answer",
        summary_turns=["summary-1"],
        turn_rewards={"summary-1": 1.0},
        retrieved_docids=["doc-1", "doc-2"],
        tool_call_counts={"search": 1, "get_document": 1},
    )

    record = build_run_record(result)

    assert record == {
        "query_id": "q1",
        "tool_call_counts": {"search": 1, "get_document": 1},
        "status": "completed",
        "retrieved_docids": ["doc-1", "doc-2"],
        "result": [
            {
                "type": "output_text",
                "output": "answer",
            }
        ],
    }


def test_runtime_result_export_populates_tool_call_counts() -> None:
    backend = FakeBackend(search_index={"clue": ["doc-1"]}, documents={"doc-1": "fact"})
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "clue"}}',
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")
    record = build_run_record(result)

    assert result.tool_call_counts == {"search": 1, "get_document": 0}
    assert record["tool_call_counts"] == {"search": 1, "get_document": 0}
    assert result.retrieved_docids == ["doc-1"]
    assert record["retrieved_docids"] == ["doc-1"]


def test_runtime_export_tracks_all_retrieved_documents_from_search_and_get_document() -> None:
    backend = FakeBackend(search_index={"clue": ["doc-1"]}, documents={"doc-1": "fact"})
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "clue"}}',
            '{"tool_name": "get_document", "arguments": {"doc_id": "doc-1"}}',
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")
    record = build_run_record(result)

    assert result.tool_call_counts == {"search": 1, "get_document": 1}
    assert result.retrieved_docids == ["doc-1"]
    assert record["tool_call_counts"] == {"search": 1, "get_document": 1}
    assert record["retrieved_docids"] == ["doc-1"]
