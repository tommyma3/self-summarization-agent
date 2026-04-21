import contextlib
import io
import json

import main as cli_entrypoint
from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.cli import build_smoke_run_record
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel, parse_model_tool_call
from self_summarization_agent.trajectory import extract_trainable_samples


class RecordingModel(ScriptedModel):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs=outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return super().generate(prompt)


def test_fake_backend_returns_search_hits_and_document() -> None:
    backend = FakeBackend(
        search_index={"who won": ["doc-1"]},
        documents={"doc-1": "doc-1 body"},
    )

    assert backend.search("who won") == [{"docid": "doc-1", "snippet": "doc-1 body"}]
    assert backend.get_document("doc-1") == "doc-1 body"


def test_runtime_injects_summary_after_threshold_crossing() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "q"}}',
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=5, max_context_tokens=64)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.summary_turns == []
    assert result.final_answer == "done"


def test_runtime_stops_on_malformed_tool_call() -> None:
    backend = FakeBackend(search_index={}, documents={})
    model = ScriptedModel(outputs=['{"tool_name": "search"}'])
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.turn_rewards == {"tool-1": -1.0}


def test_parse_model_tool_call_accepts_thinking_and_fenced_json() -> None:
    raw_output = """
<think>
I should search first.
</think>

```json
{"tool_name": "search", "arguments": {"query": "focused query"}}
```
"""

    parsed = parse_model_tool_call(raw_output)

    assert parsed is not None
    payload, normalized_output = parsed
    assert payload == {"tool_name": "search", "arguments": {"query": "focused query"}}
    assert normalized_output == '{"tool_name": "search", "arguments": {"query": "focused query"}}'


def test_parse_model_tool_call_uses_first_valid_action_when_model_outputs_multiple() -> None:
    raw_output = """
```json
{"tool_name": "search", "arguments": {"query": "first query"}}
```
```json
{"tool_name": "finish", "arguments": {"answer": "unsupported"}}
```
"""

    parsed = parse_model_tool_call(raw_output)

    assert parsed is not None
    payload, _ = parsed
    assert payload == {"tool_name": "search", "arguments": {"query": "first query"}}


def test_runtime_records_clean_tool_call_when_model_outputs_thinking() -> None:
    backend = FakeBackend(search_index={"focused query": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = ScriptedModel(
        outputs=[
            '<think>I should search.</think>\n```json\n{"tool_name": "search", "arguments": {"query": "focused query"}}\n```',
            '<think>The document supports it.</think>\n{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.turn_records[0]["completion"] == '{"tool_name": "finish", "arguments": {"answer": "done"}}'


def test_runtime_second_step_finish_sees_raw_history_and_succeeds() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = RecordingModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "q"}}',
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.final_answer == "done"
    assert len(model.prompts) == 2
    assert "### SYSTEM" in model.prompts[1]
    assert "exactly one JSON object" in model.prompts[1]
    assert "Available tools:" in model.prompts[1]
    assert "### USER\nquestion" in model.prompts[1]
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "q"}}' in model.prompts[1]
    assert '### TOOL_RESULT\n[{"docid": "doc-1", "snippet": "fact from doc-1"}]' in model.prompts[1]
    assert "### NEXT_ACTION" in model.prompts[1]


def test_runtime_attributes_malformed_penalty_to_second_tool_turn() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={})
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "q"}}',
            '{"tool_name": "search"}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.turn_rewards == {"tool-2": -1.0}


def test_runtime_uses_summary_plus_unsummarized_raw_tail_after_compaction() -> None:
    backend = FakeBackend(
        search_index={
            "first": ["old-doc"],
            "second": ["trigger-doc"],
        },
        documents={},
    )
    model = RecordingModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "first"}}',
            '{"tool_name": "search", "arguments": {"query": "second"}}',
            "summary of old-doc only",
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=256,
        token_counter=lambda text: text.count("trigger-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert len(model.prompts) == 4
    acting_prompt_after_summary = model.prompts[3]
    assert "### SYSTEM" in acting_prompt_after_summary
    assert "exactly one JSON object" in acting_prompt_after_summary
    assert "Available tools:" in acting_prompt_after_summary
    assert "### USER\nquestion" in acting_prompt_after_summary
    assert "### SUMMARY\nsummary of old-doc only" in acting_prompt_after_summary
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "first"}}' not in acting_prompt_after_summary
    assert '### TOOL_RESULT\n[{"docid": "old-doc", "snippet": ""}]' not in acting_prompt_after_summary
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "second"}}' in acting_prompt_after_summary
    assert '### TOOL_RESULT\n[{"docid": "trigger-doc", "snippet": ""}]' in acting_prompt_after_summary
    assert "### NEXT_ACTION" in acting_prompt_after_summary


def test_runtime_summarizes_two_of_three_raw_rounds_and_leaves_newest_raw() -> None:
    backend = FakeBackend(
        search_index={
            "first": ["doc-1"],
            "second": ["doc-2"],
            "third": ["doc-3"],
        },
        documents={},
    )
    model = RecordingModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "first"}}',
            '{"tool_name": "search", "arguments": {"query": "second"}}',
            '{"tool_name": "search", "arguments": {"query": "third"}}',
            "summary of first two rounds",
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=256,
        token_counter=lambda text: text.count("doc-3"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.summary_turns == ["summary-1"]
    assert len(model.prompts) == 5
    summary_prompt = model.prompts[3]
    assert '{"query": "first"}' in summary_prompt
    assert '{"query": "second"}' in summary_prompt
    assert '{"query": "third"}' not in summary_prompt
    acting_prompt_after_summary = model.prompts[4]
    assert "### SUMMARY\nsummary of first two rounds" in acting_prompt_after_summary
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "first"}}' not in acting_prompt_after_summary
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "second"}}' not in acting_prompt_after_summary
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "third"}}' in acting_prompt_after_summary


def test_runtime_keeps_single_raw_round_in_next_prompt_without_summary() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = RecordingModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "q"}}',
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=256,
        token_counter=lambda text: text.count("doc-1"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.summary_turns == []
    assert len(model.prompts) == 2
    acting_prompt = model.prompts[1]
    assert "### SUMMARY" not in acting_prompt
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "q"}}' in acting_prompt
    assert '### TOOL_RESULT\n[{"docid": "doc-1", "snippet": "fact from doc-1"}]' in acting_prompt


def test_runtime_empty_summary_does_not_retire_older_rounds() -> None:
    backend = FakeBackend(
        search_index={
            "first": ["old-doc"],
            "second": ["trigger-doc"],
        },
        documents={},
    )
    model = RecordingModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "first"}}',
            '{"tool_name": "search", "arguments": {"query": "second"}}',
            "   ",
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=256,
        token_counter=lambda text: text.count("trigger-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.summary_turns == []
    assert len(model.prompts) == 4
    acting_prompt_after_empty_summary = model.prompts[3]
    assert "### SUMMARY" not in acting_prompt_after_empty_summary
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "first"}}' in acting_prompt_after_empty_summary
    assert '### TOOL_RESULT\n[{"docid": "old-doc", "snippet": ""}]' in acting_prompt_after_empty_summary
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "second"}}' in acting_prompt_after_empty_summary
    assert '### TOOL_RESULT\n[{"docid": "trigger-doc", "snippet": ""}]' in acting_prompt_after_empty_summary


def test_runtime_records_trainable_summary_and_final_answer_turns() -> None:
    backend = FakeBackend(
        search_index={
            "first": ["old-doc"],
            "second": ["trigger-doc"],
        },
        documents={},
    )
    model = RecordingModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "first"}}',
            '{"tool_name": "search", "arguments": {"query": "second"}}',
            "summary of old-doc only",
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=256,
        token_counter=lambda text: text.count("trigger-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert [record["kind"] for record in result.turn_records] == ["summary", "final_answer"]
    assert [record["query_id"] for record in result.turn_records] == ["q1", "q1"]
    assert result.turn_records[0]["turn_id"] == "summary-1"
    assert result.turn_records[0]["completion"] == "summary of old-doc only"
    assert result.turn_records[1]["turn_id"] == "final-answer"
    assert result.turn_records[1]["completion"] == '{"tool_name": "finish", "arguments": {"answer": "done"}}'
    assert result.turn_rewards == {"summary-1": 1.0, "final-answer": 1.0}


def test_runtime_completed_result_feeds_trajectory_extraction() -> None:
    backend = FakeBackend(
        search_index={
            "first": ["old-doc"],
            "second": ["trigger-doc"],
        },
        documents={},
    )
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "first"}}',
            '{"tool_name": "search", "arguments": {"query": "second"}}',
            "summary of old-doc only",
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=256,
        token_counter=lambda text: text.count("trigger-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    samples = extract_trainable_samples(result.turn_records, result.turn_rewards)

    assert [sample.turn_id for sample in samples] == ["summary-1", "final-answer"]
    assert [sample.reward for sample in samples] == [1.0, 1.0]


def test_runtime_malformed_result_feeds_trajectory_extraction() -> None:
    backend = FakeBackend(search_index={}, documents={})
    model = ScriptedModel(outputs=['{"tool_name": "search"}'])
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")

    samples = extract_trainable_samples(result.turn_records, result.turn_rewards)

    assert result.turn_records == [{"turn_id": "tool-1", "kind": "tool"}]
    assert result.turn_rewards == {"tool-1": -1.0}
    assert samples == []


def test_runtime_stops_with_budget_exhausted_after_tool_limit() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={})
    model = ScriptedModel(outputs=['{"tool_name": "search", "arguments": {"query": "q"}}'])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=100,
        max_context_tokens=256,
        max_tool_calls=1,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "budget_exhausted"
    assert result.final_answer is None
    assert result.turn_rewards == {}


def test_runtime_applies_fit_check_to_full_summary_generation_prompt() -> None:
    backend = FakeBackend(
        search_index={
            "first": ["old-doc"],
            "second": ["trigger-doc"],
        },
        documents={},
    )
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "first"}}',
            '{"tool_name": "search", "arguments": {"query": "second"}}',
            "summary",
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=15,
        token_counter=lambda text: 100 if "Write a clean summary containing only the essential information needed" in text else 1,
    )

    try:
        runtime.run(query_id="q1", user_prompt="question")
    except ValueError as exc:
        assert "Packed prompt exceeds safe limit" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized full summary-generation prompt")


def test_runtime_raises_when_acting_prompt_exceeds_fit_limit() -> None:
    backend = FakeBackend(search_index={}, documents={})
    model = ScriptedModel(outputs=['{"tool_name": "finish", "arguments": {"answer": "done"}}'])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=100,
        max_context_tokens=3,
        token_counter=lambda text: len(text.split()),
    )

    try:
        runtime.run(query_id="q1", user_prompt="question with too many words")
    except ValueError as exc:
        assert "Packed prompt exceeds safe limit" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized acting prompt")


def test_cli_smoke_helper_returns_run_record() -> None:
    record = build_smoke_run_record()

    assert record["query_id"] == "smoke-q1"
    assert record["status"] == "completed"
    assert record["retrieved_docids"] == ["smoke-doc"]


def test_cli_entrypoint_prints_smoke_record_json() -> None:
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        cli_entrypoint.main()

    record = json.loads(stdout.getvalue())

    assert record["query_id"] == "smoke-q1"
    assert record["status"] == "completed"
    assert record["retrieved_docids"] == ["smoke-doc"]
    assert record["result"] == [{"type": "output_text", "output": "smoke answer"}]
