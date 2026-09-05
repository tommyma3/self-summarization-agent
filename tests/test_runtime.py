import contextlib
import io
import json

import main as cli_entrypoint
from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.cli import build_smoke_run_record
from self_summarization_agent.launcher_utils import serialize_runtime_result
from self_summarization_agent.generation import GenerationResult
from self_summarization_agent.models import Message, ToolCall
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel, extract_summary_output, parse_model_tool_call
from self_summarization_agent.trajectory import (
    REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT,
    build_rollout_native_training_cache,
    extract_trainable_samples,
)


class RecordingModel(ScriptedModel):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs=outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return super().generate(prompt)


def tool_output(json_text: str, thinking: str = "thinking") -> str:
    return f"<think>{thinking}</think>\n{json_text}"


class NativeMetadataModel:
    supports_native_tools = True
    require_exact_token_ids = True

    def __init__(self, outputs: list[GenerationResult]) -> None:
        self.outputs = outputs
        self.prompts = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(str(prompt).split())

    def generate_batch_with_metadata(self, prompts: list[str]) -> list[GenerationResult]:
        self.prompts.extend(prompts)
        outputs, self.outputs = self.outputs[: len(prompts)], self.outputs[len(prompts) :]
        return outputs


def test_fake_backend_returns_search_hits_and_document() -> None:
    backend = FakeBackend(
        search_index={"who won": ["doc-1"]},
        documents={"doc-1": "doc-1 body"},
    )

    assert backend.search("who won") == [{"docid": "doc-1", "snippet": "doc-1 body"}]
    assert backend.get_document("doc-1") == "doc-1 body"


def test_native_runtime_links_tools_and_persists_exact_collection_ids() -> None:
    model = NativeMetadataModel(
        [
            GenerationResult(
                text="search first</think>",
                prompt_token_ids=[1, 2],
                completion_token_ids=[10, 11],
                cumulative_logprob=-0.3,
                token_logprobs=[-0.1, -0.2],
                token_logprobs_mode="raw_logprobs",
                message=Message(
                    role="assistant",
                    reasoning_content="search first",
                    tool_calls=[ToolCall(id="call-search", name="search", arguments={"query": "q"})],
                ),
                finish_reason="tool_calls",
            ),
            GenerationResult(
                text="raw finish",
                prompt_token_ids=[1, 2, 10, 11, 20],
                completion_token_ids=[30, 31],
                cumulative_logprob=-0.7,
                token_logprobs=[-0.3, -0.4],
                token_logprobs_mode="raw_logprobs",
                message=Message(
                    role="assistant",
                    reasoning_content="answer",
                    tool_calls=[ToolCall(id="call-finish", name="finish", arguments={"answer": "done"})],
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["doc-1"]}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=2048,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.final_answer == "done"
    trajectory = result.trajectory_records[0]
    assert [message["role"] for message in trajectory["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert trajectory["messages"][3]["tool_call_id"] == "call-search"
    assert trajectory["collection_tokens"]["full_token_ids"] == [1, 2, 10, 11, 20, 30, 31]
    assert [generation["full_token_ids"] for generation in trajectory["collection_tokens"]["generations"]] == [
        [1, 2, 10, 11],
        [1, 2, 10, 11, 20, 30, 31],
    ]
    assert trajectory["collection_tokens"]["assistant_token_mask"] == [
        False,
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert "training_cache" not in trajectory
    cache = build_rollout_native_training_cache(trajectory["collection_tokens"])
    assert cache is not None
    assert cache["input_ids"] == [1, 2, 10, 11, 20, 30]
    assert cache["labels"] == [2, 10, 11, 20, 30, 31]
    assert cache["completion_mask"] == [False, True, True, False, True, True]
    assert cache["reference_logprobs"] == [0.0, -0.1, -0.2, 0.0, -0.3, -0.4]
    assert cache["reference_logprob"] == -0.25
    assert cache["reference_logprob_source"] == REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT
    assert cache["state_prefix_length"] == 2
    assert result.token_usage["budget_consumed_tokens"] == (
        result.token_usage["total_generated_tokens"] + result.token_usage["tool_result_tokens"]
    )


def test_native_summary_appends_user_control_without_changing_tools_and_keeps_exact_interval_ids() -> None:
    model = NativeMetadataModel(
        [
            GenerationResult(
                text="search first</think>",
                prompt_token_ids=[3],
                completion_token_ids=[10],
                cumulative_logprob=-0.1,
                token_logprobs=[-0.1],
                token_logprobs_mode="raw_logprobs",
                message=Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="call-search", name="search", arguments={"query": "q"})],
                ),
            ),
            GenerationResult(
                text="raw summary",
                prompt_token_ids=[3, 10, 4],
                completion_token_ids=[11],
                cumulative_logprob=-0.2,
                token_logprobs=[-0.2],
                token_logprobs_mode="raw_logprobs",
                message=Message(
                    role="assistant",
                    reasoning_content="compact",
                    content="<summary>state</summary>",
                ),
            ),
            GenerationResult(
                text="raw finish",
                prompt_token_ids=[5, 6],
                completion_token_ids=[12],
                cumulative_logprob=-0.3,
                token_logprobs=[-0.3],
                token_logprobs_mode="raw_logprobs",
                message=Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="call-finish", name="finish", arguments={"answer": "done"})],
                ),
            ),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["doc-1"]}, documents={}),
        context_threshold_tokens=1,
        max_context_tokens=2048,
        max_summary_tokens=32,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.final_answer == "done"
    assert result.summary_turns == ["summary-1"]
    summary_prompt = model.prompts[1]
    assert len(summary_prompt.tools) == 3
    assert summary_prompt.tool_choice == "none"
    assert summary_prompt.messages[0].content == model.prompts[0].messages[0].content
    assert summary_prompt.messages[-1].role == "user"
    assert summary_prompt.messages[-1].content.startswith("<summary_request>")
    acting_prompt_after_summary = model.prompts[2]
    assert [message.role for message in acting_prompt_after_summary.messages] == ["system", "user", "user"]
    assert acting_prompt_after_summary.messages[:2] == model.prompts[0].messages[:2]
    assert acting_prompt_after_summary.messages[2].content == "<summary>\nstate\n</summary>"
    assert [message["role"] for message in result.trajectory_records[0]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert result.trajectory_records[0]["collection_tokens"]["full_token_ids"] == [3, 10, 4, 11]
    assert result.trajectory_records[0]["collection_tokens"]["assistant_token_mask"] == [
        False,
        True,
        False,
        True,
    ]
    summary_cache = build_rollout_native_training_cache(
        result.trajectory_records[0]["collection_tokens"]
    )
    final_cache = build_rollout_native_training_cache(
        result.trajectory_records[1]["collection_tokens"]
    )
    assert summary_cache is not None
    assert final_cache is not None
    assert summary_cache["state_prefix_length"] == 1
    assert final_cache["state_prefix_length"] == 2
    assert summary_cache["reference_logprobs"] == [
        -0.1,
        0.0,
        -0.2,
    ]
    assert final_cache["reference_logprobs"] == [
        0.0,
        -0.3,
    ]


def test_rollout_native_cache_rejects_completion_without_exact_conditioning_prefix() -> None:
    collection_tokens = {
        "version": 2,
        "full_token_ids": [1, 10, 2, 11],
        "assistant_token_mask": [False, True, False, True],
        "generations": [
            {
                "prompt_token_ids": [9],
                "completion_token_ids": [10],
                "completion_token_logprobs": [-0.1],
                "logprobs_mode": "raw_logprobs",
            },
            {
                "prompt_token_ids": [1, 10, 2],
                "completion_token_ids": [11],
                "completion_token_logprobs": [-0.2],
                "logprobs_mode": "raw_logprobs",
            },
        ],
    }

    assert build_rollout_native_training_cache(collection_tokens) is None


def test_exact_collection_penalizes_action_without_closed_thinking() -> None:
    model = NativeMetadataModel(
        [
            GenerationResult(
                text="searching without ever closing the think block\n<search>q</search>",
                prompt_token_ids=[1, 2],
                completion_token_ids=[10, 11],
                message=Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="call-search", name="search", arguments={"query": "q"})],
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["doc-1"]}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=2048,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.final_answer is None
    assert result.tool_call_counts["search"] == 0
    assert len(result.trajectory_records) == 1
    trajectory = result.trajectory_records[0]
    assert trajectory["termination_kind"] == "malformed"
    assert trajectory["collection_tokens"]["full_token_ids"] == [1, 2, 10, 11]
    assert result.turn_rewards == {"trajectory-1": -1.0}


def test_exact_collection_terminates_when_provider_rewrites_interval_history() -> None:
    model = NativeMetadataModel(
        [
            GenerationResult(
                text="search first</think>",
                prompt_token_ids=[1, 2],
                completion_token_ids=[10, 11],
                message=Message(
                    role="assistant",
                    reasoning_content="search first",
                    tool_calls=[ToolCall(id="call-search", name="search", arguments={"query": "q"})],
                ),
                finish_reason="tool_calls",
            ),
            GenerationResult(
                text="raw finish",
                # The provider re-rendered the history, so this prompt does not
                # extend the previous generation's sampled tokens [1, 2, 10, 11].
                prompt_token_ids=[7, 8, 9],
                completion_token_ids=[30, 31],
                message=Message(
                    role="assistant",
                    reasoning_content="answer",
                    tool_calls=[ToolCall(id="call-finish", name="finish", arguments={"answer": "done"})],
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["doc-1"]}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=2048,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "history_rewrite_detected"
    assert result.final_answer is None
    assert result.tool_call_counts["search"] == 1
    assert len(result.trajectory_records) == 1
    trajectory = result.trajectory_records[0]
    assert trajectory["termination_kind"] == "malformed"
    generations = trajectory["collection_tokens"]["generations"]
    assert len(generations) == 1
    assert generations[0]["full_token_ids"] == [1, 2, 10, 11]
    assert trajectory["collection_tokens"]["full_token_ids"] == [1, 2, 10, 11]
    assert [message["role"] for message in trajectory["messages"]] == ["system", "user", "assistant", "tool"]
    assert result.turn_rewards == {"trajectory-1": -1.0}
    assert any(record.get("history_rewrite") for record in result.turn_records)


def test_exact_collection_penalizes_summary_prompt_that_rewrites_interval_history() -> None:
    model = NativeMetadataModel(
        [
            GenerationResult(
                text="search first</think>",
                prompt_token_ids=[1, 2],
                completion_token_ids=[10, 11],
                message=Message(
                    role="assistant",
                    reasoning_content="search first",
                    tool_calls=[ToolCall(id="call-search", name="search", arguments={"query": "q"})],
                ),
                finish_reason="tool_calls",
            ),
            GenerationResult(
                text="summary body",
                # The provider re-rendered the history for the summary request,
                # so this prompt does not extend the previous sampled tokens.
                prompt_token_ids=[7, 8, 9],
                completion_token_ids=[30, 31],
                finish_reason="stop",
            ),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["doc-1"]}, documents={}),
        context_threshold_tokens=1,
        max_context_tokens=2048,
        max_summary_tokens=128,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "history_rewrite_detected"
    assert result.final_answer is None
    assert len(result.trajectory_records) == 1
    trajectory = result.trajectory_records[0]
    assert trajectory["termination_kind"] == "malformed"
    generations = trajectory["collection_tokens"]["generations"]
    assert len(generations) == 1
    assert generations[0]["full_token_ids"] == [1, 2, 10, 11]
    assert result.turn_rewards == {"trajectory-1": -1.0}
    assert any(record.get("history_rewrite") for record in result.turn_records)


def test_runtime_completes_without_summary_below_threshold() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = ScriptedModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=1000, max_context_tokens=1024)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.summary_turns == []
    assert result.final_answer == "done"


def test_run_many_stream_emits_completions_and_refills_slots_between_rounds() -> None:
    class BatchRecordingScriptedModel(ScriptedModel):
        def __init__(self, outputs: list[str]) -> None:
            super().__init__(outputs=outputs)
            self.batch_sizes: list[int] = []

        def generate_batch(self, prompts: list[str]) -> list[str]:
            self.batch_sizes.append(len(prompts))
            return super().generate_batch(prompts)

    model = BatchRecordingScriptedModel(
        outputs=[
            tool_output('{"tool_name": "finish", "arguments": {"answer": "one"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "q2"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "two"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "three"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q2": ["doc-2"]}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=2048,
    )

    completed_batches = list(
        runtime.run_many_stream(
            [("q1", "first"), ("q2", "second"), ("q3", "third")],
            max_active_episodes=2,
        )
    )

    assert [[index for index, _ in batch] for batch in completed_batches] == [[0], [1, 2]]
    assert [
        result.final_answer
        for batch in completed_batches
        for _, result in batch
    ] == ["one", "two", "three"]
    assert model.batch_sizes == [2, 2]


def test_runtime_batches_same_step_search_calls() -> None:
    class BatchSearchBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__(
                search_index={"first": ["doc-1"], "second": ["doc-2"]},
                documents={"doc-1": "fact one", "doc-2": "fact two"},
            )
            self.search_many_calls: list[list[str]] = []

        def search_many(self, queries: list[str]):
            self.search_many_calls.append(list(queries))
            return [self.search(query) for query in queries]

    backend = BatchSearchBackend()
    model = ScriptedModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "second"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done one"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done two"}}'),
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=1000, max_context_tokens=1024)

    results = runtime.run_many([("q1", "question one"), ("q2", "question two")])

    assert backend.search_many_calls == [["first", "second"]]
    assert [result.final_answer for result in results] == ["done one", "done two"]


def test_runtime_does_not_silently_replace_failed_searches_with_empty_results() -> None:
    class FailedSearchBackend(FakeBackend):
        def search_many(self, queries: list[str]):
            raise RuntimeError(f"retrieval worker unavailable for {len(queries)} queries")

    model = ScriptedModel(
        outputs=[tool_output('{"tool_name": "search", "arguments": {"query": "first"}}')]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FailedSearchBackend(search_index={}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=1024,
    )

    try:
        runtime.run(query_id="q1", user_prompt="question")
    except RuntimeError as exc:
        assert "retrieval worker unavailable" in str(exc)
    else:
        raise AssertionError("retrieval failures must abort collection")


def test_runtime_stops_on_malformed_tool_call() -> None:
    backend = FakeBackend(search_index={}, documents={})
    model = ScriptedModel(outputs=['{"tool_name": "search"}'])
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=1000, max_context_tokens=1024)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.turn_rewards == {"trajectory-1": -1.0}


def test_runtime_terminates_episode_when_tool_result_overflows_context() -> None:
    # A large tool result is appended after the round-level budget checks, so
    # the following round's prompt can exceed the safe context limit.  The
    # episode must be penalized instead of raising out of collection.
    get_document = tool_output('{"tool_name": "get_document", "arguments": {"doc_id": "doc-1"}}')
    backend = FakeBackend(search_index={}, documents={"doc-1": "word " * 1200})
    model = ScriptedModel(outputs=[get_document])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=10**9,
        max_context_tokens=1024,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "context_length_exceeded"
    assert result.final_answer is None
    assert result.turn_rewards == {"trajectory-1": -1.0}
    assert result.trajectory_records[0]["termination_kind"] == "context_overflow"


def test_runtime_terminates_episode_when_forced_answer_prompt_overflows_context() -> None:
    # The generated-token budget trips only after a large tool result is
    # appended, so the forced-answer prompt itself can exceed the safe limit.
    get_document = tool_output('{"tool_name": "get_document", "arguments": {"doc_id": "doc-1"}}')
    backend = FakeBackend(search_index={}, documents={"doc-1": "word " * 1200})
    model = ScriptedModel(outputs=[get_document])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=10**9,
        max_context_tokens=1024,
        generated_token_budget=500,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "context_length_exceeded"
    assert result.token_usage["forced_answer_reasons"] == ["generated_token_budget"]
    assert result.turn_rewards == {"trajectory-1": -1.0}


def test_runtime_stops_and_penalizes_when_summary_exceeds_limit() -> None:
    first_search = tool_output('{"tool_name": "search", "arguments": {"query": "first"}}')
    overlong_summary = "<think>compact</think>\n<summary>one two three</summary>"
    backend = FakeBackend(search_index={"first": ["doc-1"]}, documents={})
    model = RecordingModel(outputs=[first_search, overlong_summary])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=2,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "summary_length_exceeded"
    assert result.final_answer is None
    assert result.summary_turns == []
    assert [record["kind"] for record in result.turn_records] == ["tool", "summary"]
    assert result.turn_rewards == {"trajectory-1": -1.0}
    summary_record = result.turn_records[-1]
    assert summary_record["summary_tokens"] == 3
    assert summary_record["max_summary_tokens"] == 2
    assert "at most 2 tokens" not in model.prompts[-1]
    assert "<summary>...</summary>" in model.prompts[-1]


def test_runtime_accepts_summary_at_maximum_length() -> None:
    first_search = tool_output('{"tool_name": "search", "arguments": {"query": "first"}}')
    summary = "<think>compact</think>\n<summary>one two</summary>"
    final_answer = tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')
    backend = FakeBackend(search_index={"first": ["doc-1"]}, documents={})
    runtime = EpisodeRuntime(
        model=ScriptedModel(outputs=[first_search, summary, final_answer]),
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=2,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.summary_turns == ["summary-1"]
    assert result.turn_records[1]["summary_tokens"] == 2


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


def test_parse_model_tool_call_accepts_tag_actions_without_thinking() -> None:
    parsed = parse_model_tool_call("<search>focused query</search>")

    assert parsed is not None
    payload, normalized_output = parsed
    assert payload == {"tool_name": "search", "arguments": {"query": "focused query"}}
    assert normalized_output == '{"tool_name": "search", "arguments": {"query": "focused query"}}'


def test_parse_model_tool_call_normalizes_document_and_answer_tags() -> None:
    document = parse_model_tool_call("<think>Need the full doc.</think>\n<document>doc-1</document>")
    answer = parse_model_tool_call("<think>Enough evidence.</think>\n<answer>done</answer>")

    assert document is not None
    assert document[0] == {"tool_name": "get_document", "arguments": {"doc_id": "doc-1"}}
    assert answer is not None
    assert answer[0] == {"tool_name": "finish", "arguments": {"answer": "done"}}


def test_parse_model_tool_call_rejects_mixed_or_repeated_tags() -> None:
    assert parse_model_tool_call("<search>q</search><answer>a</answer>") is None
    assert parse_model_tool_call("<search>q</search><search>q2</search>") is None


def test_parse_model_tool_call_uses_first_valid_action_when_model_outputs_multiple() -> None:
    raw_output = """
<think>
I should search first.
</think>

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


def test_parse_model_tool_call_rejects_output_without_completed_thinking() -> None:
    assert parse_model_tool_call('{"tool_name": "search", "arguments": {"query": "q"}}') is None


def test_parse_model_tool_call_scans_only_after_completed_thinking() -> None:
    raw_output = """
<think>
{"tool_name": "finish", "arguments": {"answer": "bad thinking json"}}
</think>
{"tool_name": "search", "arguments": {"query": "post-think query"}}
"""

    parsed = parse_model_tool_call(raw_output)

    assert parsed is not None
    payload, _ = parsed
    assert payload == {"tool_name": "search", "arguments": {"query": "post-think query"}}


def test_parse_model_tool_call_rejects_invalid_first_json_after_thinking() -> None:
    raw_output = """
<think>done</think>
{"note": "not a tool call"}
{"tool_name": "search", "arguments": {"query": "ignored"}}
"""

    assert parse_model_tool_call(raw_output) is None


def test_extract_summary_output_splits_thinking_from_summary_body() -> None:
    extracted = extract_summary_output(
        "<think><summary>ignore this</summary> I should preserve doc-1.</think>\n"
        "discard this prefix <summary>Summary cites doc-1.</summary> discard this suffix"
    )

    assert extracted is not None
    assert extracted.thinking == "<summary>ignore this</summary> I should preserve doc-1."
    assert extracted.summary == "Summary cites doc-1."


def test_extract_summary_output_rejects_wrapped_summary_without_completed_thinking() -> None:
    extracted = extract_summary_output("<summary>Summary without explicit thinking.</summary>")

    assert extracted is None


def test_extract_summary_output_rejects_unwrapped_output() -> None:
    extracted = extract_summary_output("<think>done</think>\nSummary without wrappers.")

    assert extracted is None


def test_runtime_records_raw_tool_call_completion_when_model_outputs_thinking() -> None:
    backend = FakeBackend(search_index={"focused query": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = ScriptedModel(
        outputs=[
            '<think>I should search.</think>\n```json\n{"tool_name": "search", "arguments": {"query": "focused query"}}\n```',
            '<think>The document supports it.</think>\n{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=1000, max_context_tokens=1024)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.turn_records[0]["kind"] == "tool"
    assert (
        result.turn_records[0]["completion"]
        == '<think>I should search.</think>\n```json\n{"tool_name": "search", "arguments": {"query": "focused query"}}\n```'
    )
    assert result.turn_records[0]["normalized_completion"] == '{"tool_name": "search", "arguments": {"query": "focused query"}}'
    assert (
        result.turn_records[1]["completion"]
        == '<think>The document supports it.</think>\n{"tool_name": "finish", "arguments": {"answer": "done"}}'
    )
    assert result.turn_records[1]["normalized_completion"] == '{"tool_name": "finish", "arguments": {"answer": "done"}}'


def test_runtime_second_step_finish_sees_raw_history_and_succeeds() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=1024,
        max_tool_calls=2,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.final_answer == "done"
    assert len(model.prompts) == 2
    assert "Tool Budget Remaining" not in model.prompts[0]
    assert "Tool Budget Remaining" not in model.prompts[1]
    assert "### SYSTEM" in model.prompts[1]
    assert "output exactly one action" in model.prompts[1]
    assert "### USER\nquestion" in model.prompts[1]
    assert "### ASSISTANT\n<think>thinking</think>" in model.prompts[1]
    assert '{"tool_name": "search", "arguments": {"query": "q"}}' in model.prompts[1]
    assert '<information>[{"docid": "doc-1", "snippet": "fact from doc-1"}]</information>' in model.prompts[1]
    assert "### NEXT_ACTION" not in model.prompts[1]


def test_runtime_attributes_malformed_penalty_to_second_tool_turn() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={})
    model = ScriptedModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "search"}'),
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=1000, max_context_tokens=1024)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.turn_rewards == {"trajectory-1": -1.0}


def test_runtime_appends_compaction_instruction_then_resets_to_system_and_summary() -> None:
    backend = FakeBackend(search_index={"first": ["old-doc"]}, documents={})
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}', thinking="retain this reasoning"),
            "<think>compact old-doc</think>\n<summary>summary of the task and old-doc</summary>",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        token_counter=lambda text: text.count("old-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert len(model.prompts) == 3
    compaction_prompt = model.prompts[1]
    assert "retain this reasoning" in compaction_prompt
    assert '<information>[{"docid": "old-doc", "snippet": ""}]</information>' in compaction_prompt
    assert "### USER\n<summary_request>" in compaction_prompt
    assert compaction_prompt.rstrip().endswith("</summary_request>")
    acting_prompt_after_summary = model.prompts[2]
    assert "### SYSTEM" in acting_prompt_after_summary
    assert "### USER\nquestion" in acting_prompt_after_summary
    assert "### USER\n<summary>\nsummary of the task and old-doc\n</summary>" in acting_prompt_after_summary
    assert "retain this reasoning" not in acting_prompt_after_summary
    assert "<search>first</search>" not in acting_prompt_after_summary


def test_runtime_puts_only_post_think_summary_into_context() -> None:
    backend = FakeBackend(search_index={"first": ["old-doc"]}, documents={})
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            "<think>reason about old-doc</think>\n"
            "outside prefix <summary>summary body for context</summary> outside suffix",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        token_counter=lambda text: text.count("old-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    summary_record = result.turn_records[1]
    assert summary_record["completion"] == (
        "<think>reason about old-doc</think>\n"
        "outside prefix <summary>summary body for context</summary> outside suffix"
    )
    assert summary_record["thinking"] == "reason about old-doc"
    assert summary_record["summary"] == "summary body for context"
    acting_prompt_after_summary = model.prompts[2]
    assert "### USER\n<summary>\nsummary body for context\n</summary>" in acting_prompt_after_summary
    assert "reason about old-doc" not in acting_prompt_after_summary
    assert "<think>" not in acting_prompt_after_summary
    assert "outside prefix" not in acting_prompt_after_summary


def test_runtime_penalizes_summary_without_complete_wrapper() -> None:
    backend = FakeBackend(search_index={"first": ["old-doc"]}, documents={})
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            "<think>reasoning only</think>   ",
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        token_counter=lambda text: text.count("old-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.summary_turns == []
    assert result.turn_records[-1]["summary"] == ""
    assert result.trajectory_records[-1]["termination_kind"] == "malformed"
    assert result.turn_rewards == {"trajectory-1": -1.0}


def test_runtime_penalizes_empty_wrapped_summary_as_empty_summary() -> None:
    backend = FakeBackend(search_index={"first": ["old-doc"]}, documents={})
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            "<think>reasoning only</think>\n<summary>   </summary>",
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        token_counter=lambda text: text.count("old-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "empty_summary"
    assert result.summary_turns == []
    assert result.turn_records[-1]["summary"] == ""


def test_runtime_records_one_training_trajectory_per_interval() -> None:
    backend = FakeBackend(search_index={"first": ["old-doc"]}, documents={})
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            "<think>compact old-doc</think>\n<summary>summary of the task and old-doc</summary>",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        token_counter=lambda text: text.count("old-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert [record["kind"] for record in result.turn_records] == ["tool", "summary", "final_answer"]
    assert [record["termination_kind"] for record in result.trajectory_records] == ["compaction", "final_answer"]
    first_interval = result.trajectory_records[0]
    assert first_interval["turn_ids"] == ["tool-1", "summary-1"]
    assert [message["role"] for message in first_interval["messages"]] == [
        "system", "user", "assistant", "user", "user", "assistant"
    ]
    assert "retain" not in first_interval["messages"][0]["content"]
    assert "Compact the agent history" in first_interval["messages"][-2]["content"]
    second_interval = result.trajectory_records[1]
    assert second_interval["turn_ids"] == ["final-answer"]
    assert [message["role"] for message in second_interval["messages"]] == [
        "system", "user", "user", "assistant"
    ]
    assert second_interval["messages"][1]["content"] == "question"
    assert second_interval["messages"][2]["content"] == (
        "<summary>\nsummary of the task and old-doc\n</summary>"
    )
    assert result.turn_records[1]["summary"] == "summary of the task and old-doc"
    assert result.turn_rewards == {"trajectory-1": 1.0, "trajectory-2": 1.0}


def test_runtime_preserves_original_prefix_and_replaces_only_history_across_compactions() -> None:
    backend = FakeBackend(
        search_index={"first": ["first-doc"], "second": ["second-doc"]},
        documents={},
    )
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}', "first reasoning"),
            "<think>compact first</think>\n<summary>first compressed history</summary>",
            tool_output('{"tool_name": "search", "arguments": {"query": "second"}}', "second reasoning"),
            "<think>compact second</think>\n<summary>second compressed history</summary>",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=2048,
        max_summary_tokens=128,
        token_counter=lambda text: text.count("doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="exact original question")

    assert result.status == "completed"
    assert result.summary_turns == ["summary-1", "summary-2"]
    assert len(result.trajectory_records) == 3
    initial_prefix = result.trajectory_records[0]["messages"][:2]
    for record in result.trajectory_records[1:]:
        assert record["messages"][:2] == initial_prefix
        assert record["messages"][1]["content"] == "exact original question"

    second_interval = result.trajectory_records[1]["messages"]
    assert second_interval[2]["content"] == "<summary>\nfirst compressed history\n</summary>"
    assert all("first reasoning" not in message["content"] for message in second_interval)

    third_interval = result.trajectory_records[2]["messages"]
    assert third_interval[2]["content"] == "<summary>\nsecond compressed history\n</summary>"
    assert all("first compressed history" not in message["content"] for message in third_interval)
    assert all("second reasoning" not in message["content"] for message in third_interval)


def test_runtime_completed_result_feeds_trajectory_extraction() -> None:
    backend = FakeBackend(search_index={"first": ["old-doc"]}, documents={})
    model = ScriptedModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            "<think>compact old-doc</think>\n<summary>summary of the task and old-doc</summary>",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        token_counter=lambda text: text.count("old-doc"),
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    samples = extract_trainable_samples(result.trajectory_records, result.turn_rewards)

    assert [sample.turn_id for sample in samples] == ["trajectory-1", "trajectory-2"]
    assert [sample.reward for sample in samples] == [1.0, 1.0]
    assert [record["segment_index"] for record in result.trajectory_records] == [1, 2]
    assert result.trajectory_records[0]["prefix_summary_turn_id"] is None
    assert result.trajectory_records[1]["prefix_summary_turn_id"] == "summary-1"


def test_runtime_forces_final_answer_after_tool_limit() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={})
    model = RecordingModel(
        outputs=[
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "best available"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=1024,
        max_tool_calls=1,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.final_answer == "best available"
    assert result.tool_call_counts == {"search": 1, "get_document": 0}
    assert result.turn_records[-1]["kind"] == "final_answer"
    assert "final-answer boundary" in model.prompts[1]
    assert "Tool Budget Remaining" not in model.prompts[1]
    assert "### SYSTEM\n<forced_answer_request>" in model.prompts[1]
    interval_messages = result.trajectory_records[0]["messages"]
    assert [message["role"] for message in interval_messages] == [
        "system", "user", "assistant", "user", "system", "assistant"
    ]
    assert "final-answer boundary" in interval_messages[-2]["content"]


def test_runtime_rejects_non_finish_action_after_tool_limit() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={})
    model = RecordingModel(outputs=[tool_output('{"tool_name": "search", "arguments": {"query": "q"}}')])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=1024,
        max_tool_calls=0,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.tool_call_counts == {"search": 0, "get_document": 0}
    assert "final-answer boundary" in model.prompts[0]


def test_runtime_forces_final_answer_after_generated_token_budget() -> None:
    search_output = tool_output('{"tool_name": "search", "arguments": {"query": "q"}}')
    answer_output = tool_output('{"tool_name": "finish", "arguments": {"answer": "best available"}}')
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={})
    model = RecordingModel(outputs=[search_output, answer_output])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=1024,
        generated_token_budget=1,
        token_counter=lambda text: 2 if text == search_output else 1,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.final_answer == "best available"
    assert result.token_usage["reasoning_generated_tokens"] == 2
    assert result.token_usage["tool_result_tokens"] == 1
    assert result.token_usage["forced_answer_generated_tokens"] == 1
    assert result.token_usage["total_generated_tokens"] == 3
    assert result.token_usage["budget_consumed_tokens"] == 4
    assert result.token_usage["forced_answer_reasons"] == ["generated_token_budget"]
    assert result.turn_records[0]["generation_kind"] == "action"
    assert result.turn_records[1]["generation_kind"] == "forced_answer"
    assert "### SYSTEM\n<forced_answer_request>" in model.prompts[1]


def test_runtime_counts_summary_tokens_toward_generated_token_budget() -> None:
    first_search = tool_output('{"tool_name": "search", "arguments": {"query": "first"}}')
    second_search = tool_output('{"tool_name": "search", "arguments": {"query": "second"}}')
    summary_output = "<think>compact</think>\n<summary>summary overhead tokens</summary>"
    final_output = tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')

    def count_tokens(text: str) -> int:
        if text in {first_search, second_search, final_output}:
            return 1
        if text == summary_output:
            return 50
        return text.count("trigger-doc")

    backend = FakeBackend(
        search_index={
            "first": ["old-doc"],
            "second": ["trigger-doc"],
        },
        documents={},
    )
    model = RecordingModel(outputs=[first_search, second_search, summary_output, final_output])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        generated_token_budget=4,
        token_counter=count_tokens,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.summary_turns == ["summary-1"]
    assert result.token_usage["reasoning_generated_tokens"] == 2
    assert result.token_usage["tool_result_tokens"] == 1
    assert result.token_usage["summary_generated_tokens"] == 50
    assert result.token_usage["forced_answer_generated_tokens"] == 1
    assert result.token_usage["total_generated_tokens"] == 53
    assert result.token_usage["budget_consumed_tokens"] == 54
    assert result.token_usage["forced_answer_reasons"] == ["generated_token_budget"]
    assert result.token_usage["summary_count"] == 1
    assert result.token_usage["retired_round_count"] == 2
    assert result.turn_records[2]["generation_kind"] == "summary"
    assert result.turn_records[3]["generation_kind"] == "forced_answer"


def test_runtime_counts_tool_result_tokens_toward_generated_token_budget() -> None:
    search_output = tool_output('{"tool_name": "search", "arguments": {"query": "q"}}')
    answer_output = tool_output('{"tool_name": "finish", "arguments": {"answer": "best available"}}')
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "large tool result"})
    model = RecordingModel(outputs=[search_output, answer_output])

    def count_tokens(text: str) -> int:
        if text == search_output or text == answer_output:
            return 1
        if "large tool result" in text:
            return 10
        return 1

    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=1024,
        generated_token_budget=5,
        token_counter=count_tokens,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.final_answer == "best available"
    assert result.token_usage["reasoning_generated_tokens"] == 1
    assert result.token_usage["tool_result_tokens"] == 10
    assert result.token_usage["total_generated_tokens"] == 2
    assert result.token_usage["budget_consumed_tokens"] == 12
    assert result.token_usage["forced_answer_generated_tokens"] == 1
    assert result.token_usage["forced_answer_reasons"] == ["generated_token_budget"]
    assert result.turn_records[1]["generation_kind"] == "forced_answer"


def test_runtime_continues_when_combined_budget_is_not_exhausted() -> None:
    search_output = tool_output('{"tool_name": "search", "arguments": {"query": "q"}}')
    answer_output = tool_output('{"tool_name": "finish", "arguments": {"answer": "best available"}}')
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "small tool result"})
    model = RecordingModel(outputs=[search_output, answer_output])

    def count_tokens(text: str) -> int:
        if text in {search_output, answer_output}:
            return 1
        if "small tool result" in text:
            return 3
        return 1

    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=1024,
        generated_token_budget=5,
        token_counter=count_tokens,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.token_usage["total_generated_tokens"] == 2
    assert result.token_usage["tool_result_tokens"] == 3
    assert result.token_usage["budget_consumed_tokens"] == 5
    assert result.token_usage["forced_answer_reasons"] == []
    assert result.turn_records[1]["generation_kind"] == "action"


def test_generated_token_budget_has_priority_over_compaction_threshold() -> None:
    search_output = tool_output('{"tool_name": "search", "arguments": {"query": "q"}}')
    forced_answer = tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')
    backend = FakeBackend(search_index={"q": ["trigger-doc"]}, documents={})
    model = RecordingModel(outputs=[search_output, forced_answer])

    def count_tokens(text: str) -> int:
        if text == search_output:
            return 2
        if text == forced_answer:
            return 1
        return text.count("trigger-doc")

    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1,
        max_context_tokens=1024,
        max_summary_tokens=128,
        generated_token_budget=3,
        token_counter=count_tokens,
    )

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "completed"
    assert result.summary_turns == []
    assert [record["generation_kind"] for record in result.turn_records] == ["action", "forced_answer"]
    assert result.token_usage["forced_answer_reasons"] == ["generated_token_budget"]
    assert result.token_usage["budget_consumed_tokens"] == 4
    assert "final-answer boundary" in model.prompts[1]
    assert "Compact the entire preceding task trajectory" not in model.prompts[1]


def test_serialize_runtime_result_includes_token_usage() -> None:
    backend = FakeBackend(search_index={}, documents={})
    model = RecordingModel(outputs=[tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=1024,
        token_counter=lambda text: 7 if text.startswith("<think>") else 3,
    )

    result = runtime.run(query_id="q1", user_prompt="question")
    payload = serialize_runtime_result(result, query_text="question")

    assert payload["token_usage"] == result.token_usage
    assert payload["token_usage"]["reasoning_generated_tokens"] == 7
    assert payload["token_usage"]["prompt_tokens_by_turn"] == [
        {
            "turn_id": "final-answer",
            "kind": "final_answer",
            "generation_kind": "action",
            "prompt_tokens": 3,
        }
    ]


def test_runtime_terminates_episode_when_acting_prompt_exceeds_fit_limit() -> None:
    backend = FakeBackend(search_index={}, documents={})
    model = ScriptedModel(outputs=[tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')])
    runtime = EpisodeRuntime(
        model=model,
        backend=backend,
        context_threshold_tokens=1000,
        max_context_tokens=3,
        token_counter=lambda text: len(text.split()),
    )

    result = runtime.run(query_id="q1", user_prompt="question with too many words")

    assert result.status == "context_length_exceeded"
    assert result.final_answer is None
    assert result.turn_rewards == {}
    assert result.trajectory_records == []


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
