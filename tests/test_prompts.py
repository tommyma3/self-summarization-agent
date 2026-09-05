from self_summarization_agent.models import EpisodeState
from self_summarization_agent.prompts import (
    build_compacted_messages,
    build_forced_answer_system_prompt,
    build_initial_messages,
    build_native_tool_system_prompt,
    build_summary_prompt,
    build_summary_system_prompt,
    build_system_prompt,
    format_tool_result,
    format_tool_response,
)


def test_build_forced_answer_system_prompt_allows_only_finish() -> None:
    prompt = build_forced_answer_system_prompt()
    assert "final-answer boundary" in prompt
    assert "Tool Budget Remaining" not in prompt
    assert "<answer>best supported answer</answer>" in prompt
    assert prompt.startswith("<forced_answer_request>")
    assert prompt.endswith("</forced_answer_request>")


def test_build_summary_system_prompt_is_concise_and_uses_summary_request_tags() -> None:
    prompt = build_summary_system_prompt(max_summary_tokens=2048)

    assert "2048" not in prompt
    assert "Compact the agent history" in prompt
    assert prompt.startswith("<summary_request>")
    assert prompt.endswith("</summary_request>")


def test_system_prompt_lists_normal_action_formats() -> None:
    prompt = build_system_prompt()

    assert "<search> your query </search>" in prompt
    assert "<document> docid </document>" in prompt
    assert "<answer> </answer>" in prompt
    assert "<summary_request>" in prompt
    assert "do not search, retrieve, or answer" in prompt


def test_native_tool_system_prompt_defines_stable_compaction_exception() -> None:
    prompt = build_native_tool_system_prompt()

    assert "Normally, use exactly one provided function per turn" in prompt
    assert "<summary_request>" in prompt
    assert "do not call a function" in prompt
    assert "Normal function-calling mode resumes afterward" in prompt


def test_tool_result_wrapper_preserves_raw_result_text() -> None:
    assert format_tool_result("raw </tag> body") == "<information>raw </tag> body</information>"
    assert format_tool_response("raw </tag> body") == (
        "<tool_response>\n<information>raw </tag> body</information>\n</tool_response>"
    )


def test_compacted_messages_preserve_original_prefix_and_wrap_generated_state() -> None:
    for native_tools in (False, True):
        initial_messages = build_initial_messages("original user query", native_tools=native_tools)
        messages = build_compacted_messages(
            "original user query",
            "compressed task state",
            native_tools=native_tools,
        )

        assert messages[:2] == initial_messages
        assert messages[1].role == "user"
        assert messages[1].content == "original user query"
        assert messages[2].role == "user"
        assert messages[2].content == "<summary>\ncompressed task state\n</summary>"
        assert messages[2].content.count("<summary>") == 1
        assert messages[2].content.count("</summary>") == 1


def test_initial_messages_leave_raw_user_query_unwrapped() -> None:
    messages = build_initial_messages("original user query")

    assert messages[1].role == "user"
    assert messages[1].content == "original user query"


def test_episode_state_starts_with_empty_summary() -> None:
    state = EpisodeState(query_id="q1", user_prompt="question", context_threshold_tokens=1024)
    assert state.latest_summary is None
    assert state.summary_count == 0
