from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message
from self_summarization_agent.prompts import build_initial_messages


def test_threshold_counts_raw_reasoning_and_tool_results() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=64)
    state = EpisodeState(
        query_id="q1",
        user_prompt="user prompt",
        context_threshold_tokens=10,
        messages=build_initial_messages("user prompt"),
    )
    state.messages.extend(
        [
            Message(role="assistant", content="reason carefully then search for clue"),
            Message(
                role="user",
                content="<tool_response>\n<information>doc-1 doc-2 doc-3 doc-4 doc-5</information>\n</tool_response>",
            ),
        ]
    )

    assert manager.should_summarize(state) is True


def test_summary_instruction_is_appended_to_unchanged_context() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=128)
    state = EpisodeState(
        query_id="q1",
        user_prompt="user prompt",
        context_threshold_tokens=10,
        messages=build_initial_messages("user prompt"),
    )
    state.messages.append(Message(role="assistant", content="private reasoning and action"))

    prompt = manager.build_summary_context(state, max_summary_tokens=32)

    assert list(prompt.messages[:-1]) == state.messages
    assert prompt.messages[-1].role == "user"
    assert prompt.messages[-1].content.startswith("<summary_request>")
    assert prompt.messages[-1].content.endswith("</summary_request>")
    assert "Compact the agent history" in prompt.messages[-1].content
    assert "<summary>...</summary>" in prompt.messages[-1].content
    assert "remain present verbatim" in prompt.messages[-1].content
    assert "32" not in prompt.messages[-1].content


def test_assert_fits_clamps_effective_limit_when_margin_is_too_large() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=1, safety_margin_tokens=4)

    manager.assert_fits("hello")
