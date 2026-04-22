from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, ToolCallRecord, ToolRound
from self_summarization_agent.prompts import build_summary_system_prompt, build_system_prompt


def test_threshold_crossing_marks_summary_after_completed_round() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=64)
    state = EpisodeState(query_id="q1", user_prompt="user prompt", context_threshold_tokens=10)
    state.rounds.append(
        ToolRound(
            assistant_message=Message(role="assistant", content="search for clue"),
            tool_call=ToolCallRecord(tool_name="search", arguments={"query": "clue"}, raw_output='{"query": "clue"}'),
            tool_result=Message(role="tool", content="doc-1 doc-2 doc-3 doc-4 doc-5"),
        )
    )

    assert manager.should_summarize(state) is True


def test_pack_summary_input_uses_all_rounds_it_is_given() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=64)
    state = EpisodeState(query_id="q1", user_prompt="user prompt", context_threshold_tokens=5)
    state.rounds.extend(
        [
            ToolRound(
                assistant_message=Message(role="assistant", content="older search"),
                tool_call=ToolCallRecord(tool_name="search", arguments={"query": "older"}, raw_output='{"query": "older"}'),
                tool_result=Message(role="tool", content="old result"),
            ),
            ToolRound(
                assistant_message=Message(role="assistant", content="latest search"),
                tool_call=ToolCallRecord(tool_name="search", arguments={"query": "latest"}, raw_output='{"query": "latest"}'),
                tool_result=Message(role="tool", content="latest result"),
            ),
        ]
    )

    packed = manager.build_summary_context(state)
    assert packed.startswith(build_summary_system_prompt())
    assert build_system_prompt() not in packed
    assert "older search" in packed
    assert '"query": "older"' in packed
    assert "latest search" in packed


def test_pack_summary_input_keeps_lone_completed_round() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=64)
    state = EpisodeState(query_id="q1", user_prompt="user prompt", context_threshold_tokens=5)
    state.rounds.append(
        ToolRound(
            assistant_message=Message(role="assistant", content="only search"),
            tool_call=ToolCallRecord(tool_name="search", arguments={"query": "only"}, raw_output='{"query": "only"}'),
            tool_result=Message(role="tool", content="only result"),
        )
    )

    packed = manager.build_summary_context(state)
    assert "only search" in packed
    assert '"query": "only"' in packed


def test_large_tool_call_payload_affects_summarization_decision() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=64)
    base_state = EpisodeState(query_id="q1", user_prompt="user prompt", context_threshold_tokens=0)
    threshold = manager.current_token_count(base_state) + 1

    state = EpisodeState(query_id="q1", user_prompt="user prompt", context_threshold_tokens=threshold)
    state.rounds.append(
        ToolRound(
            assistant_message=Message(role="assistant", content="search for clue"),
            tool_call=ToolCallRecord(
                tool_name="search",
                arguments={"query": "clue", "payload": " ".join(["extra"] * 50)},
                raw_output=" ".join(["result"] * 50),
            ),
            tool_result=Message(role="tool", content="doc-1"),
        )
    )

    assert manager.should_summarize(state) is True


def test_assert_fits_clamps_effective_limit_when_margin_is_too_large() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=1, safety_margin_tokens=4)

    manager.assert_fits("hello")
