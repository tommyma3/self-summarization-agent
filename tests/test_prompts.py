from self_summarization_agent.models import EpisodeState
from self_summarization_agent.prompts import (
    build_forced_answer_system_prompt,
    build_summary_prompt,
    build_summary_system_prompt,
    build_system_prompt,
    format_tool_result,
)


def test_build_forced_answer_system_prompt_allows_only_finish() -> None:
    prompt = build_forced_answer_system_prompt()
    assert "final-answer boundary" in prompt
    assert "Tool Budget Remaining" not in prompt
    assert "Search and document actions are no longer available" in prompt
    assert "<answer>best supported answer</answer>" in prompt


def test_build_summary_system_prompt_includes_configured_length_cap() -> None:
    prompt = build_summary_system_prompt(max_summary_tokens=2048)

    assert "at most 2048 tokens" in prompt


def test_system_prompt_defers_to_appended_runtime_boundaries() -> None:
    prompt = build_system_prompt()

    assert "runtime may append a compaction or forced-answer instruction" in prompt
    assert "follow that final boundary instruction" in prompt


def test_tool_result_wrapper_preserves_raw_result_text() -> None:
    assert format_tool_result("raw </tag> body") == "<information>raw </tag> body</information>"


def test_episode_state_starts_with_empty_summary() -> None:
    state = EpisodeState(query_id="q1", user_prompt="question", context_threshold_tokens=1024)
    assert state.latest_summary is None
    assert state.summary_count == 0
