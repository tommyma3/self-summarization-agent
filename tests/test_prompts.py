from self_summarization_agent.models import EpisodeState
from self_summarization_agent.prompts import (
    build_forced_answer_system_prompt,
    build_summary_prompt,
    build_summary_system_prompt,
    build_system_prompt,
)


def test_build_system_prompt_mentions_tools() -> None:
    prompt = build_system_prompt()
    assert "search" in prompt
    assert "document" in prompt
    assert "answer" in prompt
    assert "choose exactly one action" in prompt
    assert "Tool Budget Remaining" not in prompt
    assert "```" not in prompt
    assert "Think first" in prompt
    assert "Do not output more than one action" in prompt
    assert "<search>query</search>" in prompt
    assert "<document>doc_id</document>" in prompt
    assert "<answer>answer</answer>" in prompt


def test_build_forced_answer_system_prompt_allows_only_finish() -> None:
    prompt = build_forced_answer_system_prompt()
    assert "final-answer boundary" in prompt
    assert "Tool Budget Remaining" not in prompt
    assert "Search and document actions are no longer available" in prompt
    assert "<answer>best supported answer</answer>" in prompt


def test_build_summary_prompt_mentions_doc_ids() -> None:
    prompt = build_summary_prompt()
    assert "doc_id" in prompt
    assert "essential information" in prompt


def test_build_summary_system_prompt_replaces_tool_call_contract() -> None:
    prompt = build_summary_system_prompt()
    assert "summarize the previous research context" in prompt
    assert "Return only the summary text" in prompt
    assert "Do not emit a JSON tool call" in prompt
    assert "exactly one JSON object" not in prompt


def test_episode_state_starts_with_empty_summary() -> None:
    state = EpisodeState(query_id="q1", user_prompt="question", context_threshold_tokens=1024)
    assert state.latest_summary is None
    assert state.summary_count == 0
