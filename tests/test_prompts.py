from self_summarization_agent.models import EpisodeState
from self_summarization_agent.prompts import build_summary_prompt, build_system_prompt


def test_build_system_prompt_mentions_tools() -> None:
    prompt = build_system_prompt()
    assert "search" in prompt
    assert "get_document" in prompt
    assert "finish" in prompt
    assert "exactly one JSON object" in prompt
    assert "Do not wrap the JSON in ``` fences" in prompt
    assert "After any internal reasoning" in prompt
    assert "Never call finish from background knowledge or a guess" in prompt
    assert '{"tool_name": "search", "arguments": {"query": "..."}}' in prompt
    assert '{"tool_name": "get_document", "arguments": {"doc_id": "..."}}' in prompt
    assert '{"tool_name": "finish", "arguments": {"answer": "..."}}' in prompt


def test_build_summary_prompt_mentions_doc_ids() -> None:
    prompt = build_summary_prompt()
    assert "doc_id" in prompt
    assert "essential information" in prompt


def test_episode_state_starts_with_empty_summary() -> None:
    state = EpisodeState(query_id="q1", user_prompt="question", context_threshold_tokens=1024)
    assert state.latest_summary is None
    assert state.summary_count == 0
