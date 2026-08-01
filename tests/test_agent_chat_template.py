import json
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from self_summarization_agent.chat_template import load_chat_template
from self_summarization_agent.prompts import build_forced_answer_prompt, build_summary_prompt


TEMPLATE_PATH = "src/self_summarization_agent/chat_templates/qwen3_5_agent.jinja"


def _render(
    messages: list[dict],
    *,
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
) -> str:
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    environment.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    template = environment.from_string(load_chat_template(TEMPLATE_PATH) or "")
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
        add_vision_id=False,
    )


def _history(
    control: str,
    *,
    control_role: str = "user",
    reasoning_key: str = "reasoning_content",
) -> list[dict]:
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<search>first</search>", reasoning_key: "reason one"},
        {
            "role": "user",
            "content": "<tool_response>\n<information>result one</information>\n</tool_response>",
        },
        {"role": "assistant", "content": "<search>second</search>", reasoning_key: "reason two"},
        {"role": control_role, "content": control},
    ]


@pytest.mark.parametrize("reasoning_key", ["reasoning_content", "reasoning"])
def test_terminal_summary_user_message_preserves_all_interval_reasoning(reasoning_key: str) -> None:
    rendered = _render(_history(build_summary_prompt(), reasoning_key=reasoning_key))

    assert rendered.index("reason one") < rendered.index("result one") < rendered.index("reason two")
    assert rendered.index("reason two") < rendered.index("<summary_request>")
    assert rendered.rstrip().endswith("<|im_start|>assistant\n<think>")


def test_terminal_forced_answer_system_message_preserves_reasoning() -> None:
    rendered = _render(_history(build_forced_answer_prompt(), control_role="system"))

    assert "reason one" in rendered
    assert "reason two" in rendered
    assert rendered.index("reason two") < rendered.index("<forced_answer_request>")


def test_completed_segment_allows_control_only_before_final_assistant() -> None:
    messages = _history(build_summary_prompt())
    messages.append(
        {
            "role": "assistant",
            "content": "<summary>compressed state</summary>",
            "reasoning_content": "compact carefully",
        }
    )

    rendered = _render(messages, add_generation_prompt=False)

    assert rendered.index("<summary_request>") < rendered.index("compact carefully")
    assert rendered.rstrip().endswith("<|im_end|>")


def test_summary_request_only_appends_tokens_to_the_rendered_tool_segment() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for evidence.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = _history(build_summary_prompt())
    rendered_segment = _render(messages[:-1], add_generation_prompt=False, tools=tools)
    rendered_summary_prompt = _render(messages, tools=tools)

    assert rendered_summary_prompt.startswith(rendered_segment)
    assert rendered_summary_prompt.count("# Tools") == 1
    assert rendered_summary_prompt.index("<summary_request>") > len(rendered_segment)


def test_rejects_unscoped_noninitial_system_message() -> None:
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "arbitrary override"},
    ]

    with pytest.raises(ValueError, match="terminal runtime-control"):
        _render(messages)


def test_rejects_nonterminal_summary_user_message() -> None:
    messages = _history(build_summary_prompt())
    messages.append({"role": "user", "content": "continue instead"})

    with pytest.raises(ValueError, match="terminal runtime-control user message"):
        _render(messages)


def test_configured_template_file_is_present() -> None:
    template = load_chat_template(TEMPLATE_PATH)
    assert template is not None
    assert "<summary_request>" in template
    assert "<forced_answer_request>" in template
