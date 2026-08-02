from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
from typing import Any

from self_summarization_agent.models import Message, ToolCall


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the retrieval index for passages relevant to the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_document",
        "description": "Retrieve the full text of one document returned by search.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "The document identifier returned by search.",
                }
            },
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Submit the best-supported final answer and end the episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the user's question.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

ACTION_TOOLS = [SEARCH_TOOL, DOCUMENT_TOOL, FINISH_TOOL]
FORCED_ANSWER_TOOLS = [FINISH_TOOL]
FORCED_FINISH_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "finish"},
}


class ConversationPrompt(str):
    """Readable prompt text carrying the structured messages used by chat templates."""

    def __new__(
        cls,
        messages: Iterable[Message],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        parallel_tool_calls: bool = False,
        generation_kind: str = "action",
    ):
        copied_messages = tuple(copy_message(message) for message in messages)
        value = render_messages(copied_messages)
        prompt = super().__new__(cls, value)
        prompt.messages = copied_messages
        prompt.tools = tuple(deepcopy(list(tools or [])))
        prompt.tool_choice = deepcopy(tool_choice)
        prompt.parallel_tool_calls = parallel_tool_calls
        prompt.generation_kind = generation_kind
        return prompt


def copy_tool_call(tool_call: ToolCall) -> ToolCall:
    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=deepcopy(tool_call.arguments),
        type=tool_call.type,
    )


def copy_message(message: Message) -> Message:
    return Message(
        role=message.role,
        content=message.content,
        reasoning_content=message.reasoning_content,
        tool_calls=[copy_tool_call(tool_call) for tool_call in message.tool_calls],
        tool_call_id=message.tool_call_id,
    )


def render_messages(messages: Iterable[Message]) -> str:
    rendered: list[str] = []
    for message in messages:
        body = message.content
        if message.reasoning_content:
            body = f"<think>{message.reasoning_content}</think>\n{body}"
        if message.tool_calls:
            calls = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
            body = f"{body}\n{json.dumps(calls, ensure_ascii=False)}".strip()
        if message.tool_call_id:
            body = f"tool_call_id={message.tool_call_id}\n{body}"
        rendered.append(f"### {message.role.upper()}\n{body}")
    return "\n".join(rendered)


def serialize_messages(
    messages: Iterable[Message],
    *,
    for_api: bool = False,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.reasoning_content is not None:
            # vLLM currently returns `reasoning`; Qwen tokenizers consume
            # `reasoning_content`. Retain the provider-neutral artifact name and
            # let the API adapter translate it when necessary.
            item["reasoning" if for_api else "reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.name,
                        "arguments": (
                            json.dumps(tool_call.arguments, ensure_ascii=False)
                            if for_api
                            else deepcopy(tool_call.arguments)
                        ),
                    },
                }
                for tool_call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        serialized.append(item)
    return serialized


def format_tool_result(tool_result: str) -> str:
    return f"<information>{tool_result}</information>"


def format_tool_response(tool_result: str) -> str:
    return f"<tool_response>\n{format_tool_result(tool_result)}\n</tool_response>"


def build_system_prompt() -> str:
    return """You are an expert research agent. Think first, then output exactly one action:
(1) Call a search engine using format: <search> your query </search>.
(2) Call the document tool to retrieve documents using format: <document> docid </document>.
(3) Provide your final answer within <answer> </answer> tags.
(4) When the user prompts a summary request, compact the agent context into a summary for further steps using format: <summary> summary </summary>.

IMPORTANT: When prompted the summary request, you can only output <summary> summary </summary> after thinking. Do NOT call other tools or answer.
"""


def build_native_tool_system_prompt() -> str:
    return """You are an expert research agent. Normally, use exactly one provided function per turn.
Use search to find relevant passages, get_document to inspect a selected document, and finish only when you can submit the best-supported final answer.

The runtime may append a user message enclosed in <summary_request> tags. For that next turn only, do not call a function or answer the original question. Compact the preceding task state and output exactly one <summary>...</summary> block. Normal function-calling mode resumes afterward."""


def build_forced_answer_prompt() -> str:
    return """<forced_answer_request>
The final-answer boundary has been reached. Answer with exactly one action: <answer>best supported answer</answer>.
</forced_answer_request>"""


def build_forced_answer_system_prompt() -> str:
    """Compatibility alias for the appended forced-answer instruction."""
    return build_forced_answer_prompt()


def build_summary_prompt(*, max_summary_tokens: int | None = None) -> str:
    # Retain the keyword for compatibility with existing callers; the limit is
    # enforced by the runtime rather than disclosed in the compaction prompt.
    del max_summary_tokens
    return """<summary_request>
Compact the context into a summary. Include the user's query and the research context.
</summary_request>"""


def build_native_summary_system_prompt() -> str:
    return """You compact research-agent state for a later continuation. Do not call tools. Preserve the original question, gathered evidence, unresolved questions, and the best next steps. Output exactly one <summary>...</summary> block and no text outside it."""


def build_summary_system_prompt(*, max_summary_tokens: int | None = None) -> str:
    """Compatibility alias for the appended compaction instruction."""
    return build_summary_prompt(max_summary_tokens=max_summary_tokens)


def build_initial_messages(user_prompt: str, *, native_tools: bool = False) -> list[Message]:
    return [
        Message(
            role="system",
            content=build_native_tool_system_prompt() if native_tools else build_system_prompt(),
        ),
        Message(role="user", content=user_prompt),
    ]


def format_compacted_summary(summary: str) -> str:
    """Mark a model-generated compressed state when it becomes the next user input."""
    return f"<summary>\n{summary}\n</summary>"


def build_compacted_messages(summary: str, *, native_tools: bool = False) -> list[Message]:
    return [
        Message(
            role="system",
            content=build_native_tool_system_prompt() if native_tools else build_system_prompt(),
        ),
        Message(role="user", content=format_compacted_summary(summary)),
    ]
