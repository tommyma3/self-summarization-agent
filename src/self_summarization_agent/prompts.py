from collections.abc import Iterable

from self_summarization_agent.models import Message


class ConversationPrompt(str):
    """Readable prompt text carrying the structured messages used by chat templates."""

    def __new__(cls, messages: Iterable[Message]):
        copied_messages = tuple(Message(role=message.role, content=message.content) for message in messages)
        value = render_messages(copied_messages)
        prompt = super().__new__(cls, value)
        prompt.messages = copied_messages
        return prompt


def render_messages(messages: Iterable[Message]) -> str:
    return "\n".join(f"### {message.role.upper()}\n{message.content}" for message in messages)


def serialize_messages(messages: Iterable[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def format_tool_result(tool_result: str) -> str:
    return f"<information>{tool_result}</information>"


def build_system_prompt() -> str:
    return """You are an expert research agent.
Think first and then choose exactly one action:
(1) Call a search engine using format: <search> your query </search>.
(2) Call the document tool to retrieve documents using format: <document> docid </document>.
(3) Provide your final answer within <answer> </answer> tags."""


def build_forced_answer_prompt() -> str:
    return """The final-answer boundary has been reached.
Search and document actions are no longer available.
Think first, then answer with exactly one action:
<answer>best supported answer</answer>."""


def build_forced_answer_system_prompt() -> str:
    """Compatibility alias for the appended forced-answer instruction."""
    return build_forced_answer_prompt()


def build_summary_prompt(*, max_summary_tokens: int | None = None) -> str:
    # Retain the keyword for compatibility with existing callers; the limit is
    # enforced by the runtime rather than disclosed in the compaction prompt.
    del max_summary_tokens
    return """Compact the preceding task state into a summary for future steps.
Preserve the task, constraints, key evidence, unresolved work, and next steps.
Think first. After </think>, put only the compressed state inside <summary> </summary> tags."""


def build_summary_system_prompt(*, max_summary_tokens: int | None = None) -> str:
    """Compatibility alias for the appended compaction instruction."""
    return build_summary_prompt(max_summary_tokens=max_summary_tokens)


def build_initial_messages(user_prompt: str) -> list[Message]:
    return [
        Message(role="system", content=build_system_prompt()),
        Message(role="user", content=user_prompt),
    ]


def build_compacted_messages(summary: str) -> list[Message]:
    return [
        Message(role="system", content=build_system_prompt()),
        Message(role="user", content=summary),
    ]
