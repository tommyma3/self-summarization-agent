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
    return """You are an expert research agent answering the user's question step by step.

Normally, think first and then choose exactly one action:
(1) Call a search engine using format: <search> your query </search>.
(2) Call the document tool to retrieve documents using format: <document> docid </document>.
(3) Provide your final answer within <answer> </answer> tags.

The runtime may append a compaction or forced-answer instruction. When it does,
follow that final boundary instruction instead of taking a normal action."""


def build_forced_answer_prompt() -> str:
    return """The final-answer boundary has been reached.
Search and document actions are no longer available.
Think first, then answer with exactly one action:
<answer>best supported answer</answer>

Use only the preceding task state, reasoning, and tool results."""


def build_forced_answer_system_prompt() -> str:
    """Compatibility alias for the appended forced-answer instruction."""
    return build_forced_answer_prompt()


def build_summary_prompt(*, max_summary_tokens: int | None = None) -> str:
    length_instruction = (
        f"Keep the summary body to at most {max_summary_tokens} tokens.\n"
        if max_summary_tokens is not None
        else ""
    )
    return f"""Compact the entire preceding task trajectory into a self-summary for the next interval.
Preserve the original task, constraints, established evidence, hypotheses, unresolved work, and useful next steps.
Keep the summary structured and concise. Use short sentences.
{length_instruction}Think first, then return the compressed state after </think>."""


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
