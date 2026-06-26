def _tag_text(text: object) -> str:
    return str(text).replace("</", "< /").strip()


def format_action_tag(tool_name: str, arguments: dict[str, object]) -> str:
    if tool_name == "search":
        return f"<search>{_tag_text(arguments.get('query', ''))}</search>"
    if tool_name == "get_document":
        return f"<document>{_tag_text(arguments.get('doc_id', ''))}</document>"
    if tool_name == "finish":
        return f"<answer>{_tag_text(arguments.get('answer', ''))}</answer>"
    return ""


def format_history_round(tool_name: str, arguments: dict[str, object], tool_result: str) -> str:
    action = format_action_tag(tool_name, arguments)
    information = f"<information>{_tag_text(tool_result)}</information>"
    return "\n".join(part for part in (action, information) if part)


def build_system_prompt() -> str:
    return """You are an expert research agent answering the user's question step by step.

Think first. Then choose exactly one action:
(1) Call a search engine using format: <search> your query </search>.
(2) Call the document tool to retrieve documents using format: <document> docid </document>.
(3) Provide your final answer within <answer> </answer> tags."""


def build_forced_answer_system_prompt() -> str:
    return """You are an expert research agent. You are at the final-answer boundary.

Search and document actions are no longer available.
Think first, then answer with exactly one action:
<answer>best supported answer</answer>

Use only the conversation, summary, and tool results."""


def build_summary_system_prompt() -> str:
    return """You are a context summarization AI agent.

Your task is to summarize the previous research context so the research agent can continue the task. The summary should contain only the essential information needed to continue solving the task. Keep the summary structured. Use short sentences. Return a summary using format: <summary> your summary </summary>. 
"""


def build_summary_prompt() -> str:
    return (
        ""
    )
