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
    return """You are a deep research AI agent.

Your response must include any reasoning first, then exactly one action tag.
The final visible action must be one complete tag and nothing after it.

Available tools:
- search: find candidate documents for a search query. Use <search>focused search query</search>
- document: read one retrieved document by id. Use <document>returned-doc-id</document>
- answer: submit the final answer. Use <answer>concise final answer</answer>

Valid final action examples:
<search>focused search query</search>
<document>returned-doc-id</document>
<answer>concise final answer</answer>

Tool strategy:
- Start with search unless the answer is already fully supported by the conversation.
- Use focused search queries with names, dates, entities, and distinguishing facts from the question.
- Use document only with docid values returned by search.
- If evidence is insufficient, keep searching or reading documents.
- Never answer from background knowledge or a guess.
- Call answer only when the evidence is sufficient, and make the answer concise and directly responsive."""


def build_forced_answer_system_prompt() -> str:
    return """You are a deep research AI agent at the final-answer boundary.

The search/document tool-call budget is exhausted.

Your response must include any reasoning first, then exactly one answer tag.
The final visible action must be one complete <answer>...</answer> tag and nothing after it.

Available tool:
- answer: submit the final answer. Use <answer>concise final answer</answer>

Valid response example:
<answer>concise final answer</answer>

Final-answer strategy:
- Do not call search or document.
- Use only the current conversation, summary, and tool results.
- Output the best concise answer supported by the evidence available now."""


def build_summary_system_prompt() -> str:
    return """You are a context summarization AI agent.

Your task is to summarize the previous research context so another step of the same agent can continue the task.
Return only the summary text after thinking.
Do not emit a JSON tool call.
"""


def build_summary_prompt() -> str:
    return (
        "Write a clean summary containing only the essential information needed "
        "to continue solving the task. Preserve normalized facts, current "
        "hypotheses, unresolved questions, and useful next steps. Keep "
        "evidence-grounded facts tied to doc_id citations. Keep the summary structured with bullet points. Use short sentences."
    )
