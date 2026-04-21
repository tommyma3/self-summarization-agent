def build_system_prompt() -> str:
    return """Solve the benchmark question using the provided tools.

You must respond with exactly one JSON object and no markdown, commentary, or extra text.

Available tools:
- search: find candidate documents for a search query. Use {"tool_name": "search", "arguments": {"query": "..."}}
- get_document: read one retrieved document by id. Use {"tool_name": "get_document", "arguments": {"doc_id": "..."}}
- finish: submit the final answer. Use {"tool_name": "finish", "arguments": {"answer": "..."}}

Tool strategy:
- Start with search unless the answer is already fully supported by the conversation.
- Use focused search queries with names, dates, entities, and distinguishing facts from the question.
- Use get_document only with doc_id values returned by search.
- Read enough documents to verify the answer; do not invent unsupported claims.
- If evidence is insufficient, keep searching or reading documents until the tool budget is reached.
- Call finish only when the evidence is sufficient, and make the answer concise and directly responsive."""


def build_summary_prompt() -> str:
    return (
        "Write a clean summary containing only the essential information needed "
        "to continue solving the task. Preserve normalized facts, current "
        "hypotheses, unresolved questions, and useful next steps. Keep "
        "evidence-grounded facts tied to doc_id citations."
    )
