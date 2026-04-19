def build_system_prompt() -> str:
    return (
        "Solve the benchmark question using the provided tools. "
        "Use search and get_document to gather evidence. "
        "Do not invent unsupported claims. "
        "Return a concise final answer when the evidence is sufficient."
    )


def build_summary_prompt() -> str:
    return (
        "Write a clean summary containing only the essential information needed "
        "to continue solving the task. Preserve normalized facts, current "
        "hypotheses, unresolved questions, and useful next steps. Keep "
        "evidence-grounded facts tied to doc_id citations."
    )
