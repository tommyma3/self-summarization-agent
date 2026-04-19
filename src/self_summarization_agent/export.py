from self_summarization_agent.models import RuntimeResult


def build_run_record(result: RuntimeResult) -> dict[str, object]:
    return {
        "query_id": result.query_id,
        "tool_call_counts": dict(result.tool_call_counts),
        "status": result.status,
        "retrieved_docids": list(result.retrieved_docids),
        "result": [
            {
                "type": "output_text",
                "output": result.final_answer or "",
            }
        ],
    }
