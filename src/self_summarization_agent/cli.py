import json

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.export import build_run_record
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel


def build_smoke_run_record() -> dict[str, object]:
    backend = FakeBackend(
        search_index={"smoke question": ["smoke-doc"]},
        documents={"smoke-doc": "smoke body"},
    )
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "smoke question"}}',
            '{"tool_name": "get_document", "arguments": {"doc_id": "smoke-doc"}}',
            '{"tool_name": "finish", "arguments": {"answer": "smoke answer"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)
    result = runtime.run(query_id="smoke-q1", user_prompt="smoke question")
    return build_run_record(result)


def main() -> None:
    print(json.dumps(build_smoke_run_record(), indent=2, sort_keys=True))
