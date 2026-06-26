import json

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.export import build_run_record
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel


def _tool_output(action: str) -> str:
    return f"<think>select next action</think>\n{action}"


def build_smoke_run_record() -> dict[str, object]:
    backend = FakeBackend(
        search_index={"smoke question": ["smoke-doc"]},
        documents={"smoke-doc": "smoke body"},
    )
    model = ScriptedModel(
        outputs=[
            _tool_output("<search>smoke question</search>"),
            _tool_output("<document>smoke-doc</document>"),
            _tool_output("<answer>smoke answer</answer>"),
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=1000, max_context_tokens=1024)
    result = runtime.run(query_id="smoke-q1", user_prompt="smoke question")
    return build_run_record(result)


def main() -> None:
    print(json.dumps(build_smoke_run_record(), indent=2, sort_keys=True))
