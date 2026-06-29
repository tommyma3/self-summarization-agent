import argparse
import importlib.util
from pathlib import Path

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import ModelConfig, RolloutConfig, RuntimeConfig
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.launcher_utils import build_runtime
from self_summarization_agent.runtime import ScriptedModel


PROBE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_compaction_comparison.py"
PROBE_SPEC = importlib.util.spec_from_file_location("probe_compaction_comparison", PROBE_SCRIPT)
probe_comparison = importlib.util.module_from_spec(PROBE_SPEC)
assert PROBE_SPEC.loader is not None
PROBE_SPEC.loader.exec_module(probe_comparison)

SIMULATE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_collection.py"
SIMULATE_SPEC = importlib.util.spec_from_file_location("simulate_collection", SIMULATE_SCRIPT)
simulate_collection = importlib.util.module_from_spec(SIMULATE_SPEC)
assert SIMULATE_SPEC.loader is not None
SIMULATE_SPEC.loader.exec_module(simulate_collection)


class ConfigForProbe:
    model = ModelConfig(model_path="model", max_model_len=32768)
    rollout = RolloutConfig(max_model_len=49152, max_new_tokens=8192)


class RecordingModel(ScriptedModel):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs=outputs)
        self.prompts: list[str] = []
        self.max_model_len = 49152

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return super().generate(prompt)


def _tool_output(json_text: str, thinking: str = "thinking") -> str:
    return f"<think>{thinking}</think>\n{json_text}"


def test_probe_rollout_model_config_uses_rollout_max_model_len() -> None:
    args = argparse.Namespace(
        tensor_parallel_size=2,
        attention_backend="TORCH_SDPA",
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        do_sample=None,
    )

    rollout_model_config = probe_comparison.build_rollout_model_config(ConfigForProbe, args)

    assert rollout_model_config.max_model_len == 49152
    assert rollout_model_config.max_new_tokens == 8192


def test_trace_collection_forces_answer_after_generated_token_budget(tmp_path: Path) -> None:
    search_output = _tool_output('{"tool_name": "search", "arguments": {"query": "q"}}', thinking="one two")
    answer_output = _tool_output('{"tool_name": "finish", "arguments": {"answer": "best available"}}')
    model = RecordingModel(outputs=[search_output, answer_output])
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "document body"})
    runtime = build_runtime(
        model,
        backend,
        RuntimeConfig(
            context_threshold_tokens=1000,
            max_context_tokens=4096,
            tool_budget=16,
            generated_token_budget=1,
        ),
    )
    output_path = tmp_path / "trace.txt"

    simulate_collection.trace_collection(
        runtime=runtime,
        generator=model,
        example=QueryExample(query_id="q1", query="question", answer="best available"),
        sample_index=0,
        output_path=output_path,
        include_formatted_prompt=False,
    )

    trace = output_path.read_text(encoding="utf-8")
    assert "Round 2 Forced Answer Reasons" in trace
    assert '"generated_token_budget"' in trace
    assert "Round 2 Acting Context" not in trace
    assert "final_answer: best available" in trace
