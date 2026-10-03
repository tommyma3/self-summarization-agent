"""Real-backend token transport probe with synthetic terminal observations.

This does not execute model commands or measure Terminal-Bench performance.
Use Harbor/Docker trials for benchmark verification.
"""
import argparse
import json
from pathlib import Path

from self_summarization_agent.benchmarks.terminal_bench.scaffold import TerminusKiraScaffold
from self_summarization_agent.generation import VLLMGenerator
from self_summarization_agent.runtime import EpisodeRuntime
from self_summarization_agent.trajectory import _extract_collection_tokens


class ProbeTerminal:
    def execute(self, name, arguments, *, query_id):
        return "Synthetic terminal observation: the fixture's result.txt contains exactly hello. The task is satisfied."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = VLLMGenerator(model_path=args.model, max_new_tokens=2048,
        temperature=0, top_p=1, do_sample=False, tensor_parallel_size=1,
        max_model_len=16384, enable_thinking=True, language_model_only=True,
        gpu_memory_utilization=0.4, enable_prefix_caching=True,
        chat_template_path="src/self_summarization_agent/chat_templates/qwen3_5_agent.jinja")
    try:
        for name, threshold in (("no-compaction", 16000), ("compaction", 1)):
            backend = ProbeTerminal()
            runtime = EpisodeRuntime(model=generator, backend=backend,
                scaffold=TerminusKiraScaffold(backend), context_threshold_tokens=threshold,
                max_context_tokens=16384, max_summary_tokens=2048, max_tool_calls=4,
                generated_token_budget=12000, token_counter=generator.count_tokens)
            result = runtime.run("transport-fixture",
                "This is a synthetic tool-transport test. First call execute_commands with analysis and plan "
                "and exactly one command: cat result.txt followed by a newline. Read the observation, "
                "then use task_complete and follow its confirmation checklist to end. Do not execute other commands.")
            path = args.output_dir / f"{name}.records.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in result.trajectory_records))
            for record in result.trajectory_records:
                _extract_collection_tokens(record, turn_id=record["turn_id"])
            print(json.dumps(dict(profile=name, status=result.status,
                intervals=len(result.trajectory_records), summaries=len(result.summary_turns),
                token_usage=result.token_usage, path=str(path))), flush=True)
            if result.status != "completed" or (name == "compaction" and not result.summary_turns):
                raise RuntimeError(f"Probe did not reach the required boundary: {name}: {result.status}")
    finally:
        core = getattr(getattr(generator.llm, "llm_engine", None), "engine_core", None)
        if core is not None:
            core.shutdown()


if __name__ == "__main__":
    main()
