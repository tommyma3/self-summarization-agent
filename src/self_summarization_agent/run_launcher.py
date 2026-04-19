from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.config import load_run_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.export import build_run_record
from self_summarization_agent.generation import build_generator
from self_summarization_agent.launcher_utils import (
    append_jsonl,
    build_runtime,
    dataclass_to_jsonable,
    ensure_dir,
    seed_everything,
    serialize_runtime_result,
    utc_timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-summarization experiments on local BrowseComp-Plus.")
    parser.add_argument("--config", required=True, help="Path to the run YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset.limit.")
    parser.add_argument("--output-root", default=None, help="Override experiment.output_root.")
    parser.add_argument("--model-path", default=None, help="Override model.model_path.")
    parser.add_argument("--retrieval-backend", default=None, help="Override retrieval.backend.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional dotted config overrides, e.g. runtime.tool_budget=8",
    )
    return parser.parse_args()


def _merge_launcher_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_cli_overrides(args.overrides)
    if args.limit is not None:
        overrides["dataset.limit"] = args.limit
    if args.output_root is not None:
        overrides["experiment.output_root"] = args.output_root
    if args.model_path is not None:
        overrides["model.model_path"] = args.model_path
    if args.retrieval_backend is not None:
        overrides["retrieval.backend"] = args.retrieval_backend
    return overrides


def run_experiment(
    config,
    *,
    examples: list[QueryExample] | None = None,
    backend: Any | None = None,
    generator: Any | None = None,
) -> Path:
    seed_everything(config.experiment.seed)
    examples = examples or load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=False,
        seed=config.experiment.seed,
    )
    backend = backend or build_backend(config.experiment.bc_plus_root, config.retrieval)
    generator = generator or build_generator(config.model)
    runtime = build_runtime(generator, backend, config.runtime)

    run_dir = ensure_dir(Path(config.experiment.output_root) / "runs" / config.experiment.name)
    trajectory_path = run_dir / "trajectories.jsonl"
    if trajectory_path.exists():
        trajectory_path.unlink()

    for example in examples:
        result = runtime.run(query_id=example.query_id, user_prompt=example.query)
        write_json(run_dir / f"{example.query_id}.json", build_run_record(result))
        append_jsonl(
            trajectory_path,
            serialize_runtime_result(result, query_text=example.query),
        )

    manifest = {
        "timestamp_utc": utc_timestamp(),
        "config": dataclass_to_jsonable(config),
        "query_count": len(examples),
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config, _merge_launcher_overrides(args))
    run_dir = run_experiment(config)
    print(run_dir)


if __name__ == "__main__":
    main()
