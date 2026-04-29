from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import load_query_examples
from self_summarization_agent.launcher_utils import ensure_dir


def build_openrlhf_label(*, query_id: str, answer: str | None) -> str:
    return json.dumps(
        {
            "query_id": query_id,
            "answer": answer,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def export_openrlhf_prompt_data(config: Any, output_path: str | Path) -> Path:
    examples = load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    train_examples = examples if config.dataset.train_limit is None else examples[: config.dataset.train_limit]
    if not train_examples:
        raise ValueError("No training queries available for OpenRLHF export")

    output = Path(output_path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        for example in train_examples:
            row = {
                "prompt": example.query,
                "label": build_openrlhf_label(query_id=example.query_id, answer=example.answer),
                "query_id": example.query_id,
                "answer": example.answer,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BrowseComp+ prompts for OpenRLHF agent training.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--output", required=True, help="Output JSONL path for OpenRLHF --prompt_data.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    output = export_openrlhf_prompt_data(config, args.output)
    print(output)


if __name__ == "__main__":
    main()
