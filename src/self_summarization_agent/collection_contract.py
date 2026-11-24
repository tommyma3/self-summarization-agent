"""Artifact lineage checks performed before any collection/resume shortcuts."""
from hashlib import sha256
from functools import lru_cache
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from self_summarization_agent.chat_template import REPO_ROOT, load_chat_template
from self_summarization_agent.config import runtime_config_identity
from self_summarization_agent.rewards import BINARY_REWARD
from self_summarization_agent.token_stream import TITO_CONTRACT, TITO_COLLECTION_VERSION


# The earlier profile format hashed the full template source, even though
# token-in/token-out collection used it only to start intervals. These two
# releases produce identical initial IDs. Other revisions must match exactly.
_INITIAL_TEMPLATE_SHA256 = "58a2057f9253a97d937ad8fe81550da5a520fa6d8071567c67b8c1d95cb033e2"
_RETIRED_TEMPLATE_SHA256 = "5827769d61e84cebe01f8c6133717b9a34cf6724a4e192f2e88d370c0891bf22"


def equivalent_initial_templates(current: str | None, previous: str | None) -> bool:
    return (isinstance(current, str) and isinstance(previous, str)
            and sha256(current.encode()).hexdigest() == _INITIAL_TEMPLATE_SHA256
            and sha256(previous.encode()).hexdigest() == _RETIRED_TEMPLATE_SHA256)


@lru_cache(maxsize=1)
def _retired_profile_source() -> str | None:
    # Historical hash input only. Never install or render this blob.
    try:
        blob = subprocess.check_output(
            ["git", "cat-file", "blob", "3cbbd420be54b574f66379ad4a01824a5ea4b83d"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return blob.decode("utf-8")


def saved_collection_profile(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as rows:
        for line in rows:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a rollout row in {path}")
                return row.get("collection_profile_id")
    return None


def collection_profile_id(config: Any, checkpoint: Path, *, existing_id: str | None = None) -> str:
    identity = dict(contract=TITO_CONTRACT, template=load_chat_template(config.model.chat_template_path),
                    thinking=config.model.enable_thinking, backend=config.rollout.backend,
                    checkpoint=str(checkpoint), tokenizer_files={})
    if getattr(config.rollout, "browsecomp_native_tools", False):
        from self_summarization_agent.prompts import ACTION_TOOLS, build_native_tool_system_prompt
        identity["browsecomp_native_tools"] = [build_native_tool_system_prompt(), ACTION_TOOLS]
    if getattr(getattr(config, "benchmark", None), "name", "browsecomp") in {"terminal-bench", "swe-dev"}:
        from dataclasses import asdict
        from self_summarization_agent.benchmarks.harbor_common.dataset import benchmark_identity
        identity["benchmark"] = benchmark_identity(config.benchmark)
        identity["runtime"] = runtime_config_identity(config.runtime)
        if config.benchmark.name == "swe-dev":
            # Catch local checkpoint replacement before accepting a stale cache.
            # Checkpoints are immutable once published; metadata avoids repeatedly
            # streaming multi-GB weights during the staged pipeline.
            identity["policy_weight_files"] = [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                for p in sorted(checkpoint.iterdir()) if p.is_file()
                and p.suffix in {".safetensors", ".bin"}] if checkpoint.is_dir() else []
            identity["selection"] = dict(train_task_count=config.collection.train_task_count,
                eval_task_count=config.collection.eval_task_count, group_size=config.training.group_size,
                eval_attempts=config.evaluation.samples_per_task, dataset=asdict(config.dataset),
                seed=config.experiment.seed)
    elif (dataset := getattr(config, "dataset", None)) is not None and dataset.train_limit is not None:
        # Reject resume shortcuts for artifacts collected under a different
        # partition, including the old experiment-seed/shuffle-based split.
        identity["dataset_split"] = {
            "version": "query-id-sorted-seeded-v1",
            "seed": dataset.split_seed,
            "offset": dataset.offset,
            "limit": dataset.limit,
            "train_limit": dataset.train_limit,
            "eval_limit": dataset.eval_limit,
        }
    profile = getattr(getattr(config, "benchmark", None), "reward_profile", BINARY_REWARD)
    if profile != BINARY_REWARD:
        identity["reward_profile"] = profile
    runtime = getattr(config, "runtime", None)
    if getattr(runtime, "max_compactions", None) is not None and "runtime" not in identity:
        identity["runtime"] = dict(max_compactions=runtime.max_compactions,
            context_threshold_tokens=runtime.context_threshold_tokens,
            max_context_tokens=runtime.max_context_tokens, max_summary_tokens=runtime.max_summary_tokens)
        if getattr(runtime, "force_answer_at_compaction_limit", False):
            identity["runtime"]["force_answer_at_compaction_limit"] = True
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"):
        path = checkpoint / name
        if path.is_file():
            identity["tokenizer_files"][name] = sha256(path.read_bytes()).hexdigest()
    current = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if existing_id is not None and current != existing_id:
        previous = _retired_profile_source()
        if equivalent_initial_templates(identity["template"], previous):
            identity["template"] = previous
            if sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == existing_id:
                return existing_id
    return current


def validate_artifact_lineage(paths: Iterable[Path | None], *, config: Any, checkpoint: Path) -> None:
    # Scripted/test collectors without a project template keep the legacy path.
    # Production token renderers independently require the approved template.
    if not config.model.chat_template_path:
        return
    from self_summarization_agent.trajectory import _extract_collection_tokens, validate_trajectory_schema
    for path in paths:
        if path is None or not Path(path).is_file():
            continue
        expected = None
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if expected is None:
                    expected = collection_profile_id(config, checkpoint, existing_id=row.get("collection_profile_id"))
                if row.get("collection_profile_id") != expected or row.get("collection_contract") != TITO_CONTRACT:
                    raise ValueError(f"Cannot resume {path}:{number}: collection contract changed; use a fresh output lineage")
                records = row.get("trajectory_records")
                validate_trajectory_schema(records, context=f"Cannot resume {path}:{number}")
                for record in records:
                    payload = record.get("collection_tokens") or {}
                    if payload.get("version") != TITO_COLLECTION_VERSION:
                        raise ValueError(f"Cannot resume {path}:{number}: missing TITO collection tokens")
                    _extract_collection_tokens(record, turn_id=str(record.get("turn_id")))
