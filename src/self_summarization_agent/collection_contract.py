"""Artifact lineage checks performed before any collection/resume shortcuts."""
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from self_summarization_agent.chat_template import load_chat_template
from self_summarization_agent.token_stream import TITO_CONTRACT, TITO_COLLECTION_VERSION


def collection_profile_id(config: Any, checkpoint: Path) -> str:
    identity = dict(contract=TITO_CONTRACT, template=load_chat_template(config.model.chat_template_path),
                    thinking=config.model.enable_thinking, backend=config.rollout.backend,
                    checkpoint=str(checkpoint), tokenizer_files={})
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"):
        path = checkpoint / name
        if path.is_file():
            identity["tokenizer_files"][name] = sha256(path.read_bytes()).hexdigest()
    return sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def validate_artifact_lineage(paths: Iterable[Path | None], *, config: Any, checkpoint: Path) -> None:
    # Scripted/test collectors without a project template keep the legacy path.
    # Production token renderers independently require the approved template.
    if not config.model.chat_template_path:
        return
    expected = collection_profile_id(config, checkpoint)
    from self_summarization_agent.trajectory import _extract_collection_tokens, validate_trajectory_schema
    for path in paths:
        if path is None or not Path(path).is_file():
            continue
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("collection_profile_id") != expected or row.get("collection_contract") != TITO_CONTRACT:
                    raise ValueError(f"Cannot resume {path}:{number}: collection contract changed; use a fresh output lineage")
                records = row.get("trajectory_records")
                validate_trajectory_schema(records, context=f"Cannot resume {path}:{number}")
                for record in records:
                    payload = record.get("collection_tokens") or {}
                    if payload.get("version") != TITO_COLLECTION_VERSION:
                        raise ValueError(f"Cannot resume {path}:{number}: missing TITO collection tokens")
                    _extract_collection_tokens(record, turn_id=str(record.get("turn_id")))
