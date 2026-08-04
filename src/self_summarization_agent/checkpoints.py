from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


LATEST_CHECKPOINT_FILE = "latest"
CHECKPOINT_COMPLETE_FILE = ".complete"


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    checkpoint_id: str
    path: Path


def checkpoint_id_from_path(path: str | Path) -> str:
    return Path(path).name


def mark_checkpoint_complete(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    marker = checkpoint_path / CHECKPOINT_COMPLETE_FILE
    marker.write_text("complete\n", encoding="utf-8")
    return marker


def is_vllm_loadable_checkpoint(path: str | Path) -> bool:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_dir():
        return False
    if not (checkpoint_path / CHECKPOINT_COMPLETE_FILE).exists():
        return False
    has_config = (checkpoint_path / "config.json").exists()
    has_model_weights = any(
        child.name.startswith(("model", "pytorch_model", "adapter_model"))
        and child.suffix in {".bin", ".safetensors"}
        for child in checkpoint_path.iterdir()
        if child.is_file()
    )
    has_sharded_index = any(
        child.name.endswith(".index.json")
        for child in checkpoint_path.iterdir()
        if child.is_file()
    )
    return has_config and (has_model_weights or has_sharded_index)


def write_latest_checkpoint(root: str | Path, checkpoint_path: str | Path) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint = Path(checkpoint_path).resolve()
    latest_path = root_path / LATEST_CHECKPOINT_FILE
    temp_path = root_path / f".{LATEST_CHECKPOINT_FILE}.tmp"
    temp_path.write_text(str(resolved_checkpoint), encoding="utf-8")
    os.replace(temp_path, latest_path)
    return latest_path


def resolve_latest_checkpoint(root: str | Path) -> CheckpointRef:
    latest_path = Path(root) / LATEST_CHECKPOINT_FILE
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing latest checkpoint pointer: {latest_path}")
    raw_path = latest_path.read_text(encoding="utf-8").strip()
    if not raw_path:
        raise ValueError(f"Latest checkpoint pointer is empty: {latest_path}")
    checkpoint_path = Path(raw_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = latest_path.parent / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    return CheckpointRef(checkpoint_id=checkpoint_id_from_path(checkpoint_path), path=checkpoint_path)


def advance_latest_checkpoint(root: str | Path, checkpoint_path: str | Path) -> CheckpointRef:
    resolved = Path(checkpoint_path).resolve()
    if not is_vllm_loadable_checkpoint(resolved):
        raise ValueError(f"Checkpoint is not complete or vLLM-loadable: {resolved}")
    write_latest_checkpoint(root, resolved)
    return CheckpointRef(checkpoint_id=checkpoint_id_from_path(resolved), path=resolved)
