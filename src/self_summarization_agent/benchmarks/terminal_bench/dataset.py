"""Pinned official task manifest and explicit local training/evaluation tasks."""
from dataclasses import asdict
from fnmatch import fnmatch
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path

from . import HARBOR_VERSION, KIRA_REVISION, SCAFFOLD_VERSION
from .scaffold import TerminusKiraScaffold


MANIFEST = json.loads(files(__package__).joinpath("resources/dataset.json").read_text())[0]


def task_manifest(config, *, split):
    paths = config.train_task_paths if split == "train" else config.task_paths
    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported split: {split}")
    if paths:
        tasks = []
        for path in paths:
            root = Path(path).resolve()
            if not (root / "instruction.md").is_file() or not (root / "task.toml").is_file():
                raise ValueError(f"Not a Harbor task directory: {root}")
            digest = sha256()
            for file in sorted(root.rglob("*")):
                if file.is_file():
                    digest.update(str(file.relative_to(root)).encode())
                    digest.update(file.read_bytes())
            tasks.append(dict(name=root.name, path=str(root), checksum=digest.hexdigest()))
    else:
        if split == "train" and not config.allow_benchmark_training:
            raise ValueError("Official Terminal-Bench tasks are evaluation-only; set train_task_paths or explicitly allow_benchmark_training")
        if config.dataset != "terminal-bench@2.0":
            raise ValueError("Only the pinned terminal-bench@2.0 manifest is supported; use task_paths for other tasks")
        tasks = [dict(t) for t in MANIFEST["tasks"]]
        if config.dataset_revision and any(t["git_commit_id"] != config.dataset_revision for t in tasks):
            raise ValueError("dataset_revision disagrees with the pinned benchmark manifest")
    if config.task_names:
        tasks = [t for t in tasks if any(fnmatch(t["name"], p) for p in config.task_names)]
    if not tasks or len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("Task selection must be nonempty with unique names")
    return sorted(tasks, key=lambda t: t["name"])


def benchmark_identity(config):
    scaffold = TerminusKiraScaffold(None, image_enabled=config.image_profile == "local-vision")
    identity = dict(config=asdict(config), harbor=HARBOR_VERSION, kira=KIRA_REVISION,
                    scaffold=SCAFFOLD_VERSION, scaffold_digest=scaffold.fingerprint,
                    reward_mapping="verifier-binary-plus-runtime-penalties-v1",
                    manifest=MANIFEST)
    for key in ("task_paths", "train_task_paths"):
        if getattr(config, key):
            identity[key] = task_manifest(config, split="train" if key == "train_task_paths" else "eval")
    if config.vision_model_path:
        root = Path(config.vision_model_path)
        # Checkpoint metadata identifies the frozen service and catches local replacements.
        identity["vision_checkpoint"] = [
            [str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(root.rglob("*")) if p.is_file()]
    return identity
