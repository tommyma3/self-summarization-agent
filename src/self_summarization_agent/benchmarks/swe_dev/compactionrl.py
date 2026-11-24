"""Resolve the historical CompactionRL Flash manifest against prepared Harbor tasks."""
from hashlib import sha256
import json
from pathlib import Path

from harbor.models.task.paths import TaskPaths
from harbor.models.task.task import Task

REVISION = "c0727751da3d1edfdfb5cc55fba5c6e30f67ce00"
MANIFEST_SHA256 = "0ec26e0f6e7be16a476dfdc7a7dff248c2f428ff65d6fcf2cd969ca7ca929f90"


def training_tasks(config):
    if config.dataset_revision != REVISION:
        raise ValueError(f"CompactionRL reference manifest requires dataset_revision {REVISION}")
    manifest = Path(config.train_manifest_path).read_bytes()
    if sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError("CompactionRL training manifest differs from the pinned 1,105-ID release")
    instance_ids = json.loads(manifest)
    root = Path(config.prepared_task_root).resolve()
    missing = [name for name in instance_ids if not (root / name).is_dir()]
    if missing:
        raise ValueError(f"Missing {len(missing)} of {len(instance_ids)} CompactionRL task directories "
                         f"in {root} (including {', '.join(missing[:3])}). "
                         "Set benchmark.prepared_task_root to the prepared SWE-Dev Harbor dataset.")

    tasks = []
    for name in instance_ids:
        directory = root / name
        if not TaskPaths(directory).is_valid():
            raise ValueError(f"Incomplete CompactionRL Harbor task: {directory}")
        task = Task(directory)
        tasks.append(dict(name=name, path=str(task.task_dir), checksum=task.checksum,
                          manifest_digest=MANIFEST_SHA256, fixture_only=False,
                          source=dict(origin="compactionrl-swe-dev", instance_id=name),
                          image=task.config.environment.docker_image,
                          verifier_mode="harbor-task"))
    return tasks
