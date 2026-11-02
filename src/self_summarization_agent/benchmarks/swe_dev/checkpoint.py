"""SWE training identity independent of the changing behavior checkpoint path."""
from dataclasses import asdict
import json
from pathlib import Path

from .dataset import benchmark_identity
from ...chat_template import load_chat_template
from ...config import resolved_rollout_sampling_profile
from ...separate_value import digest, method_spec, atomic_json

CONTRACT_FILE = "swe_training_contract.json"


def training_contract(config):
    return dict(version=1, benchmark_digest=digest(benchmark_identity(config.benchmark)),
        reward_profile=config.benchmark.reward_profile,
        method=method_spec(config.training.value, config.training.advantage_estimator),
        value_enabled=config.training.value.enabled, value_architecture=config.training.value.architecture,
        group_size=config.training.group_size, runtime=asdict(config.runtime),
        train_sampling=resolved_rollout_sampling_profile(config, split="train"),
        eval_sampling=resolved_rollout_sampling_profile(config, split="eval"),
        template=load_chat_template(config.model.chat_template_path),
        selection=dict(train_count=config.collection.train_task_count,
                       eval_count=config.collection.eval_task_count, dataset=asdict(config.dataset)),
        backend=config.training.backend, worker_backend=config.training.verl.worker_backend,
        max_sequence_length=config.training.max_sequence_length)


def validate_checkpoint(path, config):
    path = Path(path)
    artifact = path / CONTRACT_FILE
    if artifact.is_file():
        if json.loads(artifact.read_text()) != training_contract(config):
            raise ValueError("SWE checkpoint reward/method/task/runtime contract changed")
        native = config.training.backend == "verl_ray" and config.training.verl.worker_backend == "verl_fsdp"
        if native:
            marker = path / "swe_actor_optimizer.json"
            if not marker.is_file():
                raise ValueError("SWE checkpoint missing native actor optimizer identity")
            state = json.loads(marker.read_text())
            world = state.get("world_size")
            if (state.get("version") != 1 or state.get("reward_profile") != "binary_zero_one"
                    or type(world) is not int or world < 1):
                raise ValueError("Invalid SWE actor optimizer identity")
            if not all((path / f"{kind}_world_size_{world}_rank_{rank}.pt").is_file()
                       for rank in range(world) for kind in ("model", "optim", "extra_state")):
                raise ValueError("SWE checkpoint missing native optimizer/model/RNG shards")
        elif not (path / "actor_training_state.pt").is_file():
            raise ValueError("SWE checkpoint missing persistent actor optimizer state")
    elif (path / ".complete").exists() and path.name != "iteration-00000":
        raise ValueError("SWE resume requires its training contract; use a fresh backbone initialization")


def save_contract(path, config):
    atomic_json(Path(path) / CONTRACT_FILE, training_contract(config))
