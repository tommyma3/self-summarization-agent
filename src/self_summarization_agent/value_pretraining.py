"""Read-only rollout views and private MC critic initialization checkpoints."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from self_summarization_agent.checkpoints import checkpoint_id_from_path
from self_summarization_agent.collection_contract import collection_profile_id, validate_artifact_lineage
from self_summarization_agent.config import resolved_rollout_sampling_profile, sampling_profile_id
from self_summarization_agent.separate_value import (
    CRITIC_MANIFEST, CRITIC_STATE, atomic_json, digest, model_files_complete,
    sample_key, source_critic_identity, validate_samples,
)
from self_summarization_agent.token_stream import TITO_CONTRACT
from self_summarization_agent.value_model import VALUE_HEAD_FILENAME, VALUE_HEAD_MANIFEST_FILENAME

DATASET = "dataset_manifest.json"
STATE = "state.json"
MANIFEST = "pretraining_manifest.json"
RNG = "pretraining_rng.pt"


def validate_output_layout(config, actor, output, rollout_path=None):
    """Keep private checkpoint pruning disjoint from all initializer evidence."""
    output = Path(output).resolve()
    for source in (Path(actor).resolve(), Path(config.training.value.model_path).resolve()):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Pretraining output and model initialization directories must be disjoint")
    if (output / "epochs").is_symlink():
        raise ValueError("Pretraining epoch directory must not be a symlink")
    if rollout_path is not None:
        source = Path(rollout_path).resolve()
        if output / "epochs" in source.parents:
            raise ValueError("Pretraining source data cannot live in private epoch directories")


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def initialization_binding(config, actor):
    from self_summarization_agent.dataset import default_decrypted_path

    actor = Path(actor).resolve()
    dataset_path = Path(config.dataset.decrypted_path or default_decrypted_path(config.experiment.bc_plus_root))
    return {
        "version": 1, "actor_path": str(actor), "actor_id": checkpoint_id_from_path(actor),
        "actor_digest": source_critic_identity(actor, resume=False),
        "critic_path": str(Path(config.training.value.model_path).resolve()),
        "critic_digest": source_critic_identity(config.training.value.model_path, resume=False),
        "collection_profile_id": collection_profile_id(config, actor),
        "sampling_profile_id": sampling_profile_id(resolved_rollout_sampling_profile(config, split="train")),
        "pretraining": asdict(config.training.value_pretraining), "online_value_config": asdict(config.training.value),
        "runtime": asdict(config.runtime), "dataset": asdict(config.dataset),
        "dataset_digest": file_digest(dataset_path) if dataset_path.is_file() else None,
        "retrieval": asdict(config.retrieval), "judge": asdict(config.judge), "model_dtype": config.model.dtype,
    }


def prepare_dataset(config, actor, path, allowed_queries):
    """Validate whole question groups without modifying source rows."""
    from self_summarization_agent.train_step import _load_rollout_rows, samples_from_rollout_rows

    path, actor = Path(path).resolve(), Path(actor).resolve()
    settings = config.training.value_pretraining
    validate_artifact_lineage([path], config=config, checkpoint=actor)
    rows = _load_rollout_rows(path)
    if len(rows) != settings.task_count * settings.samples_per_task:
        raise ValueError("Pretraining requires exactly task_count * samples_per_task rollouts")
    profile = resolved_rollout_sampling_profile(config, split="train")
    profile_id, collection_id = sampling_profile_id(profile), collection_profile_id(config, actor)
    checkpoint_id = checkpoint_id_from_path(actor)
    queries, keys, samples = {}, set(), []
    for row in rows:
        query, index = row.get("query_id"), row.get("rollout_index")
        if (not isinstance(query, str) or query not in allowed_queries or row.get("query") != allowed_queries[query]
                or isinstance(index, bool) or not isinstance(index, int)
                or not 0 <= index < settings.samples_per_task or (query, index) in keys):
            raise ValueError("Invalid, duplicate, or non-training pretraining rollout")
        if (row.get("collection_contract") != TITO_CONTRACT
                or row.get("collection_profile_id") != collection_id
                or row.get("sampling_profile_id") != profile_id or row.get("sampling_profile") != profile
                or row.get("policy_checkpoint_id") != checkpoint_id
                or not isinstance(row.get("policy_checkpoint_path"), str)
                or Path(row["policy_checkpoint_path"]).resolve() != actor
                or row.get("rollout_split") != "train"
                or row.get("rollout_samples_per_task") != settings.samples_per_task):
            raise ValueError("Pretraining actor, collection, or sampling lineage mismatch")
        judge, rewards = row.get("judge"), row.get("turn_rewards")
        if not isinstance(judge, dict) or not isinstance(rewards, dict):
            raise ValueError("Pretraining requires judged rollout rewards")
        expected = 1.0 if judge.get("outcome") == "correct_answer" else -1.0
        if ((expected > 0 and row.get("status") != "completed")
                or any(isinstance(r, bool) or r != expected for r in rewards.values())):
            raise ValueError("Pretraining rewards disagree with the terminal judge decision")
        keys.add((query, index))
        queries.setdefault(query, set()).add(index)
        extracted = samples_from_rollout_rows([row], expected_checkpoint_id=checkpoint_id, retain_critic_only_states=True)
        validate_samples(extracted, config.training.max_sequence_length)
        samples.extend(extracted)
    if (len(queries) != settings.task_count
            or any(indices != set(range(settings.samples_per_task)) for indices in queries.values())):
        raise ValueError("Pretraining requires complete, distinct question rollout groups")
    ordered = sorted(queries)
    random.Random(settings.seed).shuffle(ordered)
    held_out = set(ordered[:settings.validation_task_count])
    train, validation = [s for s in samples if s.query_id not in held_out], [s for s in samples if s.query_id in held_out]
    if not train or not validation:
        raise ValueError("Pretraining requires nonempty eligible fitting and validation states")
    manifest = {
        "version": 1, "source": str(path), "source_digest": file_digest(path), "actor_id": checkpoint_id,
        "collection_profile_id": collection_id, "sampling_profile_id": profile_id, "seed": settings.seed,
        "requested_rollout_count": len(rows), "rollout_keys": [[q, i] for q, i in sorted(keys)],
        "train_query_ids": sorted(set(queries) - held_out), "validation_query_ids": sorted(held_out),
        "train_rollout_count": len({s.rollout_id for s in train}),
        "validation_rollout_count": len({s.rollout_id for s in validation}),
        "train_state_count": len(train), "validation_state_count": len(validation),
        "excluded_rollout_count": len(rows) - len({s.rollout_id for s in samples}),
    }
    manifest["digest"] = digest(manifest)
    return train, validation, manifest


def _read_dataset(output):
    body = json.loads((output / DATASET).read_text())
    if (body.get("digest") != digest({k: v for k, v in body.items() if k != "digest"})
            or body["source_digest"] != file_digest(body["source"])):
        raise ValueError("Pretraining dataset changed")
    return body


def _epoch_path(output, epoch):
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Invalid pretraining epoch")
    return output / "epochs" / f"epoch-{epoch:05d}"


def _validate_epoch(output, epoch, fingerprint, identity):
    path = _epoch_path(output, epoch)
    body = json.loads((path / "epoch.json").read_text())
    critic = json.loads((path / CRITIC_MANIFEST).read_text())
    if (body != {"epoch": epoch, "fingerprint": fingerprint, "critic_identity": identity}
            or critic.get("identity") != identity or not model_files_complete(path)
            or not all((path / name).is_file() for name in (
                VALUE_HEAD_FILENAME, VALUE_HEAD_MANIFEST_FILENAME, CRITIC_STATE, RNG,
                "tokenizer.json", "tokenizer_config.json"))):
        raise ValueError("Incomplete or mismatched pretraining epoch checkpoint")
    return path


def completed_pretraining(config, actor, output):
    """Validate a published initializer without allocating a model or CUDA context."""
    output = Path(output).resolve()
    if not (output / ".complete").is_file():
        return None
    manifest = json.loads((output / MANIFEST).read_text())
    dataset = _read_dataset(output)
    binding = initialization_binding(config, actor)
    fingerprint = digest({"binding": binding, "dataset": dataset["digest"]})
    marker = json.loads((output / ".complete").read_text())
    if (manifest.get("fingerprint") != fingerprint or manifest.get("binding") != binding
            or marker != {"manifest_digest": digest(manifest)}):
        raise ValueError("Completed pretraining belongs to different inputs or configuration")
    best = _validate_epoch(output, manifest["best_epoch"], fingerprint, manifest["best_identity"])
    if ((output / "best").resolve() != best
            or source_critic_identity(best, resume=False) != manifest["best_digest"]):
        raise ValueError("Selected pretrained critic changed")
    return output / "best"


def _rng_state(device):
    import torch
    return {"cpu": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _restore_rng(path, device):
    import torch
    state = torch.load(path / RNG, map_location="cpu", weights_only=True)
    torch.set_rng_state(state["cpu"])
    random.setstate(state["python"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _links_and_prune(output, state):
    # Only state.json commits an epoch. Links and private orphan checkpoints can
    # be repaired after a crash without ever resuming uncommitted weights.
    retained = {_epoch_path(output, state[key]) for key in ("epoch", "best_epoch")}
    for name, key in (("last", "epoch"), ("best", "best_epoch")):
        temporary = output / f".{name}.tmp"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(_epoch_path(output, state[key]).relative_to(output), target_is_directory=True)
        os.replace(temporary, output / name)
    for path in (output / "epochs").iterdir():
        if path not in retained and (path.name.startswith("epoch-") or path.name.startswith(".epoch-")):
            shutil.rmtree(path)
    metrics = output / ".metrics.tmp"
    metrics.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in state["history"]))
    os.replace(metrics, output / "metrics.jsonl")


def _commit_epoch(critic, output, state):
    import torch

    destination = _epoch_path(output, state["epoch"])
    temporary = destination.with_name(f".{destination.name}.incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    identity = critic.save(temporary, f"value-pretraining-{state['epoch']:05d}")
    torch.save(_rng_state(critic.device), temporary / RNG)
    atomic_json(temporary / "epoch.json", {
        "epoch": state["epoch"], "fingerprint": state["fingerprint"], "critic_identity": identity})
    if destination.exists():  # Only an orphan left before the state commit.
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    state["last_identity"] = identity
    if state["best_epoch"] == state["epoch"]:
        state["best_identity"] = identity
    atomic_json(output / STATE, state)
    _links_and_prune(output, state)


def fit_pretraining(config, actor, rollout_path, output, allowed_queries, *, resume=False,
                    device="cuda:0", trainer_factory=None):
    """Fit only the separate critic; return its selected initialization path."""
    import torch
    from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
    from self_summarization_agent.value_diagnostics import value_diagnostics
    from self_summarization_agent.trainer import _write_training_progress

    settings = config.training.value_pretraining
    if not settings.enabled or not config.training.value.separate or config.benchmark.name != "browsecomp":
        raise ValueError("Value pretraining requires an enabled separate BrowseComp critic")
    output, actor = Path(output).resolve(), Path(actor).resolve()
    validate_output_layout(config, actor, output, rollout_path)
    complete = completed_pretraining(config, actor, output)
    if complete is not None:
        return complete
    if (output / "epochs").is_symlink():
        raise ValueError("Pretraining epoch directory must not be a symlink")
    train, validation, dataset = prepare_dataset(config, actor, rollout_path, allowed_queries)
    output.mkdir(parents=True, exist_ok=True)
    if (output / DATASET).exists():
        if json.loads((output / DATASET).read_text()) != dataset:
            raise ValueError("Pretraining dataset/split changed; use a fresh output")
    else:
        if (output / "epochs").exists():
            raise ValueError("Refusing to reuse an unmanaged pretraining checkpoint directory")
        atomic_json(output / DATASET, dataset)
    binding = initialization_binding(config, actor)
    fingerprint = digest({"binding": binding, "dataset": dataset["digest"]})
    state = json.loads((output / STATE).read_text()) if (output / STATE).exists() else None
    if state is not None:
        if not resume:
            raise ValueError("Interrupted pretraining exists; use --resume")
        if state.get("fingerprint") != fingerprint:
            raise ValueError("Pretraining initialization or configuration changed")
        source = _validate_epoch(output, state["epoch"], fingerprint, state["last_identity"])
        _validate_epoch(output, state["best_epoch"], fingerprint, state["best_identity"])
        _links_and_prune(output, state)
    else:
        source = config.training.value.model_path
    value_config = replace(config.training.value, fitting_epochs=1, learning_rate=settings.learning_rate,
                           head_learning_rate=settings.head_learning_rate, eval_before_fit=False,
                           eval_every_n_epochs=0, seed=settings.seed)
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    critic = (trainer_factory or SeparateValueTrainer)(
        source, actor, value_config, dtype=config.model.dtype, device=device, resume=state is not None,
        activation_checkpointing=config.training.activation_checkpointing,
        progress_path=str(output / "progress.json"), advantage_estimator="compaction_mc_value")
    if state is not None:
        _restore_rng(Path(source), critic.device)
    train_rewards = {s.rollout_id: s.reward for s in train}
    prevalence = (sum(train_rewards.values()) / len(train_rewards) + 1) / 2
    validation_rewards = {s.rollout_id: s.reward for s in validation}
    constant_brier = sum((prevalence - (r + 1) / 2) ** 2 for r in validation_rewards.values()) / len(validation_rewards)

    def evaluate(epoch, train_metrics=None):
        values, losses = critic.evaluate(validation)
        metrics = value_diagnostics(validation, values, losses, prefix="validation")
        if not math.isfinite(metrics["validation/brier_score"]):
            raise ValueError("Nonfinite pretraining validation score")
        return {"epoch": epoch, "optimizer_steps": critic.update_count, "train_metrics": train_metrics,
                **metrics, "constant_success_probability": prevalence, "constant_brier_score": constant_brier}

    if state is None:
        record = evaluate(0)
        state = {"version": 1, "fingerprint": fingerprint, "epoch": 0, "best_epoch": 0,
                 "best_score": record["validation/brier_score"], "non_improving_epochs": 0,
                 "history": [record]}
        _commit_epoch(critic, output, state)
    targets = {sample_key(sample): sample.reward for sample in train}
    while state["epoch"] < settings.max_epochs and state["non_improving_epochs"] < settings.patience:
        epoch = state["epoch"] + 1
        _write_training_progress(str(output / "progress.json"), "value_pretraining_fit",
                                 epoch=epoch, max_epochs=settings.max_epochs)
        started = time.monotonic()
        _, train_metrics = critic.fit(train, max_sequence_length=config.training.max_sequence_length,
                                      scalar_targets=targets)
        record = evaluate(epoch, train_metrics)
        record["fit_and_validation_seconds"] = time.monotonic() - started
        improved = record["validation/brier_score"] < state["best_score"] - settings.min_delta
        state = {**state, "epoch": epoch, "history": [*state["history"], record],
                 "non_improving_epochs": 0 if improved else state["non_improving_epochs"] + 1}
        if improved:
            state.update(best_epoch=epoch, best_score=record["validation/brier_score"])
        _commit_epoch(critic, output, state)
        print(f"[value_pretraining] epoch={epoch}/{settings.max_epochs} "
              f"validation_brier={record['validation/brier_score']:.6f} best_epoch={state['best_epoch']}", flush=True)
    best = _validate_epoch(output, state["best_epoch"], fingerprint, state["best_identity"])
    timings_path = output / "phase_timings.jsonl"
    collection_seconds = sum(json.loads(line)["elapsed_seconds"] for line in timings_path.read_text().splitlines()) if timings_path.exists() else 0.
    manifest = {
        "version": 1, "fingerprint": fingerprint, "binding": binding, "dataset_digest": dataset["digest"],
        "target_kind": "terminal_mc", "online_optimizer": "fresh", "best_epoch": state["best_epoch"],
        "best_identity": state["best_identity"], "best_digest": source_critic_identity(best, resume=False),
        "best_validation_brier": state["best_score"], "constant_brier_score": constant_brier,
        "beats_constant": state["best_score"] < constant_brier, "completed_epochs": state["epoch"],
        "optimizer_steps": critic.update_count, "collection_seconds": collection_seconds,
        "fit_and_validation_seconds": sum(row.get("fit_and_validation_seconds", 0.) for row in state["history"]),
        "stop_reason": "patience" if state["non_improving_epochs"] >= settings.patience else "max_epochs",
    }
    atomic_json(output / MANIFEST, manifest)
    atomic_json(output / ".complete", {"manifest_digest": digest(manifest)})
    _write_training_progress(str(output / "progress.json"), "value_pretraining_complete",
                             best_epoch=state["best_epoch"], completed_epochs=state["epoch"])
    return output / "best"
