"""Versioned compact frozen token baselines, bound to exact interval evidence."""

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from .separate_value import (
    ADVANTAGES,
    CRITIC_DIR,
    CRITIC_MANIFEST,
    FROZEN_TARGETS,
    atomic_json,
    digest,
    method_spec,
    sample_fingerprint,
    sample_key,
    source_critic_identity,
)
from .token_value import (
    TokenAdvantages,
    token_diagnostics,
    token_positions,
    validate_token_advantages,
    validate_token_samples,
)

TOKEN_SHARD_DIR = "token_values"


def shard_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def token_provenance(samples):
    return [
        dict(
            key=list(sample_key(s)),
            source=s.reference_logprob_source,
            mode=s.reference_logprobs_mode,
        )
        for s in samples
    ]


def token_fingerprint(samples):
    return digest(
        dict(
            evidence=sample_fingerprint(samples),
            logprob_provenance=token_provenance(samples),
        )
    )


def freeze_tokens(
    output, samples, values, behavior, config, source_identity, tokenizer_signature
):
    validate_token_samples(samples)
    if len(values) != len(samples):
        raise ValueError("Token baseline interval coverage mismatch")
    positions = []
    flat = []
    rewards = []
    rows = []
    for s, vs in zip(samples, values):
        ps = token_positions(s)
        if len(ps) != len(vs):
            raise ValueError("Token baseline position coverage mismatch")
        rows.append(
            dict(
                key=list(sample_key(s)),
                offset=len(flat),
                count=len(ps),
            )
        )
        positions.extend(ps)
        flat.extend(vs)
        rewards.extend([s.reward] * len(ps))
    v = torch.tensor(flat, dtype=torch.float32)
    tensors = dict(
        positions=torch.tensor(positions, dtype=torch.int64),
        values=v,
        advantages=torch.tensor(rewards, dtype=torch.float32) - v,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shard_directory = output / TOKEN_SHARD_DIR
    shard_directory.mkdir(parents=True, exist_ok=True)
    temporary = shard_directory / f".token-values.{os.getpid()}.tmp"
    save_file(tensors, str(temporary))
    checksum = shard_digest(temporary)
    filename = f"{TOKEN_SHARD_DIR}/token-values-{checksum}.safetensors"
    os.replace(temporary, output / filename)
    body = dict(
        version=3,
        behavior_checkpoint_id=behavior,
        source_critic_identity=source_identity,
        tokenizer_signature=tokenizer_signature,
        method=method_spec(config, "token_mc_value"),
        sample_fingerprint=token_fingerprint(samples),
        behavior_logprob_provenance=[
            dict(
                key=list(sample_key(s)),
                source=s.reference_logprob_source,
                mode=s.reference_logprobs_mode,
            )
            for s in samples
        ],
        config_fingerprint=digest(asdict(config)),
        rows=rows,
        rollout_rewards={str(s.rollout_id): s.reward for s in samples},
        shard=dict(
            file=filename,
            sha256=checksum,
            shape=[len(flat)],
            dtypes=dict(positions="int64", values="float32", advantages="float32"),
        ),
    )
    body["digest"] = digest(body)
    validate_token_frozen(
        body, output, samples, behavior, config, source_identity, tokenizer_signature
    )
    atomic_json(output / FROZEN_TARGETS, body)
    return body


def read_shard(body, output):
    shard = body["shard"]
    if (
        body.get("version") != 3
        or body.get("digest")
        != digest({k: v for k, v in body.items() if k != "digest"})
        or Path(shard["file"]).parent != Path(TOKEN_SHARD_DIR)
        or shard["dtypes"]
        != dict(positions="int64", values="float32", advantages="float32")
    ):
        raise ValueError("Invalid frozen token manifest")
    path = Path(output) / shard["file"]
    if shard_digest(path) != shard["sha256"]:
        raise ValueError("Corrupt frozen token shard")
    try:
        tensors = load_file(str(path))
    except SafetensorError as error:
        raise ValueError("Invalid frozen token shard encoding") from error
    if set(tensors) != {"positions", "values", "advantages"}:
        raise ValueError("Invalid token shard tensors")
    for name, tensor in tensors.items():
        dtype = torch.int64 if name == "positions" else torch.float32
        if list(tensor.shape) != shard["shape"] or tensor.dtype != dtype:
            raise ValueError("Invalid token shard shape/dtype")
    v, a = tensors["values"], tensors["advantages"]
    if (
        not torch.isfinite(v).all()
        or not torch.isfinite(a).all()
        or (v.abs() > 1).any()
        or (a.abs() > 2).any()
    ):
        raise ValueError("Invalid frozen token ranges")
    expected_rollouts = {str(row["key"][0]) for row in body["rows"]}
    if set(body["rollout_rewards"]) != expected_rollouts:
        raise ValueError(
            "Frozen token rewards must cover exactly the contributing rollout evidence"
        )
    offset = 0
    seen = set()
    for row in body["rows"]:
        key = tuple(row["key"])
        count = row["count"]
        reward = body["rollout_rewards"][str(key[0])]
        if (
            key in seen
            or type(count) is not int
            or count < 0
            or row["offset"] != offset
            or isinstance(reward, bool)
            or reward not in (-1, 1)
        ):
            raise ValueError("Invalid token interval membership")
        seen.add(key)
        stop = offset + count
        if stop > len(v) or not torch.allclose(
            a[offset:stop], reward - v[offset:stop], atol=2e-7, rtol=0
        ):
            raise ValueError("Invalid frozen token advantage relationship")
        ps = tensors["positions"][offset:stop].tolist()
        if ps != sorted(set(ps)) or any(p < 0 for p in ps):
            raise ValueError("Invalid ordered token positions")
        offset = stop
    if offset != len(v):
        raise ValueError("Extra frozen token values")
    return tensors


def validate_token_frozen(
    body, output, samples, behavior, config, source_identity, tokenizer_signature=None
):
    validate_token_samples(samples)
    if (
        body.get("behavior_checkpoint_id") != behavior
        or body.get("source_critic_identity") != source_identity
        or not source_identity
        or body.get("method") != method_spec(config, "token_mc_value")
        or body.get("sample_fingerprint") != token_fingerprint(samples)
        or body.get("behavior_logprob_provenance") != token_provenance(samples)
        or body.get("config_fingerprint") != digest(asdict(config))
        or not body.get("tokenizer_signature")
        or (
            tokenizer_signature is not None
            and body["tokenizer_signature"] != tokenizer_signature
        )
        or [r["key"] for r in body["rows"]] != [list(sample_key(s)) for s in samples]
    ):
        raise ValueError("Frozen token provenance mismatch")
    tensors = read_shard(body, output)
    result = {}
    for s, row in zip(samples, body["rows"]):
        start = row["offset"]
        stop = start + row["count"]
        ps = tuple(tensors["positions"][start:stop].tolist())
        if (
            ps != token_positions(s)
            or body["rollout_rewards"][str(s.rollout_id)] != s.reward
        ):
            raise ValueError("Frozen token coverage mismatch")
        result[sample_key(s)] = TokenAdvantages(
            ps, tuple(tensors["advantages"][start:stop].tolist())
        )
    validate_token_advantages(samples, result)
    return result


def verify_token_pair(output, pair, manifest, advantages):
    frozen = json.loads((Path(output) / FROZEN_TARGETS).read_text())
    read_shard(frozen, output)
    return (
        advantages.get("version") == 3
        and advantages.get("frozen_digest") == frozen["digest"]
        and manifest.get("frozen_digest") == frozen["digest"]
        and frozen["method"] == manifest["method"]
        and frozen["source_critic_identity"] == manifest["source_critic_identity"]
        and frozen["behavior_checkpoint_id"] == pair["actor_source"]
        and frozen["tokenizer_signature"] == manifest["tokenizer_signature"]
    )


def load_token_advantages(path, samples, behavior, config, source_identity):
    output = Path(path).parent
    body = json.loads(Path(path).read_text())
    frozen = json.loads((output / FROZEN_TARGETS).read_text())
    manifest = json.loads((output / CRITIC_DIR / CRITIC_MANIFEST).read_text())
    pair = dict(actor_source=behavior)
    if body.get("critic_identity") != manifest["identity"] or not verify_token_pair(
        output, pair, manifest, body
    ):
        raise ValueError("Token advantages belong to a different fitted/source critic")
    result = validate_token_frozen(
        frozen,
        output,
        samples,
        behavior,
        config,
        source_identity,
        manifest["tokenizer_signature"],
    )
    return result, body.get("metrics", {})


def run_token_critic(
    config,
    checkpoint,
    output,
    output_id,
    samples,
    source,
    resume,
    critic,
    rollouts,
    phases=None,
):
    started = time.monotonic()
    timings = dict(phases or {})
    source_identity = source_critic_identity(source, resume=resume)
    frozen_path = output / FROZEN_TARGETS
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        validate_token_frozen(
            frozen,
            output,
            samples,
            checkpoint.name,
            config.training.value,
            source_identity,
            critic.tokenizer_hash,
        )
        prefit = {}
    else:
        old, losses = critic.evaluate(samples)
        prefit = token_diagnostics(
            samples,
            old,
            losses,
            prefix="value/prefit",
            detailed=critic.detailed_diagnostics,
        )
        frozen = freeze_tokens(
            output,
            samples,
            old,
            checkpoint.name,
            config.training.value,
            source_identity,
            critic.tokenizer_hash,
        )
    timings["value/frozen_target_seconds"] = time.monotonic() - started
    critic.frozen_digest = frozen["digest"]
    critic.source_critic_identity = source_identity
    _, metrics = critic.fit(
        samples, max_sequence_length=config.training.max_sequence_length
    )
    metrics.update(prefit)
    save_started = time.monotonic()
    identity = critic.save(output / CRITIC_DIR, output_id)
    timings["value/checkpoint_save_seconds"] = time.monotonic() - save_started
    from .critic_step import evaluate_on_eval_collection, scheduled_iteration

    if scheduled_iteration(config.training.value.evalset_interval, output_id):
        metrics.update(evaluate_on_eval_collection(critic, rollouts, checkpoint.name))
    else:
        metrics["value/evalset_skipped"] = (
            "disabled"
            if config.training.value.evalset_interval == 0
            else "not_scheduled"
        )
    advantages = read_shard(frozen, output)["advantages"]
    if advantages.numel():
        metrics.update(
            {
                "value/advantage_mean": float(advantages.mean()),
                "value/advantage_std": float(advantages.std(unbiased=False)),
                "value/advantage_min": float(advantages.min()),
                "value/advantage_max": float(advantages.max()),
                "value/advantage_near_zero_fraction": float(
                    (advantages.abs() < 1e-6).float().mean()
                ),
            }
        )
    timings["value/critic_total_seconds"] = (
        time.monotonic()
        - started
        + sum(
            timings.get(key, 0.0)
            for key in ("value/model_load_seconds", "value/sample_load_seconds")
        )
    )
    metrics.update(timings)
    atomic_json(output / "critic_phase_timings.json", timings)
    atomic_json(
        output / ADVANTAGES,
        dict(
            version=3,
            critic_identity=identity,
            frozen_digest=frozen["digest"],
            metrics=metrics,
        ),
    )
