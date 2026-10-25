"""Independent critic provenance and exact-token training views.

These helpers never mutate collection records or reconstruct token histories.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

from self_summarization_agent.value_model import (
    VALUE_HEAD_FILENAME, VALUE_HEAD_MANIFEST_FILENAME, value_support, value_head_manifest,
)

CRITIC_DIR = "critic"
CRITIC_STATE = "training_state.pt"
CRITIC_MANIFEST = "critic_manifest.json"
PAIR_MANIFEST = "training_manifest.json"
ADVANTAGES = "fitted_advantages.json"
FROZEN_TARGETS = "frozen_value_targets.json"


def method_spec(config, estimator="compaction_mc_value"):
    return {"estimator": estimator, "baseline_timing": config.baseline_timing,
            "num_bins": config.num_bins, "support": value_support(config.num_bins),
            "encoding": "two_hot", "gamma": 1.0,
            "lambda": config.gae_lambda if estimator == "compaction_gae_value" else 1.0}


def source_critic_identity(source, *, resume):
    source = Path(source)
    if resume:
        return json.loads((source / CRITIC_MANIFEST).read_text())["identity"]
    # Fresh initialization has no checkpoint UUID. Hash the actual initialization
    # files, not only its mutable directory name (also covers an existing head).
    files = sorted(p for p in source.rglob("*") if p.is_file() and
                   (p.suffix in {".json", ".safetensors", ".bin"}))
    if not files or not model_files_complete(source):
        raise ValueError("Incomplete critic initialization")
    result = hashlib.sha256()
    for path in files:
        result.update(str(path.relative_to(source)).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                result.update(chunk)
    return result.hexdigest()


def _legacy_config_fingerprints(config):
    current = asdict(config)
    candidates = {digest(current)}
    if config.num_bins == 2 and config.baseline_timing == "post_fit" and config.gae_lambda == 0.95:
        candidates.add(digest({k: v for k, v in current.items()
                               if k not in {"num_bins", "baseline_timing", "gae_lambda"}}))
    return candidates


def atomic_json(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sample_key(sample):
    return (str(sample.rollout_id), sample.turn_id)


def validate_samples(samples, max_sequence_length=None):
    seen, rewards = set(), {}
    for sample in samples:
        key = sample_key(sample)
        if sample.rollout_id is None or key in seen:
            raise ValueError("Separate critic requires unique rollout/interval keys")
        seen.add(key)
        if sample.reward not in (-1, 1) or rewards.setdefault(sample.rollout_id, sample.reward) != sample.reward:
            raise ValueError("Separate critic requires consistent binary rollout rewards")
        if not sample.has_exact_collection_tokens or not sample.has_token_reference_logprobs:
            raise ValueError("Separate critic requires exact collected IDs and cached token log probabilities")
        ids, labels, mask = sample.input_ids, sample.labels, sample.completion_mask
        if not ids or labels is None or mask is None or not (len(ids) == len(labels) == len(mask) == len(sample.reference_logprobs)):
            raise ValueError("Misaligned exact interval tensors")
        if ids != sample.collection_full_token_ids[:-1] or labels != sample.collection_full_token_ids[1:]:
            raise ValueError("Cached IDs differ from collected interval")
        if mask != sample.collection_assistant_token_mask[1:]:
            raise ValueError("Separate MC requires the exact all-assistant mask")
        if not all(math.isfinite(x) for x in sample.reference_logprobs):
            raise ValueError("Nonfinite behavior log probabilities")
        anchor = sample.state_prefix_length
        if isinstance(anchor, bool) or not isinstance(anchor, int) or not 1 <= anchor <= len(ids):
            raise ValueError("Invalid exact critic state anchor")
        if max_sequence_length is not None and len(ids) > max_sequence_length:
            raise ValueError("Over-length interval; prefixes are never truncated")


def sample_fingerprint(samples):
    return digest([{
        "key": sample_key(s), "ids": s.input_ids, "labels": s.labels,
        "mask": s.completion_mask, "anchor": s.state_prefix_length,
        "reward": s.reward, "old_logprobs": s.reference_logprobs,
    } for s in samples])


def validate_advantages(samples, advantages):
    keys = [sample_key(s) for s in samples]
    if len(set(keys)) != len(keys) or set(keys) != set(advantages):
        raise ValueError("Fixed advantages must cover exactly the unique interval keys")
    values = [float(advantages[k]) for k in keys]
    if any(not math.isfinite(x) or abs(x) > 2 for x in values):
        raise ValueError("MC advantages must be finite and in [-2, 2]")
    return values


def load_advantages(path, samples, behavior_id, value_config, *, estimator="compaction_mc_value",
                    chains=None, source_identity=None):
    body = json.loads(Path(path).read_text())
    if body.get("version") == 2:
        frozen = json.loads((Path(path).parent / FROZEN_TARGETS).read_text())
        validate_frozen_targets(frozen, samples, behavior_id, value_config, estimator,
                                chains=chains, source_identity=source_identity)
        if body.get("frozen_digest") != frozen["digest"] or body.get("rows") != frozen["rows"]:
            raise ValueError("Frozen advantage artifact mismatch")
        manifest = json.loads((Path(path).parent / CRITIC_DIR / CRITIC_MANIFEST).read_text())
        if (body.get("critic_identity") != manifest.get("identity")
                or manifest.get("frozen_digest") != frozen["digest"]
                or manifest.get("source_critic_identity") != source_identity
                or manifest.get("method") != frozen["method"]):
            raise ValueError("Frozen advantages belong to a different critic")
        return {tuple(row["key"]): row["advantage"] for row in frozen["rows"]}, body.get("metrics", {})
    if estimator != "compaction_mc_value" or value_config.baseline_timing != "post_fit":
        raise ValueError("Legacy fitted advantages cannot supply pre-fit/GAE training")
    if (body.get("version") != 1 or body.get("behavior_checkpoint_id") != behavior_id
            or body.get("sample_fingerprint") != sample_fingerprint(samples)
            or body.get("config_fingerprint") not in _legacy_config_fingerprints(value_config)):
        raise ValueError("Fitted advantage provenance mismatch")
    rows = body["rows"]
    result = {tuple(row["key"]): row["advantage"] for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate fitted advantage keys")
    validate_advantages(samples, result)
    rewards = {sample_key(s): s.reward for s in samples}
    for row in rows:
        value = row["value"]
        if not math.isfinite(value) or not -1 <= value <= 1 or abs(row["advantage"] - (rewards[tuple(row["key"])] - value)) > 1e-7:
            raise ValueError("Invalid fitted value/advantage relationship")
    critic = Path(path).parent / CRITIC_DIR
    manifest = json.loads((critic / CRITIC_MANIFEST).read_text())
    if body.get("critic_identity") != manifest.get("identity"):
        raise ValueError("Fitted advantages belong to a different critic")
    return result, body.get("metrics", {})


def frozen_target_artifact(samples, values, behavior_id, config, estimator, chains, source_identity):
    from self_summarization_agent.segment_returns import segment_gae
    if len(values) != len(samples):
        raise ValueError("Pre-fit values do not align with samples")
    old = {sample_key(s): float(v) for s, v in zip(samples, values)}
    if estimator == "compaction_gae_value":
        computed = segment_gae(chains, old, config.gae_lambda)
    else:
        computed = {sample_key(s): {"key": list(sample_key(s)), "value": old[sample_key(s)],
                    "target": s.reward, "advantage": s.reward - old[sample_key(s)]} for s in samples}
    body = {"version": 2, "behavior_checkpoint_id": behavior_id,
            "source_critic_identity": source_identity, "method": method_spec(config, estimator),
            "sample_fingerprint": sample_fingerprint(samples), "config_fingerprint": digest(asdict(config)),
            "chains": chains, "chain_fingerprint": digest(chains),
            "rows": [computed[sample_key(s)] for s in samples]}
    body["digest"] = digest(body)
    validate_frozen_targets(body, samples, behavior_id, config, estimator,
                            chains=chains, source_identity=source_identity)
    return body


def validate_frozen_targets(body, samples, behavior_id, config, estimator, *, chains, source_identity):
    from self_summarization_agent.segment_returns import segment_gae
    if (chains is None or source_identity is None or body.get("version") != 2
            or body.get("digest") != digest({k: v for k, v in body.items() if k != "digest"})
            or body.get("behavior_checkpoint_id") != behavior_id
            or body.get("source_critic_identity") != source_identity
            or body.get("sample_fingerprint") != sample_fingerprint(samples)
            or body.get("config_fingerprint") != digest(asdict(config))
            or body.get("method") != method_spec(config, estimator)
            or body.get("chains") != chains or body.get("chain_fingerprint") != digest(chains)):
        raise ValueError("Frozen target provenance mismatch")
    rows = body["rows"]
    keyed = {tuple(row["key"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate frozen target keys")
    validate_advantages(samples, {key: row["advantage"] for key, row in keyed.items()})
    old = {key: row["value"] for key, row in keyed.items()}
    if any(not math.isfinite(v) or not -1 <= v <= 1 for v in old.values()):
        raise ValueError("Invalid frozen old value")
    expected = (segment_gae(chains, old, config.gae_lambda) if estimator == "compaction_gae_value"
                else {sample_key(s): {"target": s.reward, "advantage": s.reward - old[sample_key(s)]}
                      for s in samples})
    for key, row in keyed.items():
        for field in ("target", "advantage"):
            if not math.isfinite(row[field]) or abs(row[field] - expected[key][field]) > 1e-7:
                raise ValueError("Invalid frozen return/advantage recurrence")
        if not -1 <= row["target"] <= 1:
            raise ValueError("Invalid frozen return range")
        if estimator == "compaction_gae_value" and row.get("transition_reward") != expected[key]["transition_reward"]:
            raise ValueError("Invalid frozen transition reward")


def model_files_complete(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    indexes = list(path.glob("*.index.json"))
    if indexes:
        return all(all((path / f).is_file() for f in json.loads(index.read_text())["weight_map"].values()) for index in indexes)
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()


def separate_checkpoint_complete(path, *, expected_id=None):
    path = Path(path)
    critic = path / CRITIC_DIR
    try:
        pair = json.loads((path / PAIR_MANIFEST).read_text())
        manifest = json.loads((critic / CRITIC_MANIFEST).read_text())
        logical_id = expected_id or path.name.removeprefix(".").removesuffix(".incomplete")
        version = manifest.get("version")
        if version == 2:
            method = manifest["method"]
            bins = method["num_bins"]
            head = json.loads((critic / VALUE_HEAD_MANIFEST_FILENAME).read_text())
            if (method["support"] != value_support(bins)
                    or head != value_head_manifest(head["hidden_size"], bins)
                    or pair.get("method") != method or pair.get("version") != 2):
                return False
            advantages = json.loads((path / ADVANTAGES).read_text())
            if (pair.get("advantages_digest") != digest(advantages)
                    or advantages.get("critic_identity") != manifest.get("identity")):
                return False
            if method["baseline_timing"] == "pre_fit":
                frozen = json.loads((path / FROZEN_TARGETS).read_text())
                if (advantages.get("version") != 2 or frozen.get("version") != 2
                        or frozen.get("digest") != digest({k: v for k, v in frozen.items() if k != "digest"})
                        or advantages.get("frozen_digest") != frozen["digest"]
                        or manifest.get("frozen_digest") != frozen["digest"]
                        or advantages.get("rows") != frozen.get("rows")
                        or frozen.get("method") != method
                        or frozen.get("source_critic_identity") != manifest.get("source_critic_identity")
                        or frozen.get("behavior_checkpoint_id") != pair.get("actor_source")):
                    return False
        return bool(
            pair.get("version") == version and pair.get("architecture") == "separate"
            and pair.get("checkpoint_id") == logical_id
            and pair.get("critic_identity") == manifest.get("identity")
            and version in {1, 2} and manifest.get("checkpoint_id") == logical_id
            and (version == 2 or manifest.get("class_order") == [-1, 1])
            and all((critic / f).is_file() for f in (
                CRITIC_STATE, VALUE_HEAD_FILENAME, VALUE_HEAD_MANIFEST_FILENAME,
                "tokenizer.json", "tokenizer_config.json",
            )) and model_files_complete(critic) and model_files_complete(path)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_resume_checkpoint(path):
    """Allow a fresh base model, but never silently reset an existing critic."""
    path = Path(path)
    requires_resume = (path / CRITIC_DIR).exists() or (path / PAIR_MANIFEST).exists() or (
        path.name.startswith("iteration-") and path.name != "iteration-00000")
    if requires_resume and not (
        (path / ".complete").is_file() and model_files_complete(path) and separate_checkpoint_complete(path)
    ):
        raise ValueError("Separate-critic resume requires a complete matching actor/critic checkpoint")
    return requires_resume
