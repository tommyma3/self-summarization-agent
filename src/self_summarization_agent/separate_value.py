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

from self_summarization_agent.value_model import VALUE_HEAD_FILENAME, VALUE_HEAD_MANIFEST_FILENAME

CRITIC_DIR = "critic"
CRITIC_STATE = "training_state.pt"
CRITIC_MANIFEST = "critic_manifest.json"
PAIR_MANIFEST = "training_manifest.json"
ADVANTAGES = "fitted_advantages.json"


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


def load_advantages(path, samples, behavior_id, value_config):
    body = json.loads(Path(path).read_text())
    if (body.get("version") != 1 or body.get("behavior_checkpoint_id") != behavior_id
            or body.get("sample_fingerprint") != sample_fingerprint(samples)
            or body.get("config_fingerprint") != digest(asdict(value_config))):
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
        return bool(
            pair.get("version") == 1 and pair.get("architecture") == "separate"
            and pair.get("checkpoint_id") == logical_id
            and pair.get("critic_identity") == manifest.get("identity")
            and manifest.get("version") == 1 and manifest.get("checkpoint_id") == logical_id
            and manifest.get("class_order") == [-1, 1]
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
