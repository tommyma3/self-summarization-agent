from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


VALUE_HEAD_SUBDIR = "compaction_value"
VALUE_HEAD_FILENAME = f"{VALUE_HEAD_SUBDIR}/compaction_value_head.safetensors"
VALUE_HEAD_MANIFEST_FILENAME = f"{VALUE_HEAD_SUBDIR}/compaction_value_config.json"
VALUE_HEAD_VERSION = 1


def reward_class_indices(rewards: torch.Tensor) -> torch.Tensor:
    if not bool(torch.all((rewards == -1) | (rewards == 1))):
        raise ValueError("Compaction Monte Carlo value targets must be exactly -1 or +1")
    return (rewards > 0).to(torch.long)


def expected_binary_value(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities[..., 1] - probabilities[..., 0]


def value_support(num_bins: int) -> list[float]:
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("Value support requires at least two bins")
    return [-1.0 + 2.0 * k / (num_bins - 1) for k in range(num_bins)]


def expected_categorical_value(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 2:
        return expected_binary_value(logits)
    support = logits.new_tensor(value_support(logits.shape[-1]), dtype=torch.float32)
    return (logits.float().softmax(-1) * support).sum(-1)


def two_hot_targets(targets: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Detached interpolation on [-1, 1]; only roundoff-sized overflow is clamped."""
    value_support(num_bins)
    targets = targets.detach().float()
    if not bool(torch.isfinite(targets).all()) or bool((targets.abs() > 1 + 1e-6).any()):
        raise ValueError("Value targets must be finite and in [-1, 1]")
    positions = ((targets.clamp(-1, 1) + 1) * ((num_bins - 1) / 2)).clamp(0, num_bins - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp(max=num_bins - 1)
    fraction = positions - lower
    result = targets.new_zeros((*targets.shape, num_bins))
    result.scatter_add_(-1, lower.unsqueeze(-1), (1 - fraction).unsqueeze(-1))
    result.scatter_add_(-1, upper.unsqueeze(-1), fraction.unsqueeze(-1))
    return result


def outcome_log_losses(logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Binary outcome log loss implied by the categorical mean, computed stably."""
    support = logits.new_tensor(value_support(logits.shape[-1]), dtype=torch.float32)
    success = (support + 1) / 2
    log_probs = logits.float().log_softmax(-1)
    positive = torch.logsumexp(log_probs + success.log(), dim=-1)
    negative = torch.logsumexp(log_probs + (1 - success).log(), dim=-1)
    return -torch.where(rewards > 0, positive, negative)


def value_head_manifest(hidden_size: int, num_bins: int) -> dict:
    if num_bins == 2:
        return {"version": 1, "hidden_size": hidden_size, "class_order": [-1, 1]}
    return {"version": 2, "hidden_size": hidden_size, "num_bins": num_bins,
            "support": value_support(num_bins), "target_encoding": "two_hot"}


def rollout_value_weights(rollout_ids: list[str]) -> list[float]:
    if not rollout_ids:
        return []
    counts: dict[str, int] = {}
    for rollout_id in rollout_ids:
        counts[rollout_id] = counts.get(rollout_id, 0) + 1
    rollout_count = len(counts)
    return [1.0 / (rollout_count * counts[rollout_id]) for rollout_id in rollout_ids]


def normalized_value_evaluation_loss(
    per_state_losses: list[float],
    rollout_weights: list[float],
) -> float:
    """Frozen-model CE, normalized once over all real collection states.

    Pass cross-entropies computed directly from logits, not reconstructed from
    expected rewards (which can round to +/-1 for confident predictions).
    """
    if not per_state_losses or len(per_state_losses) != len(rollout_weights):
        raise ValueError("Frozen value losses and rollout weights must be nonempty and aligned")
    if any(not math.isfinite(x) or x < 0 for x in [*per_state_losses, *rollout_weights]):
        raise ValueError("Frozen value losses and weights must be finite and nonnegative")
    weight_sum = math.fsum(rollout_weights)
    if weight_sum <= 0:
        raise ValueError("Frozen value evaluation requires positive total weight")
    return math.fsum(loss * weight for loss, weight in zip(per_state_losses, rollout_weights)) / weight_sum


def rollout_normalized_value_loss(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    rollout_weights: torch.Tensor,
) -> torch.Tensor:
    losses = torch.nn.functional.cross_entropy(
        logits.float(), reward_class_indices(rewards), reduction="none"
    )
    weights = rollout_weights.to(device=losses.device, dtype=losses.dtype)
    if losses.shape != weights.shape:
        raise ValueError("Value losses and rollout weights must have the same shape")
    return (losses * weights).sum()


class CompactionValueHead(torch.nn.Module):
    def __init__(self, hidden_size: int, *, zero_initialize: bool = True, num_bins: int = 2) -> None:
        super().__init__()
        value_support(num_bins)
        self.projection = torch.nn.Linear(hidden_size, num_bins)
        if zero_initialize:
            torch.nn.init.zeros_(self.projection.weight)
            torch.nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


def model_hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    raise ValueError("Cannot determine the policy model hidden size")


def capture_lm_head_input(model: Any):
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The policy model does not expose output embeddings")
    captured: list[torch.Tensor] = []

    def hook(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("The language-model head did not receive hidden states")
        captured.append(args[0])

    return captured, output_embeddings.register_forward_pre_hook(hook)


def state_hidden_states(hidden_states: torch.Tensor, state_prefix_lengths: torch.Tensor) -> torch.Tensor:
    indices = state_prefix_lengths.to(device=hidden_states.device, dtype=torch.long) - 1
    if hidden_states.ndim != 3 or bool(torch.any(indices < 0)) or bool(
        torch.any(indices >= hidden_states.shape[1])
    ):
        raise ValueError("A value state prefix is outside the encoded interval")
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, indices]


def save_value_head(head: CompactionValueHead, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    weights_path = path / VALUE_HEAD_FILENAME
    manifest_path = path / VALUE_HEAD_MANIFEST_FILENAME
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in head.state_dict().items()},
        str(weights_path),
    )
    manifest_path.write_text(
        json.dumps(
            value_head_manifest(head.projection.in_features, head.projection.out_features),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_value_head(
    checkpoint_path: str | Path,
    *,
    hidden_size: int,
    zero_initialize: bool,
    num_bins: int = 2,
) -> tuple[CompactionValueHead, bool]:
    path = Path(checkpoint_path)
    weights_path = path / VALUE_HEAD_FILENAME
    manifest_path = path / VALUE_HEAD_MANIFEST_FILENAME

    # Backward compatibility: older checkpoints stored the sidecar at the top
    # level, where vLLM's weight loader would treat it as a model shard.
    legacy_weights_path = path / "compaction_value_head.safetensors"
    legacy_manifest_path = path / "compaction_value_config.json"
    if not weights_path.exists() and legacy_weights_path.exists():
        weights_path = legacy_weights_path
    if not manifest_path.exists() and legacy_manifest_path.exists():
        manifest_path = legacy_manifest_path

    head = CompactionValueHead(hidden_size, zero_initialize=zero_initialize, num_bins=num_bins)
    if not weights_path.exists() and not manifest_path.exists():
        return head, False
    if not weights_path.exists() or not manifest_path.exists():
        raise ValueError(f"Incomplete compaction value-head checkpoint in {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != value_head_manifest(hidden_size, num_bins):
        raise ValueError("Compaction value-head manifest does not match the policy model")
    head.load_state_dict(load_file(str(weights_path)))
    return head, True


def migrate_compaction_value_head_sidecar(checkpoint_path: str | Path) -> bool:
    """Move legacy top-level value-head sidecars into a vLLM-ignored subdirectory.

    vLLM's checkpoint loader globs ``*.safetensors`` in the model directory.  The
    value head is not part of the policy model, so it must live in a
    subdirectory.  This helper is idempotent: it renames legacy sidecars when it
    first sees them and removes any leftover top-level copies once the new
    layout is in place.
    """
    path = Path(checkpoint_path)
    old_weights = path / "compaction_value_head.safetensors"
    old_manifest = path / "compaction_value_config.json"
    new_weights = path / VALUE_HEAD_FILENAME
    new_manifest = path / VALUE_HEAD_MANIFEST_FILENAME
    migrated = False

    def _relocate(old: Path, new: Path) -> bool:
        if not old.exists():
            return False
        if new.exists():
            old.unlink()
            return True
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        return True

    if _relocate(old_weights, new_weights):
        migrated = True
    if _relocate(old_manifest, new_manifest):
        migrated = True
    return migrated
