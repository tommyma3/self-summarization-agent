"""Frozen critic diagnostics; no token reconstruction or optimizer side effects."""
from __future__ import annotations

import math
from typing import Any, Sequence

from self_summarization_agent.value_model import normalized_value_evaluation_loss, rollout_value_weights


def prediction_metrics(
    rewards: Sequence[float],
    values: Sequence[float],
    losses: Sequence[float],
    rollout_ids: Sequence[str],
    *, detailed: bool = True,
) -> dict[str, Any]:
    """Normalize once over real states, with equal total weight per rollout."""
    size = len(rewards)
    if not size or not (len(values) == len(losses) == len(rollout_ids) == size):
        raise ValueError("Critic diagnostics require nonempty, aligned predictions and labels")
    if any(r not in (-1, 1) for r in rewards):
        raise ValueError("Critic diagnostics require binary -1/+1 rewards")
    if any(not math.isfinite(v) or not -1 <= v <= 1 for v in values):
        raise ValueError("Critic predictions must be finite expected rewards in [-1, 1]")
    if any(not math.isfinite(loss) or loss < 0 for loss in losses):
        raise ValueError("Critic losses must be finite and nonnegative")
    weights = rollout_value_weights(list(rollout_ids))
    weight_sum = math.fsum(weights)
    weights = [w / weight_sum for w in weights]

    def mean(xs):
        return math.fsum(w * x for w, x in zip(weights, xs))

    def variance(xs):
        avg = mean(xs)
        return mean([(x - avg) ** 2 for x in xs])

    probabilities = [(v + 1) / 2 for v in values]
    targets = [(r + 1) / 2 for r in rewards]
    residuals = [r - v for r, v in zip(rewards, values)]
    reward_variance = variance(rewards)
    bins = []
    for index in range(10 if detailed else 0):
        selected = [i for i, p in enumerate(probabilities) if min(9, int(p * 10)) == index]
        mass = math.fsum(weights[i] for i in selected)
        bins.append({
            "lower": index / 10, "upper": (index + 1) / 10,
            "count": len(selected), "weight_mass": mass,
            "predicted_success": math.fsum(weights[i] * probabilities[i] for i in selected) / mass if mass else None,
            "observed_success": math.fsum(weights[i] * targets[i] for i in selected) / mass if mass else None,
        })
    result = {
        "state_count": size,
        "rollout_count": len(set(rollout_ids)),
        "weight_sum": weight_sum,
        "bce": normalized_value_evaluation_loss(list(losses), weights),
        "brier_score": mean([(p - y) ** 2 for p, y in zip(probabilities, targets)]),
        "explained_variance": 1 - variance(residuals) / reward_variance if reward_variance > 0 else None,
        "mean_reward": mean(rewards),
        "mean_value": mean(values),
        "value_std": math.sqrt(variance(values)),
        "value_min": min(values),
        "value_max": max(values),
        "mean_success_probability": mean(probabilities),
        "classification_accuracy": mean([float((v >= 0) == (r > 0)) for v, r in zip(values, rewards)]),
    }
    if detailed:
        result.update({
            "calibration_error": math.fsum(
                b["weight_mass"] * abs(b["predicted_success"] - b["observed_success"])
                for b in bins if b["weight_mass"]
            ),
            "calibration_bins": bins,
        })
    return result


def value_diagnostics(samples, values, losses, *, prefix: str, detailed: bool = True) -> dict[str, Any]:
    """Include metadata-derived state strata; absent legacy metadata is unknown.

    Each stratum recomputes rollout weights among its own real eligible states.
    Training-batch evaluations must not be labeled held-out validation.
    """
    if len(samples) != len(values) or len(samples) != len(losses):
        raise ValueError("Critic predictions do not align with samples")
    if any(sample.rollout_id is None for sample in samples):
        raise ValueError("Critic diagnostics require rollout identities")
    groups = {"": list(range(len(samples)))}
    for kind in (("initial", "post_compaction", "unknown") if detailed else ()):
        indices = [i for i, s in enumerate(samples) if getattr(s, "state_kind", "unknown") == kind]
        if indices:
            groups[kind + "/"] = indices
    result = {}
    for group, indices in groups.items():
        metrics = prediction_metrics(
            [samples[i].reward for i in indices], [values[i] for i in indices],
            [losses[i] for i in indices], [str(samples[i].rollout_id) for i in indices],
            detailed=detailed,
        )
        result.update({f"{prefix}/{group}{key}": value for key, value in metrics.items()})
    return result
