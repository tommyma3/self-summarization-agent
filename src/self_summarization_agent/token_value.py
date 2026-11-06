"""Training-only token views. Authoritative interval evidence is never modified."""

import math
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

from .separate_value import sample_key, validate_samples
from .trajectory import (
    REFERENCE_LOGPROB_SOURCE_POLICY_RESCORE,
    REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT,
)


def token_positions(sample):
    return tuple(i for i, enabled in enumerate(sample.completion_mask) if enabled)


def validate_token_samples(samples, max_sequence_length=None):
    validate_samples(samples, max_sequence_length)
    for s in samples:
        if s.reference_logprob_source not in {
            REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT,
            REFERENCE_LOGPROB_SOURCE_POLICY_RESCORE,
        }:
            raise ValueError("Token MC requires behavior-logprob provenance")
        if (
            s.reference_logprob_source == REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT
            and s.reference_logprobs_mode != "raw_logprobs"
        ):
            raise ValueError("Token MC requires raw rollout log probabilities")
        if (
            len(s.collection_assistant_token_mask) != len(s.collection_full_token_ids)
            or s.collection_assistant_token_mask[0]
            or any(type(m) is not bool for m in s.collection_assistant_token_mask)
            or any(type(t) is not int or t < 0 for t in s.collection_full_token_ids)
            or isinstance(s.reward, bool)
            or (
                token_positions(s)
                and token_positions(s)[0] != s.state_prefix_length - 1
            )
            or s.input_ids[1:] != s.labels[:-1]
        ):
            raise ValueError("Invalid exact token ledger alignment or mask")


@dataclass(frozen=True)
class TokenAdvantages:
    positions: tuple[int, ...]
    values: tuple[float, ...]

    def dense(self, sample):
        if self.positions != token_positions(sample) or len(self.values) != len(
            self.positions
        ):
            raise ValueError(
                "Token advantages must cover exactly the ordered sampled positions"
            )
        if any(not math.isfinite(v) or abs(v) > 2 for v in self.values):
            raise ValueError("Token advantages must be finite and in [-2, 2]")
        result = [0.0] * len(sample.labels)
        for p, v in zip(self.positions, self.values):
            result[p] = v
        return result


def validate_token_advantages(samples, advantages):
    validate_token_samples(samples)
    keys = [sample_key(s) for s in samples]
    if len(set(keys)) != len(keys) or set(advantages) != set(keys):
        raise ValueError("Token advantages require complete unique interval keys")
    result = []
    for s in samples:
        a = advantages[sample_key(s)]
        if not isinstance(a, TokenAdvantages):
            raise ValueError("Token mode requires explicit TokenAdvantages")
        a.dense(s)
        result.append(a)
    return result


def advantage_mean(advantages):
    values = [
        v
        for a in advantages
        for v in (a.values if isinstance(a, TokenAdvantages) else (a,))
    ]
    return math.fsum(values) / max(1, len(values))


def token_weights(samples):
    lengths = [sum(s.completion_mask) for s in samples]
    counts = Counter(s.rollout_id for s, n in zip(samples, lengths) if n)
    return [
        [1 / (len(counts) * counts[s.rollout_id] * n)] * n if n else []
        for s, n in zip(samples, lengths)
    ]


def token_diagnostics(
    samples, values, losses, *, prefix, detailed=True, _with_strata=True
):
    if len(samples) != len(values) or len(samples) != len(losses):
        raise ValueError("Token predictions do not align with intervals")
    weights = token_weights(samples)
    rs, vs, ls, ws = [], [], [], []
    for s, v, loss, weight in zip(samples, values, losses, weights):
        if not len(v) == len(loss) == len(weight):
            raise ValueError("Token predictions do not cover sampled positions")
        rs.extend([s.reward] * len(v))
        vs.extend(v)
        ls.extend(loss)
        ws.extend(weight)
    if any(not math.isfinite(v) or abs(v) > 1 for v in vs) or any(
        not math.isfinite(token_loss) or token_loss < 0 for token_loss in ls
    ):
        raise ValueError("Invalid token value/loss")

    def mean(xs):
        return math.fsum(w * x for w, x in zip(ws, xs))

    def var(xs):
        m = mean(xs)
        return mean([(x - m) ** 2 for x in xs])

    residuals = [r - v for r, v in zip(rs, vs)]
    probs = [(v + 1) / 2 for v in vs]
    targets = [(r + 1) / 2 for r in rs]
    result = dict(
        token_count=len(vs),
        state_count=sum(bool(v) for v in values),
        empty_interval_count=sum(not v for v in values),
        rollout_count=len({s.rollout_id for s, v in zip(samples, values) if v}),
        weight_sum=math.fsum(ws),
        bce=mean(ls),
        ce=mean(ls),
        brier_score=mean([(p - y) ** 2 for p, y in zip(probs, targets)]),
        value_mse=mean([e * e for e in residuals]),
        mean_value=mean(vs),
        mean_reward=mean(rs),
        explained_variance=1 - var(residuals) / var(rs) if len(set(rs)) > 1 else None,
        token_span_strata="unavailable_without_authoritative_token_span_metadata",
        token_span_covered_count=0,
        token_span_unclassified_count=len(vs),
    )
    if detailed:
        bins = []
        for b in range(10):
            selected = [i for i, p in enumerate(probs) if min(9, int(p * 10)) == b]
            mass = math.fsum(ws[i] for i in selected)
            bins.append(
                dict(
                    lower=b / 10,
                    upper=(b + 1) / 10,
                    count=len(selected),
                    weight_mass=mass,
                    predicted_success=math.fsum(ws[i] * probs[i] for i in selected)
                    / mass
                    if mass
                    else None,
                    observed_success=math.fsum(ws[i] * targets[i] for i in selected)
                    / mass
                    if mass
                    else None,
                )
            )
        result["calibration_bins"] = bins
        result["calibration_error"] = math.fsum(
            b["weight_mass"] * abs(b["predicted_success"] - b["observed_success"])
            for b in bins
            if b["weight_mass"]
        )
    if not _with_strata:
        result.update(
            token_span_strata="authoritative_collection_spans_subset",
            token_span_covered_count=len(vs),
            token_span_unclassified_count=0,
        )
    metrics = {f"{prefix}/{k}": v for k, v in result.items()}
    if _with_strata and any(
        getattr(s, "collection_token_spans", None) for s in samples
    ):
        metrics[f"{prefix}/token_span_strata"] = "authoritative_collection_spans"
        strata = {}
        for sample, predictions, token_losses in zip(samples, values, losses):
            spans = [
                span
                for span in (getattr(sample, "collection_token_spans", None) or [])
                if span["sampled"]
            ]
            selected = {}
            for offset, position in enumerate(token_positions(sample)):
                matching = [
                    span
                    for span in spans
                    if span["start"] <= position + 1 < span["end"]
                ]
                if len(matching) != 1:
                    if spans:
                        raise ValueError(
                            "Authoritative spans must own each sampled diagnostic token exactly once"
                        )
                    continue
                span = matching[0]
                kind = span["kind"]
                if kind == "action":
                    if span is spans[-1] and sample.trainable_kind in {
                        "final_answer",
                        "forced_answer",
                    }:
                        kind = "final_answer"
                    elif span is spans[-1] and sample.trainable_kind == "malformed":
                        kind = "malformed"
                    else:
                        kind = "research_tool_calls"
                elif kind == "forced_answer":
                    kind = "final_answer"
                selected.setdefault(kind, []).append(offset)
            for kind, offsets in selected.items():
                mask = [False] * len(sample.completion_mask)
                positions = token_positions(sample)
                for offset in offsets:
                    mask[positions[offset]] = True
                proxy = SimpleNamespace(
                    rollout_id=sample.rollout_id,
                    reward=sample.reward,
                    completion_mask=mask,
                )
                rows = strata.setdefault(kind, ([], [], []))
                rows[0].append(proxy)
                rows[1].append([predictions[offset] for offset in offsets])
                rows[2].append([token_losses[offset] for offset in offsets])
        for kind, (
            selected_samples,
            selected_values,
            selected_losses,
        ) in strata.items():
            metrics.update(
                token_diagnostics(
                    selected_samples,
                    selected_values,
                    selected_losses,
                    prefix=f"{prefix}/span/{kind}",
                    detailed=detailed,
                    _with_strata=False,
                )
            )
    return metrics
