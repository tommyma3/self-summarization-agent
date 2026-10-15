import math
from types import SimpleNamespace

import pytest

from self_summarization_agent.value_diagnostics import prediction_metrics, value_diagnostics


def test_rollout_weighted_diagnostics_and_calibration():
    m = prediction_metrics([-1, 1, 1, 1], [0] * 4, [math.log(2)] * 4, ["a", "b", "b", "b"])
    assert m["bce"] == pytest.approx(math.log(2))
    assert m["brier_score"] == pytest.approx(0.25)
    assert m["mean_reward"] == pytest.approx(0)
    assert m["explained_variance"] == pytest.approx(0)
    assert m["classification_accuracy"] == pytest.approx(0.5)
    assert m["calibration_error"] == pytest.approx(0)
    assert m["calibration_bins"][5]["weight_mass"] == pytest.approx(1)


def test_perfect_predictions_and_constant_targets():
    m = prediction_metrics([-1, 1], [-1, 1], [0, 0], ["a", "b"])
    assert m["brier_score"] == 0
    assert m["explained_variance"] == pytest.approx(1)
    assert m["calibration_error"] == 0
    assert m["calibration_bins"][9]["count"] == 1
    assert prediction_metrics([1, 1], [0, 1], [0.69, 0], ["a", "b"])["explained_variance"] is None


def test_bce_uses_logits_loss_even_when_value_saturates():
    m = prediction_metrics([-1], [1], [100], ["a"])
    assert m["bce"] == 100
    assert m["brier_score"] == 1


def test_state_strata_and_legacy_unknown_are_explicit():
    samples = [
        SimpleNamespace(rollout_id="a", reward=-1, state_kind="initial"),
        SimpleNamespace(rollout_id="b", reward=1, state_kind="post_compaction"),
        SimpleNamespace(rollout_id="b", reward=1),
    ]
    m = value_diagnostics(samples, [0] * 3, [0.69] * 3, prefix="value/eval_before")
    assert m["value/eval_before/rollout_count"] == 2
    for kind in ("initial", "post_compaction", "unknown"):
        assert m[f"value/eval_before/{kind}/state_count"] == 1


@pytest.mark.parametrize("values,losses", [([float("nan")], [0.1]), ([2], [0.1]), ([0], [float("inf")]), ([], [])])
def test_invalid_diagnostics_fail(values, losses):
    with pytest.raises(ValueError):
        prediction_metrics([1], values, losses, ["a"])


def test_state_origin_uses_only_explicit_record_metadata():
    from self_summarization_agent.trajectory import _interval_state_kind
    assert _interval_state_kind({"segment_index": 1, "prefix_summary_turn_id": None}) == "initial"
    assert _interval_state_kind({"segment_index": 2, "prefix_summary_turn_id": "summary-1"}) == "post_compaction"
    assert _interval_state_kind({"turn_id": "trajectory-1", "prompt": "<summary>text</summary>"}) == "unknown"
    assert _interval_state_kind({"segment_index": True, "prefix_summary_turn_id": None}) == "unknown"
