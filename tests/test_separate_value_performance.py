from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_separate_value import tiny, critic, samples
from self_summarization_agent.config import CompactionValueConfig, config_to_dict, load_train_config
from self_summarization_agent.value_diagnostics import value_diagnostics


def variable_samples():
    data = samples()
    for sample, length in zip(data, [3, 1, 2]):
        sample.state_prefix_length = length
    return data


def test_batched_evaluation_preserves_order_padding_and_modes(tiny):
    trainer = critic(tiny)
    torch.nn.init.normal_(trainer.head.projection.weight, std=0.2)
    data = variable_samples()
    before = deepcopy(data)
    trainer.config = replace(trainer.config, length_bucketing=False)
    expected = trainer.evaluate(data)
    assert len(set(expected[0])) > 1  # Nonconstant predictions expose reordering errors.
    trainer.config = replace(trainer.config, length_bucketing=True, eval_prefix_microbatch_size=3)
    trainer.model.train()
    trainer.head.eval()
    rng = torch.get_rng_state().clone()
    actual = trainer.evaluate(data)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=2e-7, rtol=1e-6)
    assert trainer.model.training and not trainer.head.training
    assert torch.equal(rng, torch.get_rng_state())
    assert data == before


def test_token_budget_uses_padded_length_and_keeps_oversized_singleton(tiny):
    trainer = critic(tiny)
    trainer.config = replace(trainer.config, max_prefix_tokens_per_batch=4)
    prepared = trainer._prepare(variable_samples())
    assert list(trainer._batches(prepared, range(3), 3)) == [[1, 2], [0]]
    trainer.config = replace(trainer.config, max_prefix_tokens_per_batch=1)
    assert list(trainer._batches(prepared, range(3), 3)) == [[1], [2], [0]]


def test_bucketed_fit_matches_rollout_weighted_reference_with_short_minibatch(tiny):
    one = critic(tiny)
    one.config = replace(one.config, rollout_minibatch_size=2, length_bucketing=False)
    # Compare the weighted gradient itself: Adam can amplify roundoff in GPT2's
    # theoretically zero attention-key bias gradient after padded reductions.
    one.optimizer = torch.optim.SGD([
        {"params": one.model.parameters(), "lr": 0.01},
        {"params": one.head.parameters(), "lr": 0.02},
    ])
    two = deepcopy(one)
    two.config = replace(two.config, prefix_microbatch_size=8, eval_prefix_microbatch_size=8,
                         length_bucketing=True, max_prefix_tokens_per_batch=4)
    data = variable_samples()
    data.append(replace(data[-1], rollout_id="c", turn_id="3"))
    expected, metrics = one.fit(data)
    actual, other = two.fit(data)
    assert metrics["value/optimizer_step_count"] == other["value/optimizer_step_count"] == 4
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=1e-6, rtol=1e-5)
    for a, b in zip(one.model.parameters(), two.model.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("compactions,malformed", [(0, False), (1, False), (2, False), (1, True)])
def test_optimized_fit_preserves_runtime_intervals(tiny, tmp_path, compactions, malformed):
    from test_tito import TokenModel, runtime, SEARCH, SUMMARY, FINISH
    from self_summarization_agent.trajectory import build_rollout_native_training_cache, extract_training_samples
    from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
    from transformers import GPT2Config, GPT2Model
    outputs = [SEARCH, SUMMARY] * compactions + [FINISH]
    if malformed:
        outputs = [SEARCH, "</think>bad summary<|im_end|>"]
    result = runtime(TokenModel(outputs), context_threshold_tokens=1 if compactions else 15000).run("q", "question")
    for record in result.trajectory_records:
        record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
    data = extract_training_samples(result.trajectory_records,
        {r["turn_id"]: -1.0 if malformed else 1.0 for r in result.trajectory_records}, rollout_id="q:0")
    before = deepcopy((data, result.trajectory_records))
    path, _, config = tiny
    # Runtime's sampled-only ID 999999 stays in the full ledger, outside the
    # critic's exact initial/summary prefix. No historical text is tokenized.
    prefixes = [s.input_ids[:s.state_prefix_length] for s in data]
    assert all(999999 not in prefix for prefix in prefixes)
    GPT2Model(GPT2Config(vocab_size=512, n_positions=max(map(len, prefixes)) + 1,
        n_embd=8, n_layer=1, n_head=2, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)).save_pretrained(path)
    config = replace(config, fitting_epochs=1, prefix_microbatch_size=8, eval_prefix_microbatch_size=8,
                     max_prefix_tokens_per_batch=8192, activation_checkpointing=False)
    trainer = SeparateValueTrainer(path, path, config, dtype="float32", device="cpu")
    values, metrics = trainer.fit(data, max_sequence_length=max(len(s.input_ids) for s in data))
    assert len(values) == (1 if malformed else compactions + 1)
    assert metrics["value/optimizer_step_count"] == 1
    assert (data, result.trajectory_records) == before
    with pytest.raises(ValueError, match="Over-length"):
        trainer.fit(data, max_sequence_length=1)


def test_diagnostics_do_not_change_fit_and_final_predictions_run_once(tiny, monkeypatch):
    fast = critic(tiny)
    fast.config = replace(fast.config, fitting_epochs=3)
    full = deepcopy(fast)
    full.config = replace(full.config, eval_before_fit=True, eval_every_n_epochs=1)
    full.detailed_diagnostics = True
    calls = []
    original = fast._evaluate_prepared
    def tracked(prepared):
        calls.append(fast.update_count)
        return original(prepared)
    monkeypatch.setattr(fast, "_evaluate_prepared", tracked)
    expected, metrics = fast.fit(samples())
    actual, detailed = full.fit(samples())
    assert calls == [3]
    assert "value/epoch_3/bce" not in metrics  # Final predictions/metrics have one name.
    assert "value/eval_after/calibration_bins" not in metrics
    assert "value/eval_after/calibration_bins" in detailed
    assert "value/eval_before/bce" in detailed
    assert "value/epoch_2/bce" in detailed
    assert actual == expected
    for a, b in zip(fast.model.parameters(), full.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert metrics["value/train_seconds"] > 0
    assert metrics["value/final_predictions/seconds"] > 0
    assert metrics["value/fit_seconds"] >= metrics["value/train_seconds"]


def test_prepares_prefixes_once_per_fit(tiny, monkeypatch):
    trainer = critic(tiny)
    original = trainer._prepare
    calls = []
    def prepare(data):
        calls.append(len(data))
        return original(data)
    monkeypatch.setattr(trainer, "_prepare", prepare)
    trainer.fit(samples())
    assert calls == [3]


def test_basic_diagnostics_preserve_scalar_metrics():
    data = samples()
    full = value_diagnostics(data, [-0.3, 0.5, 0.1], [0.4, 0.2, 0.6], prefix="v")
    basic = value_diagnostics(data, [-0.3, 0.5, 0.1], [0.4, 0.2, 0.6], prefix="v", detailed=False)
    assert basic == {k: v for k, v in full.items() if k in basic}
    assert "v/bce" in basic and "v/rollout_count" in basic
    assert "v/calibration_bins" not in basic and "v/initial/state_count" not in basic


@pytest.mark.parametrize("name,value", [
    ("eval_prefix_microbatch_size", 0), ("eval_prefix_microbatch_size", True),
    ("max_prefix_tokens_per_batch", -1), ("max_prefix_tokens_per_batch", 1.5),
    ("activation_checkpointing", "false"), ("length_bucketing", 1),
    ("eval_before_fit", 0), ("eval_every_n_epochs", -1),
    ("evalset_interval", True), ("diagnostics_interval", -1),
    ("progress_interval_seconds", 0), ("progress_interval_seconds", float("nan")),
])
def test_performance_config_validation(name, value):
    with pytest.raises(ValueError, match=name):
        CompactionValueConfig(enabled=True, architecture="separate", model_path="unused", **{name: value})


def test_performance_preset_roundtrip_and_checkpointing_override(tiny):
    config = load_train_config("configs/train/compact_separate_mc_ppo.yaml")
    value = CompactionValueConfig(**config_to_dict(config)["training"]["value"])
    assert value.prefix_microbatch_size == 8 and value.eval_prefix_microbatch_size == 32
    assert value.fitting_epochs == 3 and value.rollout_minibatch_size == 32
    assert value.activation_checkpointing is False
    path, _, small = tiny
    from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
    model = SeparateValueTrainer(path, path, replace(small, activation_checkpointing=False),
                                 dtype="float32", device="cpu", activation_checkpointing=True)
    assert not model.model.is_gradient_checkpointing


def test_progress_is_throttled_but_phase_boundaries_are_visible(tiny, monkeypatch):
    from self_summarization_agent import separate_value_trainer as module
    trainer = critic(tiny)
    times = iter([1., 2., 3., 40.])
    events = []
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(module, "_write_training_progress", lambda path, stage, **kw: events.append(stage))
    trainer._progress("first")
    trainer._progress("skip")
    trainer._progress("phase", force=True)
    trainer._progress("heartbeat")
    assert events == ["first", "phase", "heartbeat"]
