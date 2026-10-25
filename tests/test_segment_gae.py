from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest
import torch

from test_separate_value import tiny, samples
from self_summarization_agent.config import CompactionValueConfig, TrainingConfig, load_train_config, ModelConfig
from self_summarization_agent.segment_returns import ordered_rollout_chains, segment_gae
from self_summarization_agent.separate_value import (
    ADVANTAGES, FROZEN_TARGETS, CRITIC_DIR, CRITIC_MANIFEST, PAIR_MANIFEST,
    sample_key, frozen_target_artifact, validate_frozen_targets, source_critic_identity,
    load_advantages, atomic_json, digest, separate_checkpoint_complete, sample_fingerprint,
)
from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
from self_summarization_agent.value_model import (
    CompactionValueHead, expected_categorical_value, two_hot_targets, value_support,
    outcome_log_losses, save_value_head, load_value_head,
)


def small_collection():
    data = samples()
    for sample in data:
        sample.rollout_id = "q:0" if sample.rollout_id == "a" else "q:1"
    rows = []
    for rollout in ("q:0", "q:1"):
        selected = [s for s in data if s.rollout_id == rollout]
        records = []
        for index, sample in enumerate(selected):
            terminal = index == len(selected) - 1
            records.append({"query_id": "q", "turn_id": sample.turn_id, "segment_index": index + 1,
                "termination_kind": "final_answer" if terminal else "compaction",
                "turn_ids": [f"summary-{index}"],
                "prefix_summary_turn_id": None if index == 0 else f"summary-{index - 1}"})
        rows.append({"query_id": "q", "rollout_index": int(rollout[-1]), "trajectory_records": records,
                     "turn_rewards": {s.turn_id: s.reward for s in selected}})
    return rows, data


@pytest.mark.parametrize("lam", [0.0, 0.9, 1.0])
def test_returns_count_reward_once_and_preserve_order(lam):
    rows, data = small_collection()
    before = deepcopy((rows, data))
    chains = ordered_rollout_chains(rows, list(reversed(data)))
    values = {sample_key(s): v for s, v in zip(data, [0.3, 0.2, 0.8])}
    result = segment_gae(chains, values, lam)
    assert result[sample_key(data[0])]["target"] == -1
    assert result[sample_key(data[2])]["target"] == 1
    assert result[sample_key(data[1])]["target"] == pytest.approx((1 - lam) * 0.8 + lam)
    assert sum(row["transition_reward"] for row in result.values()) == 0
    for sample in data:
        row = result[sample_key(sample)]
        assert row["advantage"] == pytest.approx(row["target"] - values[sample_key(sample)])
        if lam == 1:
            assert row["advantage"] == pytest.approx(sample.reward - values[sample_key(sample)])
    assert (rows, data) == before


@pytest.mark.parametrize("kind", ["final_answer", "forced_answer", "malformed", "context_overflow"])
def test_terminal_kinds_never_bootstrap(kind):
    rows, data = small_collection()
    rows[0]["trajectory_records"][0]["termination_kind"] = kind
    chain = ordered_rollout_chains(rows[:1], data[:1])
    result = segment_gae(chain, {sample_key(data[0]): 0.8}, 0.0)
    assert result[sample_key(data[0])]["target"] == -1
    assert result[sample_key(data[0])]["advantage"] == -1.8


def test_empty_and_critic_only_target_batches(tiny):
    cfg = replace(tiny[2], baseline_timing="pre_fit", num_bins=51)
    empty = frozen_target_artifact([], [], "base", cfg, "compaction_gae_value", [], "source")
    validate_frozen_targets(empty, [], "base", cfg, "compaction_gae_value", chains=[], source_identity="source")
    rows, data = small_collection()
    data[-1].completion_mask = [False] * 3
    data[-1].collection_assistant_token_mask = [False] * 4
    chain = ordered_rollout_chains(rows, data)
    result = segment_gae(chain, {sample_key(s): 0.5 for s in data}, 0.9)
    assert set(result) == {sample_key(s) for s in data}
    assert result[sample_key(data[-2])]["target"] == pytest.approx(0.95)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "segment", "terminal", "summary", "reward", "rewrite"])
def test_chain_validation_fails_closed(fault):
    rows, data = small_collection()
    if fault == "missing":
        data.pop(1)
    elif fault == "duplicate":
        rows.append(deepcopy(rows[-1]))
    elif fault == "segment":
        rows[1]["trajectory_records"][1]["segment_index"] = 3
    elif fault == "terminal":
        rows[1]["trajectory_records"][-1]["termination_kind"] = "compaction"
    elif fault == "summary":
        rows[1]["trajectory_records"][1]["prefix_summary_turn_id"] = "other"
    elif fault == "reward":
        rows[1]["turn_rewards"][data[-1].turn_id] = -1
    else:
        rows[1]["status"] = "history_rewrite_detected"
    with pytest.raises(ValueError):
        ordered_rollout_chains(rows, data)


@pytest.mark.parametrize("bins", [2, 51, 201])
def test_categorical_targets_preserve_mean_and_gradients(bins, tmp_path):
    targets = torch.tensor([-1., -0.73, 0., 0.35, 1.], requires_grad=True)
    encoded = two_hot_targets(targets, bins)
    support = torch.tensor(value_support(bins))
    torch.testing.assert_close(encoded.sum(-1), torch.ones(5))
    torch.testing.assert_close((encoded * support).sum(-1), targets, atol=2e-7, rtol=1e-6)
    assert not encoded.requires_grad
    assert (encoded > 0).sum(-1).max() <= 2
    head = CompactionValueHead(3, num_bins=bins)
    logits = head(torch.ones(5, 3))
    torch.nn.functional.cross_entropy(logits, encoded).backward()
    assert torch.isfinite(head.projection.weight.grad).all()
    assert targets.grad is None
    save_value_head(head, tmp_path)
    loaded, found = load_value_head(tmp_path, hidden_size=3, num_bins=bins, zero_initialize=True)
    assert found
    torch.testing.assert_close(loaded(torch.ones(5, 3)), logits)
    with pytest.raises(ValueError, match="manifest"):
        load_value_head(tmp_path, hidden_size=3, num_bins=bins + 1, zero_initialize=True)
    if bins == 2:
        torch.testing.assert_close(encoded[:, 1], (targets + 1) / 2)


@pytest.mark.parametrize("target", [float("nan"), float("inf"), 1.1, -1.1])
def test_bad_categorical_target_rejected(target):
    with pytest.raises(ValueError, match="finite"):
        two_hot_targets(torch.tensor([target]), 51)


def test_categorical_outcome_loss_is_binary_mean_loss():
    logits = torch.tensor([[0., 3., 1.], [1., 4., 0.]])
    rewards = torch.tensor([-1., 1.])
    success = (expected_categorical_value(logits) + 1) / 2
    expected = torch.nn.functional.binary_cross_entropy(success, (rewards + 1) / 2, reduction="none")
    torch.testing.assert_close(outcome_log_losses(logits, rewards), expected)
    # Stable even when the expected value rounds to an endpoint.
    extreme = torch.tensor([[-1000., -1000., 1000.]])
    assert torch.isfinite(outcome_log_losses(extreme, torch.tensor([-1.]))).all()


def test_gae_config_and_invalid_combinations(tiny):
    preset = load_train_config("configs/train/compact_separate_gae_ppo.yaml")
    assert preset.training.value.num_bins == 51
    assert preset.training.value.baseline_timing == "pre_fit"
    assert preset.training.advantage_estimator == "compaction_gae_value"
    for overrides in ({"num_bins": True}, {"num_bins": 1}, {"gae_lambda": float("nan")},
                      {"gae_lambda": 1.1}, {"baseline_timing": "later"}):
        with pytest.raises(ValueError):
            replace(tiny[2], **overrides)
    with pytest.raises(ValueError, match="pre_fit"):
        TrainingConfig(advantage_estimator="compaction_gae_value", value=tiny[2])
    with pytest.raises(ValueError, match="separate"):
        CompactionValueConfig(num_bins=51)


def test_gae_cache_scorer_has_no_critic_dependency(monkeypatch):
    from self_summarization_agent import cache_step
    cfg = load_train_config("configs/train/compact_separate_gae_ppo.yaml")
    seen = []
    monkeypatch.setattr(cache_step, "TransformersPolicyTrainer", lambda model, training: seen.append(training))
    cache_step.build_cache_scorer(cfg, checkpoint_path="unused-actor")
    assert len(seen) == 1
    assert not seen[0].value.enabled
    assert seen[0].advantage_estimator == "group_relative"
    assert cfg.training.value.num_bins == 51


def test_frozen_artifact_rejects_stale_source_and_modified_recurrence(tiny):
    rows, data = small_collection()
    cfg = replace(tiny[2], num_bins=51, baseline_timing="pre_fit")
    chains = ordered_rollout_chains(rows, data)
    body = frozen_target_artifact(data, [0.1, 0.2, 0.3], "base", cfg, "compaction_gae_value", chains, "old")
    args = (data, "base", cfg, "compaction_gae_value")
    validate_frozen_targets(body, *args, chains=chains, source_identity="old")
    with pytest.raises(ValueError, match="provenance"):
        validate_frozen_targets(body, *args, chains=chains, source_identity="new")
    for change in ("target", "advantage", "transition_reward"):
        altered = deepcopy(body)
        altered["rows"][1][change] -= 0.1
        altered["digest"] = digest({k: v for k, v in altered.items() if k != "digest"})
        with pytest.raises(ValueError):
            validate_frozen_targets(altered, *args, chains=chains, source_identity="old")


@pytest.mark.parametrize("compactions,malformed", [(0, False), (1, False), (2, False), (1, True), (2, True)])
def test_gae_fit_preserves_runtime_tokens(tiny, compactions, malformed):
    from test_tito import TokenModel, runtime, SEARCH, SUMMARY, FINISH
    from self_summarization_agent.trajectory import build_rollout_native_training_cache, extract_training_samples
    from transformers import GPT2Config, GPT2Model
    outputs = [SEARCH, SUMMARY] * compactions + [FINISH]
    if malformed:
        outputs = [SEARCH, SUMMARY] * (compactions - 1) + [SEARCH, "</think>bad summary<|im_end|>"]
    result = runtime(TokenModel(outputs), context_threshold_tokens=1 if compactions else 15000).run("q", "question")
    records = result.trajectory_records
    for record in records:
        record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
    rewards = {r["turn_id"]: -1. if malformed else 1. for r in records}
    data = extract_training_samples(records, rewards, rollout_id="q:0")
    rows = [{"query_id": "q", "rollout_index": 0, "trajectory_records": records, "turn_rewards": rewards}]
    before = deepcopy((rows, data))
    path, _, cfg = tiny
    prefixes = [s.input_ids[:s.state_prefix_length] for s in data]
    GPT2Model(GPT2Config(vocab_size=512, n_positions=max(map(len, prefixes)) + 1,
        n_embd=8, n_layer=1, n_head=2, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)).save_pretrained(path)
    cfg = replace(cfg, num_bins=51, baseline_timing="pre_fit", fitting_epochs=1)
    critic = SeparateValueTrainer(path, path, cfg, dtype="float32", device="cpu", activation_checkpointing=False,
                                  advantage_estimator="compaction_gae_value")
    old, _ = critic.evaluate(data)
    targets = segment_gae(ordered_rollout_chains(rows, data), dict(zip(map(sample_key, data), old)), 0.95)
    critic.fit(data, scalar_targets={k: v["target"] for k, v in targets.items()})
    assert (rows, data) == before
    assert len(targets) == (compactions if malformed else compactions + 1)
    if malformed:
        assert targets[sample_key(data[-1])]["target"] == -1


@pytest.mark.parametrize("failure,estimator,bins", [
    (failure, "compaction_gae_value", 51)
    for failure in [None, "freeze", "fit", "critic_save", "actor_update", "actor_save"]
] + [(None, "compaction_gae_value", 2), (None, "compaction_mc_value", 2)])
def test_two_iterations_use_prefit_values_and_retry_immutable_targets(tiny, tmp_path, monkeypatch,
                                                                    failure, estimator, bins):
    from self_summarization_agent import train_step, critic_step, separate_value_trainer as value_module
    from self_summarization_agent import trainer as actor_module
    from transformers import GPT2LMHeadModel
    path, _, value = tiny
    value = replace(value, baseline_timing="pre_fit", num_bins=bins, evalset_interval=0)
    cfg = load_train_config("configs/train/compact_separate_gae_ppo.yaml")
    cfg.model = ModelConfig(model_path=str(path), dtype="float32")
    cfg.training = TrainingConfig(advantage_estimator=estimator, value=value,
                                 update_epochs=2, max_sequence_length=27000)
    rows, data = small_collection()
    before = deepcopy((rows, data))
    chains = ordered_rollout_chains(rows, data)
    monkeypatch.setattr(train_step, "samples_from_rollout_rows", lambda *a, **kw: deepcopy(data))
    original = value_module.SeparateValueTrainer
    failed = []
    snapshots = []
    evaluated = []

    def fail(stage):
        if stage == failure and not failed:
            failed.append(stage)
            raise RuntimeError("injected failure")

    class Critic(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "device": "cpu"})
        def evaluate(self, batch):
            values, losses = super().evaluate(batch)
            evaluated.append(values)
            return values, losses
        def fit(self, batch, **kwargs):
            snapshots.append(deepcopy(kwargs["scalar_targets"]))
            values, metrics = super().fit(batch, **kwargs)
            assert kwargs["scalar_targets"] == snapshots[-1]
            fail("fit")
            return values, metrics
        def save(self, *args):
            identity = super().save(*args)
            fail("critic_save")
            return identity
    monkeypatch.setattr(value_module, "SeparateValueTrainer", Critic)
    original_atomic = critic_step.atomic_json
    def atomic(path, body):
        original_atomic(path, body)
        if Path(path).name == FROZEN_TARGETS:
            fail("freeze")
    from pathlib import Path
    monkeypatch.setattr(critic_step, "atomic_json", atomic)
    def run_subprocess(command, **kwargs):
        args = {command[i][2:]: command[i+1] for i in range(3, len(command), 2)}
        critic_step.run(load_train_config(args["config"]), args["checkpoint"], args["rollouts"],
                        args["output"], args["output-id"])
    monkeypatch.setattr(train_step.subprocess, "run", run_subprocess)
    monkeypatch.setattr(actor_module.AutoModelForMultimodalLM, "from_pretrained",
                        lambda source, **kw: GPT2LMHeadModel.from_pretrained(source))
    monkeypatch.setattr(actor_module, "configure_tokenizer_chat_template", lambda *a: None)
    expected_actor = {}
    class Actor(actor_module.TransformersPolicyTrainer):
        def step(self, grouped_samples, *, fixed_advantages):
            assert fixed_advantages == pytest.approx(expected_actor)
            fail("actor_update")
            return super().step(grouped_samples, fixed_advantages=fixed_advantages)
        def save_checkpoint(self, output):
            super().save_checkpoint(output)
            fail("actor_save")
    monkeypatch.setattr(train_step, "TransformersPolicyTrainer", Actor)
    rollouts = tmp_path / "rollouts.jsonl"
    for iteration in (1, 2):
        source = path if iteration == 1 else path / CRITIC_DIR
        probe = original(source, path, value, dtype="float32", device="cpu", activation_checkpointing=False,
                         resume=iteration > 1, advantage_estimator=estimator)
        expected_values, _ = probe.evaluate(data)
        identity = source_critic_identity(source, resume=iteration > 1)
        expected = segment_gae(chains, dict(zip(map(sample_key, data), expected_values)),
                               value.gae_lambda if estimator == "compaction_gae_value" else 1.0)
        expected_actor = {key: row["advantage"] for key, row in expected.items()}
        rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = tmp_path / f"iteration-{iteration:05d}"
        kwargs = dict(checkpoint_path=path, rollout_path=rollouts, output_checkpoint_path=output)
        if iteration == 1 and failure:
            with pytest.raises(RuntimeError, match="injected"):
                train_step.run_train_step(cfg, **kwargs)
            assert not (output / ".complete").exists()
            frozen_bytes = (output / FROZEN_TARGETS).read_bytes()
        train_step.run_train_step(cfg, **kwargs)
        if iteration == 1 and failure:
            assert (output / FROZEN_TARGETS).read_bytes() == frozen_bytes
        assert separate_checkpoint_complete(output)
        actual, metrics = load_advantages(output / ADVANTAGES, data, path.name, value,
            estimator=estimator, chains=chains if estimator == "compaction_gae_value" else [],
            source_identity=identity)
        assert actual == pytest.approx(expected_actor)
        assert metrics["value/total_optimizer_steps"] == iteration * 2
        assert json.loads((output / FROZEN_TARGETS).read_text())["rows"][1]["value"] == pytest.approx(expected_values[1])
        path = output
    # A retry reuses the frozen prediction pass, even after a completed fit/save.
    assert len(evaluated) == 2
    assert (rows, data) == before
    with pytest.raises(ValueError, match="method/support"):
        original(path / CRITIC_DIR, path, replace(value, num_bins=2 if bins == 51 else 51),
                 dtype="float32", device="cpu", resume=True, advantage_estimator=estimator)
    broken = json.loads((path / FROZEN_TARGETS).read_text())
    broken["rows"][0]["advantage"] += 0.1
    atomic_json(path / FROZEN_TARGETS, broken)
    assert not separate_checkpoint_complete(path)


def test_historical_mc_fingerprint_remains_loadable(tiny, tmp_path):
    data = samples()
    cfg = tiny[2]
    old_config = {key: val for key, val in asdict(cfg).items()
                  if key not in {"num_bins", "baseline_timing", "gae_lambda"}}
    atomic_json(tmp_path / CRITIC_DIR / CRITIC_MANIFEST, {"identity": "historical"})
    body = {"version": 1, "behavior_checkpoint_id": "base", "critic_identity": "historical",
            "sample_fingerprint": sample_fingerprint(data), "config_fingerprint": digest(old_config),
            "rows": [{"key": sample_key(s), "value": 0.2, "advantage": s.reward - 0.2} for s in data]}
    atomic_json(tmp_path / ADVANTAGES, body)
    fixed, _ = load_advantages(tmp_path / ADVANTAGES, data, "base", cfg)
    assert fixed == {sample_key(s): s.reward - 0.2 for s in data}
    with pytest.raises(ValueError):
        load_advantages(tmp_path / ADVANTAGES, data, "base", replace(cfg, num_bins=51))
