import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast

from self_summarization_agent.config import (
    CompactionValueConfig,
    TrainingConfig,
    load_train_config,
)
from self_summarization_agent.separate_value import (
    CRITIC_MANIFEST,
    atomic_json,
    method_spec,
    sample_key,
)
from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
from self_summarization_agent.token_value import (
    TokenAdvantages,
    token_diagnostics,
    token_positions,
    token_weights,
    validate_token_advantages,
    validate_token_samples,
)
from self_summarization_agent.token_value_artifacts import (
    freeze_tokens,
    read_shard,
    validate_token_frozen,
)
from self_summarization_agent.trainer import _clipped_grpo_token_losses
from self_summarization_agent.trajectory import RLSample


def sample(rollout="r", turn="0", reward=1.0, mask=None):
    ids = [1, 2, 3, 4, 5, 6, 7]
    mask = mask if mask is not None else [False, False, True, False, False, True, True]
    return RLSample(
        query_id="q",
        turn_id=turn,
        prompt="",
        completion="",
        reward=reward,
        trainable_kind="final_answer",
        rollout_id=rollout,
        collection_full_token_ids=ids,
        collection_assistant_token_mask=mask,
        input_ids=ids[:-1],
        labels=ids[1:],
        completion_mask=mask[1:],
        reference_logprob=-2.0,
        reference_logprobs=[-2.0] * 6,
        state_prefix_length=2,
        reference_logprob_source="vllm_raw_rollout",
        reference_logprobs_mode="raw_logprobs",
    )


@pytest.fixture
def critic(tmp_path):
    path = tmp_path / "base"
    torch.manual_seed(11)
    GPT2Model(
        GPT2Config(
            vocab_size=12,
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({str(i): i for i in range(12)}, unk_token="0")
        ),
        unk_token="0",
        pad_token="0",
        eos_token="1",
    )
    tokenizer.save_pretrained(path)
    config = CompactionValueConfig(
        enabled=True,
        architecture="separate",
        model_path=str(path),
        num_bins=2,
        baseline_timing="pre_fit",
        fitting_epochs=1,
        learning_rate=0.01,
        rollout_minibatch_size=2,
        prefix_microbatch_size=1,
    )
    return SeparateValueTrainer(
        path,
        path,
        config,
        dtype="float32",
        device="cpu",
        activation_checkpointing=False,
        advantage_estimator="token_mc_value",
    )


def test_presets_and_invalid_modes():
    cfg = load_train_config("configs/train/compact_separate_token_mc_ppo.yaml")
    assert cfg.training.advantage_estimator == "token_mc_value"
    assert (
        method_spec(cfg.training.value, "token_mc_value")["state_anchor"]
        == "before_each_sampled_token"
    )
    for overrides in (
        {"training.value.num_bins": 3},
        {"training.value.baseline_timing": "post_fit"},
        {"rollout.require_exact_token_ids": False},
        {"training.train_compaction_tokens": False},
        {
            "training.advantage_estimator": "token_mc_value",
            "training.value.architecture": "shared",
        },
    ):
        with pytest.raises(ValueError):
            load_train_config(
                "configs/train/compact_separate_token_mc_ppo.yaml", overrides
            )
    control = load_train_config("configs/train/compact_separate_mc_prefit_control.yaml")
    assert control.training.value == cfg.training.value


def test_sparse_shifted_positions_and_weights():
    data = [
        sample(),
        sample("b", "0", -1),
        sample("b", "1", -1, [False, False, True, False, False, False, False]),
        sample("b", "2", -1, [False] * 7),
        sample("empty", "0", -1, [False] * 7),
    ]
    assert token_positions(data[0]) == (1, 4, 5)
    weights = token_weights(data)
    assert weights[0] == [1 / 6] * 3
    assert sum(weights[1]) == pytest.approx(0.25)
    assert weights[2] == [0.25]
    assert weights[3:] == [[], []]
    assert sum(map(sum, weights)) == pytest.approx(1)
    bad = deepcopy(data)
    bad[0].labels[0] = 10
    with pytest.raises(ValueError):
        validate_token_samples(bad)
    bad = deepcopy(data)
    bad[0].collection_assistant_token_mask[0] = True
    with pytest.raises(ValueError):
        validate_token_samples(bad)


def test_real_causal_model_pre_token_alignment(critic):
    s = sample()
    prepared = critic._prepare_tokens([s])
    critic.model.eval()
    critic.head.eval()
    # Nonzero head proves equality isn't caused by a constant zero head.
    torch.nn.init.normal_(critic.head.projection.weight) if hasattr(
        critic.head, "projection"
    ) else None
    with torch.no_grad():
        for p in critic.head.parameters():
            p.normal_()
        logits = critic._token_logits(prepared, [0])[0]
        first = critic._logits_prepared(critic._prepare([s]), [0])[0]
        torch.testing.assert_close(logits[0], first)
        for offset, k in enumerate(token_positions(s)):
            changed = deepcopy(s)
            full = list(s.collection_full_token_ids)
            full[k + 1 :] = [8] * (len(full) - k - 1)
            changed.collection_full_token_ids = full
            changed.input_ids = full[:-1]
            changed.labels = full[1:]
            other = critic._token_logits(critic._prepare_tokens([changed]), [0])[0]
            torch.testing.assert_close(
                logits[offset], other[offset], atol=1e-6, rtol=1e-6
            )


def test_fit_microbatch_partition_and_empty_intervals(critic):
    data = [
        sample(),
        sample("b", "0", -1),
        sample("b", "1", -1, [False, False, True, False, False, False, False]),
        sample("empty", "0", -1, [False] * 7),
    ]
    before = deepcopy(data)
    with torch.no_grad():
        for parameter in critic.head.parameters():
            parameter.normal_(std=0.1)
    # SGD exposes reduction differences without Adam amplifying roundoff near zero.
    critic.optimizer = torch.optim.SGD(
        [*critic.model.parameters(), *critic.head.parameters()], lr=0.01
    )
    other = deepcopy(critic)
    other.config = replace(critic.config, prefix_microbatch_size=3)
    v, m = critic.fit(data, max_sequence_length=32)
    other.fit(data, max_sequence_length=32)
    for a, b in zip(
        [*critic.model.parameters(), *critic.head.parameters()],
        [*other.model.parameters(), *other.head.parameters()],
    ):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    assert m["value/optimizer_step_count"] == 1
    assert m["value/effective_weight_mass"] == pytest.approx(1)
    assert len(v[0]) == 3 and v[-1] == []
    assert data == before
    _, m = critic.fit([data[-1]])
    assert m["value/optimizer_step_count"] == 0


def test_ppo_distinct_token_advantages_and_scalar_equivalence():
    logp = torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], requires_grad=True)
    old = torch.zeros_like(logp)
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
    a = torch.tensor([[1.0, 0.0, -1.0], [-0.5, 1.5, 0.0]])
    loss, *_ = _clipped_grpo_token_losses(logp, old, a, mask, clip_range=0.2)
    expected = -torch.minimum(logp.exp() * a, logp.exp().clamp(0.8, 1.2) * a) * mask
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss.sum(), logp, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), logp)[0],
    )
    scalar = torch.tensor([0.4, -0.6])
    x = _clipped_grpo_token_losses(logp, old, scalar, mask, clip_range=0.2)[0]
    y = _clipped_grpo_token_losses(
        logp, old, scalar[:, None].expand_as(logp), mask, clip_range=0.2
    )[0]
    torch.testing.assert_close(x, y)


def test_frozen_shard_retry_and_corruption(critic, tmp_path):
    data = [sample(), sample("b", "0", -1)]
    values, _ = critic.evaluate(data)
    output = tmp_path / "frozen"
    body = freeze_tokens(
        output, data, values, "behavior", critic.config, "source", critic.tokenizer_hash
    )
    assert not list(output.glob("*.safetensors")), "vLLM scans root weight files"
    result = validate_token_frozen(
        body, output, data, "behavior", critic.config, "source", critic.tokenizer_hash
    )
    assert result[sample_key(data[0])].dense(data[0]) == [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    again = freeze_tokens(
        output, data, values, "behavior", critic.config, "source", critic.tokenizer_hash
    )
    assert again["digest"] == body["digest"]
    for source, behavior, changed in [
        ("wrong", "behavior", data),
        ("source", "wrong", data),
        ("source", "behavior", list(reversed(data))),
    ]:
        with pytest.raises(ValueError):
            validate_token_frozen(
                body, output, changed, behavior, critic.config, source
            )
    changed = deepcopy(data)
    changed[0].completion_mask[2] = True
    with pytest.raises(ValueError):
        validate_token_frozen(
            body, output, changed, "behavior", critic.config, "source"
        )
    path = output / body["shard"]["file"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        read_shard(body, output)


def test_explicit_advantages_and_diagnostics():
    data = [sample(), sample("b", "0", -1)]
    advantages = {
        sample_key(s): TokenAdvantages(token_positions(s), (0.0,) * 3) for s in data
    }
    assert len(validate_token_advantages(data, advantages)) == 2
    with pytest.raises(ValueError):
        validate_token_advantages(data, {sample_key(s): 0.0 for s in data})
    values = [[0.5] * 3, [-0.5] * 3]
    losses = [[0.3] * 3, [0.3] * 3]
    m = token_diagnostics(data, values, losses, prefix="v")
    assert m["v/brier_score"] == pytest.approx(0.0625)
    assert m["v/rollout_count"] == 2 and m["v/token_count"] == 6
    assert m["v/ce"] == pytest.approx(0.3)
    assert m["v/weight_sum"] == pytest.approx(1)


def test_transformers_actor_token_update_and_resume(critic, monkeypatch, tmp_path):
    from transformers import GPT2LMHeadModel

    from self_summarization_agent import trainer as module
    from self_summarization_agent.config import ModelConfig
    from self_summarization_agent.trainer import TransformersPolicyTrainer

    monkeypatch.setattr(
        module.AutoModelForMultimodalLM,
        "from_pretrained",
        lambda *a, **kw: GPT2LMHeadModel(critic.model.config),
    )
    monkeypatch.setattr(module, "configure_tokenizer_chat_template", lambda *a: None)
    cfg = TrainingConfig(
        advantage_estimator="token_mc_value",
        value=critic.config,
        update_epochs=2,
        minibatch_size=2,
        learning_rate=0.001,
        max_sequence_length=32,
    )
    torch.manual_seed(42)
    a = TransformersPolicyTrainer(
        ModelConfig(model_path=critic.source, dtype="float32", device_map=None), cfg
    )
    torch.manual_seed(42)
    b = TransformersPolicyTrainer(
        ModelConfig(model_path=critic.source, dtype="float32", device_map=None),
        replace(cfg, gradient_accumulation_microbatch_size=2),
    )
    data = [sample(), sample("b", "0", -1)]
    fixed = {
        sample_key(s): TokenAdvantages(token_positions(s), (1.0, -1.0, 0.0))
        for s in data
    }
    before = deepcopy(data)
    metrics = a.step({"q": data}, fixed_advantages=fixed)
    b.step({"q": data}, fixed_advantages=fixed)
    assert metrics.optimizer_step_count == 2 and data == before
    assert metrics.mean_advantage == pytest.approx(0.0)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=1e-5)
    output = tmp_path / "saved"
    critic.save(output, "saved")
    with pytest.raises(ValueError, match="method"):
        SeparateValueTrainer(
            output,
            critic.source,
            critic.config,
            dtype="float32",
            device="cpu",
            resume=True,
            advantage_estimator="compaction_mc_value",
        )


def test_real_verl_dataproto_alignment():
    # Use the installed pinned DataProto, including jagged response arrays.
    from self_summarization_agent.verl_ray_trainer import (
        build_verl_value_actor_dataproto,
    )

    data = [sample(), sample("b", "0", -1)]
    batch = build_verl_value_actor_dataproto(
        data,
        advantages=[
            TokenAdvantages(token_positions(s), (1.0, -1.0, 0.25)) for s in data
        ],
        value_weights=[0.0, 0.0],
        checkpoint_id="behavior",
        alignment=4,
        max_sequence_length=32,
    )
    rows = batch.batch["advantages"].unbind()
    assert rows[0].tolist() == [0.0, 1.0, 0.0, 0.0, -1.0, 0.25]
    assert rows[2].tolist() == [0.0] * 6
    assert batch.batch["response_mask"].unbind()[2].tolist() == [False] * 6


def test_compaction_extraction_and_token_warmup(tmp_path):
    from value_pretraining_fixtures import make_batch

    from self_summarization_agent.value_pretraining import (
        completed_pretraining,
        fit_pretraining,
        prepare_dataset,
    )

    batch = make_batch(tmp_path)
    batch.config.training.advantage_estimator = "token_mc_value"
    batch.config.training.value = replace(batch.config.training.value, num_bins=2)
    # Collection identity is unchanged by the training-only estimator.
    before = deepcopy(batch.rows)
    train, validation, manifest = prepare_dataset(
        batch.config, batch.actor, batch.path, batch.allowed
    )
    assert batch.rows == before
    assert any(s.state_kind == "post_compaction" for s in train + validation)
    assert all(s.reward in (-1.0, 1.0) for s in train + validation)
    assert all(not s.collection_assistant_token_mask[0] for s in train + validation)
    output = tmp_path / "warmup"
    best = fit_pretraining(
        batch.config, batch.actor, batch.path, output, batch.allowed, device="cpu"
    )
    assert completed_pretraining(batch.config, batch.actor, output) == best
    body = json.loads((best / CRITIC_MANIFEST).read_text())
    assert body["method"]["estimator"] == "token_mc_value"
    history = json.loads((output / "state.json").read_text())["history"]
    assert all(
        row["validation/token_count"] > row["validation/rollout_count"]
        for row in history
    )


def test_token_pair_retry_reload_and_second_update(critic, tmp_path):
    from types import SimpleNamespace

    from self_summarization_agent.separate_value import (
        ADVANTAGES,
        CRITIC_DIR,
        FROZEN_TARGETS,
        PAIR_MANIFEST,
        digest,
        separate_checkpoint_complete,
        source_critic_identity,
    )
    from self_summarization_agent.token_value_artifacts import (
        load_token_advantages,
        run_token_critic,
    )

    data = [sample(), sample("b", "0", -1)]
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            value=replace(critic.config, evalset_interval=0), max_sequence_length=32
        )
    )
    critic.config = cfg.training.value
    output = tmp_path / "iteration-00001"
    source = Path(critic.source)
    checkpoint = source
    run_token_critic(
        cfg, checkpoint, output, output.name, data, source, False, critic, "unused"
    )
    first = json.loads((output / FROZEN_TARGETS).read_text())["digest"]
    # A retry starts from the committed initializer, while reusing frozen values.
    retry = SeparateValueTrainer(
        source,
        source,
        cfg.training.value,
        dtype="float32",
        device="cpu",
        activation_checkpointing=False,
        advantage_estimator="token_mc_value",
    )
    run_token_critic(
        cfg, checkpoint, output, output.name, data, source, False, retry, "unused"
    )
    assert json.loads((output / FROZEN_TARGETS).read_text())["digest"] == first
    source_id = source_critic_identity(source, resume=False)
    fixed, _ = load_token_advantages(
        output / ADVANTAGES, data, checkpoint.name, cfg.training.value, source_id
    )
    assert fixed[sample_key(data[0])].values == (1.0, 1.0, 1.0)
    retry.model.save_pretrained(output)
    cm = json.loads((output / CRITIC_DIR / CRITIC_MANIFEST).read_text())
    atomic_json(
        output / PAIR_MANIFEST,
        dict(
            version=2,
            architecture="separate",
            checkpoint_id=output.name,
            actor_source=checkpoint.name,
            critic_identity=cm["identity"],
            method=cm["method"],
            advantages_digest=digest(json.loads((output / ADVANTAGES).read_text())),
        ),
    )
    assert separate_checkpoint_complete(output)
    resumed = SeparateValueTrainer(
        output / CRITIC_DIR,
        source,
        cfg.training.value,
        dtype="float32",
        device="cpu",
        resume=True,
        activation_checkpointing=False,
        advantage_estimator="token_mc_value",
    )
    second = tmp_path / "iteration-00002"
    run_token_critic(
        cfg,
        output,
        second,
        second.name,
        data,
        output / CRITIC_DIR,
        True,
        resumed,
        "unused",
    )
    assert resumed.update_count == 2
    assert (
        json.loads((second / FROZEN_TARGETS).read_text())["source_critic_identity"]
        == cm["identity"]
    )


def test_hand_weighted_token_ce_gradient(critic):
    data = [sample(), sample("b", "0", -1), sample("b", "1", -1)]
    shortened = data[-1]
    shortened.collection_full_token_ids = shortened.collection_full_token_ids[:4]
    shortened.collection_assistant_token_mask = (
        shortened.collection_assistant_token_mask[:4]
    )
    shortened.input_ids = shortened.collection_full_token_ids[:-1]
    shortened.labels = shortened.collection_full_token_ids[1:]
    shortened.completion_mask = shortened.collection_assistant_token_mask[1:]
    shortened.reference_logprobs = shortened.reference_logprobs[:3]
    with torch.no_grad():
        for p in critic.head.parameters():
            p.normal_(std=0.1)
    critic.optimizer = torch.optim.SGD(
        [*critic.model.parameters(), *critic.head.parameters()], lr=0.01
    )
    reference = deepcopy(critic)
    prepared = reference._prepare_tokens(data)
    logits = reference._token_logits(prepared, [0, 1, 2])
    loss = sum(
        weight
        * torch.nn.functional.cross_entropy(
            token_logits,
            torch.full((len(token_logits),), int(s.reward == 1), dtype=torch.long),
        )
        for s, token_logits, weight in zip(data, logits, [0.5, 0.25, 0.25])
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [*reference.model.parameters(), *reference.head.parameters()],
        reference.config.max_grad_norm,
    )
    reference.optimizer.step()
    _, metrics = critic.fit(data)
    assert metrics["value/epoch_1/target_ce"] == pytest.approx(
        float(loss.detach()), abs=1e-7
    )
    for a, b in zip(
        [*critic.model.parameters(), *critic.head.parameters()],
        [*reference.model.parameters(), *reference.head.parameters()],
    ):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize(
    "case", ["plain", "compact", "repeated", "forced", "bad", "empty", "overlimit"]
)
def test_all_interval_boundaries_preserve_raw_token_coverage(case):
    import hashlib

    from test_tito import FINISH, SEARCH, SUMMARY, TokenModel, runtime

    from self_summarization_agent.trajectory import (
        build_rollout_native_training_cache,
        extract_trainable_samples,
    )

    if case in {"bad", "empty", "overlimit"}:
        summary = {
            "bad": "</think>bad<|im_end|>",
            "empty": "</think><summary></summary><|im_end|>",
            "overlimit": "</think><summary>" + "x" * 101 + "</summary><|im_end|>",
        }[case]
        outputs = [SEARCH, summary]
        options = {"context_threshold_tokens": 1}
        expected_intervals = 1
    else:
        compactions = {"plain": 0, "compact": 1, "repeated": 2, "forced": 0}[case]
        outputs = (
            [SEARCH, SUMMARY] * compactions + [FINISH]
            if compactions
            else [SEARCH, FINISH]
        )
        options = {"context_threshold_tokens": 1 if compactions else 15000}
        if case == "forced":
            options["max_tool_calls"] = 1
        expected_intervals = compactions + 1
    model = TokenModel(outputs)
    result = runtime(model, **options).run("q", "question")

    def evidence_hash():
        return hashlib.sha256(
            json.dumps(result.trajectory_records, sort_keys=True).encode()
        ).hexdigest()

    before = evidence_hash()
    records = deepcopy(result.trajectory_records)
    for record in records:
        record["training_cache"] = build_rollout_native_training_cache(
            record["collection_tokens"]
        )
    data = extract_trainable_samples(
        records, {r["turn_id"]: -1.0 for r in records}, rollout_id="q:0"
    )
    validate_token_samples(data)
    assert len(data) == expected_intervals
    assert sum(len(token_positions(s)) for s in data) == sum(
        len(g.completion_token_ids) for g in model.results
    )
    assert all(s.reward == -1.0 for s in data)
    assert evidence_hash() == before
    if case in {"bad", "empty", "overlimit"}:
        assert result.trajectory_records[-1]["termination_kind"] == "malformed"


def test_launcher_retry_preserves_only_verified_token_inputs(critic, tmp_path):
    from self_summarization_agent.iteration_launcher import _restart_staged_checkpoint
    from self_summarization_agent.separate_value import FROZEN_TARGETS

    data = [sample()]
    values, _ = critic.evaluate(data)
    stage = tmp_path / ".iteration-00001.incomplete"
    frozen = freeze_tokens(
        stage, data, values, "behavior", critic.config, "source", critic.tokenizer_hash
    )
    (stage / "partial-actor.bin").write_bytes(b"uncommitted")
    archived = _restart_staged_checkpoint(stage, retain_token_frozen=True)
    assert (archived / "partial-actor.bin").is_file()
    assert set(p.name for p in stage.iterdir()) == {
        FROZEN_TARGETS,
        Path(frozen["shard"]["file"]).parts[0],
    }
    assert (
        json.loads((stage / FROZEN_TARGETS).read_text())["digest"] == frozen["digest"]
    )
    validate_token_frozen(
        frozen, stage, data, "behavior", critic.config, "source", critic.tokenizer_hash
    )
    (stage / frozen["shard"]["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        _restart_staged_checkpoint(stage, retain_token_frozen=True)
    assert (stage / FROZEN_TARGETS).is_file()


def test_final_short_rollout_minibatch_has_unit_weight(critic):
    import math

    data = [sample("a"), sample("b", reward=-1.0), sample("c")]
    critic.optimizer = torch.optim.SGD(
        [*critic.model.parameters(), *critic.head.parameters()], lr=0.0
    )
    _, metrics = critic.fit(data)
    assert metrics["value/optimizer_step_count"] == 2
    assert metrics["value/effective_weight_mass"] == pytest.approx(2.0)
    assert metrics["value/epoch_1/target_ce"] == pytest.approx(math.log(2), abs=1e-7)


def test_authoritative_span_strata_do_not_train_copied_context():
    data = sample()
    data.collection_token_spans = [
        dict(start=0, end=2, sampled=False, kind="initial"),
        dict(start=2, end=3, sampled=True, kind="action"),
        dict(start=3, end=5, sampled=False, kind="tool_result"),
        dict(start=5, end=7, sampled=True, kind="summary"),
    ]
    before = deepcopy(data)
    metrics = token_diagnostics(
        [data], [[0.2, 0.4, 0.6]], [[0.5, 0.4, 0.3]], prefix="v"
    )
    assert metrics["v/span/research_tool_calls/token_count"] == 1
    assert metrics["v/span/summary/token_count"] == 2
    assert metrics["v/span/summary/weight_sum"] == pytest.approx(1.0)
    assert data == before
    data.collection_token_spans[-1]["kind"] = "action"
    metrics = token_diagnostics(
        [data], [[0.2, 0.4, 0.6]], [[0.5, 0.4, 0.3]], prefix="v"
    )
    assert metrics["v/span/final_answer/token_count"] == 2
    assert metrics["v/span/research_tool_calls/token_count"] == 1


def test_token_diagnostic_schedule_preserves_training_rng_and_weights(critic):
    data = [sample(), sample("b", reward=-1.0)]
    critic.config = replace(critic.config, fitting_epochs=2)
    for module in critic.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.1
    other = deepcopy(critic)
    other.config = replace(other.config, eval_before_fit=True, eval_every_n_epochs=1)
    torch.manual_seed(99)
    critic.fit(data)
    torch.manual_seed(99)
    _, metrics = other.fit(data)
    assert "value/eval_before/brier_score" in metrics
    assert "value/epoch_1/brier_score" in metrics
    for a, b in zip(
        [*critic.model.parameters(), *critic.head.parameters()],
        [*other.model.parameters(), *other.head.parameters()],
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_batched_sparse_head_matches_individual_intervals(critic):
    data = [sample(), sample('b', mask=[False, False, True, False, False, False, False])]
    before = deepcopy(data)
    prepared = critic._prepare_tokens(data)
    batched = critic._token_logits(prepared, [0, 1])
    for i in range(2):
        torch.testing.assert_close(batched[i], critic._token_logits(prepared, [i])[0])
    assert data == before
    assert prepared.prefixes[0].tolist() == data[0].collection_full_token_ids[:-1]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires multiple GPUs')
def test_parallel_token_update_matches_single_gpu(critic):
    devices = min(4, torch.cuda.device_count())
    cfg = replace(critic.config, prefix_microbatch_size=devices,
                  eval_prefix_microbatch_size=devices, rollout_minibatch_size=5)
    single = SeparateValueTrainer(critic.source, critic.source, cfg, dtype='float32',
                                  activation_checkpointing=False,
                                  advantage_estimator='token_mc_value')
    parallel = SeparateValueTrainer(critic.source, critic.source,
                                    replace(cfg, gpu_ids=list(range(devices))),
                                    dtype='float32', activation_checkpointing=False,
                                    advantage_estimator='token_mc_value')
    with torch.no_grad():
        for p in single.head.parameters():
            p.normal_(std=0.1)
    parallel.head.load_state_dict(single.head.state_dict())
    # Uneven rollout interval counts, sparse masks and a short final microbatch.
    data = [sample('a'), sample('a', '1'), sample('b', reward=-1),
            sample('c'), sample('d', reward=-1)]
    data[-1].collection_assistant_token_mask[-1] = False
    data[-1].completion_mask[-1] = False
    for trainer in (single, parallel):
        trainer.optimizer = torch.optim.SGD(
            [*trainer.model.parameters(), *trainer.head.parameters()], lr=0.01)
    before = deepcopy(data)
    expected, _ = single.fit(data)
    actual, metrics = parallel.fit(data)
    for a, b in zip([*single.model.parameters(), *single.head.parameters()],
                    [*parallel.model.parameters(), *parallel.head.parameters()]):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    for a, b in zip(actual, expected):
        assert a == pytest.approx(b, abs=2e-6)
    assert metrics['value/effective_weight_mass'] == pytest.approx(1)
    assert metrics['value/data_parallel_devices'] == devices
    assert before == data


@pytest.mark.parametrize('ids', [[0, 0], [-1], [True], '0,1'])
def test_value_gpu_ids_reject_invalid_devices(ids):
    with pytest.raises(ValueError, match='gpu_ids'):
        CompactionValueConfig(gpu_ids=ids)
