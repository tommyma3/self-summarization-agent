from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from self_summarization_agent.config import CompactionValueConfig, TrainingConfig, ModelConfig, load_train_config
from self_summarization_agent.separate_value import (
    validate_samples, validate_advantages, sample_key, separate_checkpoint_complete,
    atomic_json, CRITIC_DIR, CRITIC_STATE, PAIR_MANIFEST, CRITIC_MANIFEST,
)
from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
from self_summarization_agent.trajectory import RLSample
from self_summarization_agent.trainer import TransformersPolicyTrainer


def samples():
    return [RLSample(query_id="q", turn_id=str(i), prompt="", completion="", reward=reward,
        trainable_kind="summary" if i == 1 else "final_answer", rollout_id=rollout,
        input_ids=[1, 2 if i == 0 else 5, 3], labels=[2 if i == 0 else 5, 3, 4], completion_mask=[False, True, True],
        reference_logprob=-2., reference_logprobs=[0., -2., -2.], state_prefix_length=2,
        collection_full_token_ids=[1, 2 if i == 0 else 5, 3, 4], collection_assistant_token_mask=[False, False, True, True],
        state_kind="post_compaction" if i == 2 else "initial")
        for i, (rollout, reward) in enumerate([("a", -1.), ("b", 1.), ("b", 1.)])]


@pytest.fixture
def tiny(tmp_path):
    path = tmp_path / "base"
    config = GPT2Config(vocab_size=8, n_positions=16, n_embd=8, n_layer=1, n_head=2,
                        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
    GPT2Model(config).save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i): i for i in range(8)}, unk_token="0")), unk_token="0", pad_token="0", eos_token="1")
    tokenizer.save_pretrained(path)
    value = CompactionValueConfig(enabled=True, architecture="separate", model_path=str(path),
        fitting_epochs=2, learning_rate=0.01, head_learning_rate=0.02,
        rollout_minibatch_size=2, prefix_microbatch_size=1, max_grad_norm=10.)
    return path, config, value


def critic(tiny, **kwargs):
    path, _, value = tiny
    return SeparateValueTrainer(path, path, value, dtype="float32", device="cpu",
                                activation_checkpointing=False, **kwargs)


def test_config_preset_and_compatibility():
    config = load_train_config("configs/train/compact_separate_mc_ppo.yaml")
    assert config.training.value.separate
    assert config.training.update_epochs == 2
    assert config.training.value.fitting_epochs == 3
    assert load_train_config("configs/train/compact_value_mc.yaml").training.value.shared
    with pytest.raises(ValueError, match="model_path"):
        CompactionValueConfig(enabled=True, architecture="separate")
    with pytest.raises(ValueError, match="shared-mode-only"):
        CompactionValueConfig(enabled=True, architecture="separate", model_path="x", loss_coefficient=0.1)
    with pytest.raises(ValueError, match="supports"):
        TrainingConfig(backend="fsdp2_context_parallel", advantage_estimator="compaction_mc_value", value=config.training.value)


def test_exact_prefix_and_validation(tiny):
    trainer = critic(tiny)
    data = samples()
    before = deepcopy(data)
    values, _ = trainer.evaluate(data)
    data[0].input_ids[-1] = 6
    data[0].labels[-2:] = [6, 7]
    data[0].collection_full_token_ids[-2:] = [6, 7]
    assert trainer.evaluate(data)[0] == values
    validate_samples(data, 3)
    with pytest.raises(ValueError, match="Over-length"):
        validate_samples(data, 2)
    data[0].input_ids[0] = 7
    with pytest.raises(ValueError, match="Cached IDs"):
        validate_samples(data)
    assert before[1:] == data[1:]
    with pytest.raises(ValueError, match="exactly"):
        validate_advantages(before, {})


def test_fit_epochs_weighting_and_resume(tiny, tmp_path):
    one = critic(tiny)
    two = critic(tiny)
    two.config = replace(two.config, prefix_microbatch_size=3)
    data = samples()
    before = deepcopy(data)
    backbone_before = deepcopy(one.model.state_dict())
    values, metrics = one.fit(data)
    other_values, _ = two.fit(data)
    torch.testing.assert_close(torch.tensor(values), torch.tensor(other_values), atol=1e-6, rtol=1e-6)
    for a, b in zip(one.model.parameters(), two.model.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    assert metrics["value/optimizer_step_count"] == 2
    assert any(not torch.equal(v, one.model.state_dict()[k]) for k, v in backbone_before.items())
    assert data == before
    output = tmp_path / "iteration-00001"
    identity = one.save(output / CRITIC_DIR, output.name)
    one.model.save_pretrained(output)
    atomic_json(output / PAIR_MANIFEST, {"version": 1, "architecture": "separate",
        "checkpoint_id": output.name, "critic_identity": identity})
    assert separate_checkpoint_complete(output)
    resumed = SeparateValueTrainer(output / CRITIC_DIR, tiny[0], tiny[2], dtype="float32", device="cpu",
                                   activation_checkpointing=False, resume=True)
    assert resumed.update_count == 2
    assert resumed.evaluate(data)[0] == values
    _, next_metrics = resumed.fit(data)
    assert next_metrics["value/total_optimizer_steps"] == 4
    (output / CRITIC_DIR / CRITIC_STATE).unlink()
    assert not separate_checkpoint_complete(output)


def test_rollout_weighting_matches_reference(tiny):
    trainer = critic(tiny)
    trainer.config = replace(trainer.config, fitting_epochs=1)
    reference = deepcopy(trainer)
    data = samples()
    trainer.fit(data)
    reference.optimizer.zero_grad()
    logits = reference.logits(data)
    loss = (torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1, 1]), reduction="none")
            * torch.tensor([0.5, 0.25, 0.25])).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([*reference.model.parameters(), *reference.head.parameters()], 10.)
    reference.optimizer.step()
    for a, b in zip(trainer.head.parameters(), reference.head.parameters()):
        torch.testing.assert_close(a, b)


def make_actor(tiny, monkeypatch, microbatch=1):
    from self_summarization_agent import trainer as module
    path, config, value = tiny
    torch.manual_seed(9)
    monkeypatch.setattr(module.AutoModelForMultimodalLM, "from_pretrained", lambda *a, **kw: GPT2LMHeadModel(config))
    monkeypatch.setattr(module, "configure_tokenizer_chat_template", lambda *a: None)
    training = TrainingConfig(advantage_estimator="compaction_mc_value", value=value,
        update_epochs=2, minibatch_size=3, gradient_accumulation_microbatch_size=microbatch,
        learning_rate=0.001, max_sequence_length=16)
    return TransformersPolicyTrainer(ModelConfig(model_path=str(path), dtype="float32", device_map=None), training)


def test_actor_fixed_advantages_and_accumulation(tiny, monkeypatch):
    a = make_actor(tiny, monkeypatch)
    b = make_actor(tiny, monkeypatch, microbatch=3)
    data = samples()
    data[0].completion_mask = [True, True, True]
    data[0].collection_assistant_token_mask = [False, True, True, True]
    before = deepcopy(data)
    advantages = {sample_key(s): s.reward - 0.2 for s in data}
    assert a.value_head is None
    with pytest.raises(ValueError, match="fitted advantages"):
        a.step({"q": data})
    metrics = a.step({"q": data}, fixed_advantages=advantages)
    b.step({"q": data}, fixed_advantages=advantages)
    assert metrics.optimizer_step_count == 2
    assert metrics.mean_advantage == pytest.approx(sum(advantages.values()) / 3)
    assert data == before
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=1e-5)


def test_actor_same_reward_and_critic_only(tiny, monkeypatch):
    actor = make_actor(tiny, monkeypatch)
    data = samples()
    for sample in data:
        sample.reward = 1.
    data[0].completion_mask = [False] * 3
    data[0].collection_assistant_token_mask = [False] * 4
    metrics = actor.step({"q": data}, fixed_advantages={sample_key(s): 0.7 for s in data})
    assert metrics.optimizer_step_count == 2
    assert metrics.sample_count == 3


def test_tokenizer_mismatch(tiny, tmp_path):
    path, config, value = tiny
    other = tmp_path / "other"
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"x": 0, "y": 1}, unk_token="x")))
    tokenizer.save_pretrained(other)
    with pytest.raises(ValueError, match="mappings"):
        SeparateValueTrainer(path, other, value, dtype="float32", device="cpu")


def test_fixed_artifact_provenance(tiny, tmp_path):
    from dataclasses import asdict
    from self_summarization_agent.separate_value import (
        ADVANTAGES, digest, sample_fingerprint, load_advantages,
    )
    data = samples()
    model = critic(tiny)
    values, metrics = model.fit(data)
    identity = model.save(tmp_path / CRITIC_DIR, "iteration-00001")
    body = {"version": 1, "behavior_checkpoint_id": "base", "critic_identity": identity,
        "sample_fingerprint": sample_fingerprint(data), "config_fingerprint": digest(asdict(tiny[2])),
        "rows": [{"key": sample_key(s), "value": v, "advantage": s.reward - v} for s, v in zip(data, values)],
        "metrics": metrics}
    target = tmp_path / ADVANTAGES
    atomic_json(target, body)
    fixed, _ = load_advantages(target, data, "base", tiny[2])
    assert list(fixed.values()) == [s.reward - v for s, v in zip(data, values)]
    with pytest.raises(ValueError, match="provenance"):
        load_advantages(target, data, "other-policy", tiny[2])
    body["rows"].append(body["rows"][0])
    atomic_json(target, body)
    with pytest.raises(ValueError, match="Duplicate"):
        load_advantages(target, data, "base", tiny[2])


def test_fsdp_fixed_batches_keep_rows_and_old_logprobs(tiny):
    from self_summarization_agent.verl_ray_trainer import VerlRayPolicyTrainer
    data = samples()
    calls = []
    class Group:
        data_parallel_size = 2
        def update_actor(self, batch):
            calls.append(batch)
            return {"actor/pg_loss": 0.3, "actor/ppo_kl": 0.01, "actor/pg_clipfrac": 0.2}
    trainer = VerlRayPolicyTrainer.__new__(VerlRayPolicyTrainer)
    trainer.training_config = TrainingConfig(backend="verl_ray", advantage_estimator="compaction_mc_value",
        value=tiny[2], update_epochs=2, minibatch_size=2, target_kl=0.03)
    trainer.training_config.verl.worker_backend = "verl_fsdp"
    trainer._worker_group = Group()
    trainer.checkpoint_id = "base"
    trainer.progress_path = None
    fixed = {sample_key(s): s.reward - 0.3 for s in data}
    metrics = trainer.step({"q": data}, fixed_advantages=fixed)
    assert metrics.optimizer_step_count == 4
    assert [c.meta_info["sample_count"] for c in calls] == [2, 1, 2, 1]
    for call in calls:
        assert call.meta_info["epochs"] == 1
        if call.meta_info["sample_count"] == 1:
            assert not call.batch["response_mask"][1].any()
        for index in range(call.meta_info["sample_count"]):
            key = (call.non_tensor_batch["rollout_ids"][index], call.non_tensor_batch["turn_ids"][index])
            assert call.batch["advantages"][index].tolist() == pytest.approx([fixed[key]] * 3)
            assert call.batch["old_log_probs"][index].tolist() == [0., -2., -2.]
    calls.clear()
    trainer.training_config.target_kl = 0.001
    assert trainer.step({"q": data}, fixed_advantages=fixed).optimizer_step_count == 1


def test_distributed_loss_matches_token_mean(monkeypatch):
    from self_summarization_agent import verl_separate_actor_worker as module
    from verl.utils.metric import Metric, AggregationType
    from self_summarization_agent.verl_ray_trainer import _reduce_worker_metric
    class Batch(dict):
        def select(self, *keys):
            return self
        def to_padded_tensor(self):
            return self
    monkeypatch.setattr(module, "no_padding_2_padding", lambda output, data: output)
    logprobs = torch.tensor([[-1.1, -2.3, -0.8], [-1.3, -0.6, -1.9]], requires_grad=True)
    old = torch.full_like(logprobs, -1.)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    advantages = torch.tensor([[1., 1., 1.], [-0.4, -0.4, -0.4]])
    outputs = []
    losses = []
    for rank in range(2):
        data = Batch(response_mask=mask[rank:rank+1], old_log_probs=old[rank:rank+1],
            advantages=advantages[rank:rank+1], batch_num_tokens=3, dp_size=2)
        loss, metrics = module.fixed_advantage_ppo_loss(SimpleNamespace(clip_ratio=0.2),
            {"log_probs": logprobs[rank:rank+1]}, data)
        losses.append(loss)
        outputs.append(metrics)
    actual = sum(losses) / 2
    ratio = (logprobs - old).exp()
    expected = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)[mask].mean()
    torch.testing.assert_close(actual, expected)
    assert _reduce_worker_metric("actor/ppo_kl", [o["actor/ppo_kl"] for o in outputs], Metric,
                                 AggregationType) == pytest.approx(float((old - logprobs)[mask].mean()))


@pytest.mark.parametrize("failure_stage", [None, "update", "save"])
def test_two_iteration_orchestration_uses_post_fit_values(tiny, tmp_path, monkeypatch, failure_stage):
    import json
    from self_summarization_agent import train_step, critic_step
    from self_summarization_agent import separate_value_trainer as value_module
    from self_summarization_agent.trainer import UpdateMetrics
    from self_summarization_agent.separate_value import ADVANTAGES, load_advantages
    path, _, value = tiny
    config = load_train_config("configs/train/compact_separate_mc_ppo.yaml")
    config.model = ModelConfig(model_path=str(path), dtype="float32")
    config.training = TrainingConfig(advantage_estimator="compaction_mc_value", value=value,
                                    update_epochs=2, max_sequence_length=27000)
    data = samples()
    before = deepcopy(data)
    monkeypatch.setattr(train_step, "samples_from_rollout_rows", lambda *a, **kw: deepcopy(data))
    original = value_module.SeparateValueTrainer
    monkeypatch.setattr(value_module, "SeparateValueTrainer", lambda *a, **kw: original(*a, **{**kw, "device": "cpu"}))
    order = []
    def subprocess_run(command, **kwargs):
        args = {command[i][2:]: command[i+1] for i in range(3, len(command), 2)}
        order.append("critic")
        critic_step.run(load_train_config(args["config"]), args["checkpoint"], args["rollouts"],
                        args["output"], args["output-id"], args.get("progress"))
    monkeypatch.setattr(train_step.subprocess, "run", subprocess_run)
    failures = []
    from self_summarization_agent import trainer as actor_module
    monkeypatch.setattr(actor_module.AutoModelForMultimodalLM, "from_pretrained",
                        lambda source, **kw: GPT2LMHeadModel.from_pretrained(source))
    monkeypatch.setattr(actor_module, "configure_tokenizer_chat_template", lambda *a: None)
    class Actor(TransformersPolicyTrainer):
        def __init__(self, model_config, training_config, **kw):
            assert order[-1] == "critic"
            super().__init__(model_config, training_config, **kw)
            order.append("actor")
        def step(self, grouped_samples, *, fixed_advantages):
            if failure_stage == "update" and not failures:
                failures.append("update")
                raise RuntimeError("injected update failure")
            assert fixed_advantages != {sample_key(s): s.reward for s in data}
            assert set(fixed_advantages) == {sample_key(s) for s in data}
            return super().step(grouped_samples, fixed_advantages=fixed_advantages)
        def save_checkpoint(self, output):
            if failure_stage == "save" and not failures:
                failures.append("save")
                raise RuntimeError("injected save failure")
            super().save_checkpoint(output)
    monkeypatch.setattr(train_step, "TransformersPolicyTrainer", Actor)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("{}\n")
    for iteration in (1, 2):
        output = tmp_path / f"iteration-{iteration:05d}"
        if iteration == 1 and failure_stage:
            with pytest.raises(RuntimeError, match="injected"):
                train_step.run_train_step(config, checkpoint_path=path, rollout_path=rollouts,
                                         output_checkpoint_path=output)
            assert not (output / ".complete").exists()
        train_step.run_train_step(config, checkpoint_path=path, rollout_path=rollouts,
                                 output_checkpoint_path=output)
        assert separate_checkpoint_complete(output)
        fixed, metrics = load_advantages(output / ADVANTAGES, data, path.name, value)
        assert metrics["value/total_optimizer_steps"] == iteration * 2
        assert (output / ".complete").exists()
        path = output
    assert order == ["critic", "actor"] * (3 if failure_stage else 2)
    assert data == before
    (path / CRITIC_DIR / CRITIC_STATE).unlink()
    with pytest.raises(ValueError, match="complete matching"):
        train_step.run_train_step(config, checkpoint_path=path, rollout_path=rollouts,
                                 output_checkpoint_path=tmp_path / "iteration-00003")


def test_resume_rejects_uncommitted_pair(tiny, tmp_path):
    from self_summarization_agent.separate_value import validate_resume_checkpoint
    assert validate_resume_checkpoint(tiny[0]) is False
    output = tmp_path / 'iteration-00001'
    model = critic(tiny)
    identity = model.save(output / CRITIC_DIR, output.name)
    model.model.save_pretrained(output)
    atomic_json(output / PAIR_MANIFEST, {"version": 1, "architecture": "separate",
        "checkpoint_id": output.name, "critic_identity": identity})
    assert separate_checkpoint_complete(output)
    with pytest.raises(ValueError, match="complete matching"):
        validate_resume_checkpoint(output)


@pytest.mark.parametrize("compactions,malformed", [(0, False), (1, False), (2, False), (1, True)])
def test_runtime_intervals_remain_immutable_during_fixed_update(compactions, malformed):
    # Reuse the exact-token runtime fixture: its tokenizer rejects history re-encoding.
    from test_tito import TokenModel, runtime, SEARCH, SUMMARY, FINISH
    from self_summarization_agent.trajectory import build_rollout_native_training_cache, extract_training_samples
    from self_summarization_agent.separate_value import sample_key
    outputs = [SEARCH, SUMMARY] * compactions + [FINISH]
    if malformed:
        outputs = [SEARCH, "</think>bad summary<|im_end|>"]
    result = runtime(TokenModel(outputs), context_threshold_tokens=1 if compactions else 15000).run("q", "question")
    for record in result.trajectory_records:
        record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
    rewards = {r["turn_id"]: -1. if malformed else 1. for r in result.trajectory_records}
    data = extract_training_samples(result.trajectory_records, rewards, rollout_id="q:0")
    assert len(data) == (1 if malformed else compactions + 1)
    before = deepcopy((result.trajectory_records, data))
    class TokenOnlyActor(TransformersPolicyTrainer):
        def _sequence_token_logprobs_and_mask(self, samples):
            assert len(samples) == 1
            s = samples[0]
            return self.model.weight.expand(1, len(s.input_ids)), torch.tensor([s.completion_mask])
        def _model_device(self):
            return torch.device("cpu")
    actor = TokenOnlyActor.__new__(TokenOnlyActor)
    actor.training_config = TrainingConfig(advantage_estimator="compaction_mc_value",
        value=CompactionValueConfig(enabled=True, architecture="separate", model_path="unused"),
        update_epochs=2, minibatch_size=2)
    actor.model = torch.nn.Linear(1, 1, bias=False)
    actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.01)
    metrics = actor.step({"q": data}, fixed_advantages={sample_key(s): s.reward - 0.1 for s in data})
    assert metrics.optimizer_step_count >= 2
    assert (result.trajectory_records, data) == before


def test_publication_requires_actor_weights(tiny, tmp_path):
    from self_summarization_agent.iteration_launcher import _promote_staged_checkpoint
    from self_summarization_agent.checkpoints import mark_checkpoint_complete
    model = critic(tiny)
    staged = tmp_path / ".iteration-00001.incomplete"
    identity = model.save(staged / CRITIC_DIR, "iteration-00001")
    atomic_json(staged / PAIR_MANIFEST, {"version": 1, "architecture": "separate",
        "checkpoint_id": "iteration-00001", "critic_identity": identity})
    mark_checkpoint_complete(staged)
    destination = tmp_path / "iteration-00001"
    assert not separate_checkpoint_complete(staged)
    with pytest.raises(RuntimeError, match="complete checkpoint"):
        _promote_staged_checkpoint(staged, destination, require_value_head=False, require_separate_critic=True)
    assert not destination.exists()
    model.model.save_pretrained(staged)
    assert separate_checkpoint_complete(staged)
    _promote_staged_checkpoint(staged, destination, require_value_head=False, require_separate_critic=True)
    assert separate_checkpoint_complete(destination)
    assert not staged.exists()
