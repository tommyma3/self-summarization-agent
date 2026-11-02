from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest
import torch
from transformers import GPT2LMHeadModel

from swe_dev_fixtures import pilot_config, FixturePolicy, outputs
from self_summarization_agent.benchmarks.terminal_bench.collection import collect_terminal_rollouts
from self_summarization_agent.benchmarks.swe_dev.checkpoint import validate_checkpoint
from self_summarization_agent.cache_step import run_cache_step
from self_summarization_agent.judge_step import judge_rollouts
from self_summarization_agent.train_step import run_train_step, samples_from_rollout_rows
from self_summarization_agent.trainer import TransformersPolicyTrainer
from self_summarization_agent.trajectory import extract_trainable_samples
from self_summarization_agent.separate_value import separate_checkpoint_complete, CRITIC_DIR, CRITIC_STATE


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("compactions", [0, 1, 2])
def test_executable_fixture_collection_verification_tokens_and_resume(tmp_path, compactions):
    cfg, base = pilot_config(tmp_path, compactions=compactions)
    raw = tmp_path / "raw.jsonl"
    policy = FixturePolicy(outputs(compactions), raw_logprobs=True)
    collect_terminal_rollouts(cfg, checkpoint_path=base, output_path=raw, generator=policy, split="train")
    collected = rows(raw)
    assert [r["benchmark_passed"] for r in collected] == [True, False, False, True]
    assert all(r["swe_metadata"]["fixture_only"] for r in collected)
    assert all(set(r["turn_rewards"].values()) == {float(r["benchmark_passed"])} for r in collected)
    checker = runpy.run_path("scripts/verify_tito_records.py")
    for row in collected:
        assert checker["check_successor_prefix"](row["trajectory_records"], policy.tokenizer) == []
        for i, record in enumerate(row["trajectory_records"]):
            assert checker["check_interval"](record, i, 2) == []
    # Real candidate state survives all model context boundaries.
    first = Path(collected[0]["swe_metadata"]["candidate"]["snapshot"]["path"])
    assert (first / "names.py").read_text().endswith("return value.strip()\n")
    collect_terminal_rollouts(cfg, checkpoint_path=base, output_path=raw, split="train", resume=True)
    assert rows(raw) == collected
    cfg.benchmark.max_output_chars += 1
    with pytest.raises(ValueError, match="changed"):
        collect_terminal_rollouts(cfg, checkpoint_path=base, output_path=raw, split="train", resume=True)


@pytest.mark.parametrize("mode", ["grpo", "value_mc", "separate_mc_ppo", "separate_gae_ppo"])
def test_four_profiles_rescore_update_and_checkpoint_resume(tmp_path, monkeypatch, mode):
    cfg, base = pilot_config(tmp_path, mode)
    # The fixture uses a real tiny causal LM; production uses the configured Qwen model.
    import self_summarization_agent.trainer as module
    monkeypatch.setattr(module, "AutoModelForMultimodalLM", GPT2LMHeadModel)
    # Keep process isolation in production, but run the real critic phase on CPU
    # in this test without requiring a GPU or spawning an unpatched model loader.
    import self_summarization_agent.train_step as step
    from self_summarization_agent import critic_step
    from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
    real_run = step.subprocess.run
    def critic_process(command, **kwargs):
        if isinstance(command, list) and "self_summarization_agent.critic_step" in command:
            values = {command[i]: command[i+1] for i in range(3, len(command), 2)}
            from self_summarization_agent.config import load_train_config
            critic_step.run(load_train_config(values["--config"]), values["--checkpoint"],
                values["--rollouts"], values["--output"], values["--output-id"])
            return
        return real_run(command, **kwargs)
    monkeypatch.setattr(step.subprocess, "run", critic_process)
    original_init = SeparateValueTrainer.__init__
    def cpu_init(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return original_init(self, *args, **kwargs)
    monkeypatch.setattr(SeparateValueTrainer, "__init__", cpu_init)

    checkpoint = base
    prior_actor_steps = 0
    for iteration in range(2):
        raw, judged, cached = [tmp_path / f"{iteration}-{kind}.jsonl" for kind in ["raw", "judged", "cached"]]
        policy = FixturePolicy(outputs(), raw_logprobs=False)
        collect_terminal_rollouts(cfg, checkpoint_path=checkpoint, output_path=raw, generator=policy, split="train")
        judge_rollouts(cfg, checkpoint_path=checkpoint, rollout_path=raw, output_path=judged, split="train")
        before = deepcopy(rows(judged))
        # This frozen scorer computes probabilities of exact saved IDs, never tokenizes history.
        scorer = TransformersPolicyTrainer(replace(cfg.model, model_path=str(checkpoint)), cfg.training)
        run_cache_step(cfg, checkpoint_path=checkpoint, rollout_path=judged, output_path=cached, scorer=scorer)
        cached_rows = rows(cached)
        for old, new in zip(before, cached_rows):
            for a, b in zip(old["trajectory_records"], new["trajectory_records"]):
                assert a == {k: v for k, v in b.items() if k != "training_cache"}
                assert b["training_cache"]["reference_logprob_source"] == "policy_rescore"
        samples = samples_from_rollout_rows(cached_rows, expected_checkpoint_id=checkpoint.name)
        assert {s.reward for s in samples} == {0., 1.}
        assert len({s.rollout_id for s in samples}) == 4
        target = tmp_path / f"iteration-{iteration+1:05d}"
        metric = tmp_path / f"metrics-{iteration}.json"
        run_train_step(cfg, checkpoint_path=checkpoint, rollout_path=cached,
            output_checkpoint_path=target, metrics_path=metric)
        metrics = json.loads(metric.read_text())
        assert metrics["optimizer_step_count"] > 0 and torch.isfinite(torch.tensor(metrics["loss"]))
        assert (target / "actor_training_state.pt").is_file()
        state = torch.load(target / "actor_training_state.pt", weights_only=True)
        steps = [v["step"].item() for v in state["optimizer"]["state"].values()]
        assert min(steps) == prior_actor_steps + metrics["optimizer_step_count"]
        prior_actor_steps = min(steps)
        from safetensors.torch import load_file
        old_weights = load_file(str(checkpoint / "model.safetensors"))
        new_weights = load_file(str(target / "model.safetensors"))
        assert any(not torch.equal(old_weights[k], v) for k, v in new_weights.items())
        if cfg.training.value.separate:
            assert separate_checkpoint_complete(target)
            critic_state = torch.load(target / CRITIC_DIR / CRITIC_STATE, weights_only=True)
            assert critic_state["update_count"] == iteration + 1
        validate_checkpoint(target, cfg)
        checkpoint = target
    cfg.runtime.max_summary_tokens += 1
    with pytest.raises(ValueError, match="contract changed"):
        validate_checkpoint(checkpoint, cfg)
