from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from swe_dev_fixtures import pilot_config, FixturePolicy, action, outputs
from test_tito import SUMMARY
from self_summarization_agent.benchmarks.swe_dev.local_fixture import FIX_COMMAND
from self_summarization_agent.benchmarks.terminal_bench.collection import collect_terminal_rollouts
from self_summarization_agent.benchmarks.terminal_bench.verification import apply_verifier_reward
from self_summarization_agent.benchmarks.swe_dev.selection import select_training_rows, validate_attempt_keys
from self_summarization_agent.cache_step import run_cache_step
from self_summarization_agent.value_model import (expected_binary_value, expected_categorical_value,
    reward_class_indices, two_hot_targets, value_support, CompactionValueHead, save_value_head, load_value_head)
from self_summarization_agent.value_diagnostics import prediction_metrics
from self_summarization_agent.segment_returns import segment_gae
from self_summarization_agent.iteration_launcher import _has_complete_raw_rollouts


@pytest.mark.parametrize("summary,status", [
    ("</think><summary></summary><|im_end|>", "empty_summary"),
    ("</think>missing summary<|im_end|>", "malformed_tool_call"),
    ("</thinking><summary>good</summary><|im_end|>", "malformed_tool_call"),
    ("</think><summary>" + "x" * 101 + "</summary><|im_end|>", "summary_length_exceeded"),
    ("</think><summary>good</summary>", "malformed_tool_call"),
])
def test_failed_summary_retains_raw_and_binary_penalty_is_trainable(tmp_path, summary, status):
    cfg, base = pilot_config(tmp_path)
    model = FixturePolicy([action(FIX_COMMAND), summary, *list(outputs())[5:]], raw_logprobs=True)
    raw = tmp_path / "raw.jsonl"
    collect_terminal_rollouts(cfg, checkpoint_path=base, output_path=raw, generator=model, split="train")
    collected = [json.loads(line) for line in raw.read_text().splitlines()]
    row = collected[0]
    assert row["status"] == status and row["benchmark_passed"]
    assert set(row["turn_rewards"].values()) == {0.}
    assert row["trajectory_records"][-1]["completion"].endswith(summary)
    assert len(row["trajectory_records"]) == 1
    selected, report = select_training_rows(collected, cfg)
    assert report["selected_groups"] == 1 and selected[0]["training_selection"]["selected"]
    before = deepcopy(collected)
    collected[1]["benchmark_verifier"]["exception"] = {"exception_type": "host_failure"}
    collected[1] = apply_verifier_reward(collected[1])
    selected, report = select_training_rows(collected, cfg)
    assert report["selected_groups"] == 0 and all(r["trainable_sample_count"] == 0 for r in selected)
    assert before[0]["trajectory_records"] == selected[0]["trajectory_records"]
    # An infrastructure failure excludes the group rather than imputing zero.
    assert not selected[1]["training_eligible"] and selected[1]["turn_rewards"] == {}
    raw.write_text("".join(json.dumps(r) + "\n" for r in collected))
    with pytest.raises(ValueError, match="No complete eligible"):
        run_cache_step(cfg, checkpoint_path=base, rollout_path=raw, output_path=tmp_path / "cached.jsonl")


def test_complete_attempt_keys_rewrites_and_native_cache_resume(tmp_path):
    cfg, base = pilot_config(tmp_path)
    raw, cached = tmp_path / "raw.jsonl", tmp_path / "cached.jsonl"
    collect_terminal_rollouts(cfg, checkpoint_path=base, output_path=raw,
                             generator=FixturePolicy(outputs(), raw_logprobs=True), split="train")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert _has_complete_raw_rollouts(raw, checkpoint_id=base.name, expected_count=4)
    assert not validate_attempt_keys(rows[:-1], complete=True)
    with pytest.raises(ValueError, match="Duplicate"): validate_attempt_keys([*rows[:-1], rows[0]], complete=True)
    selected, report = select_training_rows(rows[:-1], cfg)
    assert report["selected_groups"] == 0
    rewrite = deepcopy(rows)
    rewrite[0]["status"] = "history_rewrite_detected"
    assert select_training_rows(rewrite, cfg)[1]["selected_groups"] == 0
    run_cache_step(cfg, checkpoint_path=base, rollout_path=raw, output_path=cached)
    native = [json.loads(line) for line in cached.read_text().splitlines()]
    for row in native:
        for record in row["trajectory_records"]:
            cache, tokens = record["training_cache"], record["collection_tokens"]
            assert cache["reference_logprob_source"] == "vllm_raw_rollout"
            assert cache["input_ids"] == tokens["full_token_ids"][:-1]
            assert cache["labels"] == tokens["full_token_ids"][1:]
            assert cache["completion_mask"] == tokens["assistant_token_mask"][1:]
    run_cache_step(cfg, checkpoint_path=base, rollout_path=raw, output_path=cached, resume=True)
    # Same behavior checkpoint but different rewards must not reuse cached views.
    rows[0]["benchmark_verifier"]["rewards"] = {"reward": 0}
    rows[0] = apply_verifier_reward(rows[0])
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="cached evidence/rewards/selection changed"):
        run_cache_step(cfg, checkpoint_path=base, rollout_path=raw, output_path=cached, resume=True)


def test_binary_support_diagnostics_and_resume_rejection(tmp_path):
    profile = "binary_zero_one"
    logits = torch.zeros((2, 2))
    torch.testing.assert_close(expected_binary_value(logits, profile), torch.tensor([.5, .5]))
    assert reward_class_indices(torch.tensor([0., 1.]), profile).tolist() == [0, 1]
    with pytest.raises(ValueError): reward_class_indices(torch.tensor([-1., 1.]), profile)
    assert value_support(51, profile)[0] == 0 and value_support(51, profile)[-1] == 1
    targets = two_hot_targets(torch.tensor([0., .31, 1.]), 51, profile)
    torch.testing.assert_close(targets @ torch.tensor(value_support(51, profile)), torch.tensor([0., .31, 1.]))
    assert expected_categorical_value(torch.zeros(1, 51), profile).item() == pytest.approx(.5)
    metrics = prediction_metrics([0., 1.], [.5, .5], [.693, .693], ["a", "b"], reward_profile=profile)
    assert metrics["brier_score"] == .25 and metrics["classification_accuracy"] == .5
    head = CompactionValueHead(4, reward_profile=profile)
    save_value_head(head, tmp_path)
    assert load_value_head(tmp_path, hidden_size=4, zero_initialize=True, reward_profile=profile)[1]
    with pytest.raises(ValueError, match="manifest"): load_value_head(tmp_path, hidden_size=4, zero_initialize=True)


def test_binary_gae_hand_recurrence_mc_and_single_interval():
    chain = [dict(key=["rollout", str(i)], terminal=i == 2, reward=1.) for i in range(3)]
    old = {("rollout", str(i)): value for i, value in enumerate([.2, .7, .4])}
    result = segment_gae([chain], old, .5, "binary_zero_one")
    assert [result[tuple(item["key"])]["target"] for item in chain] == pytest.approx([.7, .7, 1.])
    assert [result[tuple(item["key"])]["advantage"] for item in chain] == pytest.approx([.5, 0., .6])
    assert all(r["target"] == pytest.approx(1.) for r in segment_gae([chain], old, 1., "binary_zero_one").values())
    failed = [[dict(key=["a", "0"], terminal=True, reward=0.)]]
    assert segment_gae(failed, {("a", "0"): .8}, .95, "binary_zero_one")[("a", "0")]["advantage"] == -.8
