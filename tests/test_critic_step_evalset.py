import json
import math
from copy import deepcopy
from dataclasses import replace

import pytest

from self_summarization_agent import critic_step
from self_summarization_agent import separate_value_trainer as separate_value_trainer_module
from self_summarization_agent.config import CompactionValueConfig, load_train_config
from self_summarization_agent.separate_value import ADVANTAGES
from self_summarization_agent.trajectory import build_rollout_native_training_cache

SEARCH = "</think>\n<search>q</search><|im_end|>"
SUMMARY = "</think>\n<summary>evidence</summary><|im_end|>"
FINISH = "</think>\n<answer>done</answer><|im_end|>"


def rollout_row(query_id, policy_id, reward, *, compaction):
    from test_tito import TokenModel, runtime

    outputs = [SEARCH, SUMMARY] * compaction + [FINISH]
    result = runtime(TokenModel(outputs), context_threshold_tokens=1 if compaction else 15000).run(query_id, "question")
    records = result.trajectory_records
    turn_rewards = {record["turn_id"]: float(reward) for record in records}
    return {
        "policy_checkpoint_id": policy_id,
        "query_id": query_id,
        "rollout_index": 0,
        "turn_records": [],
        "trajectory_records": records,
        "turn_rewards": turn_rewards,
        "trainable_sample_count": len(records),
    }


def attach_native_caches(row):
    row = deepcopy(row)
    for record in row["trajectory_records"]:
        record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
    return row


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ConstantCritic:
    """CPU stand-in recording call order; constant zero values, ln(2) per-state loss."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def fit(self, samples, *, max_sequence_length=None):
        self.calls.append("fit")
        return [0.0] * len(samples), {"value/fit_marker": 1.0}

    def save(self, path, checkpoint_id):
        self.calls.append("save")
        return "identity"

    def evaluate(self, samples):
        self.calls.append("evaluate")
        return [0.0] * len(samples), [math.log(2)] * len(samples)


def separate_config(tmp_path):
    config = load_train_config("configs/train/compact_separate_mc_ppo.yaml")
    value = CompactionValueConfig(enabled=True, architecture="separate",
        model_path=str(tmp_path / "unused-base"), fitting_epochs=1)
    config.training.value = value
    config.training.max_sequence_length = 512
    config.model = replace(config.model, dtype="float32")
    return config


def test_evalset_metrics_on_same_policy_eval_states(tmp_path):
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    row_a = rollout_row("q-a", "iteration-00000", 1.0, compaction=1)
    row_b = rollout_row("q-b", "iteration-00000", -1.0, compaction=0)
    eval_file = rollouts / "iteration-00000.eval.jsonl"
    write_jsonl(eval_file, [row_a, row_b])
    before = eval_file.read_bytes()

    metrics = critic_step.evaluate_on_eval_collection(ConstantCritic(), rollouts / "iteration-00001.jsonl", "iteration-00000")

    assert metrics["value/evalset/policy_id"] == "iteration-00000"
    assert metrics["value/evalset/data_policy_id"] == "iteration-00000"
    assert metrics["value/evalset/eval_scope"] == "same_policy_eval_collection"
    assert metrics["value/evalset/native_cache_fail_count"] == 0
    assert metrics["value/evalset/state_count"] == 3
    assert metrics["value/evalset/rollout_count"] == 2
    assert metrics["value/evalset/initial/state_count"] == 2
    assert metrics["value/evalset/post_compaction/state_count"] == 1
    # Constant zero value: correct exactly on successful rollouts.
    assert metrics["value/evalset/classification_accuracy"] == pytest.approx(0.5)
    assert metrics["value/evalset/bce"] == pytest.approx(math.log(2))
    assert metrics["value/evalset/explained_variance"] == pytest.approx(0.0)
    # In-memory only: the eval collection on disk must stay cache-free.
    assert eval_file.read_bytes() == before
    assert "training_cache" not in json.loads(before.splitlines()[0])["trajectory_records"][0]


def test_evalset_records_skip_reasons_and_policy_filter(tmp_path):
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    missing = critic_step.evaluate_on_eval_collection(ConstantCritic(), rollouts / "x.jsonl", "iteration-00000")
    assert missing["value/evalset_skipped"] == "missing_eval_rollouts"

    write_jsonl(rollouts / "iteration-00000.eval.jsonl", [rollout_row("q-a", "iteration-00001", 1.0, compaction=0)])
    filtered = critic_step.evaluate_on_eval_collection(ConstantCritic(), rollouts / "x.jsonl", "iteration-00000")
    assert filtered["value/evalset_skipped"] == "no_rows_for_policy"


def test_evalset_drops_records_with_failed_native_cache_and_counts_them(tmp_path):
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    row = rollout_row("q-a", "iteration-00000", 1.0, compaction=1)
    broken = deepcopy(row)
    broken["query_id"] = "q-b"
    broken["trajectory_records"] = [deepcopy(row["trajectory_records"][0])]
    broken["trajectory_records"][0]["collection_tokens"] = {"version": 0}
    broken["turn_rewards"] = {broken["trajectory_records"][0]["turn_id"]: 1.0}
    write_jsonl(rollouts / "iteration-00000.eval.jsonl", [row, broken])

    metrics = critic_step.evaluate_on_eval_collection(ConstantCritic(), rollouts / "x.jsonl", "iteration-00000")

    assert "value/evalset_skipped" not in metrics
    assert metrics["value/evalset/native_cache_fail_count"] == 1
    assert metrics["value/evalset/state_count"] == 2
    assert metrics["value/evalset/rollout_count"] == 1


def test_critic_step_run_records_evalset_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(separate_value_trainer_module, "SeparateValueTrainer", ConstantCritic)
    config = separate_config(tmp_path)
    checkpoint = tmp_path / "iteration-00000"
    checkpoint.mkdir()
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    train_row = attach_native_caches(rollout_row("q-train", "iteration-00000", 1.0, compaction=1))
    write_jsonl(rollouts / "iteration-00001.jsonl", [train_row])
    eval_rows = [rollout_row("q-eval-1", "iteration-00000", 1.0, compaction=1),
                 rollout_row("q-eval-2", "iteration-00000", -1.0, compaction=0)]
    write_jsonl(rollouts / "iteration-00000.eval.jsonl", eval_rows)
    output = tmp_path / "iteration-00001"

    critic_step.run(config, checkpoint, rollouts / "iteration-00001.jsonl", output, output.name)

    body = json.loads((output / ADVANTAGES).read_text())
    metrics = body["metrics"]
    assert metrics["value/fit_marker"] == 1.0
    assert metrics["value/evalset/classification_accuracy"] == pytest.approx(0.5)
    assert metrics["value/evalset/bce"] == pytest.approx(math.log(2))
    assert metrics["value/evalset/policy_id"] == "iteration-00000"
    assert metrics["value/evalset/state_count"] == 3
    # Advantage rows still cover exactly the train-collection intervals.
    assert len(body["rows"]) == 2
    assert body["behavior_checkpoint_id"] == "iteration-00000"
