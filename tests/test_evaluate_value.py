import copy
import json
from types import SimpleNamespace

import pytest

from self_summarization_agent import evaluate_value as module
from self_summarization_agent.config import load_train_config
from self_summarization_agent.trajectory import RLSample
from self_summarization_agent.value_model import VALUE_HEAD_FILENAME


def setup_evaluation(monkeypatch, tmp_path):
    config = load_train_config("configs/train/compact_value_mc_head_lr.yaml")
    records = tmp_path / "heldout.jsonl"
    records.write_text(json.dumps({"query_id": "heldout", "trajectory_records": [{"query_id": "heldout"}]}) + "\n")
    checkpoint = tmp_path / "checkpoint"
    head = checkpoint / VALUE_HEAD_FILENAME
    head.parent.mkdir(parents=True)
    head.write_bytes(b"fixture checked by injected trainer")
    sample = RLSample(
        query_id="heldout", turn_id="trajectory-1", rollout_id="heldout:0", reward=1,
        prompt="", completion="", trainable_kind="final_answer",
        input_ids=[1, 2], labels=[2, 3], completion_mask=[False, True],
        reference_logprob=-1, reference_logprobs=[0, -1], state_prefix_length=1,
        collection_full_token_ids=[1, 2, 3], collection_assistant_token_mask=[False, False, True],
        state_kind="initial",
    )

    def extract(rows, **kwargs):
        assert kwargs["expected_checkpoint_id"] == "origin"
        assert kwargs["retain_critic_only_states"]
        return [sample]

    monkeypatch.setattr(module, "samples_from_rollout_rows", extract)
    args = dict(
        checkpoint_path=checkpoint, rollout_path=records, generating_checkpoint_id="origin",
        training_query_ids={"train"}, policy_eval_query_ids={"final-eval"},
    )
    return config, args, sample, records


def test_read_only_evaluation_uses_origin_labels_and_preserves_records(monkeypatch, tmp_path):
    config, args, sample, records = setup_evaluation(monkeypatch, tmp_path)
    before_sample, before_records = copy.deepcopy(sample), records.read_bytes()
    calls = []

    class Evaluator:
        def evaluate_compaction_values(self, samples, *, microbatch_size):
            calls.append(samples)
            return [0.5], [0.28768207]

        def step(self, *args):
            pytest.fail("Held-out evaluator must never train")

        def save_checkpoint(self, *args):
            pytest.fail("Held-out evaluator must never save weights")

    metrics = module.evaluate_value_checkpoint(config, **args, trainer_factory=lambda *a: Evaluator())
    assert calls == [[sample]]
    assert metrics["generating_checkpoint_id"] == "origin"
    assert metrics["value/eval_scope"] == "fixed_held_out_records"
    assert metrics["value/held_out/brier_score"] == pytest.approx(0.0625)
    assert metrics["value/held_out/initial/state_count"] == 1
    assert sample == before_sample
    assert records.read_bytes() == before_records


@pytest.mark.parametrize("partition", ["training_query_ids", "policy_eval_query_ids"])
def test_overlap_rejected_before_loading_model(monkeypatch, tmp_path, partition):
    config, args, sample, records = setup_evaluation(monkeypatch, tmp_path)
    args[partition] = {"heldout"}
    with pytest.raises(ValueError, match="overlap"):
        module.evaluate_value_checkpoint(config, **args, trainer_factory=lambda *a: pytest.fail("Must not load"))


def test_missing_exact_evidence_rejected(monkeypatch, tmp_path):
    config, args, sample, records = setup_evaluation(monkeypatch, tmp_path)
    sample.collection_full_token_ids = None
    with pytest.raises(ValueError, match="exact collected"):
        module.evaluate_value_checkpoint(config, **args, trainer_factory=lambda *a: pytest.fail("Must not load"))


def test_query_manifest_validation(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text('["q1", "q2"]')
    assert module.load_query_ids(path) == {"q1", "q2"}
    path.write_text('{"q1": true}')
    with pytest.raises(ValueError, match="JSON array"):
        module.load_query_ids(path)
