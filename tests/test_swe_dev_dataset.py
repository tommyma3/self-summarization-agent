from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_dev_fixtures import pilot_config
from self_summarization_agent.benchmarks.swe_dev.audit import audit_rows
from self_summarization_agent.benchmarks.swe_dev.dataset import task_manifest, benchmark_identity
from self_summarization_agent.benchmarks.harbor_common.dataset import selected_tasks
from self_summarization_agent.collection_contract import collection_profile_id
from self_summarization_agent.iteration_launcher import _expected_train_rollout_count, _expected_eval_rollout_count


def test_audit_counts_duplicates_and_never_infers_from_transcripts():
    row = dict(input=[dict(role="user", content="repo=made-up base=0123 tests=all")],
               instance_id="", **{"from": "swe-dev"})
    result = audit_rows([row, row, dict(row, instance_id="issue-1")])
    assert result["rows"] == 3 and result["duplicate_transcripts"] == 2
    assert result["blank_instance_ids"] == 2 and result["nonblank_origin_instance_pairs"] == 1
    assert result["complete_explicit_specifications"] == result["validated_upstream_tasks"] == 0
    assert result["missing_assets"]["missing_base_commit"] == 3


def test_manifest_validation_counts_and_content_lineage(tmp_path):
    config, checkpoint = pilot_config(tmp_path)
    tasks = task_manifest(config.benchmark, split="train")
    assert len(tasks) == 1 and tasks[0]["fixture_only"]
    assert task_manifest(config.benchmark, split="eval") == []
    assert _expected_train_rollout_count(config) == 4
    assert _expected_eval_rollout_count(config) == 0
    identity = collection_profile_id(config, checkpoint)
    # Changing image/verifier metadata also changes collection identity even
    # when model-visible instructions remain unchanged.
    path = Path(config.benchmark.train_manifest_path)
    body = json.loads(path.read_text())
    body["tasks"][0]["verifier"]["limits"]["memory_mb"] = 1024
    path.write_text(json.dumps(body))
    assert collection_profile_id(config, checkpoint) != identity


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "Duplicate"), ("blank_task", "namespaced"),
    ("unrecovered_source", "source metadata"), ("verifier_digest", "verifier digest"),
    ("bundle_digest", "bundle digest"), ("validation_digest", "validation report digest"),
    ("build_digest", "build context digest"), ("revision", "revision"),
])
def test_manifest_rejects_unrecoverable_or_changed_tasks(tmp_path, mutation, match):
    config, _ = pilot_config(tmp_path)
    path = Path(config.benchmark.train_manifest_path)
    body = json.loads(path.read_text())
    task = body["tasks"][0]
    if mutation == "duplicate": body["tasks"].append(deepcopy(task))
    elif mutation == "blank_task": task["task_id"] = ""
    elif mutation == "unrecovered_source": task["source"] = dict(kind="upstream", origin="swe-dev", instance_id="")
    elif mutation == "revision": body["dataset_revision"] = "changed"
    else:
        key = {"verifier_digest": None, "bundle_digest": "bundle_digest",
               "validation_digest": "validation_digest", "build_digest": "build_context_digest"}[mutation]
        if key: task[key] = "bad"
        else: task["verifier"]["digest"] = "bad"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=match): task_manifest(config.benchmark, split="train")


def test_manifests_reject_task_and_content_overlap_and_missing_eval(tmp_path):
    config, _ = pilot_config(tmp_path)
    body = json.loads(Path(config.benchmark.train_manifest_path).read_text())
    body["split"] = "eval"
    Path(config.benchmark.eval_manifest_path).write_text(json.dumps(body))
    with pytest.raises(ValueError, match="Overlapping"): task_manifest(config.benchmark, split="train")
    config.benchmark.eval_manifest_path = None
    with pytest.raises(ValueError, match="explicit"): task_manifest(config.benchmark, split="train")


def test_eval_membership_and_order_ignore_iteration_seed(monkeypatch):
    import self_summarization_agent.benchmarks.harbor_common.dataset as shared
    monkeypatch.setattr(shared, "task_manifest", lambda *_args, **_kw: [dict(name=str(i)) for i in range(10)])
    cfg = SimpleNamespace(benchmark=SimpleNamespace(), experiment=SimpleNamespace(seed=3),
        dataset=SimpleNamespace(shuffle=True, offset=1, limit=8),
        collection=SimpleNamespace(train_task_count=4, eval_task_count=4))
    assert selected_tasks(cfg, split="eval", sample_seed=0) == selected_tasks(cfg, split="eval", sample_seed=999)
    assert selected_tasks(cfg, split="train", sample_seed=0) != selected_tasks(cfg, split="train", sample_seed=999)
