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


def test_compactionrl_reference_presets_require_complete_prepared_dataset(tmp_path):
    from self_summarization_agent.config import load_train_config
    from self_summarization_agent.benchmarks.swe_dev.compactionrl import REVISION

    for path in Path("configs/train").glob("swe_dev_*.yaml"):
        config = load_train_config(path)
        assert config.benchmark.dataset_revision == REVISION
        assert config.collection.train_task_count is None
        if config.benchmark.environment == "local-tmux":
            config.benchmark.prepared_task_root = str(tmp_path)
            with pytest.raises(ValueError, match="prepare_tmux"):
                selected_tasks(config, split="train")
            continue
        assert config.benchmark.environment == "docker"
        config.benchmark.prepared_task_root = str(tmp_path)
        with pytest.raises(ValueError, match="Missing 1105 of 1105"):
            selected_tasks(config, split="train")

    config = load_train_config("configs/train/swe_dev_grpo.yaml")
    config.benchmark.train_manifest_path = str(tmp_path / "short.json")
    Path(config.benchmark.train_manifest_path).write_text('["amalmurali47_swagroutes-2"]')
    with pytest.raises(ValueError, match="differs from the pinned"):
        selected_tasks(config, split="train")


def test_compactionrl_selects_exact_ids_and_tracks_native_task_content(tmp_path, monkeypatch):
    from hashlib import sha256
    from self_summarization_agent.config import load_train_config
    from self_summarization_agent.benchmarks.swe_dev import compactionrl

    config = load_train_config("configs/train/swe_dev_grpo.yaml")
    ids = json.loads(Path(config.benchmark.train_manifest_path).read_bytes())
    assert len(set(ids)) == len(ids) == 1105
    ids = ids[:2]
    # Exercise actual Harbor task parsing and hashing on two small task directories.
    manifest = tmp_path / "subset.json"
    manifest.write_text(json.dumps(ids))
    monkeypatch.setattr(compactionrl, "MANIFEST_SHA256", sha256(manifest.read_bytes()).hexdigest())
    config.benchmark.train_manifest_path = str(manifest)
    config.benchmark.prepared_task_root = str(tmp_path)
    for name in [*ids, "unlisted-task"]:
        directory = tmp_path / name
        (directory / "environment").mkdir(parents=True)
        (directory / "tests").mkdir()
        (directory / "tests/test.sh").write_text("exit 0")
        (directory / "task.toml").write_text('version = "1.0"')
        (directory / "instruction.md").write_text(name)
    tasks = selected_tasks(config, split="train")
    assert [t["name"] for t in tasks] == ids
    assert all(t["verifier_mode"] == "harbor-task" for t in tasks)
    assert selected_tasks(config, split="eval") == []
    verifier_path = tmp_path / ids[0] / "tests/test.sh"
    verifier_path.write_text("exit 1")
    assert selected_tasks(config, split="train")[0]["checksum"] != tasks[0]["checksum"]
    verifier_path.unlink()
    with pytest.raises(ValueError, match="Incomplete CompactionRL Harbor task"):
        selected_tasks(config, split="train")
