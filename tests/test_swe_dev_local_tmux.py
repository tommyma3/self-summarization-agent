"""Exercise the tmux transport and real pytest grading without a network or model download."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from swe_dev_fixtures import FixturePolicy, action, pilot_config
from test_terminal_bench import COMPLETE
from test_tito import SUMMARY

from self_summarization_agent.benchmarks.swe_dev import local_tmux, tmux_tasks
from self_summarization_agent.benchmarks.terminal_bench.collection import (
    collect_terminal_rollouts,
)


@pytest.mark.skipif(not shutil.which("tmux"), reason="tmux executable required")
@pytest.mark.parametrize("compact", [False, True])
def test_tmux_collector_grades_final_workspace(tmp_path, monkeypatch, compact):
    cfg, checkpoint = pilot_config(tmp_path, compactions=int(compact))
    cfg.training.group_size = 2
    cfg.benchmark.environment = "local-tmux"
    cfg.benchmark.dataset = "compactionrl-swe-dev"
    cfg.benchmark.eval_manifest_path = None
    cfg.benchmark.prepared_task_root = str(tmp_path)
    # The test runner's interpreter and tmp_path can be private to its Unix account.
    for module in (local_tmux, tmux_tasks):
        monkeypatch.setattr(module, "worker_command", lambda args, *unused: args)
        monkeypatch.setattr(module, "writable_workspace", lambda *unused: None)
    bundle = tmp_path / "bundle"
    base = bundle / "base"
    base.mkdir(parents=True)
    (base / "feature.py").write_text("def value(): return 0\n")
    (bundle / "venv").symlink_to(sys.prefix, target_is_directory=True)
    trusted = bundle / "trusted"
    trusted.mkdir()
    (trusted / "test_feature.py").write_text(
        "from feature import value\ndef test_value(): assert value() == 1\ndef test_type(): assert isinstance(value(), int)\n"
    )
    (bundle / "spec.json").write_text(
        json.dumps(
            {
                "problem_statement": "Make value return 1.",
                "FAIL_TO_PASS": ["test_feature.py::test_value"],
                "PASS_TO_PASS": ["test_feature.py::test_type"],
            }
        )
    )
    task = {
        "name": "tiny",
        "path": str(bundle),
        "checksum": "tiny",
        "manifest_digest": "test",
        "verifier_mode": "tmux-pytest",
        "image": None,
        "fixture_only": False,
        "source": {},
    }
    monkeypatch.setattr(tmux_tasks, "load_tasks", lambda _: [task])
    generations = []
    for cmd in ("cat feature.py\n", "printf 'def value(): return 1\\n' > feature.py\n"):
        generations.extend(
            [action(cmd), SUMMARY, COMPLETE]
            if compact
            else [action(cmd), COMPLETE, COMPLETE]
        )
    raw = tmp_path / "raw.jsonl"
    collect_terminal_rollouts(
        cfg,
        checkpoint_path=checkpoint,
        output_path=raw,
        split="train",
        generator=FixturePolicy(generations, raw_logprobs=True),
    )
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert [r["benchmark_verifier"]["rewards"]["reward"] for r in rows] == [0, 1]
    assert [r["benchmark_passed"] for r in rows] == [False, True]
    assert all(
        r["training_eligible"] and r["judge"]["source"] == "tmux-pytest" for r in rows
    )
    assert all(len(r["trajectory_records"]) == 1 + int(compact) for r in rows)
    assert (base / "feature.py").read_text() == "def value(): return 0\n"
    assert (
        (Path(rows[1]["swe_metadata"]["candidate"]["snapshot"]["path"]) / "feature.py")
        .read_text()
        .endswith("return 1\n")
    )


def test_validated_pool_detects_bundle_and_dependency_drift(tmp_path):
    from self_summarization_agent.benchmarks.swe_dev.dataset import tree_digest
    from self_summarization_agent.config import load_train_config

    cfg = load_train_config("configs/train/swe_dev_tmux_grpo.yaml").benchmark
    cfg.prepared_task_root = str(tmp_path)
    name = "en0_dnry-config-4"
    bundle = tmp_path / name
    for part in ("base", "trusted", "venv/bin"):
        (bundle / part).mkdir(parents=True)
    (bundle / "venv/bin/python").symlink_to(sys.executable)
    package = bundle / "venv/lib/python3.9/site-packages/sample_module-1.0.dist-info"
    package.mkdir(parents=True)
    (bundle / "packages.txt").write_text("sample-module==1.0\n")
    tmux_tasks.write_json(
        bundle / "spec.json",
        {
            "instance_id": "en0__dnry-config-4",
            "revision": "test",
            "repo": "en0/dnry-config",
            "base_commit": "1" * 40,
            "FAIL_TO_PASS": ["regression"],
            "PASS_TO_PASS": ["existing"],
        },
    )
    tmux_tasks.write_json(
        bundle / "validation.json",
        [
            {
                "baseline": {
                    "reward": 0,
                    "outcomes": {"regression": "failed", "existing": "passed"},
                },
                "oracle": {
                    "reward": 1,
                    "outcomes": {"regression": "passed", "existing": "passed"},
                },
            }
        ]
        * 2,
    )
    record = {p: tree_digest(bundle / p) for p in ("base", "trusted")}
    record.update(
        {
            p: tmux_tasks.file_hash(bundle / p)
            for p in ("spec.json", "packages.txt", "validation.json")
        }
    )
    tmux_tasks.write_json(
        tmp_path / "validated.json", {name: dict(record, directory=name)}
    )
    assert [t["name"] for t in tmux_tasks.load_tasks(cfg)] == [name]
    package.rmdir()
    with pytest.raises(ValueError, match="dependencies changed"):
        tmux_tasks.load_tasks(cfg)
    package.mkdir()
    (bundle / "base/edited").touch()
    with pytest.raises(ValueError, match="source or verifier changed"):
        tmux_tasks.load_tasks(cfg)
