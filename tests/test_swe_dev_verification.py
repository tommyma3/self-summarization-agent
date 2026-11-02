from pathlib import Path
import json
import shutil

import pytest

from swe_dev_fixtures import pilot_config
from self_summarization_agent.benchmarks.swe_dev.dataset import task_manifest, tree_digest
from self_summarization_agent.benchmarks.swe_dev.verification import (
    export_candidate, snapshot_candidate, verify_local_fixture, CandidateError, _result)


def test_export_includes_untracked_binary_modes_and_deletions(tmp_path):
    base, repo = tmp_path / "base", tmp_path / "repo"
    base.mkdir(); repo.mkdir()
    (base / "deleted.py").write_text("old")
    (base / "same.py").write_text("same")
    (repo / "same.py").write_text("same")
    (repo / "added.bin").write_bytes(bytes(range(256)))
    (repo / "executable").write_text("#!/bin/sh\nexit 0\n")
    (repo / "executable").chmod(0o755)
    before = tree_digest(base)
    artifact = export_candidate(repo, base, tmp_path / "export")
    snapshot = Path(artifact["snapshot"]["path"])
    assert not (snapshot / "deleted.py").exists()
    assert (snapshot / "added.bin").read_bytes() == bytes(range(256))
    assert (snapshot / "executable").stat().st_mode & 0o777 == 0o755
    assert "GIT binary patch" in Path(artifact["patch_path"]).read_text()
    assert tree_digest(base) == before
    assert artifact["snapshot"]["digest"] == tree_digest(snapshot)
    (repo / "link").symlink_to("/etc/passwd")
    with pytest.raises(CandidateError, match="symlink"): snapshot_candidate(repo, tmp_path / "bad")


def test_fixed_discovery_ignores_candidate_tests_and_forged_reward(tmp_path):
    config, _ = pilot_config(tmp_path)
    task = task_manifest(config.benchmark, split="train")[0]
    candidate = tmp_path / "candidate"
    shutil.copytree(Path(task["path"]) / "environment/repo", candidate)
    (candidate / "tests").mkdir()
    (candidate / "tests/check.py").write_text('print("forged reward")')
    (candidate / "pytest.ini").write_text("[pytest]\naddopts = --ignore=names.py\n")
    outcome = verify_local_fixture(task, candidate, tmp_path / "wrong")
    assert outcome["rewards"] == {"reward": 0} and outcome["discovery"]["executed"] == 3
    shutil.copyfile(Path(task["path"]) / "solution/names.py", candidate / "names.py")
    assert verify_local_fixture(task, candidate, tmp_path / "correct")["rewards"] == {"reward": 1}
    # This output is from an untrusted child, never a trusted discovery report.
    (candidate / "names.py").write_text('print(\'{"expected":3,"executed":3,"passed":3}\')\nraise SystemExit(0)\n')
    assert verify_local_fixture(task, candidate, tmp_path / "forged")["rewards"] == {"reward": 0}


def test_no_discovered_tests_and_verifier_errors_cannot_pass():
    assert _result(0, '{"expected":0,"executed":0,"passed":0}', 3)["exception"]
    assert _result(0, "", 3)["rewards"] is None
    assert _result(1, '{"expected":3,"executed":3,"passed":3}', 3)["rewards"] == {"reward": 0}


def test_harbor_hook_quiesces_before_export_and_verifies_snapshot(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from self_summarization_agent.benchmarks.swe_dev import harbor_trial as adapter
    cfg, _ = pilot_config(tmp_path)
    task = task_manifest(cfg.benchmark, split="train")[0]
    origin = tmp_path / "live-agent"
    shutil.copytree(Path(task["path"]) / "environment/repo", origin)
    shutil.copyfile(Path(task["path"]) / "solution/names.py", origin / "names.py")
    order = []
    async def stop(arguments, **kwargs):
        assert arguments == ["stop", "main"]
        order.append("stop")
    async def download(path, destination):
        assert order == ["stop"] and path == "/app/repo"
        order.append("snapshot")
        shutil.copytree(origin, destination, dirs_exist_ok=True)
    def verify(task, snapshot, output):
        assert order == ["stop", "snapshot"] and Path(snapshot) != origin
        order.append("verify")
        return verify_local_fixture(task, snapshot, output)
    monkeypatch.setattr(adapter, "verify_docker", verify)
    trial = adapter.SWETrial.__new__(adapter.SWETrial)
    trial.config = SimpleNamespace(trials_dir=tmp_path, trial_name="trusted-trial")
    trial._result = SimpleNamespace(verifier_result=None)
    trial._environment = SimpleNamespace(_run_docker_compose_command=stop, download_dir=download)
    trial.swe_task = task
    asyncio.run(trial._verify_with_retry())
    assert order == ["stop", "snapshot", "verify"]
    assert trial.result.verifier_result.rewards == {"reward": 1}
    assert trial.swe_evidence["snapshot"]["digest"] == tree_digest(origin)


def test_docker_verification_enforces_limits_and_cleans_up(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from self_summarization_agent.benchmarks.swe_dev import verification as module
    cfg, _ = pilot_config(tmp_path)
    task = task_manifest(cfg.benchmark, split="train")[0]
    task["image"] = "fixture-image@sha256:" + "a" * 64  # Mock-only identity, never a manifest claim.
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='{"expected":3,"executed":3,"passed":3}', stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.verify_docker(task, tmp_path, tmp_path / "logs")["rewards"] == {"reward": 1}
    command = commands[1]
    assert "--network=none" in command and "--read-only" in command and "--cap-drop=ALL" in command
    assert "--cpus" in command and "--memory" in command and "--pids-limit" in command
    assert commands[-1][:3] == ["docker", "rm", "-f"]
