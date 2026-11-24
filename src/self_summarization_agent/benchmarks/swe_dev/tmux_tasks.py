"""Native pytest grading for independently recovered SWE-Dev task specifications."""

import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import tempfile
from hashlib import file_digest, sha256
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, body):
    Path(path).write_text(json.dumps(body, indent=2) + "\n")


def worker_command(args, account="nobody"):
    if os.getuid() == 0:
        user = pwd.getpwnam(account)
        return [
            "setpriv",
            f"--reuid={user.pw_uid}",
            f"--regid={user.pw_gid}",
            "--clear-groups",
            "--no-new-privs",
            *args,
        ]
    return args


def writable_workspace(path, account="nobody"):
    if os.getuid() == 0:
        user = pwd.getpwnam(account)
        for entry in [path, *path.rglob("*")]:
            os.chown(entry, user.pw_uid, user.pw_gid, follow_symlinks=False)


def task_env(home, repo, python):
    return {
        "HOME": str(home),
        "PATH": f"{python.parent}:/usr/local/bin:/usr/bin:/bin",
        "VIRTUAL_ENV": str(python.parent.parent),
        "TERM": "xterm",
        "SHELL": "/bin/bash",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, [repo, repo / "src"])),
    }


def grade(bundle, candidate, output):
    """Execute exactly the published node IDs in a fresh copy with trusted tests restored."""
    bundle, output = Path(bundle), Path(output)
    spec = json.loads((bundle / "spec.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    nodes = [*spec["FAIL_TO_PASS"], *spec["PASS_TO_PASS"]]
    python = bundle / "venv/bin/python"
    with tempfile.TemporaryDirectory(prefix="swe-grade-") as dirname:
        home = Path(dirname)
        repo = home / "repo"
        shutil.copytree(
            candidate, repo, ignore=shutil.ignore_patterns(".git", "__pycache__")
        )
        for src in (bundle / "trusted").rglob("*"):
            if src.is_file():
                dest = repo / src.relative_to(bundle / "trusted")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
        runner = home / "grade.py"
        runner.write_text(_PYTEST_RUNNER)
        writable_workspace(home, "daemon")
        with (output / "pytest.log").open("wb") as log:
            proc = subprocess.Popen(
                worker_command([str(python), str(runner), *nodes], "daemon"),
                cwd=repo,
                env=task_env(home, repo, python),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                result = {"reward": 0, "reason": "timeout", "outcomes": {}}
                write_json(output / "result.json", result)
                return result
        report = home / "outcomes.json"
        outcomes = json.loads(report.read_text()) if report.is_file() else {}
        if set(outcomes) != set(nodes) or proc.returncode not in (0, 1):
            reward, reason = None, "incomplete_test_discovery"
        else:
            reward = int(
                proc.returncode == 0 and all(v == "passed" for v in outcomes.values())
            )
            reason = "passed" if reward else "failed"
        result = {"reward": reward, "reason": reason, "outcomes": outcomes}
        write_json(output / "result.json", result)
        return result


_PYTEST_RUNNER = """import json, os, sys
from pathlib import Path
import pytest
class Outcomes:
    def __init__(self):
        self.results = {}
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed or report.skipped:
            self.results[report.nodeid] = report.outcome
plugin = Outcomes()
status = pytest.main(['-q', '-o', 'addopts=', '-p', 'no:cacheprovider', *sys.argv[1:]], plugins=[plugin])
(Path(os.environ['HOME']) / 'outcomes.json').write_text(json.dumps(plugin.results))
sys.exit(status)
"""


def load_tasks(config):
    from .compactionrl import MANIFEST_SHA256, REVISION
    from .dataset import tree_digest

    if (
        config.dataset_revision != REVISION
        or file_hash(config.train_manifest_path) != MANIFEST_SHA256
    ):
        raise ValueError("Unexpected CompactionRL reference revision or ID manifest")
    reference = json.loads(Path(config.train_manifest_path).read_text())
    root = Path(config.prepared_task_root).resolve()
    manifest = root / "validated.json"
    if not manifest.is_file():
        raise ValueError(f"Run swe_dev.prepare_tmux first; missing {manifest}")
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, dict) or not entries or entries.keys() - set(reference):
        raise ValueError(
            "Expected a nonempty validated subset of the CompactionRL training pool"
        )
    tasks = []
    for name in reference:
        if name not in entries:
            continue
        checks = entries[name]
        bundle = root / checks["directory"]
        if bundle.resolve().parent != root:
            raise ValueError(f"Invalid bundle directory for {name}")
        if any(
            tree_digest(bundle / part) != checks[part] for part in ("base", "trusted")
        ):
            raise ValueError(f"Prepared source or verifier changed for {name}")
        if any(
            file_hash(bundle / part) != checks[part]
            for part in ("spec.json", "packages.txt", "validation.json")
        ):
            raise ValueError(f"Prepared task metadata changed for {name}")
        spec = json.loads((bundle / "spec.json").read_text())
        # Distribution metadata is sufficient to detect package upgrades without
        # executing code from the task environment in the training process.
        locked = set()
        for line in (bundle / "packages.txt").read_text().splitlines():
            if "==" in line:
                package, version = line.split("==", 1)
                locked.add((re.sub(r"[-_.]+", "-", package.lower()), version))
        installed = set()
        for dist in (bundle / "venv").glob("lib/python*/site-packages/*.dist-info"):
            package, version = dist.name.removesuffix(".dist-info").rsplit("-", 1)
            installed.add((re.sub(r"[-_.]+", "-", package.lower()), version))
        if installed != locked:
            raise ValueError(f"Installed task dependencies changed for {name}")
        reports = json.loads((bundle / "validation.json").read_text())
        baseline = dict.fromkeys(spec["FAIL_TO_PASS"], "failed")
        baseline.update(dict.fromkeys(spec["PASS_TO_PASS"], "passed"))
        if len(reports) < 2 or any(
            r["baseline"]["outcomes"] != baseline
            or r["baseline"]["reward"] != 0
            or r["oracle"]["reward"] != 1
            or r["oracle"]["outcomes"] != dict.fromkeys(baseline, "passed")
            for r in reports
        ):
            raise ValueError(f"Invalid baseline/reference validation for {name}")
        if (
            spec["instance_id"].replace("__", "_").lower() != name
            or not (bundle / "venv/bin/python").is_file()
        ):
            raise ValueError(f"Inconsistent local tmux bundle for {name}")
        tasks.append(
            {
                "name": name,
                "path": str(bundle),
                "checksum": sha256(
                    json.dumps(checks, sort_keys=True).encode()
                ).hexdigest(),
                "source": {
                    "origin": "nebius/SWE-bench-extra",
                    "revision": spec["revision"],
                    "instance_id": spec["instance_id"],
                    "base_commit": spec["base_commit"],
                    "repository_url": "https://github.com/" + spec["repo"],
                },
                "image": None,
                "fixture_only": False,
                "manifest_digest": file_hash(manifest),
                "verifier_mode": "tmux-pytest",
            }
        )
    return tasks
