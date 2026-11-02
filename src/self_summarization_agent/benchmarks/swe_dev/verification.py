"""Candidate snapshots and fresh verifier execution, never live-workspace reward files."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

from .dataset import tree_digest


class CandidateError(ValueError):
    pass


def snapshot_candidate(repo, destination, *, max_bytes=16 * 1024 * 1024):
    """All regular deliverables, including untracked/binary files and executable modes.

    Snapshot reconstruction on a clean base first removes base deliverables, so
    deletions are represented too. Symlinks/special files fail closed in v1.
    Never copy the candidate's .git hooks/configuration into verification.
    """
    repo, destination = Path(repo), Path(destination)
    total, files = 0, []
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise CandidateError("Unsupported symlink/special deliverable: " + str(relative))
        total += path.stat().st_size
        if total > max_bytes:
            raise CandidateError("Candidate snapshot exceeds export limit")
        body = path.read_bytes()
        files.append((relative, body, mode & 0o777))
    destination.mkdir(parents=True, exist_ok=False)
    for relative, body, mode in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        target.chmod(mode)
    return dict(version=1, digest=tree_digest(destination), files=len(files), bytes=total,
                path=str(destination.resolve()), format="complete-regular-tree-v1")


def export_candidate(repo, base, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_candidate(repo, output / "snapshot")
    # --no-index includes additions/deletions/binaries; patch is diagnostic,
    # the complete immutable snapshot is the reconstruction authority.
    result = subprocess.run(["git", "diff", "--no-index", "--binary", "--", str(base), snapshot["path"]],
                            capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("Candidate patch export failed: " + result.stderr.decode(errors="replace"))
    patch = output / "candidate.patch"
    patch.write_bytes(result.stdout)
    evidence = dict(snapshot=snapshot, patch_path=str(patch.resolve()),
                    patch_digest=sha256(result.stdout).hexdigest(), base_digest=tree_digest(base))
    (output / "candidate.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def _result(returncode, stdout, expected):
    # Trusted fixed discovery; missing/empty/malformed results cannot pass.
    try:
        report = json.loads(stdout.strip().splitlines()[-1])
        if (report["expected"] != expected or report["executed"] != expected
                or type(report["passed"]) is not int or not 0 <= report["passed"] <= expected):
            raise ValueError("Invalid discovery counts")
    except (ValueError, KeyError, IndexError, TypeError):
        return dict(rewards=None, exception={"type": "VerifierProtocolError",
                    "message": "Missing or inconsistent trusted discovery report"},
                    classification="verifier_protocol_error")
    passed = returncode == 0 and report["passed"] == expected
    return dict(rewards={"reward": int(passed)}, exception=None,
                classification="passed" if passed else "candidate_test_failure", discovery=report)


def verify_local_fixture(task, snapshot, output):
    """CPU plumbing only: trusted fixture code, no claim of a security sandbox."""
    if not task["fixture_only"]:
        raise ValueError("Host execution is restricted to deterministic local fixtures")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A separate workspace is used; candidate tests/pytest config/sitecustomize
    # never participate in trusted discovery. Production uses Docker below.
    with tempfile.TemporaryDirectory(prefix="swe-fixture-verifier-") as directory:
        root = Path(directory) / "repo"
        shutil.copytree(snapshot, root)
        verifier = task["verifier"]
        try:
            result = subprocess.run([sys.executable, "-I", "-S", verifier["path"], str(root)],
                cwd=directory, capture_output=True, text=True, timeout=verifier["timeout_seconds"])
        except subprocess.TimeoutExpired:
            return dict(rewards={"reward": 0}, exception=None, classification="candidate_timeout")
        (output / "stdout.txt").write_text(result.stdout)
        (output / "stderr.txt").write_text(result.stderr)
        return _result(result.returncode, result.stdout, verifier["expected_tests"])


def verify_docker(task, snapshot, output):
    """Fresh pinned base with the complete candidate tree; no agent processes/mounts.

    The trusted verifier path and expected count come from the validated manifest.
    Its protocol is one JSON discovery report on stdout, never a candidate reward
    file. Resource flags are enforced on the fresh container.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = "swe-verify-" + uuid.uuid4().hex
    limits = task["verifier"]["limits"]
    verifier = task["verifier"]
    command = ["docker", "run", "--name", name, "--network=none", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit", str(limits["pids"]),
        "--cpus", str(limits["cpus"]), "--memory", str(limits["memory_mb"]) + "m",
        "--user", "65534:65534", "--tmpfs", "/tmp:rw,nosuid,size=" + str(limits["tmp_mb"]) + "m",
        "--mount", f"type=bind,src={Path(snapshot).resolve()},dst=/app/repo,readonly",
        "--mount", f"type=bind,src={Path(verifier['path']).resolve()},dst=/trusted/check.py,readonly",
        "--workdir", "/tmp", "--entrypoint", "python", task["image"],
        "-I", "-S", "/trusted/check.py", "/app/repo"]
    try:
        # An unavailable pinned image is infrastructure, never a candidate failure.
        subprocess.run(["docker", "image", "inspect", task["image"]], check=True, capture_output=True)
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=verifier["timeout_seconds"])
        except subprocess.TimeoutExpired:
            return dict(rewards={"reward": 0}, exception=None, classification="candidate_timeout")
        (output / "command.json").write_text(json.dumps(command))
        (output / "stdout.txt").write_text(result.stdout)
        (output / "stderr.txt").write_text(result.stderr)
        if result.returncode in (125, 126, 127):
            raise RuntimeError("Verifier container infrastructure/entrypoint failure: " + result.stderr)
        return _result(result.returncode, result.stdout, verifier["expected_tests"])
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
