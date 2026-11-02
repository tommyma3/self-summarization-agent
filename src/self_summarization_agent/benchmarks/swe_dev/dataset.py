"""Versioned executable manifests, independent of historical policy transcripts."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

SCHEMA_VERSION = 1
SELECTION_VERSION = "complete-task-groups-v1"


def tree_digest(root):
    root = Path(root)
    digest = sha256()
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in trusted task bundle: {path}")
        if path.is_file():
            body = path.read_bytes()
            digest.update(json.dumps([path.relative_to(root).as_posix(), path.stat().st_mode & 0o777,
                                      len(body)]).encode() + b"\0" + body)
    return digest.hexdigest()


def _path(manifest, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _load(path, split, config):
    path = Path(path).resolve()
    body = json.loads(path.read_text())
    if (body.get("version") != SCHEMA_VERSION or body.get("split") != split
            or body.get("dataset_revision") != config.dataset_revision
            or not body.get("split_version") or type(body.get("split_seed")) is not int):
        raise ValueError("Invalid SWE manifest version/split/source revision")
    tasks = []
    for entry in body["tasks"]:
        entry = dict(entry)
        if (not isinstance(entry.get("task_id"), str) or ":" not in entry["task_id"]
                or not entry.get("leakage_group")):
            raise ValueError("SWE tasks need namespaced task IDs and leakage groups")
        root = _path(path, entry["bundle_path"])
        if not all((root / f).is_file() for f in ("task.toml", "instruction.md", "environment/Dockerfile")):
            raise ValueError("Missing SWE Harbor bundle")
        if tree_digest(root) != entry["bundle_digest"]:
            raise ValueError("SWE task bundle digest changed")
        if sha256((root / "instruction.md").read_bytes()).hexdigest() != entry["instruction_digest"]:
            raise ValueError("SWE instruction digest changed")
        if entry.get("build_context_digest") != tree_digest(root / "environment"):
            raise ValueError("SWE build context digest changed")
        if (root / "environment/docker-compose.yaml").exists():
            raise ValueError("Custom compose/mounts are outside SWE text-only v1")
        source = entry["source"]
        fixture = source.get("kind") == "deterministic-local-fixture"
        if not fixture:
            if (not source.get("instance_id") or not source.get("origin") or not source.get("revision")
                    or not source.get("repository_url") or not re.fullmatch(r"[0-9a-f]{40}", source.get("base_commit", ""))):
                raise ValueError("Missing recovered SWE source metadata")
            if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", entry.get("image", "")):
                raise ValueError("SWE images must be pinned by digest")
            if not entry.get("setup_provenance") or not entry.get("reference_patch_digest"):
                raise ValueError("Missing setup/reference-patch provenance")
            base = root / "environment/repo"
            import subprocess
            revision = subprocess.run(["git", "-C", str(base), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True).stdout.strip()
            if revision != source["base_commit"]:
                raise ValueError("Recovered repository differs from pinned base commit")
        elif config.environment != "local-fixture":
            raise ValueError("Local fixtures require environment: local-fixture; no Docker validation claimed")
        verifier = entry["verifier"]
        script = _path(path, verifier["path"])
        # Keep trusted material outside the agent image build context.
        if script.is_relative_to(root / "environment"):
            raise ValueError("Verifier is inside the agent build context")
        if sha256(script.read_bytes()).hexdigest() != verifier["digest"]:
            raise ValueError("SWE verifier digest changed")
        if (not verifier.get("fail_to_pass") or not verifier.get("pass_to_pass")
                or type(verifier.get("expected_tests")) is not int or verifier["expected_tests"] < 1
                or verifier.get("timeout_seconds", 0) <= 0):
            raise ValueError("Missing trusted test discovery/resource profile")
        report_path = _path(path, entry["validation_report"])
        if sha256(report_path.read_bytes()).hexdigest() != entry["validation_digest"]:
            raise ValueError("SWE validation report digest changed")
        report = json.loads(report_path.read_text())
        if (report.get("bundle_digest") != entry["bundle_digest"] or report.get("repetitions", 0) < 2
                or not report.get("baseline_validated") or not report.get("oracle_validated")
                or report.get("verifier_digest") != verifier["digest"] or report.get("image") != entry["image"]):
            raise ValueError("SWE baseline/oracle validation required")
        tasks.append(dict(entry, name=entry["task_id"], path=str(root),
                          verifier=dict(verifier, path=str(script)), checksum=entry["bundle_digest"],
                          fixture_only=fixture, manifest_digest=sha256(path.read_bytes()).hexdigest()))
    if len({t["name"] for t in tasks}) != len(tasks) or len({t["leakage_group"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate SWE task/issue/leakage identity")
    source_keys = [(t["source"]["origin"], t["source"]["instance_id"]) for t in tasks
                   if t["source"].get("instance_id")]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Duplicate SWE source instance IDs")
    return sorted(tasks, key=lambda t: t["name"])


def manifests(config):
    if not config.train_manifest_path or not config.eval_manifest_path:
        raise ValueError("SWE requires explicit train and eval manifests")
    train = _load(config.train_manifest_path, "train", config)
    evaluation = _load(config.eval_manifest_path, "eval", config)
    for key in ("name", "leakage_group", "instruction_digest", "bundle_digest"):
        if {t[key] for t in train} & {t[key] for t in evaluation}:
            raise ValueError(f"Overlapping SWE train/eval {key}")
    # Namespaced source identities also prevent duplicate IDs under new aliases.
    def source_ids(tasks):
        return {(t["source"].get("origin"), t["source"].get("instance_id")) for t in tasks
                if t["source"].get("instance_id")}
    if source_ids(train) & source_ids(evaluation):
        raise ValueError("Overlapping SWE source instance IDs")
    return train, evaluation


def task_manifest(config, *, split):
    if split not in {"train", "eval"}:
        raise ValueError("Unsupported SWE split")
    train, evaluation = manifests(config)
    return train if split == "train" else evaluation


def benchmark_identity(config):
    from ..terminal_bench import HARBOR_VERSION, KIRA_REVISION, SCAFFOLD_VERSION
    from ..terminal_bench.scaffold import TerminusKiraScaffold
    train, evaluation = manifests(config)
    return dict(config=asdict(config), train=train, eval=evaluation, selection=SELECTION_VERSION,
                harbor=HARBOR_VERSION, kira=KIRA_REVISION, scaffold=SCAFFOLD_VERSION,
                scaffold_digest=TerminusKiraScaffold(None).fingerprint,
                verification="fresh-candidate-v1", reward_profile=config.reward_profile)
