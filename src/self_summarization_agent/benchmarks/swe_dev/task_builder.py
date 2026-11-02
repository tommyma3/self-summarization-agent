"""Build the single deterministic local pilot; no inferred upstream metadata."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from .dataset import tree_digest
from .verification import verify_local_fixture


def build_fixture(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    verifier = bundle / "tests/check.py"
    task = dict(fixture_only=True, verifier=dict(path=str(verifier), expected_tests=3, timeout_seconds=10))
    trials = []
    for repetition in range(2):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            shutil.copytree(bundle / "environment/repo", root)
            baseline = verify_local_fixture(task, root, output / f"baseline-{repetition}")
            shutil.copy2(bundle / "solution/names.py", root / "names.py")
            oracle = verify_local_fixture(task, root, output / f"oracle-{repetition}")
            if baseline["rewards"]["reward"] != 0 or baseline.get("discovery", {}).get("passed") != 2 or oracle["rewards"]["reward"] != 1:
                raise ValueError("Fixture baseline/oracle validation failed")
            trials.append(dict(baseline=baseline, oracle=oracle))
    report = dict(version=1, fixture_only=True, repetitions=2, bundle_digest=tree_digest(bundle),
                  baseline_validated=True, oracle_validated=True, trials=trials,
                  image=None, verifier_digest=sha256(verifier.read_bytes()).hexdigest())
    report_path = output / "validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    entry = dict(task_id="local-fixture:trim-name-v1", leakage_group="local-fixture:trim-name",
        source=dict(kind="deterministic-local-fixture", origin="project-fixture", instance_id=None,
                    revision="trim-name-v1", repository_url=None, base_commit=None),
        bundle_path=str(bundle), bundle_digest=tree_digest(bundle), image=None,
        build_context_digest=tree_digest(bundle / "environment"),
        instruction_digest=sha256((bundle / "instruction.md").read_bytes()).hexdigest(),
        reference_patch_digest=sha256((bundle / "solution/names.py").read_bytes()).hexdigest(),
        verifier=dict(path=str(verifier), digest=sha256(verifier.read_bytes()).hexdigest(), version="fixture-v1",
            fail_to_pass=["strip_edges"], pass_to_pass=["interior_whitespace", "empty"], expected_tests=3,
            timeout_seconds=10, limits=dict(cpus=1, memory_mb=512, pids=32, tmp_mb=32)),
        validation_report=str(report_path), validation_digest=sha256(report_path.read_bytes()).hexdigest())
    for split, tasks in (("train", [entry]), ("eval", [])):
        manifest = dict(version=1, split=split, split_version="fixture-only-v1", split_seed=0,
                        dataset_revision="local-fixture-v1", tasks=tasks)
        (output / f"{split}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "train.json", output / "eval.json"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", default="tests/fixtures/swe_dev/trim-name")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_fixture(args.bundle, args.output)

if __name__ == "__main__":
    main()
