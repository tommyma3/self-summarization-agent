"""Materialize matching Nebius tasks and publish only verified baseline/oracle pairs."""

import argparse
import fcntl
import json
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from .compactionrl import MANIFEST_SHA256
from .dataset import tree_digest
from .tmux_tasks import (
    file_hash,
    grade,
    task_env,
    worker_command,
    writable_workspace,
    write_json,
)

REVISION = "11dcbfb30e19552df2a2f8030bd764adc95c92a5"
PARQUET_HASH = "c8633563d41c1a2fde0361650f96b5103b11f74d7feac5c9abf33f58e486d946"
URL = f"https://huggingface.co/datasets/nebius/SWE-bench-extra/resolve/{REVISION}/data/train-00000-of-00001.parquet"


def run(args, cwd, **kwargs):
    result = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
        **kwargs,
    )
    if result.returncode:
        raise ValueError(
            f"{args[0]} failed: " + result.stdout.decode(errors="replace")[-1800:]
        )
    return result.stdout


def prepare(row, root, python, uv, recipe):
    name = row["instance_id"].replace("__", "_").lower()
    # Keep the venv at its final path; entry point shebangs embed it.
    bundle = Path(tempfile.mkdtemp(prefix=name + "-", dir=root))
    bundle.chmod(0o755)
    spec = {
        key: row[key]
        for key in (
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        )
    }
    spec["revision"] = REVISION
    write_json(bundle / "spec.json", spec)
    if not spec["FAIL_TO_PASS"] or not spec["PASS_TO_PASS"] or not row["test_patch"]:
        raise ValueError("missing independent fail/pass tests")
    with tempfile.TemporaryDirectory(prefix="swe-prep-") as tmp:
        home = Path(tmp)
        checkout = home / "checkout"
        run(["git", "init", "-q", str(checkout)], home)
        run(
            [
                "git",
                "fetch",
                "-q",
                "--depth=1",
                f"https://github.com/{row['repo']}.git",
                row["base_commit"],
            ],
            checkout,
        )
        run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], checkout)
        if (
            run(["git", "rev-parse", "HEAD"], checkout).decode().strip()
            != row["base_commit"]
        ):
            raise ValueError("Fetched checkout differs from the published base commit")
        shutil.copytree(
            checkout,
            bundle / "base",
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        tree_digest(bundle / "base")  # same regular-file contract as candidate export
        if (
            sum(p.stat().st_size for p in (bundle / "base").rglob("*") if p.is_file())
            > 16 * 1024 * 1024
        ):
            raise ValueError("base exceeds candidate snapshot size limit")
        run(
            ["git", "apply", "--index", "-"], checkout, input=row["test_patch"].encode()
        )
        changed = (
            run(["git", "diff", "--cached", "--name-only", "-z"], checkout)
            .decode()
            .split("\0")[:-1]
        )
        files = set(changed) | {
            node.split("::")[0] for node in spec["FAIL_TO_PASS"] + spec["PASS_TO_PASS"]
        }
        for filename in files:
            rel = Path(filename)
            if rel.is_absolute() or ".." in rel.parts or not (checkout / rel).is_file():
                raise ValueError(f"unsupported test path: {filename}")
            dest = bundle / "trusted" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkout / rel, dest)
        venv = bundle / "venv"
        run([uv, "venv", "--python", python, str(venv)], home)
        # Project build hooks run as the worker, with no trainer environment.
        writable_workspace(home)
        writable_workspace(venv)
        env = task_env(home, checkout, venv / "bin/python")
        env.update(UV_CACHE_DIR=str(home / "uv-cache"), UV_LINK_MODE="copy")
        test_requirements = []
        for filename in (
            "requirements-test.txt",
            "requirements_test.txt",
            "requirements-testing.txt",
        ):
            if (checkout / filename).is_file():
                test_requirements += ["-r", str(checkout / filename)]
        if recipe is None:
            requirements = [
                "pytest<9",
                "setuptools<81",
                "pytest-mock",
                "mock",
                "trustme",
                "pytest-asyncio<1",
                "pytest-cov",
                "freezegun",
                "requests-mock",
                "hypothesis<6.142",
                *test_requirements,
                str(checkout) + "[test,tests,testing]",
            ]
        else:
            requirements = ["--no-deps", *recipe["packages"]]
            if recipe["project_metadata"]:
                requirements.append(str(checkout))
        run(
            worker_command(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(venv / "bin/python"),
                    *requirements,
                ]
            ),
            home,
            env=env,
        )
        # Keep third-party dependencies, but force imports to use the candidate tree.
        wheels = list(
            venv.glob("lib/python*/site-packages/*.dist-info/direct_url.json")
        )
        for wheel in wheels:
            saved = home / wheel.parent.name
            shutil.copytree(wheel.parent, saved)
            metadata = wheel.with_name("METADATA").read_text()
            package = next(
                line[6:] for line in metadata.splitlines() if line.startswith("Name: ")
            )
            run(
                worker_command(
                    [
                        uv,
                        "pip",
                        "uninstall",
                        "--python",
                        str(venv / "bin/python"),
                        package,
                    ]
                ),
                home,
                env=env,
            )
            # Some projects query their installed distribution version on import.
            (saved / "direct_url.json").unlink()
            shutil.copytree(saved, wheel.parent)
        (bundle / "packages.txt").write_bytes(
            run(
                worker_command(
                    [
                        uv,
                        "pip",
                        "freeze",
                        "--quiet",
                        "--python",
                        str(venv / "bin/python"),
                    ]
                ),
                home,
                env=env,
            )
        )
        writable_workspace(venv, "root")
        oracle = home / "oracle"
        shutil.copytree(bundle / "base", oracle)
        run(["git", "apply", "-"], oracle, input=row["patch"].encode())
        validations = []
        for i in range(2):
            base = grade(bundle, bundle / "base", bundle / f"baseline-{i}")
            fixed = grade(bundle, oracle, bundle / f"oracle-{i}")
            if (
                base["outcomes"]
                != {
                    **dict.fromkeys(spec["FAIL_TO_PASS"], "failed"),
                    **dict.fromkeys(spec["PASS_TO_PASS"], "passed"),
                }
                or fixed["reward"] != 1
            ):
                raise ValueError(
                    f"baseline/oracle validation failed: {base['reason']}/{fixed['reason']}"
                )
            validations.append({"baseline": base, "oracle": fixed})
        write_json(bundle / "validation.json", validations)
    return {
        "directory": bundle.name,
        **{p: tree_digest(bundle / p) for p in ("base", "trusted")},
        **{
            p: file_hash(bundle / p)
            for p in ("spec.json", "packages.txt", "validation.json")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", default="configs/train/data/swedev_reference_1105.json"
    )
    parser.add_argument(
        "--recipes",
        default="configs/train/data/swedev_tmux_python39.json",
        help="Dependency locks for the validated Python 3.9 task pool",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--python", required=True, help="Absolute worker-accessible Python interpreter"
    )
    parser.add_argument("--parquet", type=Path)
    parser.add_argument(
        "--instance", action="append", help="Select one reference ID (repeatable)"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not Path(args.python).is_file() or not shutil.which("uv"):
        parser.error(
            "Provide an existing worker-accessible --python and install uv on PATH"
        )
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.output.resolve()
    root.mkdir(exist_ok=True, parents=True)
    lock = (root / ".prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parquet = args.parquet or root / "source.parquet"
    if not args.parquet and not parquet.exists():
        urllib.request.urlretrieve(URL, root / "source.partial")
        if file_hash(root / "source.partial") != PARQUET_HASH:
            raise ValueError(
                "Downloaded SWE-bench Extra parquet has an unexpected digest"
            )
        (root / "source.partial").replace(parquet)
    if (
        file_hash(parquet) != PARQUET_HASH
        or file_hash(args.reference) != MANIFEST_SHA256
    ):
        raise ValueError("Source parquet or reference manifest digest mismatch")
    refs = set(json.loads(Path(args.reference).read_text()))
    rows = {
        r["instance_id"].replace("__", "_").lower(): r
        for r in pq.read_table(parquet).to_pylist()
        if r["instance_id"].replace("__", "_").lower() in refs
    }
    recipes = json.loads(Path(args.recipes).read_text())
    wanted = set(args.instance or recipes)
    if wanted - rows.keys():
        raise ValueError(
            f"No matching upstream specification: {sorted(wanted - rows.keys())}"
        )
    uv = root / "uv"
    if not uv.exists():
        shutil.copy2(shutil.which("uv"), uv)
    index = root / "validated.json"
    entries = json.loads(index.read_text()) if index.exists() else {}
    excluded_path = root / "exclusions.json"
    exclusions = json.loads(excluded_path.read_text()) if excluded_path.exists() else {}

    def one(name):
        try:
            return (
                name,
                prepare(rows[name], root, args.python, str(uv), recipes.get(name)),
                None,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return name, None, str(exc)

    with ThreadPoolExecutor(args.workers) as pool:
        for name, checks, error in pool.map(one, sorted(wanted - entries.keys())):
            if error:
                exclusions[name] = error
            else:
                entries[name] = checks
                exclusions.pop(name, None)
            print(
                name, "excluded: " + error[:120] if error else "validated", flush=True
            )
    write_json(root / "validated.tmp", entries)
    (root / "validated.tmp").replace(index)
    write_json(root / "exclusions.tmp", exclusions)
    (root / "exclusions.tmp").replace(excluded_path)
    print(
        f"Validated {len(wanted & entries.keys())}/{len(wanted)} requested tasks; "
        f"{len(entries)} total in {index}.",
        flush=True,
    )
    if not (wanted & entries.keys()):
        raise SystemExit("No requested tasks passed validation")


if __name__ == "__main__":
    main()
