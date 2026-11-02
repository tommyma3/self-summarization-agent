"""Audit released conversations without manufacturing executable task metadata.

python -m self_summarization_agent.benchmarks.swe_dev.audit PARQUET... --output report.json
Download source parquet files at the revision below before invoking. No trajectory
is imported into collection or training by this command.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

REPOSITORY_REVISION = "72917b610c3749c9e1236b29effd629f8aca6a99"
DATASET_REVISION = "5888e5864886590ead78a69b6e60490a0d89a594"
PINNED_FILES = {
    "rft-00000-of-00001.parquet": "d03c081bb40575fad6489b91cd9d65bc0a210ed54036f27c98d3d5776bc26c61",
    "sft-00000-of-00002.parquet": "74cbbaef1f64f469a1b26726587dce970319b39bd38e75fbb60ec73da2bd4ea9",
    "sft-00001-of-00002.parquet": "c2c748cabfc4ae379b645d7d24a61aac5248a11687c79e0d99f231c12190501a",
}
REQUIRED_ASSETS = ["repository_url", "base_commit", "instruction", "setup",
                   "fail_to_pass", "pass_to_pass", "verifier", "image", "oracle_validation"]


def audit_rows(rows):
    origins, exclusions, ids, transcripts = Counter(), Counter(), set(), set()
    duplicate_trajectories = blank_ids = count = complete_specs = 0
    for row in rows:
        count += 1
        origins[str(row.get("from"))] += 1
        instance = row.get("instance_id")
        if not isinstance(instance, str) or not instance.strip():
            blank_ids += 1
        else:
            ids.add((row.get("from"), instance))
        key = sha256(json.dumps(row.get("input"), sort_keys=True).encode()).hexdigest()
        duplicate_trajectories += key in transcripts
        transcripts.add(key)
        complete_specs += all(row.get(asset) for asset in REQUIRED_ASSETS)
        for asset in REQUIRED_ASSETS:
            if not row.get(asset):
                exclusions["missing_" + asset] += 1
    return dict(rows=count, origins=dict(origins), nonblank_origin_instance_pairs=len(ids),
                blank_instance_ids=blank_ids, duplicate_transcripts=duplicate_trajectories,
                complete_explicit_specifications=complete_specs, validated_upstream_tasks=0,
                missing_assets=dict(exclusions),
                limitation="Counts of nonblank IDs are not counts of executable tasks; no transcript inference.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("parquet", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    reports = {}
    paths = [Path(p) for p in args.parquet]
    if {p.name for p in paths} != set(PINNED_FILES) or len(paths) != len(PINNED_FILES):
        raise ValueError("Full audit requires all three pinned source parquet files")
    def rows(path):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    for file in paths:
        with file.open("rb") as stream:
            checksum = __import__("hashlib").file_digest(stream, "sha256").hexdigest()
        if PINNED_FILES[file.name] != checksum:
            raise ValueError("Source file differs from the pinned dataset revision")
        reports[file.name] = {"sha256": checksum, "schema": str(pq.read_schema(file)),
                              **audit_rows(rows(file))}
    from itertools import chain
    aggregate = audit_rows(chain.from_iterable(rows(p) for p in paths))
    Path(args.output).write_text(json.dumps(dict(repository_revision=REPOSITORY_REVISION,
        dataset_revision=DATASET_REVISION, aggregate=aggregate, files=reports), indent=2) + "\n")

if __name__ == "__main__":
    main()
