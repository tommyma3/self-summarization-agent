# Independent task recovery gate

The production-readiness plan stops at M1 on 2026-11-02: **zero independently
recoverable upstream tasks** were found in the scoped inventory. This is a
recovery result, not completion of the production pilot. No task images were
built and no real-task Docker, Qwen/vLLM, GPU update/resume, or held-out evaluation
was run. The existing fixture remains available.

The [inventory](recovery-inventory-2026-11-02.json) records checked paths, source
revisions, access outcomes, exclusions, and the installed infrastructure. The
[generated report](recovery-report-2026-11-02.json) contains the required assets,
recovery count, gate result, and empty versioned split proposal. Both repository
HEAD and dataset revision match the prior audit. The dataset file listing is
unchanged. The existing parquet audit was reused, not rerun. GitHub's releases API
returned HTTP 403 (rate limit); newly attached release assets remain unresolved.
Local inspection covered the recorded task directories only. No maintainers were
contacted. Docker is absent; Harbor 0.1.44 and four A100 80GB devices are installed.
Hardware inventory does not validate GPU training or isolation.

## Run the gate

```bash
.venv/bin/python -m self_summarization_agent.benchmarks.swe_dev.recovery \
  docs/swe-dev/recovery-inventory-2026-11-02.json \
  --output artifacts/swe-dev/recovery.json
```

The command writes the report even when no tasks qualify and exits **2** for a
blocked gate, **0** when at least one specification can advance to bundle
validation. Invalid inventory headers, duplicate candidate IDs, or corrupt
external indexes fail closed. It reads explicit local evidence and performs no
network requests or task execution. Repeated runs with identical inputs produce
identical reports. Paths resolve against the inventory directory.

The output is a proposal, deliberately separate from executable manifests accepted
by `dataset.py`. Successful recovery does not imply baseline/oracle validation.
M2 must still build pinned bundles/images and validate them repeatedly before
admission. No rollout, renderer, token ledger, reward, cache, or training code is
changed by this gate.

## Supply missing assets

Supply one independently sourced specification per upstream task containing:

- Origin, instance identity, canonical issue URL, repository URL, and exact
  40-character base commit, with evidence separate from conversations.
- Canonical instruction, setup/dependency provenance, trusted fail-to-pass and
  pass-to-pass tests, reference fix, and trusted verifier implementation.
- For every field: source location, immutable source revision, access requirement,
  local evidence path, SHA-256, and whether it is an `upstream-release` or
  `maintainer-supplied` artifact.

Metadata evidence must be a UTF-8 file containing the exact value, optionally
with trailing whitespace, or a JSON object with that field's exact value. File
assets are hashed as complete files. Missing, empty, changed, symlinked, or
transcript-derived evidence excludes the candidate and records field-level
reasons. This is provenance/integrity validation; a maintainer still needs to
establish authenticity and audit executable verifier material before M2/M3.
Do not copy metadata out of historical agent conversations.

For example, each `candidates` entry in an inventory uses this shape (all asset
fields are required; the abbreviated example does not qualify):

```json
{
  "candidate_id": "swe-dev:upstream-id",
  "reserved_split": "train",
  "assets": {
    "origin": {
      "value": "swe-dev",
      "path": "assets/origin.txt",
      "sha256": "<SHA-256 of origin.txt>",
      "source": "<independent release URL>",
      "revision": "<immutable release revision>",
      "access": "public",
      "kind": "upstream-release"
    },
    "instruction": {
      "path": "assets/instruction.md",
      "sha256": "<SHA-256 of instruction.md>",
      "source": "<independent release URL>",
      "revision": "<immutable release revision>",
      "access": "public",
      "kind": "upstream-release"
    }
  }
}
```

The inventory header requires `version: 1`, `dataset_revision`, `split_version`,
and an integer `split_seed`. Include every other field listed in the report's
`required_assets`. Preserve this inventory as an experiment artifact before
collecting any rollouts. The seed records the experiment's split provenance;
assignment is explicit through `reserved_split`, never randomly changed on resume.

Source identity, canonical issue URL, instruction bytes, and reference-fix bytes
form leakage keys. Connected duplicate groups must have the same split; conflicts
exclude the entire group. External overlap excludes the whole connected group.
The source key is `source:` followed by JSON serialization of `[origin,
instance_id]`; the other keys use `issue:<URL>`, `instruction:<SHA-256>`, and
`patch:<SHA-256>`. Supply trusted external evaluation indexes as JSON lists of
objects containing nonempty `leakage_keys` lists, with the same evidence envelope
under `external_evaluation[].index`. An absent index remains explicitly unresolved.
Exact content checks cannot prove semantic disjointness or detect differently
encoded equivalent patches. Keep SWE-bench Verified and official Terminal-Bench
evaluation-only; resolve intended evaluation identities before claiming overlap
clearance.

Proceed to M2 only after independent specifications are supplied. Do not create
production presets backed by the fixture, invent missing assets, or launch policy
workloads to bypass this gate. Subsequent production phases and their evidence
requirements remain as specified in the readiness plan.

## Validation evidence

The final focused run passed **48 tests**: 16 recovery tests plus all 32 existing
SWE tests. These cover altered/missing evidence, transcript rejection, metadata
binding, duplicate IDs, transitive leakage groups, external overlap, deterministic
reports, blocked CLI publication, and existing fixture collection/update/resume.
They are local integrity and CPU integration evidence, not production Docker or
GPU evidence. Reproduce using the installed environment without dependency changes:

```bash
.venv/bin/python -m pytest -q \
  tests/test_swe_dev_recovery.py tests/test_swe_dev_dataset.py \
  tests/test_swe_dev_verification.py tests/test_swe_dev_rewards.py \
  tests/test_swe_dev_training.py
```

The checked report was also regenerated and compared for exact equality; Python
compilation and `git diff --check` passed.
