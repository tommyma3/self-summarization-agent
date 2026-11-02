# SWE-Dev integration and validation

The integration currently has **one deterministic local coding fixture and zero
validated upstream SWE-Dev tasks**. It connects the existing Harbor/Terminus-KIRA
collector and `EpisodeRuntime` to verifier rewards, complete-group selection,
exact-token cache construction, all four training objectives, and checkpoint
resume. Local tests use scripted completions and a tiny CPU causal LM. They prove
integration, not fresh on-policy SWE-Dev performance, Docker isolation, Qwen/vLLM
collection, distributed GPU training, or held-out benchmark accuracy.

`AGENTS.md` is unchanged. The trajectory schema, TITO contract, interval renderer,
and cache version are unchanged. Only runtime reward selection was added to the
runtime; the append-only state machine is reused. Every successful compaction
finalizes the old ledger before starting its successor, and the shell/environment
belongs to the entire rollout. Saved-record checks cover the old interval and
its successor, including summary tokens and linked native tool messages.

## Source-recoverability audit (2026-11-02)

The audit fetched and inspected these immutable source revisions:

- [SWE-Dev repository, commit 72917b6](https://github.com/THUDM/SWE-Dev/tree/72917b610c3749c9e1236b29effd629f8aca6a99).
- [SWE-Dev-train, revision 5888e58](https://huggingface.co/datasets/zai-org/SWE-Dev-train/tree/5888e5864886590ead78a69b6e60490a0d89a594).

All three parquet files were downloaded at that dataset revision and hashed.
The machine-readable [coverage report](swe-dev/source-audit-2026-11-02.json)
records complete-file schemas, checksums, counts and exclusions. It contains no
conversation contents and imports nothing into training.

| Coverage | Result |
| --- | ---: |
| Released rows | 20,147 |
| SFT rows | 17,871 |
| RFT rows | 2,276 |
| Origin `swe-dev` rows | 15,674 |
| Origin `nebius` rows | 4,473 |
| Blank instance IDs | 17,871 (all SFT rows) |
| Nonblank `(origin, instance_id)` pairs | 2,276 |
| Duplicate transcripts across the release | 2,276 |
| Complete explicit executable specifications | 0 |
| Upstream tasks validated here | 0 |

Transcript duplicates use a SHA-256 of canonical serialized `input`, independently
of split/origin. Nonblank IDs are possible recovery keys, not executable tasks.
SFT/RFT are upstream conversation partitions, not leakage-safe task splits for
this training experiment.

Both splits contain `input`, `from`, `f2p_pass`, `f2p_count`, `reward`, and
`instance_id`. Missing **independent task assets** are: repository URL and exact
base commit; a canonical task instruction outside the conversation and its digest;
pinned setup/dependency provenance; trusted fail-to-pass/pass-to-pass test bodies,
commands and discovery counts; trusted verifier version; pinned task image; and
repeatable baseline/reference-fix validation. Historical rewards/counts and
assistant-reported test outputs do not supply those assets.

The pinned Git repository contains the construction pipeline, configuration
placeholders and SWE-bench image lists. It does not contain the training outputs
referenced by its README (`results/issues/all_tasks.jsonl`,
`results/descriptions/output_f2p.jsonl`, `results/testcases/output.jsonl`,
`results/evaluation-0218/evaluated_testcases`) or an instance-indexed trusted
training bundle release. Reconstructing from issue links alone would still lack
trusted setup/tests. This audit does not prove those assets cannot exist elsewhere;
it records that they were not supplied by the inspected release. No base commit,
image digest, oracle patch, or test metadata has been guessed.

SWE-bench Verified and official Terminal-Bench remain evaluation-only by default.
No held-out overlap claim is made until recovered task/issue/patch identities are
available. Do not expand the upstream training dataset before this recovery gate
and repeated baseline/oracle validation pass.

Reproduce the full metadata audit after downloading the three pinned parquet files:

```bash
uv run --locked --no-sync python -m self_summarization_agent.benchmarks.swe_dev.audit \
  /path/to/rft-00000-of-00001.parquet \
  /path/to/sft-00000-of-00002.parquet \
  /path/to/sft-00001-of-00002.parquet \
  --output artifacts/swe-dev/source-audit.json
```

The command requires all three files and rejects checksums that differ from the
pinned release. Dataset conversations never become on-policy cache entries.

## One executable fixture

`tests/fixtures/swe_dev/trim-name` asks for a whitespace-trimming fix in
`/app/repo/names.py`. Its trusted checker executes three fixed cases: one
baseline failure and two baseline regression passes. The reference fix passes all
three, twice. The [fixture report](swe-dev/fixture-validation.json) and disjoint
[train](swe-dev/fixture-train.json)/[empty eval](swe-dev/fixture-eval.json) manifests
are committed small examples. There is no held-out score for this single fixture.

Regenerate validation logs and manifests into configurable artifact storage:

```bash
uv run --locked --no-sync python -m self_summarization_agent.benchmarks.swe_dev.task_builder \
  --output artifacts/swe-dev/pilot
```

Only `environment/` is the image build context. Trusted tests and the reference
fix are outside it. The Dockerfile is an **unbuilt fixture example**, not a pinned,
validated upstream image. Fixture source IDs/base commits/image digests are
explicitly absent. Production entries require recovered source provenance,
40-character base commits, image digests, build-context and verifier hashes,
nonempty trusted tests, and repeated validation bound to that image/verifier.
Changes to manifests, task content, image, verifier, runtime, sampling or reward
profile invalidate collection/resume identity. Duplicate task/source/leakage IDs
and overlapping train/eval task/content identities are rejected.

The `local-fixture` environment accepts only fixed inspection/edit commands from
a generator explicitly labeled `fixture_only`. It is a CPU plumbing harness,
not a sandbox for arbitrary policies. The four checked-in `swe_dev_*` training
presets select this example and resolve binary rewards; the tests replace their
large-model paths with the tiny fixture model. They are not a command to launch
an unrestricted model against the host shell.

## Collection, verification, selection and cache

Use `rollout_collection` and the staged `iteration_launcher` for recovered tasks.
The older BrowseComp-only run/train launchers reject SWE configurations. SWE
uses the existing terminal collector/scaffold; no second policy loop or renderer
is introduced. No retrieval worker or answer judge starts. `judge_step` converts
recorded verifier outcomes even when `judge.enabled` is false. Expected evaluation
counts and stable selection come from the resolved eval manifest, independently
of iteration sampling seeds.

Every fresh attempt has its own workspace. A successful summary changes model
context only. Candidate export occurs before verification and cleanup, includes
untracked files, additions/deletions, binary content and modes, and preserves a
complete snapshot plus a readable binary-capable patch and digests. Version 1
rejects symlinks/special files rather than following them, and excludes `.git` and
Python bytecode caches. The snapshot is the reconstruction authority; candidate
Git hooks/configuration are not executed.

The SWE Harbor trial hook stops the agent container, copies the repository from
the stopped environment, and verifies the frozen snapshot in a fresh pinned
container. It disables network, drops capabilities, applies CPU/memory/PID/tmp
limits, uses a read-only repository/root, and mounts the trusted verifier outside
the candidate. It never reads an agent-writable reward file. Trusted verifier
implementations must own discovery and isolate candidate code from their reward
channel; the fixture checker evaluates child function results in a trusted parent,
so candidate output is never accepted as a discovery/reward report. Missing or
inconsistent trusted discovery is infrastructure failure. Candidate correctness
failures/timeouts remain task failures.

**Docker execution remains unvalidated here:** no `docker` executable is present.
The fresh-container hook has not been exercised against recovered upstream tasks.
Production dependency/write requirements, host/container trust hardening, disk
quota enforcement, cancellation under real workloads, and measured sandbox
resource use remain operational gates. Harbor's declared storage setting is not
proof of an enforced disk quota. Resource measurements are reported unavailable,
not fabricated as zeros. Real-task verifier scripts must be audited and pinned;
the fixture Python checker protocol is not a universal SWE test converter.

SWE uses `binary_zero_one`: correct eligible output gets `1`; ordinary failures
and malformed/empty/oversized summaries get `0`; infrastructure failures get no
training reward. Runtime penalties take precedence over a passing patch unless
verification itself failed. History rewrites and broken/missing interval chains
exclude the whole rollout. All K attempts must be eligible before a task group
is selected. Raw attempts remain intact; selection metadata and sidecar reports
explain exclusions. Empty selected batches fail before writing a checkpoint.

Exact IDs remain authoritative. Cache v7 uses unchanged `F[:-1]`, `F[1:]`, and
`M[1:]`. Valid raw-policy log probabilities use `vllm_raw_rollout`; otherwise the
frozen behavior checkpoint rescoring path computes probabilities for the saved
IDs. No historical tokenization fallback is introduced. Cache resume compares
its source evidence, rewards and selection, and rejects changed inputs.

## Four training profiles and resume

| Preset | Baseline and targets |
| --- | --- |
| `swe_dev_grpo.yaml` | Group-relative normalization over distinct rollout outcomes, broadcast to intervals; zero-variance groups logged |
| `swe_dev_value_mc.yaml` | Shared binary head, frozen old `V`, advantage `R - V`; rollout-normalized critic weight |
| `swe_dev_separate_mc_ppo.yaml` | Fit separate MC critic, then freeze post-fit `R - V` for PPO |
| `swe_dev_separate_gae_ppo.yaml` | Freeze pre-fit values/GAE targets/advantages with gamma 1, lambda .95 and 51 bins before fitting |

SWE binary heads predict success probability directly; zero initialization gives
`0.5`. Categorical supports and targets are `[0,1]`, and negative advantages
remain valid. Diagnostics use the same support and a `.5` success threshold.
Signed heads, legacy targets and other benchmarks keep their historical defaults.

SWE checkpoint contracts bind task/image/verifier identity, runtime, sampling,
reward support and training method. Actor optimizer moments/RNG and separate
critic optimizer state persist. Head manifests and actor/critic pair manifests
reject incompatible reward scales or target provenance. Native verl persistence
also records topology and requires model/optimizer/RNG shards; its distributed
execution remains a separate real-GPU validation gate. The CPU integration tests
exercise the Transformers path, not the native FSDP engine.

The end-to-end tests collect four attempts with mixed `0/1` outcomes and unequal
interval counts, verify candidate code, rescore exact IDs, perform finite updates,
assert actor weights changed, then collect under the saved checkpoint and update
again. They assert monotonically accumulated actor optimizer steps and separate
critic update counts, plus incompatible-resume rejection. Value pretraining stays
disabled; its BrowseComp guard was not removed.

## Validation commands and remaining gates

```bash
uv run --locked --no-sync --extra terminal-bench pytest -q \
  tests/test_swe_dev_dataset.py tests/test_swe_dev_verification.py \
  tests/test_swe_dev_rewards.py tests/test_swe_dev_training.py
```

`--no-sync` preserves the user's existing dependency/lockfile edits and uses the
installed environment. Harbor 0.1.44 is installed there. The optional extra was
already moved into base dependencies by those edits; this implementation does
not rewrite them.

The new suite checks zero/one/multiple compactions, failed summaries, sampled
termination/thinking closure, saved-record TITO verification, native and rescored
caches, group exclusion, candidate test tampering, provenance, and update/resume
for all four methods. Existing runtime/TITO/terminal tests additionally cover
forced completion, budget priority, oversized observations, command bytes,
cancellation/confirmation and linked tool IDs. Existing benchmark tests are the
signed-reward regression gate.

Before claiming full SWE-Dev validation, recover trusted executable tasks, pin and
repeatedly validate their images/tests/reference fixes, establish task/issue/patch
disjointness, run Docker isolation/oracle trials, collect fresh Qwen/vLLM policy
rollouts, verify saved records with `scripts/verify_tito_records.py`, and perform
GPU update/resume trials and resource measurements. Only then expand task count
and measure environment/inference throughput. `LockedPolicy` serializes inference;
concurrent environments are not cross-episode model batching.

Recorded validation evidence is in [validation-2026-11-02.json](swe-dev/validation-2026-11-02.json).
The regression run passed 692 cases with four skips and two GPU cases run
separately. The two-GPU replicated-head check passed separately. The Qwen dense
checkpoint-gradient case fails with a maximum absolute difference
`2.5272369384765625e-05` against `1e-05`; the identical failure was reproduced using
an untouched archive of the original Git HEAD and the same installed dependencies.
Its implementation and tolerance were not changed. The focused SWE and signed
value-pretraining run passed 79 cases, including all 32 new SWE cases. Trial logs
remain under configurable/ignored artifact storage; no real-task accuracy is
reported from these scripted fixtures.

The Qwen gradient-test failure recorded above was subsequently investigated and
resolved as a comparison of different floating-point calculation layouts.
The independent reference now matches the token-local arithmetic layout while
still resetting the unpatched attention mixer at each interval. All existing
tolerances remain unchanged; the expanded GPU suite passes 13 checks. The
[follow-up investigation](analysis/2026-11-02-qwen-gradient-investigation.md)
records the operation trace, FP64/finite-difference evidence, kernel-precision
control, optimizer-step comparison, and the combined 117-passing-test result.
The earlier validation JSON remains the historical record of that earlier run.
