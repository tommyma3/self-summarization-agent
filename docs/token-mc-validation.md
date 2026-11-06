Token MC validation — started 2026-11-02
======================================

Collection and token-cache v7 are unchanged. Frozen token artifacts use schema
v3, compact float32 safetensors arrays under `token_values/`, and a digest-bound
JSON manifest. Existing scalar artifacts and value-configuration fingerprints
retain their previous schemas.

The targeted CPU acceptance run passed **322 tests, with four existing tests
skipped**. It covers real tiny GPT-2 pre-token causality, disjoint assistant
spans, hand-computed weighted CE/gradients, short final minibatches, microbatch
invariance, vector PPO/gradients, scalar equivalence, zero-average advantages,
empty training views, corrupt/mismatched artifacts, immutable interval evidence,
no/one/repeated compaction, forced answers, failed/empty/over-limit summaries,
query-disjoint token warmup selection, exact interrupted optimizer/RNG recovery,
paired checkpoint reload and a second update. Additional span diagnostics and
sidecar regression checks passed. Basic correctness lint and `git diff --check`
passed. Installed backend versions and the exact verl commit are recorded in
[`runtime-versions-2026-11-02.json`](token-mc/runtime-versions-2026-11-02.json).

A synthetic **critic-only** production-length probe used Qwen3.5-0.8B, the
Qwen3.5-9B actor tokenizer, bfloat16, activation checkpointing, microbatch one,
and 27,000 shifted input tokens on an A100 80 GiB. Its update and predictions
were finite. Peak allocated GPU memory was 7,976,892,928 bytes (7.43 GiB); fit
plus final evaluation took 88.97 s and total load/probe time was 99.07 s. The
26,969 supervised tokens correspond to about 303 trained tokens/s including
final evaluation and first-use kernels. Repeated synthetic IDs measure capacity,
not benchmark quality. Measurements are in
[`memory-probe-2026-11-02.json`](token-mc/memory-probe-2026-11-02.json).
Reproduce with `scripts/benchmark_token_value.py --output /path/to/probe.json`.

The real smoke used `compact_separate_token_mc_smoke.yaml` and a separate
`experiments/token-mc-bcplus-smoke` root: two training questions with two attempts,
a two-question warmup with query-level 1/1 fit/validation split, one warmup epoch,
shorter generation budgets, and a 2,000-token compaction threshold. Models,
retrieval/index, judge, dataset partition and four-rank FSDP/Ulysses topology match
the production preset. This is not a full benchmark result.

Real token warmup completed before online collection and selected epoch zero by
held-out weighted Brier score. Two real native verl/FSDP PPO updates completed;
both exported complete actor/critic pairs accepted by the repository validator.
Both checkpoints were reloaded by real vLLM, and explicit final-checkpoint
benchmark evaluation completed. Checkpoint one used 4,096 sampled tokens and
one actor optimizer step (finite loss 1.00067997); checkpoint two used 3,102
sampled tokens and two actor optimizer steps (finite loss 0.97692927). The second
update's pre-fit values ranged from -0.03733611 to 0.00311279, with token-advantage
standard deviation 0.00581304. Its source critic identity matches checkpoint
one's fitted critic identity. The critic optimizer reached four online steps.
Authoritative ledger spans supply research/tool, summary and final-answer
strata; missing metadata is explicitly counted rather than reconstructed.

Warmup raw/judged/cached evidence and the first online collection retained their
file hashes. Real retries before the sidecar correction had identical frozen
baseline digests; dedicated retry tests verify immutable sidecar reuse and
fail-closed corruption handling. Per-update measurements and current paired
provenance are saved in
[`smoke-update1-2026-11-02.json`](token-mc/smoke-update1-2026-11-02.json),
[`smoke-update2-2026-11-02.json`](token-mc/smoke-update2-2026-11-02.json), and
[`smoke-provenance-2026-11-02.json`](token-mc/smoke-provenance-2026-11-02.json).

Real validation found and fixed two integration problems. Ray initially tried
to package local checkpoints (15.9 GiB); the new presets exclude experiments,
data, VCS and virtual environments, reducing the bundle to 6.85 MiB. vLLM also
scanned a root-level frozen tensor shard as model weights. Frozen shards now live
under `token_values/`, with regression coverage; the invalid prototype export was
archived and regenerated from the unchanged committed input and cached data.
An earlier development process imported the sample class before provenance
fields landed; restarting reused the unchanged collection.

A production-size attempt was started under `experiments/token-mc-bcplus` with
50 training questions, four attempts and 100 evaluation questions, resuming the
validated smoke pair. **It was stopped during evaluation at the user's explicit
request.** No production-size iteration or full-budget benchmark result is
claimed, and the full evaluation must not be resumed without a new instruction.
`validation_initialization.json` records the initial state; partial artifacts are
retained for provenance. The matched interval control is delivered but no
full-budget control comparison or task-success improvement is claimed.

Code acceptance is based on the CPU checks, finite production-length critic
probe, real token warmup, two finite native FSDP PPO updates with resume,
matching source/fitted critic identities, unchanged evidence, and successful
smoke-checkpoint reload/evaluation. Full evaluation was intentionally omitted at
the user's request.
