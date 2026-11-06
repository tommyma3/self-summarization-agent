# SWE-Dev RL: closing the production-readiness gaps

Status: implementation reached the Phase 1 recovery gate, 2026-11-02. Recovery tooling and a scoped inventory/report are implemented; zero upstream specifications qualify. M2–M7 are pending the required independent task assets. See [recovery evidence and procedure](../swe-dev/recovery.md). No real-task Docker validation or GPU pilot run is claimed.

This plan builds on the current implementation and supersedes the outstanding-work assessment in the earlier [integration plan](2026-11-02-swe-dev-training-environment.md). Keep that document as design history. Current implementation evidence is in [docs/swe-dev.md](../swe-dev.md). The four focused SWE test modules were rerun during this assessment: **32 passed**. They exercise deterministic fixture completions and CPU training, not fresh on-policy SWE-Dev learning.

## Objective and scope

Deliver a reproducible pilot in which real, independently specified SWE-Dev tasks produce fresh Qwen/vLLM rollouts, trusted verifier rewards, exact-token training caches, GPU updates, and resumed training. Validate GRPO first, then shared value-MC, separate MC-PPO, and separate GAE-PPO. Deliver a fixed held-out evaluation and measured operating limits before increasing scale.

Reuse the existing Harbor/Terminus-KIRA collector, EpisodeRuntime, manifests, reward mapping, selection, cache, and trainers. The benchmark dispatch and four RL objectives already exist; do not reimplement them. Leave value pretraining disabled. Do not make distributed training or cross-episode batching a prerequisite for the first pilot; validate them separately when selected for deployment.

## Non-negotiable data contract

Follow [AGENTS.md](../../AGENTS.md) and the [append-only interval design](../superpowers/specs/2026-07-31-append-only-compaction-interval-design.md).

- Identify the affected interval boundary before changing runtime-adjacent code. Every non-compaction transition appends only.
- Preserve stable system instructions and original query, raw sampled output, native tool-call IDs, exact token IDs, and masks. Never reconstruct historical prompts or silently retokenize training data.
- Successful compaction finalizes the immutable old interval before initializing its successor from the unchanged system/query and wrapped extracted summary. The live environment persists across compaction.
- Malformed summaries remain recorded and receive the configured penalty; failed summaries never become state. History rewrites fail closed.
- Training consumes every eligible interval; reward normalization remains per distinct rollout. Preserve current critic weighting and baseline timing.
- Dataset transcripts and historical rewards are not fresh on-policy evidence. Hidden tests, reference patches, and future verifier results must not enter actor or critic observations.

A proposed change to this contract requires explicit approval and is outside this plan.

## Phase 1 — Recover executable task assets

**Priority: first feasibility blocker.** The existing pinned-source audit found zero complete executable task specifications. Do not repeat that audit unless inspecting new assets or revisions.

Work:

1. Inventory local/connected assets and any newly available upstream releases against the recorded missing-assets list. Record source revisions, hashes, access requirements, and recovery outcomes. Contacting maintainers is a separate user-authorized action if needed.
2. Recover a canonical instruction, repository URL/base commit, setup/dependency provenance, trusted fail-to-pass and pass-to-pass tests, a reference fix, and verifier provenance for each candidate. Resolve upstream origin and instance identities independently of conversation text.
3. Create a machine-readable recovery manifest with per-field provenance and explicit exclusion reasons. Never guess commits, tests, or images from an assistant transcript.
4. Define leakage groups using source task/issue identity and patch/content fingerprints. Reserve evaluation membership before collecting training rollouts. Record overlap checks against any intended external evaluation set.

Deliverables: updated coverage report, recoverable task specifications, exclusions, and a versioned split manifest proposal.

**Gate:** at least one independently recoverable upstream task before claiming a real-task smoke test; target 10–20 validated tasks for the subsequent pilot. If fewer are recoverable, report the actual count and limitations. If none are recoverable, stop upstream pilot expansion and document exactly which assets must be supplied. A newly generated task/test dataset is a separately named variant, not silently equivalent to the original SWE-Dev release.

## Phase 2 — Build and validate production task bundles

Primary files: `benchmarks/swe_dev/task_builder.py`, `dataset.py`, and new focused builder tests under `tests/` (source paths are relative to `src/self_summarization_agent/`).

Work:

1. Preserve the deterministic fixture builder and add a production import/build/validate path for explicit recovered specifications. Support resumable task preparation with content-based identities and per-task failure reports.
2. Construct Harbor bundles with `/app/repo` at the pinned base commit. Build environments with resolved dependencies; pin the resulting image digest. Keep trusted tests and reference fixes outside agent image build contexts and mounts.
3. Run the baseline and reference fix in fresh verifier containers at least twice each. Check the expected failing tests, passing regression tests, nonempty discovery, and all-pass reference outcome. Reject flaky or incompletely specified tasks.
4. Emit manifests accepted by the existing loader: source/base identity, instruction/build/bundle digests, image, verifier/test identities, resource limits, reference-patch digest, and validation report digest.
5. Resolve pilot compatibility constraints before generalizing: full-tree candidate export currently has a hard-coded 16 MiB cap and rejects symlinks. Measure actual task trees. Make export limits explicit/configurable and part of resume identity if needed; retain fail-closed symlink behavior unless a tested preservation design is necessary. Do not replace authoritative snapshots with a lossy diff.

Deliverables: production builder, small schema examples, pinned task bundles/images in artifact storage, baseline/oracle logs, and disjoint train/eval manifests. Keep large repositories, images, logs, and patches outside source control.

**Gate:** every admitted task passes repeated baseline/oracle checks using its exact pinned verifier/image. A single task may advance to the smoke test while additional tasks are prepared.

## Phase 3 — Validate Docker execution and trusted verification

Primary files: `benchmarks/swe_dev/harbor_trial.py`, `verification.py`, shared terminal collection/session code only where a demonstrated defect requires it.

Work:

1. Preflight the installed locked Python/Harbor environment, Docker access, image availability, host resources, and configured GPU path. Do not upgrade dependencies speculatively.
2. Exercise fresh attempts, persistent terminal state, compaction, completion, container quiescence, snapshot export, fresh verification, and cleanup against a real task. Verify Harbor can export from the stopped container as the hook assumes.
3. Adapt the current single Python-checker protocol to actual task test runners as necessary. Declare required dependencies, working directories, and writable build/cache locations; use ephemeral writable areas without exposing trusted verifier material. A read-only repository may be incompatible with some real test suites.
4. Ensure the trusted harness owns discovery and reward reporting. Test candidate test/config changes, forged reports, modified reward files, and malicious output. Do not treat candidate-controlled stdout as a trusted result merely because it parses as JSON.
5. Verify isolation and actual enforcement of network, CPU, memory, process, temporary storage, and disk limits. Test timeout, cancellation, container-start failure, reset, and cleanup without orphaned workloads. Classify infrastructure failures separately from valid candidate failures.
6. Record resource measurements and enforced limits. A Harbor configuration field alone is not evidence of a working quota.

**Gate:** repeatable real Docker baseline/oracle trials, clean independent attempts, unchanged workspace state across summaries, verified reward isolation, and bounded cleanup/cancellation. Fix compatibility defects before allowing arbitrary policy execution. The local fixture command allowlist stays intact.

## Phase 4 — Collect fresh Qwen/vLLM trajectories

Primary files: new production presets under `configs/train/` and `configs/eval/`; existing `rollout_collection.py`, terminal `collection.py`, and `collection_contract.py` only if validation reveals gaps.

Work:

1. Add separately named production pilot configs using `environment: docker`, real manifests, model/checkpoint identity, tokenizer/template, explicit GPU placement, and compatible context/training limits. Preserve fixture presets. Start with one concurrent environment and K=4 independent attempts per training task.
2. Run a one-task collection smoke test, then the validated pilot set. Use the existing collector and staged iteration launcher; no BrowseComp retrieval or model answer judge should start.
3. Collect fresh no-compaction and compacted rollouts under matched task/model/sampling budgets. Exercise one and multiple summaries, forced completion, and malformed-summary handling. Use existing controlled tests for rare failure paths; do not present scripted failures as naturally observed model behavior.
4. Verify saved records with `scripts/verify_tito_records.py`. Check immutable interval prefixes, exact IDs, native linked tool messages, summary masks, successor summaries, and workspace continuity.
5. Verify reward conversion and complete-group selection. Preserve all raw attempts, explain exclusions, and reject broken token evidence. Validate exact-ID rescoring under the frozen behavior checkpoint; compare native raw logprobs where available using a declared numerical tolerance.

**Gate:** real model-generated, verifier-backed rollouts pass saved-record integrity checks and yield a nonempty eligible training cache. Include a real successful compaction. If binary rewards produce only equal-reward groups, report that GRPO has no reward-driven signal; broaden the pilot/task difficulty or sampling in a versioned experiment rather than fabricating successes or filtering individual attempts.

## Phase 5 — GPU updates and resume for all four methods

Work sequentially through GRPO, shared value-MC, separate MC-PPO, and separate GAE-PPO. Use the same validated task set and declared initialization/sampling controls, with fresh artifacts for each method.

For each method:

1. Run collection -> verifier import -> eligibility selection -> exact-token cache -> critic phase where required -> actor update -> checkpoint publication.
2. Require finite losses/gradients and actual parameter updates on a batch with nonzero training signal. Check retained interval counts, summary-token masks, rollout-level reward normalization, and critic weights. Validate the existing pre-fit/post-fit baseline semantics; do not change them to make tests pass.
3. Resume from the saved checkpoint, collect genuinely new completions from that checkpoint, and perform a second update. Verify optimizer progress, RNG/state restoration, critic state/pairing, and frozen-target provenance.
4. Test phase interruption/reuse and incompatible task/image/verifier/runtime/checkpoint rejection. Interrupted stateful attempts restart in fresh environments; do not reconstruct an active ledger or shell history.
5. Validate the chosen production trainer backend on real hardware. If native verl/FSDP is required, additionally test its topology, distributed reductions, optimizer/RNG shards, save/resume, and memory use. CPU or replicated-head tests do not substitute for that run.

**Gate:** an evidence matrix reports update/resume coverage for each method/backend and compaction profile. Full four-method support is claimed only after all four pass. A successful GRPO pilot is a useful earlier milestone, explicitly scoped as such.

## Phase 6 — Held-out evaluation and measured scaling

1. Freeze a nonempty held-out manifest and stable evaluation sampling; keep SWE-bench Verified and official Terminal-Bench evaluation-only by default. Report unresolved overlap limitations.
2. Evaluate initialization and trained checkpoints with and without compaction. Keep task selection, sampling, tool/runtime controls, and token ceilings comparable; report early context termination rather than truncating history. Small pilots establish functionality, not statistically persuasive learning gains.
3. Measure raw verifier pass rate, training-eligible rate, infrastructure failures, malformed outputs, history rewrites, excluded/zero-variance groups, updates, tokens, summaries, latency, GPU utilization, and sandbox resource use. Report valid-attempt and all-attempt denominators.
4. Increase environment concurrency gradually and measure the bottleneck. `LockedPolicy` currently serializes inference; add cross-episode model batching only if profiling warrants it, retaining each request's exact ledger IDs and immutable prefixes.
5. Publish an operations runbook with actual tested commands, dependency/image/checkpoint identities, artifact layout, resource requirements, failure triage, and resume behavior. Update `docs/swe-dev.md` to distinguish fixture, real-task, and backend-specific evidence.

**Gate:** reproducible held-out results, measured safe operating limits, and a repeatable launch/resume procedure. Throughput optimizations are a subsequent workstream unless the measured pilot is operationally infeasible.

## Delivery sequence and acceptance evidence

| Milestone | Depends on | Reviewable result |
| --- | --- | --- |
| M1: recoverable source | New trusted task assets | Recovery manifest and explicit exclusions |
| M2: one executable task | M1 | Pinned bundle/image plus repeated baseline/oracle evidence |
| M3: trusted environment | M2 | Docker lifecycle, isolation, verifier and cleanup tests |
| M4: fresh policy collection | M3 | Real Qwen/vLLM traces, integrity report and exact-token cache |
| M5: first RL pilot | M4 | GRPO GPU update, checkpoint and second fresh iteration |
| M6: four-method coverage | M5 | Value-method update/resume evidence and required backend validation |
| M7: measured pilot | M6 and a held-out task set | Evaluation, resource/throughput report and runbook |

Task recovery/building for additional tasks may proceed alongside validation of the first admitted task. Each code change should be independently reviewable; do not combine a renderer redesign or unrelated dependency migration with these milestones.

Run focused tests for affected components and the existing SWE suite after each relevant change. Before the pilot is accepted, also run runtime/TITO/terminal, cache/selection/launcher, reward/value, and checkpoint regressions. Preserve signed-reward behavior for BrowseComp and Terminal-Bench. Record real Docker and GPU tests separately from mocks and CPU tests.

The critical path is **trusted task assets -> validated environment/verifier -> fresh exact-token collection -> real GPU update/resume**. Task recoverability and available Docker/GPU infrastructure determine the schedule; assigning a completion date before those gates would be speculative.
