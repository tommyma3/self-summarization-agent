Terminal-Bench 2.0 / Terminus-KIRA implementation plan

Status: implemented on 2026-10-03 with local tests and a real Qwen/vLLM transport probe. See [usage and validation](terminal-bench.md). Docker benchmark/oracle execution and real vision inference remain unverified on this host. The approved frozen vision helper was implemented as local inference; image bytes are not sent to an external model endpoint.

Add a Harbor external-agent adapter that combines Terminus-KIRA terminal behavior with this repository's existing append-only runtime, token ledger, and training artifacts. Harbor owns task environments and verification; the project owns every trainable policy generation and compaction boundary. Report the resulting scaffold as `terminus-kira-tito`, with its upstream revision and documented adaptations.

The governing contract remains AGENTS.md and the TITO amendment in `docs/superpowers/specs/2026-07-31-append-only-compaction-interval-design.md`. This plan does not authorize a rollout-format redesign.

**Evidence and integration constraints**

The repository already has `IntervalTokenLedger`, `TokenRequest`, `QwenAgentTokenRenderer`, token-input backend methods, exact-token trajectory extraction, and cache lineage checks. However, `EpisodeRuntime` hard-codes search/document/finish dispatch; `parse_native_completion` accepts only those tools and one string parameter; the HTTP forced-answer regex requires `finish(answer)`. Dataset loading, collection, retrieval-worker startup, and judging also assume BrowseComp. Adding only a Harbor dependency and shell backend would leave these paths incompatible.

Upstream KIRA provides `execute_commands`, `task_complete`, and `image_read`, terminal polling, and a completion confirmation checklist. Its current generation path uses LiteLLM messages, reconstructs assistant history, unwinds context after overflow, and substitutes text after truncated output. Its image tool performs a separate model call. These paths need explicit adaptation before they can satisfy the local contract. See the [KIRA implementation](https://github.com/krafton-ai/KIRA/blob/main/terminus_kira/terminus_kira.py).

[Harbor supports external agents](https://docs.harborframework.com/core-concepts/agents/custom-agents), allowing the token-aware policy loop to remain outside task containers. [Harbor tasks](https://docs.harborframework.com/core-concepts/tasks/overview) provide instructions, environments, and verifier scripts that emit numerical rewards. The official benchmark identifier is `terminal-bench@2.0`; record the resolved dataset revision as well as that label. See the [benchmark publisher's usage instructions](https://huggingface.co/datasets/harborframework/terminal-bench-2.0).

1. **Pin dependencies and establish the `uv` environment.**

   Keep the existing Python 3.12 constraint and use the repository `.venv`. Add an optional `terminal-bench` extra with an exactly selected Harbor release; commit the resolved `uv.lock`. Select the version after checking the KIRA/Harbor API combination rather than assuming today's Harbor works with any KIRA checkout. [KIRA's dependency metadata](https://github.com/krafton-ai/KIRA/blob/main/pyproject.toml) currently specifies Python >=3.12 and Harbor >=0.1.44, which establishes a lower bound, not a tested compatibility pin.

   Prefer a small attributed port of KIRA's prompt, tool schemas, terminal execution helpers, and confirmation behavior under the benchmark package. Record the exact upstream commit and retain its license notices. This avoids inheriting its message-history loop or relying on its repository-relative prompt imports. Check prompt resource packaging in the built wheel.

   During implementation, use `uv venv --python 3.12`, `uv lock`, `uv sync --locked --extra terminal-bench`, and `uv run --locked --extra terminal-bench ...`. Do not use ad hoc pip installs or a separate unmanaged environment. Check resolution against the existing CUDA, vLLM, Torch, and Transformers pins without unrelated upgrades. The README currently references development groups/extras absent from the inspected pyproject; use commands supported by the actual manifest. Docker is a host prerequisite, separate from Python dependencies.

   Exit criterion: a clean locked sync can import the adapter and Harbor on the supported host; existing BrowseComp imports still work without the new extra.

2. **Separate benchmark behavior from interval management.**

   Add a small benchmark/scaffold interface, with a BrowseComp implementation preserving current defaults and a Terminal-Bench implementation. It supplies stable system text, tool schemas, schema-aware action parsing, execution, completion handling, summary guidance, forced-answer guidance, and tool-budget accounting. Keep ledger creation, immutable appends, generated-token accounting, failure handling, and compaction in the shared runtime.

   Likely edits: `runtime.py`, `prompts.py`, `token_renderer.py`, `generation.py`, `config.py`, and `launcher_utils.py`; proposed new modules: `benchmarks/base.py`, `benchmarks/browsecomp.py`, and `benchmarks/terminal_bench/{scaffold,tools,harbor_agent}.py`.

   Move parsed-action construction out of backend-specific inference code so offline vLLM and the token-capable HTTP endpoint use the same benchmark parser. Support nested command arrays, numeric durations, escaped keystrokes, and parameterless completion. Never trim command strings or normalize sampled history. Retain the current single native tool call per assistant turn, with multiple commands inside `execute_commands`; explicitly reject ambiguous/multiple calls before execution and document this scaffold adaptation.

   Parameterize control text and generation constraints together: the actual renderer append must match the recorded control. Preserve the initial tool definitions during summaries and forced completion; generation restrictions must not alter the earlier prefix. Avoid globally changing BrowseComp's parser, prompts, or reward semantics.

   Exit criterion: existing BrowseComp contract tests pass with identical default behavior, and synthetic terminal actions use the same ledger path.

3. **Implement Harbor task and terminal lifecycle.**

   Use a Harbor external-agent class as the entry point and a per-trial terminal adapter. Each task/attempt gets an isolated environment, persistent tmux session, original instruction, runtime state, and output directory. Compaction preserves the live shell, files, working directory, processes, and completion-confirmation state; it replaces only model context.

   Port command execution, empty-command waiting, incremental output, marker polling, and timeout protection. Produce real linked `role: tool` observations after execution. Output filtering or bounding is allowed only while constructing a new observation, before its first append; save raw terminal artifacts separately and never edit committed observations. Keep an explicit observation budget that reserves space for a legal summary or forced-answer boundary.

   Implement normal completion as first `task_complete` -> appended confirmation observation -> second `task_complete` -> terminal interval and verifier. Keep this state across summaries. Budget-forced completion appends a terminal system control and generates a constrained final completion without executing additional commands; record it as forced, even if verification later succeeds. Generated-token budget exhaustion takes precedence over compaction. Hard infrastructure cancellation saves the valid prefix and a distinct failure status without fabricating an assistant answer.

   Bridge Harbor's async environment API to the runtime without blocking the orchestration event loop or sharing sessions between episodes. Retry inference only from an identical committed request when no completion was accepted. Do not automatically replay a shell command after a timeout with an unknown execution outcome. Clean up environments after verification and artifact export, including exceptions.

   Exit criterion: a local fixture modifies a file, retains shell state across turns and compaction, confirms completion, and obtains a verifier reward.

4. **Make TITO a required capability of this benchmark.**

   Reject production configurations without token-input generation, authoritative returned IDs, and the approved renderer. Start validation with offline vLLM and the vLLM-compatible integer-prompt Completions endpoint. Ordinary message-only API providers cannot satisfy this requirement.

   For every generation in an interval, assert:

   ```text
   P[k+1] = P[k] || C[k] || E[k]
   ```

   `P` is the exact submitted prompt, `C` the exact sampled IDs, and `E` only newly encoded observations, controls, separators, and the next assistant header. Preserve prompt/completion/full IDs, raw completion, finish reason, logprobs when available, tool-call IDs, and append-time masks. Every sampled token is trainable; all external tokens are conditioning. Parsed actions and Harbor trajectory exports are diagnostic views, never prompt or training reconstruction sources.

   Successful compaction must append its terminal user request and raw assistant summary to the old ledger, finalize that ledger, then initialize a successor with the byte-identical system/query and wrapped extracted summary. Keep no raw event tail. Malformed, empty, and oversized summaries terminate without installing state. Require byte-exact thinking closure and supported sampled termination for actions, summaries, natural completion, and forced completion. Never repair truncated output or unwind history.

   A mismatched backend input fails closed; protocol errors abort collection. Extraction must still raise `ProviderHistoryRewriteError` for invalid saved evidence. Reject oversized training intervals rather than truncating them.

   Exit criterion: exact-ID fixtures using noncanonical tokenization prove there is no decode/re-encode path after initialization, including backend requests and cache construction.

5. **Handle image analysis explicitly.**

   Proposed initial complete profile: implement `image_read` as a separately configured, frozen vision service. Read the image inside the trial environment, record its content hash and service/model configuration, and append its returned text as a linked conditioning-only tool observation. Report its usage separately. It must never be a hidden call to the changing trainable policy. This keeps every policy generation on the shared TITO ledger while making the auxiliary service's role explicit.

   Provide a separately named text-only profile for initial smoke tests, with a stable tool set chosen before interval initialization. It is a partial milestone, not full KIRA parity. A full image-enabled run must fail configuration validation if its vision service is unavailable; do not silently omit image tasks or change tools mid-interval.

   If image reasoning itself must be optimized as part of the trainable policy, treat that as additional design work: the current text-only token contract does not describe multimodal conditioning payloads or auxiliary policy trajectories. Do not claim that encoding a base64 string or omitting the helper's sampled tokens solves that problem.

6. **Integrate collection, verification, and training artifacts.**

   Add a benchmark selector defaulting to BrowseComp, plus dataset revision/task filters, environment provider, concurrency, timeout, scaffold revision, and image profile. Load Terminal-Bench instructions without requiring reference answers or retrieval indexes. Adapt `dataset.py`, `rollout_collection.py`, `merged_collect_step.py`, launcher configuration, and judge dispatch so terminal runs do not start retrieval or answer-judge workers.

   Preserve Harbor's raw verifier reward separately from the training reward. Use benchmark verification for success/failure, retain existing malformed-output penalty precedence, and distinguish infrastructure/verifier errors from valid task failures. Broadcast the terminal training reward to all intervals from the rollout; retain rollout-level GRPO normalization and interval-start value-MC anchors. Feed the existing cache/training pipeline stored IDs and masks directly.

   Extend `collection_contract.py` and cache/resume identity with dataset/task revision, scaffold/prompt/tool schema digest, relevant environment identity, image-service configuration, and benchmark reward mapping. Keep schema 3 evidence and the existing cache format where sufficient; version any necessary additions explicitly. Resume completed trials by stable task/attempt keys. Restart interrupted stateful trials in fresh environments unless a complete environment/session/ledger checkpoint exists.

   Default official Terminal-Bench 2.0 tasks to evaluation. Make any training dataset or deliberate benchmark-task training explicit and separately label those results. Add matching compaction/no-compaction evaluation configs and a fixture/smoke config, recording task coverage, attempts, pass rate, tokens, summaries, timeouts, malformed output, and integrity failures.

7. **Validate in increasing scope and document the supported profile.**

   Run all Python checks with `uv run --locked --extra terminal-bench`. Establish a current baseline; the September validation note reports historical failures and pending real-engine validation, not a guarantee about today's tree.

   Add focused tests for terminal schema parsing and command fidelity; linked tool IDs; multi-command order; timeout outcomes; completion confirmation; environment isolation; and unchanged shell state across summaries. Cover no compaction, one and multiple compactions, forced completion, simultaneous token-budget/summary triggers, malformed/empty/oversized summaries, incorrect thinking closure, missing end tokens, missing/mismatched exact IDs, context overflow, and concurrent trials. Verify archived intervals never change.

   Test full collection -> verifier -> exact-token extraction -> cache -> one training update for GRPO and value-MC, including rollout-level reward weighting and native/rescored logprob provenance. Corrupted evidence must fail before training. Test resume rejection after task/scaffold/tool/profile changes. Test frozen image responses as external masked spans with separately tracked usage.

   Then run Docker fixture trials, an oracle sanity check, and a small pinned official task subset with a real token-capable backend. Obtain both a no-compaction trace and a successful compaction trace; validate saved records with `scripts/verify_tito_records.py`. Expand to the full pinned dataset only after token integrity and verifier integration pass. Provide reproducible `uv` commands, adapter import path, backend requirements, deviations from upstream KIRA, and artifact locations in the README.

Implementation order: dependency pinning -> benchmark interface extraction -> terminal adapter and parsing -> exact-token boundary tests -> verifier/collector integration -> image-enabled profile -> real-backend and training smoke checks. Each stage should be reviewable independently; benchmark integration is complete only when the selected full profile and its TITO acceptance checks pass.
