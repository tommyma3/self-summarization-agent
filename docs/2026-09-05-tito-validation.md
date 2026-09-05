# TITO implementation and validation

Date: 2026-09-05. Implementation complete; real inference-engine and distributed GPU validation remain pending.

## Implemented behavior

`IntervalTokenLedger` stores immutable request snapshots and append-time token ownership. `QwenAgentTokenRenderer` formats only new tool results, runtime controls, and assistant headers after initialization. It requires the repository's Qwen template and atomic ChatML boundary tokens. No continuation renders or retokenizes historical assistant text. Only successful compaction finalizes the old ledger and initializes a successor from stable instructions, the original query, and wrapped summary state.

The runtime submits explicit IDs through offline vLLM, SGLang, the vLLM-compatible Completions endpoint, or Transformers (including trainer-owned collection). Native API calls are parsed locally for routing and linked diagnostic messages; their sampled IDs are retained verbatim. Summary/forced-answer API restrictions use structured generation. Unsupported token seams and input mismatches fail closed. The known `</thinking>` typo may continue only for an unambiguous tagged tool action with a sampled end-of-turn marker; incomplete reasoning never dispatches a tool.

Every completed interval remains a training sample. All sampled assistant tokens, including sampled stop tokens and summary output, receive the assistant mask. Tool/control/header tokens and the next interval's summary copy are conditioning. GRPO still normalizes across distinct rollouts; value-MC still uses each interval's initial state and the rollout's terminal reward. Neither objective nor distributed loss weighting was changed.

Collection-token schema 3 records the renderer fingerprint, exact generations, and append spans. Training-cache version 7 preserves this identity and validates cached IDs/masks against the collected sequence. Native raw logprobs are reused only with established provenance; otherwise training rescoring consumes the stored IDs. Resume preflights reject incompatible collection lineages before artifact-completion shortcuts.

## Local results

Tests used an isolated Python 3.12.14 runtime with pytest 9.1.1, Transformers 5.16.1, and CPU PyTorch 2.14.0. Dependencies are under ignored `.local-test-deps`; the project venv was not repaired. The public Qwen/Qwen3.5-9B tokenizer was downloaded without model weights for conformance checks.

| Check | Result |
| --- | --- |
| Dedicated `tests/test_tito.py` | 38 passed, including four real-tokenizer cases |
| Full working-tree suite | 254 passed, 11 failed, 1 skipped |
| Untouched HEAD, same local dependencies | 216 passed, the same 11 failed, 1 skipped |
| Comparison of JUnit failure identities | No new failures |
| Python AST parsing and `git diff --check` | Passed |

The dedicated suite covers noncanonical sampled IDs, no/one/multiple compactions, immutable prefixes, control boundaries, both tool protocols and thinking modes, malformed summaries and thinking, concurrency, overflow, corrupted token/mask/cache identity, each token adapter, and exact-ID rescoring followed by a CPU update in both RL modes. The rescoring test compares native and rescored token logprobs with absolute tolerance `1e-6` on a causal toy policy; it is not an inference-engine numerical parity measurement.

The existing failures were reproduced on untouched HEAD and left outside this transport change:

- `test_context`: `test_summary_instruction_is_appended_to_unchanged_context` (old expected prompt wording).
- `test_prompts`: `test_build_summary_system_prompt_is_concise_and_requires_wrappers`, `test_system_prompt_lists_normal_action_formats` (old wording).
- `test_runtime`: `test_runtime_second_step_finish_sees_raw_history_and_succeeds` (old wording); `test_native_summary_appends_user_control_without_changing_tools_and_keeps_exact_interval_ids` (legacy fixture's summary reservation exhausts its context limit).
- `test_iteration_launcher`: `test_iteration_launcher_delegates_retrieval_ownership_to_merged_collect` (command matching on Windows).
- `test_judge_worker`: both microbatch tests (mock configuration lacks `batch_timeout_seconds`).
- `test_merged_collect_step`: `test_cache_overlap_worker_microbatches_samples_across_judged_rows` (worker response type differs from fixture expectation).
- `test_rollout_collection`: `test_collect_rollouts_resume_skips_existing_rows` (resume fixture lacks trajectory records).
- `test_trainer`: `test_compaction_value_forward_selects_only_anchor_and_trainable_logits` (mock trainer lacks tokenizer).

The skipped existing test requires two CUDA GPUs. Local JUnit files are `.local-test-deps/baseline-results.xml` and `.local-test-deps/current-results.xml`.

## Server handoff

Run from the repository root with the configured Linux environment activated. Replace the tokenizer path if needed:

```bash
export TITO_TEST_TOKENIZER_PATH=/124090467/Qwen/Qwen3.5-9B
python -m pytest tests/test_tito.py -q
python -m pytest tests -q
```

Do not assume the 11 local baseline failures will all occur on Linux; compare remaining failures with this list. The local backend adapter tests use engine/client doubles. Actual vLLM, SGLang, HTTP structured generation, CUDA, NCCL, and four-rank VERL execution have not run locally.

Start with an existing offline-vLLM collection probe:

```bash
python scripts/probe_vllm_collection.py \
  --config configs/train/compact_value_mc.yaml \
  --sample-index 0 \
  --set runtime.tool_budget=4 \
  --set runtime.context_threshold_tokens=1 \
  --output /tmp/tito-compaction-probe.txt
```

Inspect the trace: a threshold of 1 requests compaction after a completed tool round, but the policy may finish or be malformed before that round. Obtain at least one successful compaction and an ordinary no-compaction rollout. Also exercise forced answers. Confirm each next prompt extends the preceding sampled sequence, and each archived interval has schema 3 tokens, consistent spans, and complete assistant ownership. Repeat real-backend smoke checks for SGLang and the token-capable vLLM API if those backends will be used; the API must support integer prompts, returned token IDs, and the configured structured-output constraints.

Then use the existing `scripts/train_iterations.sh` launcher with a fresh `--latest-root` and fresh `experiment.output_root` for each RL mode. Run a small group containing different terminal rewards and at least one compacted rollout through collection, judging, cache creation, and one GRPO update; repeat using `configs/train/compact_value_mc.yaml` for the four-GPU value-MC path. Verify retained segment counts, first-generation value anchors, native-versus-rescored logprobs at a recorded tolerance, finite gradients, and checkpoint/value-head publication. Re-run the same iteration to check completed-phase reuse, then advance one iteration to check loading and resumed collection.

Record prompt-construction time, collection throughput, peak GPU memory, token-mask coverage, integrity failures, and actual malformed-output reasons before a long run. No performance or reward recovery claim is established by the local tests.

Old raw/judged/cache files must not be resumed into this collector. Keep previous experiments intact and collect fresh TITO artifacts. Existing policy/value weights can be initialization candidates after ordinary checkpoint compatibility checks; they do not make old rollout data compatible with the new collection contract.

## 2026-09-05 addendum: versioned termination-semantics change (contract v2)

The TITO contract was bumped from `qwen-agent-tito-v1` to `qwen-agent-tito-v2`
(`token_stream.TITO_CONTRACT`). Under v2, any TITO completion whose think block is
unclosed or wrongly closed — including the `</thinking>` typo and case/whitespace
variants of `</think>` — terminates as `malformed_tool_call` (raw tokens retained and
trainable, −1 penalty) for every generation kind (action, finish, forced answer,
summary). v1 routed an unambiguous tagged action after a `</thinking>` typo instead.
This is an RL reward-semantics change: collect in a fresh output lineage, and never
resume v1 raw/judged/cache artifacts into a v2 run — lineage preflight rejects the
contract mismatch.
