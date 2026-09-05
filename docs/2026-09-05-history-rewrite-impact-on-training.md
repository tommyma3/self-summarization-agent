# History Rewrite Detection: Impact on Training and Open Problems

Date: 2026-09-05
Status: historical diagnosis; the [TITO implementation](2026-09-05-tito-implementation-plan.md) now replaces mid-interval chat rendering. [Local validation and remaining server checks](2026-09-05-tito-validation.md) track verification. The observations and metrics below describe the earlier collector.

## Background

Commit `1f131ff` ("Unclosed thinking blocks now assigned as malformed response")
added exact-token history-integrity enforcement to the episode runtime:

- Every request to the vLLM/OpenAI-compatible server re-renders the full
  message history through the chat template. If an earlier assistant turn is
  malformed (e.g. a think block that never closes, or a misspelled closing
  tag), the template normalizes it, so the next request's `prompt_token_ids`
  no longer extend the previously sampled tokens. This is a **history
  rewrite**: the model continues from tokens it never generated.
- Under `require_exact_token_ids: true` (set in all train configs, including
  `configs/train/compact_value_mc.yaml`):
  - A `search`/`get_document` action whose think block never closes
    terminates the episode as malformed with reward −1
    (`runtime.py:867`).
  - Any other prompt mismatch terminates the episode with status
    `history_rewrite_detected`, reward −1 (`rewards.py:13`).
  - At extraction, records failing the prefix check raise
    `ProviderHistoryRewriteError` and are silently skipped
    (`trajectory.py`), instead of being trainable as before.

Commit `3340b49` ("Patch broken summary prompt reconstruct history") is
plumbing only: `build_generator` now forwards `require_exact_token_ids` from
config. All configs already set `true` and the generator default was already
`true`, so this commit alone does not change behavior.

## Observed impact on the value_mc run

Run: `artifacts/train/qwen-bcplus-compact-value-mc`. Iterations 13+ were
collected after `1f131ff` (checkpoint timestamps vs. commit times).

| metric | pre-change (iter 11) | post-change (iters 13–16) |
|---|---|---|
| malformed terminations (train, per 200 rollouts) | 31 | 55–56 |
| `history_rewrite_detected` (train, per 200) | 0 | 12–20 |
| `value/classification_accuracy` | 0.883 | 0.70–0.77 |
| mean_reward | 0.766 | 0.40–0.53 |

Reported eval accuracy dropped from 0.88 to 0.62 after the change and resume.

Interpretation: the commits changed the **reward function mid-training**.
Roughly 10–25% of episodes per iteration changed reward (new −1 penalties or
discarded intervals). The value head shares the backbone with the policy and
was asked to fit a shifted return distribution on a checkpoint trained under
the old one. This distribution shift is the likely cause of the collapse,
not a resume/checkpoint bug.

Note: the old 0.88 is not a valid fallback — part of it was earned on
rewritten-history samples that the current contract (AGENTS.md) correctly
rejects. Reverting the commit would silently reintroduce training on
rewritten history in all experiments.

## Open problems

1. **Reward-semantics break at resume.** On-policy RL cannot cleanly bridge
   the pre-/post-`1f131ff` boundary. value_mc needs a fresh run (or a resume
   from iteration 00012 or earlier) under a single, fixed reward definition.

2. **Salvageability of previous GRPO baselines is unknown.** The same
   runtime/reward/extraction code applies to them. Old rollout data may
   contain rewritten-history intervals that were trainable under the old
   code. Each baseline's rejection rate under the new extraction must be
   measured before deciding whether its numbers remain valid.

3. **High malformed-output rate is itself suspicious.** 10–25% of episodes
   emitting unclosed think blocks suggests output-format drift in the policy
   (possibly an interaction between the chat template and
   `enable_thinking`), independent of the training-contract question.

4. **The check's consequence is a design choice, not a law of physics.**
   Detection is required for data validity, but −1 penalty + episode
   termination + sample discard is what caused the distribution shift.
   Candidate alternatives (all are RL data-contract changes per AGENTS.md
   rule 8 and need explicit approval):
   - **Neutral handling**: let the episode continue, mask the mismatched
     span as conditioning-only, no −1 penalty. Minimal distribution shift.
   - **Re-render-faithful extraction**: build training sequences from the
     stored server-rendered `prompt_token_ids`; train only on sampled tokens
     that survive verbatim; normalized spans become conditioning. Salvages
     already-collected data (re-extract instead of re-collect) and is
     sound because the training context then matches the inference context.
   - **Token-ID prompting**: send `prompt_token_ids` = previous
     `full_token_ids` + freshly rendered tool-result tokens, bypassing
     chat-template re-rendering entirely. Eliminates the root cause going
     forward; permitted by AGENTS.md if the stored sequence is unchanged.
   - Simply deleting the runtime check is **not** viable: it restores
     training on rewritten history.

## Recommended next steps

1. Run `extract_trainable_samples` over each old GRPO baseline's judged
   rollouts; count `ProviderHistoryRewriteError` skips per experiment.
   Near-zero rejection → baseline numbers remain usable; otherwise rerun.
2. Decide on one of the contract variants above (neutral handling /
   re-render-faithful extraction / token-ID prompting) and apply it
   uniformly to all experiments being compared.
3. Restart value_mc from a pre-change checkpoint under the chosen variant.
4. Investigate the unclosed-think-block rate in the policy's outputs.
