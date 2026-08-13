# Patches

Venv-level fixes that must survive `uv sync` / venv rebuilds.

## sglang-l2norm-do-not-specialize.patch

**Diagnosed 2026-08-12** while debugging collection stalls of the
`qwen-bcplus-no-compact-32k-train` run (see `artifacts/train/.../train-iterations-*.log`
for the runs that wedged).

### Symptom

Rollout collection stalls after 3–198 episodes: sglang schedulers pin ~85% CPU,
GPU utilization is 0%, no rollout output is written, and the stall never
recovers. `disable_radix_cache` and `max_concurrent_episodes` do not help.

### Root cause

In sglang 0.5.10.post1, `sglang/srt/layers/attention/fla/l2norm.py` declares the
sequence length `T` as `tl.constexpr`:

```python
@triton.jit
def l2norm_fwd_kernel(x, y, eps, NB: tl.constexpr, T: tl.constexpr, ...):
```

so every distinct sequence length forces a full triton recompilation
(~304 `l2norm_fwd` variants accumulated in `~/.triton/cache`, one per distinct
shape; each compile takes 3–20s of CPU). When the compiler wedges on some
shape, the batch never runs, the scheduler busy-polls forever, and collection
freezes. Every *other* fla kernel (`chunk_o`, `chunk_delta_h`,
`chunk_scaled_dot_kkt`, `cumsum`, …) already uses `do_not_specialize=["T"]`;
l2norm.py is the one file upstream missed.

Diagnosis chain: py-spy `--subprocesses --format raw` profiling of the wedged
run → scheduler main thread cycling through `run_batch` with triton
`make_ttgir` frames → triton cache diffing showing per-shape constants baked
into `l2norm_fwd_kernel` (`arith.constant <T> : i64`) → cache frozen at the
moment of the stall.

### Fix

`do_not_specialize=["T"]` + pass `T` as a 1-element int32 GPU scalar that the
kernel loads at runtime (`T = tl.load(T)`), and drop the dead `NB` constexpr.
Verified: one compilation serves all shapes; output matches reference to bf16
rounding (~1e-3 max error).

### Apply

```bash
patches/apply_sglang_l2norm_patch.sh
```

Re-run it after any venv rebuild. The script refuses to apply if the installed
sglang no longer matches the expected original (e.g. an upgrade already fixed
it) and exits cleanly when the fix is already present.

## verl-ulysses-position-ids-contiguous.patch

**Diagnosed 2026-08-13** while debugging the `train_update` failure at iteration 4
of the `qwen-bcplus-no-compact-32k-train` run (see
`artifacts/train/.../train-iterations-*.log`).

### Symptom

`train_update` crashes ~177s in on the first Ulysses-SP micro-batch (the first
run with `ulysses_sequence_parallel_size: 4`, enabled 2026-08-12). First failure
was `ValueError: Tensors must be contiguous`; after fixing contiguity, a GPU
fault (`CUDA error: an illegal memory access / illegal instruction was
encountered` on the NCCL watchdogs) followed, plus an `AssertionError` in
`offload_fsdp_model_to_cpu` and a PyTorch `SavedTensorHooks` internal assert
during unwinding (both cascades of the first failure).

### Root cause

In verl 0.9.0.dev0, the Ulysses sequence-parallel flash-attention monkey patch
(`_ulysses_flash_attention_forward` in `verl/models/transformers/monkey_patch.py`)
calls `torch.distributed.all_gather` on `position_ids` and forwards the result to
transformers' FA2 varlen path. Qwen3.5's forward
(`transformers/models/qwen3_5/modeling_qwen3_5.py:1347-1352`) expands and slices
the tensor (`position_ids[None, ...].expand(4, ...)` then `position_ids[1:]`) for
mrope, so the ids reach the gather as a **non-contiguous** `(3, 1, seq)`
tensor — two separate problems:

1. `all_gather` strictly requires a contiguous tensor → `ValueError`.
2. Even when made contiguous, the 3-D shape breaks FA2's packed-sequence
   inference: `prepare_fa_kwargs_from_position_ids` flattens the ids, so its
   `cu_seq_lens` ends at `3 * seq` while the gathered query tensor has only
   `seq` rows → the flash-attention kernel reads out of bounds → the CUDA
   fault. (The three mrope copies are identical — produced by `expand` — so
   they carry no extra information.)

### Fix

In the monkey patch, before the `all_gather`: slice 3-D mrope ids to the 2-D
`(bsz, seq)` view with `position_ids[0]`, and pass `position_ids.contiguous()`
to the gather (a no-op when already contiguous; the tensor is long dtype with
no grad).

### Apply

```bash
patches/apply_verl_position_ids_patch.sh
```

Re-run it after any venv rebuild. The script refuses to apply if the installed
verl no longer matches the expected original (e.g. an upgrade already fixed
it) and exits cleanly when the fix is already present.
