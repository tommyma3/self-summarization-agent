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
