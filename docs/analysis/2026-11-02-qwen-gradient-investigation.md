# Qwen packed-gradient failure: numerical investigation

The reported failure comes from rounding in the test's comparison, rather than
an interval-isolation or activation-checkpointing error in the tested cases.
The test grouped the feed-forward calculations differently on its two sides.
That introduced small rounding differences outside the attention patch, which
grew during the backward calculation. No production training change was needed.

The test reference now groups those calculations the same way on both sides,
while still running the original, unpatched attention mixer separately for each
interval. **No test tolerance was changed, no gradient assertion was removed,
and no failing case was skipped.**

## What the original failure actually measured

The original test ran each entire decoder interval independently for its expected
result, then ran the patched decoder over a packed sequence. Attention correctly
resets at each boundary in both paths, but the MLP's linear operations see
different matrix shapes: interval lengths 19, 12, 30 and 3 versus packed length 64.

With the original seed and installed kernels:

| Observation | Result |
| --- | --- |
| Input normalization and attention forward outputs | Bitwise identical |
| First forward difference | MLP gate/up projections, maximum `3.5762786865234375e-7` |
| Final decoder output difference | Maximum `2.384185791015625e-7` |
| Input-gradient difference | Maximum `4.291534423828125e-6`, within the existing tolerance |
| Convolution weight-gradient difference | Maximum `2.5272369384765625e-5`; 2 of 192 entries outside tolerance |
| QKV projection weight-gradient difference | Maximum `4.601478576660156e-5`; 20 of 1,536 entries outside tolerance |
| Beta projection weight-gradient difference | Maximum `1.4722347259521484e-5`; 4 of 64 entries outside tolerance |

The earlier report mentioned two entries because the original assertion stopped
at the first failing parameter tensor. Inspecting every tensor found the other
24 entries as well. The backward computation amplifies the small downstream
MLP difference; attention forward does not introduce that difference.

Turning checkpointing off produces the same discrepancies. Running the whole
decoder in the same per-interval layout on both sides, with checkpointing enabled
on the patched side, produces bitwise-identical outputs, input gradients and
parameter gradients. The forward/backward measurements are retained in
[the operation trace](qwen-gradient/operation-trace.json).

## Independent precision and update checks

`scripts/diagnose_qwen_interval_gradients.py` implements a small CPU FP64 oracle
using ordinary PyTorch operations and a direct token-by-token delta-rule
recurrence. It uses the exact original FP32 parameter and input values converted
to FP64. It does not use FLA's custom backward, fused normalization or convolution,
or the upstream fallback's explicit FP32 casts.

The oracle's packed and separate layouts agree to maximum output error
`2.220446049250313e-16` and parameter-gradient RMS error
`2.6807729027367474e-16`. Central finite differences independently validate its
gradients at the two original failing convolution entries. For example, entry
`[35, 0, 3]` has analytic derivative `2.459896242012893`; a step of `1e-5` gives
finite-difference derivative `2.459896242612558`.

There is also a **shared** kernel-precision effect that must not be confused with
the packed/separate discrepancy:

| FP32 GPU path versus CPU FP64 oracle | Default Triton precision | `TRITON_F32_DEFAULT=ieee` control |
| --- | ---: | ---: |
| Separate-layout parameter-gradient relative L2 error | `2.2351387702153375e-3` | `2.377437987445765e-6` |
| Packed-layout parameter-gradient relative L2 error | `2.2351373480657215e-3` | `2.3757756422623707e-6` |

The installed FLA kernels use Triton matrix operations, including TF32 precision
on this A100. PyTorch's `highest` matmul setting does not select the default
precision for those Triton operations. Changing the Triton default sharply
reduces the common error in **both** layouts. The control does not override
explicitly selected precisions inside individual kernels. Production precision
settings and dependency pins were left unchanged.

A real AdamW step on the original FP32 comparison, using learning rate `1e-5`,
gradient clipping at 1 and zero weight decay, produces maximum weight difference
`7.275957614183426e-12` between the packed and separate paths. The separate update
has L2 norm `9.538477170281112e-4`. This quantifies the impact in this tiny
reproduction; it is not a long-run convergence guarantee or a claim about every
model, dtype, GPU or sequence-parallel layout.

Full numerical reports: [default precision](qwen-gradient/default.json) and
[Triton IEEE-default control](qwen-gradient/ieee.json).

## Regression repair and coverage

The independent reference explicitly resets the unpatched mixer at the known
interval boundaries and keeps normalization/MLP operations at the packed shape.
It cannot pass by sharing the patched mixer's boundary handling. A separate
forward assertion still compares against independent full-decoder calls.

The unchanged output, input-gradient and every-parameter-gradient assertions now
cover dense and MoE decoders, checkpointing enabled and disabled, and seeds 3
and 9. The intervening forward call remains, so checkpoint recomputation must
recover its own boundary metadata. Existing direct-mixer tests still compare
independent interval values and gradients and demonstrate that an uncorrected
mixer leaks across boundaries.

An additional test compares both feed-forward layouts against the CPU FP64
calculation using the original FP32 comparison defaults, then checks the FP64
analytic gradient against a finite-difference derivative using the stricter
FP64 defaults. All 13 GPU tests pass with the original tolerances.

The interval renderer, token ledger, trajectory evidence, training implementation
and `AGENTS.md` are unchanged by this investigation.

## Reproduce

```bash
uv run --locked --no-sync --extra terminal-bench pytest -q tests/test_qwen_interval_attention.py

uv run --locked --no-sync python scripts/diagnose_qwen_interval_gradients.py \
  --output artifacts/qwen-gradient/default.json

TRITON_F32_DEFAULT=ieee uv run --locked --no-sync python \
  scripts/diagnose_qwen_interval_gradients.py --output artifacts/qwen-gradient/ieee.json
```

Recorded environment: PyTorch `2.10.0+cu128`, CUDA `12.8`, NVIDIA A100-SXM4-80GB.
The reports preserve numerical measurements rather than redefining an acceptance
tolerance from this particular machine.

The combined validation run passed **117 tests with four skips**, including this
GPU suite, the two-GPU replicated-head worker, shared value training, TITO
integrity, Terminal-Bench behavior, and all four SWE fixture update/resume
profiles with compaction. Its command was:

```bash
uv run --locked --no-sync --extra terminal-bench pytest -q \
  tests/test_qwen_interval_attention.py tests/test_value_worker_distributed.py \
  tests/test_value_training_integration.py tests/test_tito.py \
  tests/test_terminal_bench.py tests/test_swe_dev_training.py
```
