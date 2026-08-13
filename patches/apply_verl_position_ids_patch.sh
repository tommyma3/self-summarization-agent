#!/usr/bin/env bash
# Applies the verl Ulysses position_ids contiguity fix to the project venv.
#
# Background: verl 0.9.0.dev0's Ulysses flash-attention monkey patch
# (`_ulysses_flash_attention_forward` in verl/models/transformers/monkey_patch.py)
# calls torch.distributed.all_gather on `position_ids`.  Qwen3.5's forward
# (transformers/models/qwen3_5/modeling_qwen3_5.py) expands + slices the tensor
# (`position_ids[None, ...].expand(4, ...)` then `[1:]`), so it reaches the
# gather non-contiguous and all_gather raises `ValueError: Tensors must be
# contiguous`.  Fix: `.contiguous()` at the gather site (a no-op when already
# contiguous; position_ids is a long tensor with no grad).
#
# The patch is applied in place to the installed venv, so it must be
# re-applied after any `uv sync` / venv rebuild.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/verl-ulysses-position-ids-contiguous.patch"
TARGET="verl/models/transformers/monkey_patch.py"

SITE_PACKAGES="$(cd "${REPO_ROOT}" && .venv/bin/python3 -c 'import site; print(site.getsitepackages()[0])')"
cd "${SITE_PACKAGES}"

if ! [ -f "${TARGET}" ]; then
    echo "error: ${TARGET} not found under ${SITE_PACKAGES}" >&2
    exit 1
fi

# Refuse to patch a file that already diverges from the expected original
# (e.g. a verl upgrade changed it) unless it already contains the fix.
if grep -q 'position_ids.contiguous()' "${TARGET}"; then
    echo "already patched: ${TARGET}"
    exit 0
fi
if ! grep -q 'torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())' "${TARGET}"; then
    echo "error: ${TARGET} does not match the expected verl source; refusing to patch" >&2
    exit 1
fi

patch -p1 -N -r - --forward < "${PATCH_FILE}" || {
    echo "error: patch did not apply cleanly; check ${TARGET}" >&2
    exit 1
}
echo "patched: ${TARGET}"
