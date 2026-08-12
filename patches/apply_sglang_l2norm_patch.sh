#!/usr/bin/env bash
# Applies the sglang l2norm do_not_specialize fix to the project venv.
#
# Background: sglang 0.5.10.post1's fla l2norm_fwd_kernel declares the
# sequence length T as tl.constexpr, so every distinct sequence length
# triggers a full triton recompilation.  During long agent rollouts this
# compile storm eventually wedges the scheduler (100% CPU, 0% GPU, no
# outputs).  All other fla kernels already use do_not_specialize=["T"];
# l2norm.py was the one file missed upstream.
#
# The patch is applied in place to the installed venv, so it must be
# re-applied after any `uv sync` / venv rebuild.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/sglang-l2norm-do-not-specialize.patch"
TARGET="sglang/srt/layers/attention/fla/l2norm.py"

SITE_PACKAGES="$(cd "${REPO_ROOT}" && .venv/bin/python3 -c 'import site; print(site.getsitepackages()[0])')"
cd "${SITE_PACKAGES}"

if ! [ -f "${TARGET}" ]; then
    echo "error: ${TARGET} not found under ${SITE_PACKAGES}" >&2
    exit 1
fi

# Refuse to patch a file that already diverges from the expected original
# (e.g. an sglang upgrade changed it) unless it already contains the fix.
if grep -q 'do_not_specialize=\["T"\]' "${TARGET}"; then
    echo "already patched: ${TARGET}"
    exit 0
fi
if ! grep -q 'T: tl.constexpr' "${TARGET}"; then
    echo "error: ${TARGET} does not match the expected sglang 0.5.10.post1 source; refusing to patch" >&2
    exit 1
fi

patch -p1 -N -r - --forward < "${PATCH_FILE}" || {
    echo "error: patch did not apply cleanly; check ${TARGET}" >&2
    exit 1
}
echo "patched: ${TARGET}"
