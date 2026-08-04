#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/train_iterations.sh --iterations N --latest-root PATH [options]

Run or resume training until checkpoint iteration N is reached.

Required arguments:
  --iterations N       Target checkpoint iteration; must be a positive integer.
  --latest-root PATH   Training artifact root containing the latest pointer.

Options:
  --config PATH        Training config (default: configs/train/default.yaml).
  --python PATH        Python executable (default: $PYTHON_BIN or python).
  --set KEY=VALUE      Config override forwarded to every iteration; repeatable.
  -h, --help           Show this help message.

All stdout and stderr are streamed to the terminal and appended to a timestamped
train-iterations-*.log file directly under --latest-root. Iterations always use
--resume, so completed phases are reused safely.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TARGET_ITERATIONS=""
LATEST_ROOT=""
CONFIG_PATH="${REPO_ROOT}/configs/train/default.yaml"
PYTHON_EXECUTABLE="${PYTHON_BIN:-python}"
CONFIG_OVERRIDES=()

while (($#)); do
    case "$1" in
        --iterations)
            [[ $# -ge 2 ]] || { echo "error: --iterations requires a value" >&2; exit 2; }
            TARGET_ITERATIONS="$2"
            shift 2
            ;;
        --latest-root)
            [[ $# -ge 2 ]] || { echo "error: --latest-root requires a value" >&2; exit 2; }
            LATEST_ROOT="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "error: --config requires a value" >&2; exit 2; }
            CONFIG_PATH="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "error: --python requires a value" >&2; exit 2; }
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --set)
            [[ $# -ge 2 ]] || { echo "error: --set requires KEY=VALUE" >&2; exit 2; }
            CONFIG_OVERRIDES+=(--set "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${TARGET_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --iterations must be a positive integer" >&2
    exit 2
fi
if [[ -z "${LATEST_ROOT}" ]]; then
    echo "error: --latest-root is required" >&2
    exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "error: training config does not exist: ${CONFIG_PATH}" >&2
    exit 2
fi
if ! command -v "${PYTHON_EXECUTABLE}" >/dev/null 2>&1; then
    echo "error: Python executable was not found: ${PYTHON_EXECUTABLE}" >&2
    exit 2
fi

mkdir -p -- "${LATEST_ROOT}"
LATEST_ROOT="$(cd -- "${LATEST_ROOT}" && pwd)"
CONFIG_PATH="$(cd -- "$(dirname -- "${CONFIG_PATH}")" && pwd)/$(basename -- "${CONFIG_PATH}")"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LATEST_ROOT}/train-iterations-${TIMESTAMP}.log"
exec > >(tee -a -- "${LOG_FILE}") 2>&1

echo "Training launcher started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repository: ${REPO_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Latest root: ${LATEST_ROOT}"
echo "Target iteration: ${TARGET_ITERATIONS}"
echo "Log file: ${LOG_FILE}"

LATEST_POINTER="${LATEST_ROOT}/latest"
if [[ ! -f "${LATEST_POINTER}" ]]; then
    echo "error: missing latest checkpoint pointer: ${LATEST_POINTER}" >&2
    exit 1
fi

LATEST_CHECKPOINT="$(<"${LATEST_POINTER}")"
LATEST_CHECKPOINT="${LATEST_CHECKPOINT%$'\r'}"
CURRENT_CHECKPOINT_NAME="$(basename -- "${LATEST_CHECKPOINT}")"
if [[ ! "${CURRENT_CHECKPOINT_NAME}" =~ ^iteration-([0-9]+)$ ]]; then
    echo "error: latest must point to an iteration-N checkpoint; got: ${LATEST_CHECKPOINT}" >&2
    exit 1
fi
CURRENT_ITERATION=$((10#${BASH_REMATCH[1]}))

if ((CURRENT_ITERATION >= TARGET_ITERATIONS)); then
    echo "Target already reached: current iteration ${CURRENT_ITERATION}."
    exit 0
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${REPO_ROOT}"

for ((iteration = CURRENT_ITERATION + 1; iteration <= TARGET_ITERATIONS; iteration++)); do
    echo
    echo "=== Starting training iteration ${iteration}/${TARGET_ITERATIONS} ==="
    "${PYTHON_EXECUTABLE}" -m self_summarization_agent.iteration_launcher \
        --config "${CONFIG_PATH}" \
        --iteration "${iteration}" \
        --latest-root "${LATEST_ROOT}" \
        --resume \
        "${CONFIG_OVERRIDES[@]}"
    echo "=== Completed training iteration ${iteration}/${TARGET_ITERATIONS} ==="
done

echo "Training launcher completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Final target iteration: ${TARGET_ITERATIONS}"
echo "Log file: ${LOG_FILE}"
