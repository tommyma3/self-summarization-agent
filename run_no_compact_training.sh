#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/train/no_compact_32k.yaml"
EXPERIMENT_NAME="qwen-bcplus-no-compact-32k-train"
LOG_FILE="artifacts/train/${EXPERIMENT_NAME}/training.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting no-compact 32k training (iterations 1 -> 10)" | tee -a "$LOG_FILE"

for i in $(seq 1 10); do
    {
        echo ""
        echo "=========================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Running iteration $i"
        echo "=========================================="
        uv run -m self_summarization_agent.iteration_launcher \
            --config "$CONFIG" \
            --iteration "$i" \
            --resume
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Iteration $i complete."
    } 2>&1 | tee -a "$LOG_FILE"
done

echo "All iterations finished." | tee -a "$LOG_FILE"
