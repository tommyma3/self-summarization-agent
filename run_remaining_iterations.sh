#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/train/default.yaml"
LOG_FILE="artifacts/train/qwen-bcplus-train/training.log"
LATEST_FILE="artifacts/train/qwen-bcplus-train/latest"

mkdir -p "$(dirname "$LOG_FILE")"

# Start from the very beginning
START_ITER=1

echo "Starting training iterations ($START_ITER -> 20)" | tee -a "$LOG_FILE"

for i in $(seq "$START_ITER" 20); do
    {
        echo ""
        echo "=========================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Running iteration $i"
        echo "=========================================="
        uv run python -m self_summarization_agent.iteration_launcher \
            --config "$CONFIG" \
            --iteration "$i" \
            --resume
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Iteration $i complete."
    } 2>&1 | tee -a "$LOG_FILE"
done

echo "All iterations finished." | tee -a "$LOG_FILE"
