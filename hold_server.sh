#!/usr/bin/env bash
set -euo pipefail


while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [ -n "$gpu_procs" ]; then
        echo "[$(date)] GPU busy (pids: $(echo "$gpu_procs" | tr '\n' ' ')). Skipping probe. Sleeping 30 minutes..."
    else
        echo "[$(date)] GPU free. Running probe..."
        uv run scripts/probe_vllm_collection.py --sample-index 1
        echo "[$(date)] Probe complete. Sleeping 60 minutes..."
    fi
    sleep 3600
done
