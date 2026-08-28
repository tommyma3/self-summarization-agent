#!/usr/bin/env bash
set -euo pipefail

consecutive_free=0

while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [ -n "$gpu_procs" ]; then
        consecutive_free=0
        echo "[$(date)] GPU busy (pids: $(echo "$gpu_procs" | tr '\n' ' ')). Skipping probe. Sleeping 30 minutes..."
        sleep 1800
    else
        consecutive_free=$((consecutive_free + 1))
        if [ "$consecutive_free" -ge 2 ]; then
            echo "[$(date)] GPU free for 2 consecutive checks. Running probe..."
            uv run scripts/probe_vllm_collection.py --sample-index 1
            consecutive_free=0
            echo "[$(date)] Probe complete. Sleeping 60 minutes..."
            sleep 3600
        else
            echo "[$(date)] GPU free (${consecutive_free}/2 consecutive checks). Sleeping 30 minutes..."
            sleep 1800
        fi
    fi
done
