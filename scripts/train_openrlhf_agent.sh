#!/usr/bin/env bash
set -euo pipefail

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/default.yaml}"
PROMPT_DATA="${PROMPT_DATA:-artifacts/openrlhf/train_prompts.jsonl}"
MODEL_PATH="${MODEL_PATH:-/124090467/Qwen/Qwen3.5-9B}"
SAVE_PATH="${SAVE_PATH:-artifacts/openrlhf/checkpoints}"
AGENT_FUNC_PATH="${AGENT_FUNC_PATH:-src/self_summarization_agent/openrlhf_agent.py}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8765/judge}"

python -m self_summarization_agent.openrlhf_dataset \
  --config "${TRAIN_CONFIG}" \
  --output "${PROMPT_DATA}"

export SELF_SUMMARIZATION_OPENRLHF_CONFIG="${TRAIN_CONFIG}"
export SELF_SUMMARIZATION_JUDGE_URL="${JUDGE_URL}"

python -m openrlhf.cli.train_ppo_ray \
  --pretrain "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --prompt_data "${PROMPT_DATA}" \
  --input_key prompt \
  --label_key label \
  --agent_func_path "${AGENT_FUNC_PATH}" \
  --advantage_estimator group_norm \
  --n_samples_per_prompt 4 \
  --zero_stage 3 \
  --bf16 \
  --gradient_checkpointing \
  --packing_samples \
  --use_dynamic_batch \
  --train_batch_size 128 \
  --micro_train_batch_size 1 \
  --max_len 32768 \
  --max_new_tokens 8192 \
  --ring_attn_size 2 \
  "$@"
