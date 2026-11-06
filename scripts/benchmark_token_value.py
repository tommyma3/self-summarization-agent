"""Synthetic full-interval token-critic capacity probe; never a benchmark score."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from self_summarization_agent.config import load_train_config
from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
from self_summarization_agent.trajectory import RLSample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/train/compact_separate_token_mc_ppo.yaml"
    )
    parser.add_argument("--length", type=int, default=27000)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of visible GPUs; set CUDA_VISIBLE_DEVICES before launching")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_train_config(args.config)
    if config.training.advantage_estimator != "token_mc_value" or args.length < 32:
        raise ValueError("Probe requires token_mc_value and at least 32 shifted tokens")
    if min(args.samples, args.devices, args.microbatch) < 1:
        raise ValueError("samples, devices and microbatch must be positive")
    value = replace(config.training.value, fitting_epochs=1,
                    rollout_minibatch_size=args.samples,
                    gpu_ids=list(range(args.devices)) if args.devices > 1 else [],
                    prefix_microbatch_size=args.microbatch,
                    eval_prefix_microbatch_size=args.microbatch,
                    max_prefix_tokens_per_batch=args.length * args.microbatch)
    started = time.monotonic()
    critic = SeparateValueTrainer(
        value.model_path,
        config.model.model_path,
        value,
        dtype=config.model.dtype,
        advantage_estimator="token_mc_value",
    )
    ids = [100] * (args.length + 1)
    mask = [False] * 32 + [True] * (len(ids) - 32)
    # Fabricated IDs/logprobs for capacity only; never freeze/publish actor targets.
    sample = RLSample(
        query_id="synthetic-memory-probe",
        turn_id="0",
        prompt="",
        completion="",
        reward=1.0,
        trainable_kind="final_answer",
        rollout_id="probe",
        collection_full_token_ids=ids,
        collection_assistant_token_mask=mask,
        input_ids=ids[:-1],
        labels=ids[1:],
        completion_mask=mask[1:],
        reference_logprob=-1.0,
        reference_logprobs=[-1.0] * args.length,
        state_prefix_length=32,
        reference_logprob_source="policy_rescore",
    )
    values, metrics = critic.fit(
        [replace(sample, rollout_id=f"probe-{i}", reward=1.0 if i % 2 else -1.0,
                 collection_full_token_ids=[100 + i] * (args.length + 1),
                 input_ids=[100 + i] * args.length, labels=[100 + i] * args.length)
         for i in range(args.samples)], max_sequence_length=config.training.max_sequence_length
    )
    metrics.update(
        probe_kind="synthetic_production_length_critic_only",
        total_seconds=time.monotonic() - started,
        gpu=torch.cuda.get_device_name(),
        finite=all(torch.isfinite(torch.tensor(v)).all().item() for v in values),
        **critic.kernel_diagnostics(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
