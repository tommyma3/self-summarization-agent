"""Isolated synthetic capacity probe of the production four-GPU actor path."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=65536)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    if args.length < 4 or args.repeats < 1:
        parser.error('length must be >= 4 and repeats >= 1')

    import ray
    import torch
    from omegaconf import OmegaConf
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from self_summarization_agent.config import load_train_config
    from self_summarization_agent.trajectory import RLSample
    from self_summarization_agent.verl_interval_worker import IntervalActorRolloutRefWorker
    from self_summarization_agent.verl_ray_trainer import (
        build_verl_actor_dataproto, build_verl_fsdp_worker_config, _extract_worker_metrics,
    )

    class ProbeWorker(IntervalActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def probe_memory(self, reset=False):
            torch.cuda.synchronize()
            result = dict(rank=torch.distributed.get_rank(),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                allocated_gib=torch.cuda.memory_allocated() / 2**30)
            if reset:
                torch.cuda.reset_peak_memory_stats()
            return result

    cfg = load_train_config('configs/train/no_compaction_loss.yaml')
    train = cfg.training
    train.max_sequence_length = args.length
    train.minibatch_size = 1
    train.update_epochs = 1
    train.verl.fsdp.ppo_max_token_len_per_gpu = math.ceil(args.length / 4)
    train.verl.fsdp.log_prob_max_token_len_per_gpu = math.ceil(args.length / 4)
    train.verl.fsdp.use_torch_compile = False
    train.verl.fsdp.ppo_micro_batch_size_per_gpu = 1
    worker_config = build_verl_fsdp_worker_config(cfg.model, train)
    report = dict(kind='synthetic_actor_capacity_probe', full_input_tokens=args.length,
        shifted_training_tokens=args.length-1, model=cfg.model.model_path,
        world_size=4, sequence_parallel_size=4, dtype=cfg.model.dtype,
        activation_checkpointing=train.activation_checkpointing,
        param_offload=train.verl.fsdp.param_offload,
        optimizer_offload=train.verl.fsdp.optimizer_offload,
        worker_config=worker_config, steps=[], status='initializing')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    save()
    started = time.monotonic()
    try:
        ray.init(address='local', namespace=f'actor-capacity-{os.getpid()}',
            num_cpus=12, num_gpus=4, include_dashboard=False,
            object_store_memory=1024**3,
            runtime_env={'env_vars': {
                'PYTHONPATH': str(Path.cwd() / 'src') + ':' + str(Path.cwd()),
                'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
                'OMP_NUM_THREADS': '4',
                'RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO': '0',
            }})
        pool = RayResourcePool(process_on_nodes=[4], use_gpu=True, max_colocate_count=1,
            name_prefix=f'capacity-{os.getpid()}')
        workers = RayWorkerGroup(resource_pool=pool,
            ray_cls_with_init=RayClassWithInitArgs(ray.remote(ProbeWorker),
                config=OmegaConf.create(worker_config), role='actor'),
            name_prefix=f'capacity-{os.getpid()}')
        workers.init_model()
        report['initialization_seconds'] = time.monotonic() - started
        report['initial_memory'] = workers.probe_memory(reset=True)
        report['status'] = 'running'
        save()
        rng = random.Random(20261117)
        tokens = [rng.randrange(100, 200000) for _ in range(args.length)]
        sample = RLSample(query_id='synthetic-capacity', turn_id='synthetic-interval',
            prompt='', completion='', reward=1., trainable_kind='synthetic_probe',
            input_ids=tokens[:-1], labels=tokens[1:],
            completion_mask=[True] * (args.length-1),
            reference_logprob=-math.log(248320),
            reference_logprobs=[-math.log(248320)] * (args.length-1))
        for step in range(args.repeats):
            batch = build_verl_actor_dataproto([sample], [1.],
                checkpoint_id='synthetic-only', max_sequence_length=args.length)
            batch.meta_info.update(mini_batch_size=1, global_batch_size=1,
                epochs=1, seed=42, query_count=1, contributing_sample_count=1)
            tick = time.monotonic()
            metrics = _extract_worker_metrics(workers.update_actor(batch.to_tensordict()))
            memory = workers.probe_memory(reset=True)
            row = dict(step=step+1, seconds=time.monotonic()-tick,
                metrics=metrics, per_rank_memory=memory)
            report['steps'].append(row)
            save()
            print(json.dumps(row), flush=True)
            for key in ('loss', 'grad_norm', 'actor/pg_loss'):
                if key not in metrics or not math.isfinite(metrics[key]):
                    raise RuntimeError(f'Missing or nonfinite {key}: {metrics}')
            if metrics['grad_norm'] <= 0:
                raise RuntimeError('No nonzero gradient was measured')
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        report['total_seconds'] = time.monotonic()-started
        save()
        ray.shutdown()


if __name__ == '__main__':
    main()
