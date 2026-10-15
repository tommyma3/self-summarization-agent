"""Two-GPU smoke test of the real value worker with a tiny FSDP model."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from self_summarization_agent.value_model import load_value_head
from self_summarization_agent.value_optimizer import attach_value_head
from self_summarization_agent.verl_value_worker import CompactionValueTrainingWorker


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 3)
        self.lm_head = torch.nn.Linear(3, 2)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, features):
        return self.lm_head(self.backbone(features))


def distributed_worker(rank, init_file, output_dir):
    from verl.utils.torch_functional import get_constant_schedule_with_warmup

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        for dp_size in (2, 1):
            # dp=1 duplicates examples over two replicas, like the head after SP
            # hidden-state gathering. dp=2 distributes the examples across ranks.
            torch.manual_seed(9)
            model = TinyModel().cuda(rank)
            reference_model = deepcopy(model)
            model = FSDP(model, device_id=rank, use_orig_params=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0)
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
            worker = CompactionValueTrainingWorker.__new__(CompactionValueTrainingWorker)
            worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=3))
            worker.device_name = f"cuda:{rank}"

            def actor_step():
                model.clip_grad_norm_(1.0)
                optimizer.step()

            worker.engine = SimpleNamespace(
                module=model, optimizer=optimizer, lr_scheduler=scheduler, scaler=None,
                optimizer_config=SimpleNamespace(clip_grad=1.0), optimizer_step=actor_step,
            )
            worker.set_loss_fn = lambda fn: None
            worker.configure_compaction_value(
                checkpoint_path=str(Path(output_dir) / "base"),
                loss_coefficient=0.3, zero_initialize_head=False, head_learning_rate=1e-2,
                policy_loss_fn=lambda **kw: (kw["model_output"]["policy_loss"], {}),
            )
            worker._pop_state_logits = lambda data: worker.value_head(worker._captured_hidden_states.pop())
            reference_head = deepcopy(worker.value_head)
            # Do not copy the production head's distributed autograd hooks.
            for parameter in reference_head.parameters():
                parameter._backward_hooks = None
            reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=5e-4, weight_decay=0)
            attach_value_head(reference_optimizer, reference_head.parameters(), learning_rate=1e-2)
            features = torch.tensor([[0.2, 1.], [-0.5, 0.7]], device=rank)
            rewards = torch.tensor([-1., 1.], device=rank)
            indices = [rank] if dp_size == 2 else [0, 1]
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                output = model(features[indices])
                loss, _metrics = worker._compaction_actor_critic_loss(
                    model_output={"policy_loss": output.square().sum() * dp_size / 4},
                    data={
                        "value_probe_mask": torch.zeros(len(indices), device=rank, dtype=torch.bool),
                        "value_valid_mask": torch.ones(len(indices), device=rank, dtype=torch.bool),
                        "value_weights": torch.full((len(indices),), 0.5, device=rank),
                        "rewards": rewards[indices], "dp_size": dp_size,
                    },
                )
                loss.backward()
                worker.engine.optimizer_step()
                scheduler.step()

                hidden = reference_model.backbone(features)
                reference_loss = reference_model.lm_head(hidden).square().mean()
                reference_loss += 0.3 * torch.nn.functional.cross_entropy(reference_head(hidden), (rewards > 0).long())
                reference_loss.backward()
                torch.nn.utils.clip_grad_norm_(reference_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
                reference_optimizer.step()
                for actual, expected in zip(worker.value_head.parameters(), reference_head.parameters()):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                with FSDP.summon_full_params(model):
                    for actual, expected in zip(model.parameters(), reference_model.parameters()):
                        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                assert optimizer.param_groups[1]["lr"] == 1e-2
                assert scheduler.get_last_lr() == [5e-4, 1e-2]

            # Exercise the actual packed-prefix Ulysses gather/unpadding path.
            if dp_size == 1:
                from verl.utils.ulysses import set_ulysses_sequence_parallel_group
                set_ulysses_sequence_parallel_group(dist.group.WORLD)
                worker.engine.ulysses_sequence_parallel_size = 2
                packed = torch.arange(18, dtype=torch.float32, device=rank).reshape(6, 3) / 10
                worker._captured_hidden_states.append(packed[rank * 3:(rank + 1) * 3].unsqueeze(0))
                data = {
                    "input_ids": torch.nested.as_nested_tensor(
                        [torch.tensor([1, 2], device=rank), torch.tensor([3, 4, 5], device=rank)],
                        layout=torch.jagged,
                    ),
                    "state_prefix_lengths": torch.tensor([2, 3], device=rank),
                }
                with torch.no_grad():
                    gathered = CompactionValueTrainingWorker._pop_state_logits(worker, data)
                    expected = worker.value_head(packed[[1, 4]])
                torch.testing.assert_close(gathered, expected)
                assert not worker._captured_hidden_states
                set_ulysses_sequence_parallel_group(None)

            path = str(Path(output_dir) / f"dp-{dp_size}")
            worker.save_compaction_value_head(path)
            loaded, found = load_value_head(path, hidden_size=3, zero_initialize=True)
            assert found
            for actual, expected in zip(loaded.parameters(), worker.value_head.parameters()):
                torch.testing.assert_close(actual, expected.detach().cpu())
            worker._value_capture_handle.remove()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_replicated_head_updates_match_reference_with_fsdp(tmp_path):
    torch.multiprocessing.spawn(
        distributed_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True,
    )
