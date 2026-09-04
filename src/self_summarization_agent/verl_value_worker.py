from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch

from self_summarization_agent.value_model import (
    CompactionValueHead,
    expected_binary_value,
    load_value_head,
    reward_class_indices,
    save_value_head,
)


def _require_verl_value_worker_dependencies():
    try:
        from verl.single_controller.base.decorator import (
            Dispatch,
            make_nd_compute_dataproto_dispatch_fn,
            register,
        )
        from verl.utils.metric import AggregationType, Metric
        from verl.utils.ulysses import gather_outputs_and_unpad
        from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
    except ImportError as exc:  # pragma: no cover - exercised on the remote verl environment
        raise ImportError(
            "Compaction-value FSDP training requires the verl version pinned by uv.lock"
        ) from exc
    return {
        "Dispatch": Dispatch,
        "make_dispatch": make_nd_compute_dataproto_dispatch_fn,
        "register": register,
        "AggregationType": AggregationType,
        "Metric": Metric,
        "gather_outputs_and_unpad": gather_outputs_and_unpad,
        "ActorRolloutRefWorker": ActorRolloutRefWorker,
        "TrainingWorker": TrainingWorker,
    }


_verl = _require_verl_value_worker_dependencies()
Dispatch = _verl["Dispatch"]
make_nd_compute_dataproto_dispatch_fn = _verl["make_dispatch"]
register = _verl["register"]
AggregationType = _verl["AggregationType"]
Metric = _verl["Metric"]
gather_outputs_and_unpad = _verl["gather_outputs_and_unpad"]
ActorRolloutRefWorker = _verl["ActorRolloutRefWorker"]
TrainingWorker = _verl["TrainingWorker"]


def _hidden_size_from_worker(worker: Any) -> int:
    config = worker.model_config.hf_config
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    raise ValueError("Cannot determine the FSDP policy hidden size for the compaction value head")


def _output_embeddings(module: Any) -> Any:
    candidates = [
        module,
        getattr(module, "module", None),
        getattr(module, "_fsdp_wrapped_module", None),
    ]
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "get_output_embeddings"):
            continue
        output_embeddings = candidate.get_output_embeddings()
        if output_embeddings is not None:
            return output_embeddings
    raise ValueError("The FSDP policy does not expose output embeddings")


class CompactionValueTrainingWorker(TrainingWorker):
    """verl TrainingWorker with a shared-backbone, replicated two-class value head."""

    def configure_compaction_value(
        self,
        *,
        checkpoint_path: str,
        loss_coefficient: float,
        zero_initialize_head: bool,
        policy_loss_fn: Any,
    ) -> None:
        if getattr(self, "value_head", None) is not None:
            raise RuntimeError("Compaction value worker was configured more than once")
        hidden_size = _hidden_size_from_worker(self)
        head, loaded = load_value_head(
            checkpoint_path,
            hidden_size=hidden_size,
            zero_initialize=zero_initialize_head,
        )
        device = torch.device(self.device_name)
        model_dtype = getattr(
            self.engine,
            "_autocast_dtype",
            next(self.engine.module.parameters()).dtype,
        )
        self.value_head: CompactionValueHead = head.to(device=device, dtype=model_dtype)
        # All ranks must start from identical replicated sidecar state. This is
        # a no-op for the default zero initialization, and also makes optional
        # random initialization deterministic with respect to rank zero.
        if torch.distributed.get_world_size() > 1:
            for parameter in self.value_head.parameters():
                torch.distributed.broadcast(parameter.data, src=0)
        self.value_head_loaded = loaded
        self.value_loss_coefficient = float(loss_coefficient)
        self.policy_loss_fn = policy_loss_fn
        self._captured_hidden_states: list[torch.Tensor] = []
        if getattr(self.engine, "scaler", None) is not None:
            raise ValueError("Compaction-value FSDP currently requires bfloat16 or float32 training")

        output_embeddings = _output_embeddings(self.engine.module)

        def capture_hidden_states(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("The distributed LM head did not receive tensor hidden states")
            self._captured_hidden_states.append(args[0])

        self._value_capture_handle = output_embeddings.register_forward_pre_hook(capture_hidden_states)

        # Keep one optimizer group so the already-created scheduler continues to
        # address every trainable parameter. Optimizer state is initialized lazily.
        if self.engine.optimizer is None or not self.engine.optimizer.param_groups:
            raise RuntimeError("Compaction value training requires an initialized actor optimizer")
        self.engine.optimizer.param_groups[0]["params"].extend(list(self.value_head.parameters()))

        # The head is intentionally replicated rather than FSDP-sharded because it
        # has only 2 * hidden_size parameters. Average its gradients across the full
        # worker world; FSDP continues to synchronize the shared backbone itself.
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            for parameter in self.value_head.parameters():
                parameter.register_hook(partial(self._average_head_gradient, world_size=world_size))

        original_optimizer_step = self.engine.optimizer_step

        def optimizer_step_with_value_head():
            head_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.value_head.parameters(),
                max_norm=self.engine.optimizer_config.clip_grad,
            )
            if not bool(torch.isfinite(head_grad_norm)):
                raise FloatingPointError(
                    f"Compaction value-head gradient norm is non-finite: {head_grad_norm.item()}"
                )
            return original_optimizer_step()

        self.engine.optimizer_step = optimizer_step_with_value_head
        self.set_loss_fn(self._compaction_actor_critic_loss)

    @staticmethod
    def _average_head_gradient(gradient: torch.Tensor, *, world_size: int) -> torch.Tensor:
        gradient = gradient.contiguous()
        torch.distributed.all_reduce(gradient, op=torch.distributed.ReduceOp.SUM)
        return gradient / world_size

    def _pop_state_logits(self, data: Any) -> torch.Tensor:
        if len(self._captured_hidden_states) != 1:
            raise RuntimeError(
                "Expected exactly one LM-head hidden-state capture per distributed microbatch; "
                f"got {len(self._captured_hidden_states)}"
            )
        hidden_states = self._captured_hidden_states.pop(0)
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise RuntimeError(
                "Compaction-value FSDP currently requires verl remove-padding hidden states "
                "with shape [1, packed_tokens, hidden_size]"
            )
        packed_hidden = hidden_states.squeeze(0)
        input_ids = data["input_ids"]
        if not getattr(input_ids, "is_nested", False):
            raise RuntimeError("Compaction-value FSDP requires a jagged verl input batch")
        offsets = input_ids.offsets().to(device=packed_hidden.device, dtype=torch.long)
        total_tokens = int(offsets[-1].item())

        sequence_parallel_size = int(getattr(self.engine, "ulysses_sequence_parallel_size", 1))
        if sequence_parallel_size > 1:
            padded_tokens = packed_hidden.shape[0] * sequence_parallel_size
            padding_size = padded_tokens - total_tokens
            if not 0 <= padding_size < sequence_parallel_size:
                raise RuntimeError(
                    "Ulysses hidden-state layout does not match the packed input token count: "
                    f"local={packed_hidden.shape[0]}, sp={sequence_parallel_size}, total={total_tokens}"
                )
            packed_hidden = gather_outputs_and_unpad(
                packed_hidden,
                gather_dim=0,
                unpad_dim=0,
                padding_size=padding_size,
            )
        if packed_hidden.shape[0] != total_tokens:
            raise RuntimeError(
                "Gathered hidden-state count does not match the append-only interval tokens: "
                f"{packed_hidden.shape[0]} != {total_tokens}"
            )

        prefix_lengths = data["state_prefix_lengths"].to(
            device=packed_hidden.device,
            dtype=torch.long,
        )
        sequence_lengths = offsets.diff()
        if bool(torch.any(prefix_lengths < 1)) or bool(torch.any(prefix_lengths > sequence_lengths)):
            raise ValueError("A distributed value-state anchor is outside its encoded interval")
        anchor_indices = offsets[:-1] + prefix_lengths - 1
        state_hidden = packed_hidden.index_select(0, anchor_indices)
        return self.value_head(state_hidden)

    def _compaction_actor_critic_loss(
        self,
        *,
        model_output: dict[str, Any],
        data: Any,
        dp_group: Any = None,
    ):
        state_logits = self._pop_state_logits(data)
        value_probe = data["value_probe_mask"].to(device=state_logits.device, dtype=torch.bool)
        if bool(torch.all(value_probe)):
            model_output.pop("log_probs", None)
            # verl's postprocess builds nested tensors by unbinding along the batch
            # dimension. Keep per-state outputs at least 1-D so they are not treated
            # as zero-dimensional scalars.
            model_output["compaction_values"] = expected_binary_value(state_logits).unsqueeze(-1)
            model_output["sample_indices"] = data["sample_indices"].to(state_logits.device).unsqueeze(-1)
            return state_logits.sum() * 0.0, {}
        if bool(torch.any(value_probe)):
            raise ValueError("A distributed microbatch cannot mix value probes and trainable intervals")

        policy_loss, metrics = self.policy_loss_fn(
            model_output=model_output,
            data=data,
            dp_group=dp_group,
        )
        rewards = data["rewards"].to(device=state_logits.device, dtype=torch.float32)
        value_weights = data["value_weights"].to(device=state_logits.device, dtype=torch.float32)
        valid = data["value_valid_mask"].to(device=state_logits.device, dtype=torch.bool)
        if not bool(torch.all((rewards[valid] == -1) | (rewards[valid] == 1))):
            raise ValueError("Compaction Monte Carlo value targets must be exactly -1 or +1")
        per_state_loss = torch.nn.functional.cross_entropy(
            state_logits.float(),
            reward_class_indices(rewards),
            reduction="none",
        )
        local_value_loss = (per_state_loss * value_weights * valid.to(value_weights.dtype)).sum()
        dp_size = int(data["dp_size"])
        value_loss = local_value_loss * dp_size
        total_loss = policy_loss + self.value_loss_coefficient * value_loss.to(policy_loss.device)

        predicted = torch.where(expected_binary_value(state_logits) >= 0, 1.0, -1.0)
        correct_weight = value_weights * valid.to(value_weights.dtype)
        local_accuracy = ((predicted == rewards).to(value_weights.dtype) * correct_weight).sum()
        metrics["value/loss"] = Metric(value=value_loss, aggregation=AggregationType.SUM)
        metrics["value/classification_accuracy"] = Metric(
            value=local_accuracy * dp_size,
            aggregation=AggregationType.SUM,
        )
        metrics["value/state_count"] = Metric(
            value=valid.sum().to(torch.float32),
            aggregation=AggregationType.SUM,
        )
        return total_loss, metrics

    def compute_compaction_values(self, data: Any):
        if self._captured_hidden_states:
            raise RuntimeError("Stale LM-head captures exist before frozen value evaluation")
        output = self.infer_batch(data)
        if self._captured_hidden_states:
            raise RuntimeError("Unused LM-head captures remain after frozen value evaluation")
        return output

    def save_compaction_value_head(self, path: str) -> None:
        if torch.distributed.get_rank() == 0:
            save_value_head(self.value_head, Path(path))
        torch.distributed.barrier()


class CompactionValueActorRolloutRefWorker(ActorRolloutRefWorker):
    actor_worker_cls = CompactionValueTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor or self.actor is None:
            raise RuntimeError("Compaction value worker must be initialized with the actor role")
        value_config = self.config.compaction_value
        self.actor.configure_compaction_value(
            checkpoint_path=str(self.config.model.path),
            loss_coefficient=float(value_config.loss_coefficient),
            zero_initialize_head=bool(value_config.zero_initialize_head),
            policy_loss_fn=self.loss_fn,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_compaction_values(self, data):
        output = self.actor.compute_compaction_values(data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        self.actor.save_compaction_value_head(local_path)
