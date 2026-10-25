"""Exact-prefix categorical value fitting with an independent text model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import sys
import time
import uuid

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from self_summarization_agent.separate_value import (
    CRITIC_MANIFEST, CRITIC_STATE, atomic_json, digest, validate_samples, sample_key, method_spec,
)
from self_summarization_agent.value_model import (
    expected_categorical_value, load_value_head, model_hidden_size,
    two_hot_targets, outcome_log_losses, save_value_head,
)
from self_summarization_agent.value_diagnostics import value_diagnostics
from self_summarization_agent.trainer import _write_training_progress


def tokenizer_signature(tokenizer):
    # Do not compare chat templates: collection already rendered authoritative IDs.
    return digest({"vocab": tokenizer.get_vocab(), "special_ids": sorted(tokenizer.all_special_ids)})


@dataclass
class _PreparedPrefixes:
    # Detached copies of exact IDs, never a replacement for trajectory records.
    prefixes: list[torch.Tensor]
    lengths: list[int]
    targets: torch.Tensor
    rewards: torch.Tensor


class SeparateValueTrainer:
    def __init__(self, source, actor_path, config, *, dtype="bfloat16", device="cuda:0",
                 activation_checkpointing=True, resume=False, progress_path=None,
                 advantage_estimator="compaction_mc_value"):
        self.config = config
        self.method = method_spec(config, advantage_estimator)
        self.frozen_digest = None
        self.source_critic_identity = None
        self.progress_path = progress_path
        self.device = torch.device(device)
        self.update_count = 0
        self._last_progress = float("-inf")
        self.detailed_diagnostics = False
        self.source = str(source)
        if resume:
            manifest = json.loads((Path(source) / CRITIC_MANIFEST).read_text())
            legacy_method = {**method_spec(config), "num_bins": 2, "support": [-1.0, 1.0],
                             "baseline_timing": "post_fit"}
            if manifest.get("method", legacy_method) != self.method:
                raise ValueError("Critic method/support changed during resume; start a fresh experiment")
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        actor_tokenizer = AutoTokenizer.from_pretrained(actor_path, trust_remote_code=False)
        self.tokenizer_hash = tokenizer_signature(self.tokenizer)
        if self.tokenizer_hash != tokenizer_signature(actor_tokenizer):
            raise ValueError("Actor and critic token-ID mappings/special tokens differ")
        hf_config = AutoConfig.from_pretrained(source, trust_remote_code=False)
        text_config = getattr(hf_config, "text_config", hf_config)
        kwargs = {}
        if hasattr(hf_config, "text_config"):
            kwargs["key_mapping"] = {r"^model.language_model\.": ""}
        torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}.get(dtype)
        if torch_dtype is None or torch_dtype == torch.float16:
            raise ValueError("Separate critic supports bfloat16 or float32 (no fp16 scaler)")
        self.model, loading = AutoModel.from_pretrained(
            source, config=text_config, dtype=torch_dtype, trust_remote_code=False,
            output_loading_info=True, **kwargs,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Incomplete critic backbone load: {loading}")
        self.model.to(self.device)
        self.model.config.use_cache = False
        self.activation_checkpointing = (activation_checkpointing if config.activation_checkpointing is None
                                         else config.activation_checkpointing)
        if self.activation_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.head, found = load_value_head(source, hidden_size=model_hidden_size(self.model),
                                               zero_initialize=config.zero_initialize_head, num_bins=config.num_bins)
        if resume and not found:
            raise ValueError("Resumed critic is missing its value head")
        self.head.to(device=self.device, dtype=torch_dtype)
        self.optimizer = torch.optim.AdamW([
            {"params": list(self.model.parameters()), "lr": config.learning_rate},
            {"params": list(self.head.parameters()), "lr": config.head_learning_rate or config.learning_rate},
        ], weight_decay=0.0)
        if resume:
            state = torch.load(Path(source) / CRITIC_STATE, map_location=self.device, weights_only=True)
            if state.get("version") != 1:
                raise ValueError("Unsupported critic optimizer state")
            self.optimizer.load_state_dict(state["optimizer"])
            self.update_count = state["update_count"]
            # Explicit config overrides apply on continuation; moments persist.
            self.optimizer.param_groups[0]["lr"] = config.learning_rate
            self.optimizer.param_groups[1]["lr"] = config.head_learning_rate or config.learning_rate

    def synchronize(self):
        """Use only at phase boundaries, not for each microbatch."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def kernel_diagnostics(self):
        module = sys.modules[type(self.model).__module__]
        result = {"value/kernels/model_type": self.model.config.model_type,
                  "value/kernels/attention_backend": getattr(self.model.config, "_attn_implementation", None),
                  "value/activation_checkpointing": self.activation_checkpointing}
        if hasattr(module, "is_fast_path_available"):
            result["value/kernels/fast_path_available"] = bool(module.is_fast_path_available)
            for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                         "fused_recurrent_gated_delta_rule"):
                result[f"value/kernels/{name}_available"] = callable(getattr(module, name, None))
        return result

    def _progress(self, stage, *, force=False, **payload):
        now = time.monotonic()
        if force or now - self._last_progress >= self.config.progress_interval_seconds:
            _write_training_progress(self.progress_path, stage, **payload)
            self._last_progress = now

    def _prepare(self, samples, scalar_targets=None):
        prefixes = [s.input_ids[:s.state_prefix_length] for s in samples]
        limit = self.model.config.max_position_embeddings
        vocab_size = self.model.get_input_embeddings().num_embeddings
        if any(not p or len(p) > limit or any(t < 0 or t >= vocab_size for t in p) for p in prefixes):
            raise ValueError("Critic prefix exceeds model capacity or vocabulary")
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
        if scalar_targets is not None and set(scalar_targets) != {sample_key(s) for s in samples}:
            raise ValueError("Fixed value targets must cover exactly the interval keys")
        targets = (rewards if scalar_targets is None else
                   torch.tensor([scalar_targets[sample_key(s)] for s in samples], dtype=torch.float32))
        return _PreparedPrefixes([torch.tensor(p, dtype=torch.long) for p in prefixes],
            [len(p) for p in prefixes], two_hot_targets(targets, self.config.num_bins), rewards)

    def _batches(self, prepared, indices, batch_size):
        indices = list(indices)
        if self.config.length_bucketing:
            indices.sort(key=lambda i: prepared.lengths[i])
        batch, longest = [], 0
        budget = self.config.max_prefix_tokens_per_batch
        for index in indices:
            length = prepared.lengths[index]
            if batch and (len(batch) >= batch_size or
                          (budget is not None and max(longest, length) * (len(batch) + 1) > budget)):
                yield batch
                batch, longest = [], 0
            # A valid prefix larger than the batching budget runs alone, never truncated.
            batch.append(index)
            longest = max(longest, length)
        if batch:
            yield batch

    def _to_device(self, tensor):
        if self.device.type == "cuda":
            return tensor.pin_memory().to(self.device, non_blocking=True)
        return tensor.to(self.device)

    def _logits_prepared(self, prepared, indices):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        ids = self._to_device(torch.nn.utils.rnn.pad_sequence(
            [prepared.prefixes[i] for i in indices], batch_first=True, padding_value=pad_id))
        lengths = self._to_device(torch.tensor([prepared.lengths[i] for i in indices]))
        mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
        positions = torch.arange(ids.shape[1], device=self.device)[None, :].expand_as(ids)
        hidden = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False).last_hidden_state
        return self.head(hidden[torch.arange(len(indices), device=self.device), lengths - 1])

    def logits(self, samples):
        prepared = self._prepare(samples)
        return self._logits_prepared(prepared, list(range(len(samples))))

    def evaluate(self, samples):
        validate_samples(samples)
        return self._evaluate_prepared(self._prepare(samples))[:2]

    def _evaluate_prepared(self, prepared):
        size = len(prepared.prefixes)
        if not size:
            return [], [], {"seconds": 0.0, "states_per_second": 0.0, "prefix_tokens_per_second": 0.0,
                            "padding_fraction": 0.0}
        self.synchronize()
        started = time.monotonic()
        modes = self.model.training, self.head.training
        padded_tokens = 0
        # Diagnostic scheduling must not change the training RNG or module mode.
        devices = [self.device] if self.device.type == "cuda" else []
        try:
            self.model.eval()
            self.head.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                predictions = torch.empty((size, 2), device=self.device, dtype=torch.float32)
                completed = 0
                for indices in self._batches(prepared, range(size),
                        self.config.eval_prefix_microbatch_size or self.config.prefix_microbatch_size):
                    logits = self._logits_prepared(prepared, indices)
                    rewards = self._to_device(prepared.rewards[indices])
                    values = expected_categorical_value(logits)
                    losses = outcome_log_losses(logits, rewards)
                    predictions[self._to_device(torch.tensor(indices))] = torch.stack((values, losses), dim=1)
                    padded_tokens += len(indices) * max(prepared.lengths[i] for i in indices)
                    completed += len(indices)
                    self._progress("critic_evaluate", completed_states=completed, state_count=size)
                # Only compact scalar outputs cross to the CPU; no per-batch .item()/.cpu().
                values, losses = predictions.cpu().t().tolist()
        finally:
            self.model.train(modes[0])
            self.head.train(modes[1])
        self.synchronize()
        elapsed = time.monotonic() - started
        tokens = sum(prepared.lengths)
        return values, losses, {"seconds": elapsed, "states_per_second": size / elapsed,
            "prefix_tokens_per_second": tokens / elapsed, "padding_fraction": 1 - tokens / padded_tokens}

    def fit(self, samples, *, max_sequence_length=None, scalar_targets=None):
        self.synchronize()
        started = time.monotonic()
        if self.method["estimator"] == "compaction_gae_value" and scalar_targets is None:
            raise ValueError("GAE fitting requires frozen pre-fit scalar targets")
        validate_samples(samples, max_sequence_length)
        if not samples:
            return [], {"value/optimizer_step_count": 0}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        prepared = self._prepare(samples, scalar_targets)
        metrics = {"value/prepare_seconds": time.monotonic() - started, **self.kernel_diagnostics(),
                   "value/detailed_diagnostics": self.detailed_diagnostics}
        if self.config.eval_before_fit:
            before, before_losses, stats = self._evaluate_prepared(prepared)
            metrics.update(value_diagnostics(samples, before, before_losses, prefix="value/eval_before",
                                             detailed=self.detailed_diagnostics))
            metrics["value/eval_before_seconds"] = stats["seconds"]
        else:
            metrics["value/eval_before_skipped"] = "disabled"
        rollouts = {}
        for index, sample in enumerate(samples):
            rollouts.setdefault(sample.rollout_id, []).append(index)
        keys = list(rollouts)
        initial_steps = self.update_count
        train_seconds = 0.0
        for epoch in range(self.config.fitting_epochs):
            random.Random(self.config.seed + self.update_count + epoch).shuffle(keys)
            self.model.train()
            self.head.train()
            self._progress("critic_fit", force=True, epoch=epoch + 1, epochs=self.config.fitting_epochs)
            self.synchronize()
            epoch_started = time.monotonic()
            online_loss = torch.zeros((), device=self.device)
            padded_tokens = 0
            for start in range(0, len(keys), self.config.rollout_minibatch_size):
                selected = keys[start:start + self.config.rollout_minibatch_size]
                weights_by_index = {i: 1 / (len(selected) * len(rollouts[k]))
                                    for k in selected for i in rollouts[k]}
                self.optimizer.zero_grad(set_to_none=True)
                for indices in self._batches(prepared, weights_by_index, self.config.prefix_microbatch_size):
                    logits = self._logits_prepared(prepared, indices)
                    targets = self._to_device(prepared.targets[indices])
                    losses = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none")
                    weights = self._to_device(torch.tensor([weights_by_index[i] for i in indices]))
                    loss = (losses * weights).sum()
                    online_loss += loss.detach() * len(selected)
                    loss.backward()
                    padded_tokens += len(indices) * max(prepared.lengths[i] for i in indices)
                norm = torch.nn.utils.clip_grad_norm_(
                    [*self.model.parameters(), *self.head.parameters()], self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.update_count += 1
                self._progress("critic_fit", epoch=epoch + 1,
                    epochs=self.config.fitting_epochs, completed_rollouts=min(start + len(selected), len(keys)),
                    rollout_count=len(keys), optimizer_steps=self.update_count)
            self.synchronize()
            elapsed = time.monotonic() - epoch_started
            train_seconds += elapsed
            prefix = f"value/epoch_{epoch + 1}"
            metrics.update({f"{prefix}/train_seconds": elapsed,
                f"{prefix}/target_ce": float(online_loss) / len(keys),
                f"{prefix}/states_per_second": len(samples) / elapsed,
                f"{prefix}/prefix_tokens_per_second": sum(prepared.lengths) / elapsed,
                f"{prefix}/padding_fraction": 1 - sum(prepared.lengths) / padded_tokens})
            if (epoch + 1 < self.config.fitting_epochs and self.config.eval_every_n_epochs
                    and (epoch + 1) % self.config.eval_every_n_epochs == 0):
                values, losses, stats = self._evaluate_prepared(prepared)
                metrics.update(value_diagnostics(samples, values, losses, prefix=prefix,
                                                 detailed=self.detailed_diagnostics))
                metrics[f"{prefix}/eval_seconds"] = stats["seconds"]
            print(f"[critic] epoch={epoch + 1}/{self.config.fitting_epochs} "
                  f"target_CE={metrics[f'{prefix}/target_ce']:.6f} train_seconds={elapsed:.2f}", flush=True)
            if scalar_targets is None and self.config.num_bins == 2:
                metrics[f"{prefix}/online_bce"] = metrics[f"{prefix}/target_ce"]
        self._progress("critic_final_predictions", force=True, state_count=len(samples))
        values, losses, stats = self._evaluate_prepared(prepared)
        metrics.update({f"value/final_predictions/{key}": value for key, value in stats.items()})
        diagnostic_started = time.monotonic()
        metrics.update(value_diagnostics(samples, values, losses, prefix="value/eval_after",
                                         detailed=self.detailed_diagnostics))
        metrics["value/final_diagnostics_seconds"] = time.monotonic() - diagnostic_started
        metrics["value/train_seconds"] = train_seconds
        metrics.update({"value/optimizer_step_count": self.update_count - initial_steps,
            "value/total_optimizer_steps": self.update_count, "value/last_gradient_norm": float(norm), "value/fitting_epochs": self.config.fitting_epochs,
            "value/backbone_learning_rate": self.optimizer.param_groups[0]["lr"],
            "value/head_learning_rate": self.optimizer.param_groups[1]["lr"],
            "value/eval_scope": "training_collection", "value/fit_seconds": time.monotonic() - started,
            "value/peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "value/peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0})
        return values, metrics

    def save(self, path, checkpoint_id):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        save_value_head(self.head, path)
        torch.save({"version": 1, "optimizer": self.optimizer.state_dict(),
                    "update_count": self.update_count}, path / CRITIC_STATE)
        identity = uuid.uuid4().hex
        version = 1 if self.config.num_bins == 2 and self.config.baseline_timing == "post_fit" else 2
        atomic_json(path / CRITIC_MANIFEST, {"version": version, "identity": identity,
            "checkpoint_id": checkpoint_id,
            **({"class_order": [-1, 1]} if version == 1 else {}),
            **({"method": self.method, "frozen_digest": self.frozen_digest,
                "source_critic_identity": self.source_critic_identity} if version == 2 else {}),
            "tokenizer_signature": self.tokenizer_hash, "initialization": self.config.model_path,
            "source": self.source, "config": asdict(self.config)})
        return identity
