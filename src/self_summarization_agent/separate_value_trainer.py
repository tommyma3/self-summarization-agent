"""Prefix-only MC classification using an independently optimized text model."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import time
import uuid

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from self_summarization_agent.separate_value import (
    CRITIC_MANIFEST, CRITIC_STATE, atomic_json, digest, validate_samples,
)
from self_summarization_agent.value_model import (
    CompactionValueHead, expected_binary_value, load_value_head, model_hidden_size,
    reward_class_indices, save_value_head,
)
from self_summarization_agent.value_diagnostics import value_diagnostics
from self_summarization_agent.trainer import _write_training_progress


def tokenizer_signature(tokenizer):
    # Do not compare chat templates: collection already rendered authoritative IDs.
    return digest({"vocab": tokenizer.get_vocab(), "special_ids": sorted(tokenizer.all_special_ids)})


class SeparateValueTrainer:
    def __init__(self, source, actor_path, config, *, dtype="bfloat16", device="cuda:0",
                 activation_checkpointing=True, resume=False, progress_path=None):
        self.config = config
        self.progress_path = progress_path
        self.device = torch.device(device)
        self.update_count = 0
        self.source = str(source)
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
        if activation_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head, found = load_value_head(source, hidden_size=model_hidden_size(self.model),
                                           zero_initialize=config.zero_initialize_head)
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

    def logits(self, samples):
        prefixes = [s.input_ids[:s.state_prefix_length] for s in samples]
        limit = self.model.config.max_position_embeddings
        vocab_size = self.model.get_input_embeddings().num_embeddings
        if any(len(p) > limit or any(t < 0 or t >= vocab_size for t in p) for p in prefixes):
            raise ValueError("Critic prefix exceeds model capacity or vocabulary")
        lengths = torch.tensor([len(p) for p in prefixes], device=self.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        ids = torch.full((len(prefixes), int(lengths.max())), pad_id, dtype=torch.long, device=self.device)
        for row, prefix in enumerate(prefixes):
            ids[row, :len(prefix)] = torch.tensor(prefix, device=self.device)
        mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
        positions = torch.arange(ids.shape[1], device=self.device)[None, :].expand_as(ids)
        hidden = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False).last_hidden_state
        return self.head(hidden[torch.arange(len(samples), device=self.device), lengths - 1])

    def evaluate(self, samples):
        self.model.eval()
        self.head.eval()
        values, losses = [], []
        with torch.no_grad():
            for start in range(0, len(samples), self.config.prefix_microbatch_size):
                batch = samples[start:start + self.config.prefix_microbatch_size]
                logits = self.logits(batch)
                targets = reward_class_indices(torch.tensor([s.reward for s in batch], device=self.device))
                values.extend(expected_binary_value(logits).cpu().tolist())
                losses.extend(torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none").cpu().tolist())
        return values, losses

    def fit(self, samples, *, max_sequence_length=None):
        validate_samples(samples, max_sequence_length)
        if not samples:
            return [], {"value/optimizer_step_count": 0}
        started = time.monotonic()
        before, before_losses = self.evaluate(samples)
        metrics = value_diagnostics(samples, before, before_losses, prefix="value/eval_before")
        rollouts = {}
        for sample in samples:
            rollouts.setdefault(sample.rollout_id, []).append(sample)
        keys = list(rollouts)
        initial_steps = self.update_count
        for epoch in range(self.config.fitting_epochs):
            random.Random(self.config.seed + self.update_count + epoch).shuffle(keys)
            self.model.train()
            self.head.train()
            for start in range(0, len(keys), self.config.rollout_minibatch_size):
                selected = keys[start:start + self.config.rollout_minibatch_size]
                examples = [(s, 1 / (len(selected) * len(rollouts[k]))) for k in selected for s in rollouts[k]]
                self.optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(examples), self.config.prefix_microbatch_size):
                    chunk = examples[offset:offset + self.config.prefix_microbatch_size]
                    logits = self.logits([s for s, _ in chunk])
                    targets = reward_class_indices(torch.tensor([s.reward for s, _ in chunk], device=self.device))
                    losses = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none")
                    weights = torch.tensor([w for _, w in chunk], device=self.device)
                    (losses * weights).sum().backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [*self.model.parameters(), *self.head.parameters()], self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.update_count += 1
                _write_training_progress(self.progress_path, "critic_fit", epoch=epoch + 1,
                    epochs=self.config.fitting_epochs, completed_rollouts=min(start + len(selected), len(keys)),
                    rollout_count=len(keys), optimizer_steps=self.update_count, gradient_norm=float(norm))
            _write_training_progress(self.progress_path, "critic_evaluate", epoch=epoch + 1,
                                     epochs=self.config.fitting_epochs, state_count=len(samples))
            values, losses = self.evaluate(samples)
            epoch_metrics = value_diagnostics(samples, values, losses, prefix=f"value/epoch_{epoch + 1}")
            metrics.update(epoch_metrics)
            print(f"[critic] epoch={epoch + 1}/{self.config.fitting_epochs} CE={epoch_metrics[f'value/epoch_{epoch + 1}/bce']:.6f}", flush=True)
        metrics.update(value_diagnostics(samples, values, losses, prefix="value/eval_after"))
        metrics.update({"value/optimizer_step_count": self.update_count - initial_steps,
            "value/total_optimizer_steps": self.update_count, "value/last_gradient_norm": float(norm), "value/fitting_epochs": self.config.fitting_epochs,
            "value/backbone_learning_rate": self.optimizer.param_groups[0]["lr"],
            "value/head_learning_rate": self.optimizer.param_groups[1]["lr"],
            "value/eval_scope": "training_collection", "value/fit_seconds": time.monotonic() - started,
            "value/peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0})
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
        atomic_json(path / CRITIC_MANIFEST, {"version": 1, "identity": identity,
            "checkpoint_id": checkpoint_id, "class_order": [-1, 1],
            "tokenizer_signature": self.tokenizer_hash, "initialization": self.config.model_path,
            "source": self.source, "config": asdict(self.config)})
        return identity
