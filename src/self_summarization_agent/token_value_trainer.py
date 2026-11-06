"""Full-interval causal forward passes for the independent two-bin token critic."""

import random
import time
from dataclasses import dataclass

import torch

from .token_value import token_diagnostics, token_positions, validate_token_samples


@dataclass
class PreparedTokens:
    prefixes: list
    lengths: list
    positions: list
    position_tensors: list


class TokenValueForward(torch.nn.Module):
    """Keep the backbone and head together when replicating across devices."""

    def __init__(self, model, head):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, ids, lengths, sampled_positions):
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :].expand_as(ids)
        hidden = self.model(
            input_ids=ids,
            attention_mask=positions < lengths[:, None],
            position_ids=positions,
            use_cache=False,
        ).last_hidden_state
        # Gather only sampled states before the head; never transfer hidden states
        # between GPUs. Padding here is ignored by the caller.
        rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
        return self.head(hidden[rows, sampled_positions])


class TokenValueTrainerMixin:
    def _prepare_tokens(self, samples, max_sequence_length=None):
        validate_token_samples(samples, max_sequence_length)
        capacity = self.model.config.max_position_embeddings
        vocab = self.model.get_input_embeddings().num_embeddings
        if any(
            len(s.input_ids) > capacity
            or any(t >= vocab for t in s.collection_full_token_ids)
            for s in samples
        ):
            raise ValueError(
                "Full token interval exceeds critic capacity or vocabulary; never truncate"
            )
        positions = [token_positions(s) for s in samples]
        return PreparedTokens(
            [torch.tensor(s.input_ids, dtype=torch.long) for s in samples],
            [len(s.input_ids) for s in samples],
            positions,
            [torch.tensor(p, dtype=torch.long) for p in positions],
        )

    def _token_logits(self, prepared, indices):
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids = self._to_device(
            torch.nn.utils.rnn.pad_sequence(
                [prepared.prefixes[i] for i in indices],
                batch_first=True,
                padding_value=pad,
            )
        )
        lengths = self._to_device(torch.tensor([prepared.lengths[i] for i in indices]))
        sampled_positions = self._to_device(torch.nn.utils.rnn.pad_sequence(
            [prepared.position_tensors[i] for i in indices], batch_first=True,
        ))
        forward = getattr(self, "_token_parallel", None)
        if forward is None:
            forward = TokenValueForward(self.model, self.head)
        logits = forward(ids, lengths, sampled_positions)
        return [logits[row, :len(prepared.positions[i])] for row, i in enumerate(indices)]

    def evaluate_tokens(self, samples):
        prepared = self._prepare_tokens(samples)
        return self._evaluate_tokens(samples, prepared)

    def _evaluate_tokens(self, samples, prepared):
        values, losses = [[] for _ in samples], [[] for _ in samples]
        modes = self.model.training, self.head.training
        devices = getattr(self, "_token_devices", [self.device]) if self.device.type == "cuda" else []
        try:
            self.model.eval()
            self.head.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                eligible = [i for i, p in enumerate(prepared.positions) if p]
                for indices in self._batches(
                    prepared, eligible, self.config.eval_prefix_microbatch_size or 1
                ):
                    chunks = self._token_logits(prepared, indices)
                    logits = torch.cat(chunks).float()
                    targets = torch.cat([
                        torch.full((len(chunk),), int(samples[i].reward == 1),
                                   device=self.device, dtype=torch.long)
                        for i, chunk in zip(indices, chunks)
                    ])
                    batch_values = (logits.softmax(-1)[:, 1] * 2 - 1).cpu().tolist()
                    batch_losses = torch.nn.functional.cross_entropy(
                        logits, targets, reduction="none").cpu().tolist()
                    offset = 0
                    for i, chunk in zip(indices, chunks):
                        end = offset + len(chunk)
                        values[i], losses[i] = batch_values[offset:end], batch_losses[offset:end]
                        offset = end

        finally:
            self.model.train(modes[0])
            self.head.train(modes[1])
        return values, losses

    def fit_tokens(self, samples, *, max_sequence_length=None):
        started = time.monotonic()
        prepared = self._prepare_tokens(samples, max_sequence_length)
        prepare_seconds = time.monotonic() - started
        rollouts = {}
        for i, s in enumerate(samples):
            if prepared.positions[i]:
                rollouts.setdefault(s.rollout_id, []).append(i)
        keys = list(rollouts)
        initial = self.update_count
        padded = 0
        mass = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        metrics = self.kernel_diagnostics()
        metrics["value/detailed_diagnostics"] = self.detailed_diagnostics
        if self.config.eval_before_fit:
            before, before_losses = self._evaluate_tokens(samples, prepared)
            metrics.update(
                token_diagnostics(
                    samples,
                    before,
                    before_losses,
                    prefix="value/eval_before",
                    detailed=self.detailed_diagnostics,
                )
            )
        norm = 0.0
        update_started = time.monotonic()
        for epoch in range(self.config.fitting_epochs):
            random.Random(self.config.seed + self.update_count + epoch).shuffle(keys)
            self.model.train()
            self.head.train()
            total = torch.zeros((), device=self.device)
            for start in range(0, len(keys), self.config.rollout_minibatch_size):
                selected = keys[start : start + self.config.rollout_minibatch_size]
                weights = {
                    i: 1 / (len(selected) * len(rollouts[r]))
                    for r in selected
                    for i in rollouts[r]
                }
                self.optimizer.zero_grad(set_to_none=True)
                for indices in self._batches(
                    prepared, weights, self.config.prefix_microbatch_size
                ):
                    logits = self._token_logits(prepared, indices)
                    counts = [len(chunk) for chunk in logits]
                    targets = self._to_device(torch.tensor([
                        int(samples[i].reward == 1) for i in indices
                    ])).repeat_interleave(torch.tensor(counts, device=self.device), output_size=sum(counts))
                    token_weights = self._to_device(torch.tensor([
                        weights[i] / count for i, count in zip(indices, counts)
                    ])).repeat_interleave(torch.tensor(counts, device=self.device), output_size=sum(counts))
                    loss = (torch.nn.functional.cross_entropy(
                        torch.cat(logits).float(), targets, reduction="none"
                    ) * token_weights).sum()
                    total += loss.detach() * len(selected)
                    mass += sum(weights[i] for i in indices)
                    loss.backward()
                    padded += len(indices) * max(prepared.lengths[i] for i in indices)
                norm = torch.nn.utils.clip_grad_norm_(
                    [*self.model.parameters(), *self.head.parameters()],
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.update_count += 1
                self._progress(
                    "critic_token_fit",
                    epoch=epoch + 1,
                    completed_rollouts=start + len(selected),
                    rollout_count=len(keys),
                )
            metrics[f"value/epoch_{epoch + 1}/target_ce"] = float(total) / max(1, len(keys))
            if (
                epoch + 1 < self.config.fitting_epochs
                and self.config.eval_every_n_epochs
                and (epoch + 1) % self.config.eval_every_n_epochs == 0
            ):
                intermediate, intermediate_losses = self._evaluate_tokens(
                    samples, prepared
                )
                metrics.update(
                    token_diagnostics(
                        samples,
                        intermediate,
                        intermediate_losses,
                        prefix=f"value/epoch_{epoch + 1}",
                        detailed=self.detailed_diagnostics,
                    )
                )
        self.synchronize()
        update_seconds = time.monotonic() - update_started
        evaluation_started = time.monotonic()
        values, losses = self._evaluate_tokens(samples, prepared)
        evaluation_seconds = time.monotonic() - evaluation_started
        elapsed = time.monotonic() - started
        metrics.update(
            token_diagnostics(
                samples,
                values,
                losses,
                prefix="value/eval_after",
                detailed=self.detailed_diagnostics,
            )
        )
        metrics.update(
            {
                "value/data_parallel_devices": len(getattr(self, "_token_devices", [self.device])),
                "value/optimizer_step_count": self.update_count - initial,
                "value/total_optimizer_steps": self.update_count,
                "value/fit_seconds": elapsed,
                "value/sequence_prepare_seconds": prepare_seconds,
                "value/update_seconds": update_seconds,
                "value/final_evaluation_seconds": evaluation_seconds,
                "value/last_gradient_norm": float(norm),
                "value/effective_weight_mass": mass,
                "value/padded_tokens": padded,
                "value/full_interval_tokens_processed": sum(
                    prepared.lengths[i] for r in rollouts.values() for i in r
                )
                * self.config.fitting_epochs,
                "value/full_interval_tokens_per_second": sum(
                    prepared.lengths[i] for r in rollouts.values() for i in r
                )
                * self.config.fitting_epochs
                / max(elapsed, 1e-9),
                "value/tokens_per_second": sum(map(len, values))
                * self.config.fitting_epochs
                / max(elapsed, 1e-9),
                "value/peak_memory_bytes": torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0,
            }
        )
        return values, metrics
