"""Head-free dual-clip PPO worker with globally token-weighted diagnostics."""
from functools import partial

import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

from self_summarization_agent.verl_interval_worker import IntervalActorRolloutRefWorker


def fixed_advantage_ppo_loss(config, model_output, data, dp_group=None):
    clip_ratio_c = getattr(config, "clip_ratio_c", 3.0)
    if not clip_ratio_c > 1.0:
        raise ValueError("Dual-clip PPO requires clip_ratio_c > 1")
    clip_low = getattr(config, "clip_ratio_low", None)
    clip_high = getattr(config, "clip_ratio_high", None)
    clip_low = config.clip_ratio if clip_low is None else clip_low
    clip_high = config.clip_ratio if clip_high is None else clip_high
    logprobs = no_padding_2_padding(model_output["log_probs"], data).float()
    fields = data.select("response_mask", "old_log_probs", "advantages").to_padded_tensor()
    mask = fields["response_mask"].bool()
    # Mask before exp: zero-weight alignment rows must not create inf * 0.
    delta = torch.where(mask, logprobs - fields["old_log_probs"].float(), 0.).clamp(-20., 20.)
    ratio = delta.exp()
    advantages = fields["advantages"].float()
    unclipped_losses = -ratio * advantages
    clipped_losses = -ratio.clamp(1 - clip_low, 1 + clip_high) * advantages
    ppo_losses = torch.maximum(unclipped_losses, clipped_losses)
    dual_clip_bound = -clip_ratio_c * advantages
    losses = torch.where(advantages < 0, torch.minimum(ppo_losses, dual_clip_bound), ppo_losses)
    denominator = data["batch_num_tokens"]
    scale = data["dp_size"] / denominator
    loss = torch.where(mask, losses, 0.).sum() * scale
    kl = torch.where(mask, -delta.detach(), 0.).sum() * scale
    clip = (clipped_losses.detach() > unclipped_losses.detach()).logical_and(mask).float().sum() * scale
    dual_clip = ((ppo_losses.detach() > dual_clip_bound.detach()) & (advantages < 0)
                 & mask).float().sum() * scale
    return loss, {name: Metric(value=value.detach(), aggregation=AggregationType.SUM)
        for name, value in {"actor/pg_loss": loss, "actor/ppo_kl": kl,
                            "actor/pg_clipfrac": clip, "actor/pg_clipfrac_lower": dual_clip}.items()}


class SeparateCriticActorWorker(IntervalActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self.actor.set_loss_fn(partial(fixed_advantage_ppo_loss, config=self.config.actor))
