"""Head-free PPO worker with globally token-weighted loss and diagnostics."""
from functools import partial

import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.metric import AggregationType, Metric
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import no_padding_2_padding


def fixed_advantage_ppo_loss(config, model_output, data, dp_group=None):
    logprobs = no_padding_2_padding(model_output["log_probs"], data).float()
    fields = data.select("response_mask", "old_log_probs", "advantages").to_padded_tensor()
    mask = fields["response_mask"].bool()
    # Mask before exp: zero-weight alignment rows must not create inf * 0.
    delta = torch.where(mask, logprobs - fields["old_log_probs"].float(), 0.)
    ratio = delta.exp()
    advantages = fields["advantages"].float()
    losses = -torch.minimum(ratio * advantages,
        ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages)
    denominator = data["batch_num_tokens"]
    scale = data["dp_size"] / denominator
    loss = torch.where(mask, losses, 0.).sum() * scale
    kl = torch.where(mask, -delta.detach(), 0.).sum() * scale
    clip = ((ratio.detach() - 1).abs() > config.clip_ratio).logical_and(mask).float().sum() * scale
    return loss, {name: Metric(value=value.detach(), aggregation=AggregationType.SUM)
        for name, value in {"actor/pg_loss": loss, "actor/ppo_kl": kl,
                            "actor/pg_clipfrac": clip}.items()}


class SeparateCriticActorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self.actor.set_loss_fn(partial(fixed_advantage_ppo_loss, config=self.config.actor))
