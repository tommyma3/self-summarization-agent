"""Attach the value head without changing the joint optimizer schedule."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def attach_value_head(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Return the head group index; None preserves the historical shared group.

    verl creates LambdaLR before the sidecar exists. Extend that same scheduler,
    including its current LR, without resetting its step count. Other scheduler
    types fail before modifying the optimizer.
    """
    if not optimizer.param_groups:
        raise ValueError("Value-head training requires an initialized actor optimizer")
    params = list(parameters)
    existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not params or len({id(p) for p in params}) != len(params) or any(id(p) in existing for p in params):
        raise ValueError("Value-head parameters must be nonempty, unique, and absent from the actor optimizer")
    if learning_rate is None:
        optimizer.param_groups[0]["params"].extend(params)
        return 0
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not (
        math.isfinite(learning_rate) and learning_rate > 0
    ):
        raise ValueError("Value-head learning rate must be finite and positive")

    index = len(optimizer.param_groups)
    current_lr = float(learning_rate)
    if scheduler is not None:
        if type(scheduler) is not torch.optim.lr_scheduler.LambdaLR or scheduler.optimizer is not optimizer:
            raise ValueError("Separate value-head LR supports only the actor's LambdaLR scheduler")
        if not (len(scheduler.base_lrs) == len(scheduler.lr_lambdas) == len(scheduler.get_last_lr()) == index):
            raise ValueError("Actor LambdaLR parameter-group bookkeeping is inconsistent")
        current_lr *= scheduler.lr_lambdas[0](scheduler.last_epoch)

    group = {key: value for key, value in optimizer.param_groups[0].items()
             if key not in {"params", "lr", "initial_lr"}}
    group.update(params=params, lr=current_lr)
    if scheduler is not None:
        group["initial_lr"] = float(learning_rate)
    optimizer.add_param_group(group)
    if scheduler is not None:
        scheduler.base_lrs.append(float(learning_rate))
        scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
        scheduler._last_lr.append(current_lr)
    return index


def value_learning_rate_metrics(optimizer: torch.optim.Optimizer, head_group: int) -> dict[str, float | None]:
    backbone_lr = float(optimizer.param_groups[0]["lr"])
    head_lr = float(optimizer.param_groups[head_group]["lr"])
    return {
        "value/backbone_learning_rate": backbone_lr,
        "value/head_learning_rate": head_lr,
        "value/head_to_backbone_lr_ratio": head_lr / backbone_lr if backbone_lr else None,
    }
