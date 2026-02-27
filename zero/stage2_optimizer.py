from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from collectives import CollectiveOps, LocalCollectives, SendRecvCollectives
from .common import (
    ShardSpec,
    assign_flat_params,
    build_flat_param_metadata,
    compute_shard_spec,
    flatten_grads_fp32,
    flatten_params_fp32,
    get_rank_world_size,
)


@dataclass
class ZeROStage2Optimizer:
    """Stage 2: reduce-scatter gradients, shard optimizer states, allgather updated params."""

    model: nn.Module
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    collectives: Optional[CollectiveOps] = None

    def __post_init__(self) -> None:
        rank, world_size = get_rank_world_size()
        self.rank = rank
        self.world_size = world_size

        if self.collectives is None:
            self.collectives = SendRecvCollectives() if world_size > 1 else LocalCollectives()

        self.meta = build_flat_param_metadata(self.model)
        self.shard: ShardSpec = compute_shard_spec(self.meta.total_numel, rank=rank, world_size=world_size)

        self.step_count = 0
        self.exp_avg = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.exp_avg_sq = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def zero_grad(self) -> None:
        for p in self.meta.params:
            p.grad = None

    def _adamw_update(self, local_params: torch.Tensor, local_grads: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        beta1, beta2 = self.betas

        self.exp_avg.mul_(beta1).add_(local_grads, alpha=1.0 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(local_grads, local_grads, value=1.0 - beta2)

        bias_correction1 = 1.0 - (beta1**self.step_count)
        bias_correction2 = 1.0 - (beta2**self.step_count)

        if self.weight_decay != 0.0:
            local_params.mul_(1.0 - (self.lr * self.weight_decay))

        denom = self.exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
        step_size = self.lr / bias_correction1
        local_params.addcdiv_(self.exp_avg, denom, value=-step_size)
        return local_params

    def step(self) -> None:
        flat_params = flatten_params_fp32(self.meta)
        flat_grads = flatten_grads_fp32(self.meta)

        reduced_shard = self.collectives.reduce_scatter(flat_grads)
        reduced_shard = reduced_shard / self.world_size

        expected = self.shard.shard_numel
        if reduced_shard.numel() < expected:
            raise ValueError(f"reduce_scatter returned too few elements: {reduced_shard.numel()} < {expected}")
        local_grads = reduced_shard[:expected].clone()

        local_params = flat_params[self.shard.shard_start : self.shard.shard_end].clone()
        updated_local = self._adamw_update(local_params=local_params, local_grads=local_grads)

        full_updated = self.collectives.allgather(updated_local)
        assign_flat_params(self.meta, full_updated[: self.meta.total_numel])

    def state_dict(self) -> Dict[str, object]:
        return {
            "step_count": self.step_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "shard_start": self.shard.shard_start,
            "shard_end": self.shard.shard_end,
            "exp_avg": self.exp_avg.detach().cpu(),
            "exp_avg_sq": self.exp_avg_sq.detach().cpu(),
            "hparams": {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            },
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.step_count = int(state["step_count"])
        self.exp_avg = state["exp_avg"].to(device=self.meta.device, dtype=torch.float32)
        self.exp_avg_sq = state["exp_avg_sq"].to(device=self.meta.device, dtype=torch.float32)
