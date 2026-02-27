from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from collectives import CollectiveOps, LocalCollectives, SendRecvCollectives
from .common import assign_flat_grads, build_flat_param_metadata, flatten_grads_fp32, get_rank_world_size


@dataclass
class ZeROStage0DDP:
    """Stage 0 baseline: full replication + gradient allreduce + local AdamW step."""

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
        self.optimizer = torch.optim.AdamW(
            self.meta.params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def prepare_forward(self) -> Dict[str, float]:
        return {"prepare_comm_ms": 0.0}

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _clip_flat_grads_inplace(self, flat_grads: torch.Tensor, max_grad_norm: float) -> float:
        grad_norm = float(flat_grads.norm(2).item())
        if max_grad_norm > 0 and grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            flat_grads.mul_(scale)
        return grad_norm

    def step_with_stats(self, max_grad_norm: float = 0.0) -> Dict[str, float]:
        t0 = time.perf_counter()
        flat_grads = flatten_grads_fp32(self.meta)

        t_comm0 = time.perf_counter()
        reduced = self.collectives.allreduce(flat_grads, average=True)
        comm_ms = (time.perf_counter() - t_comm0) * 1000.0

        grad_norm = self._clip_flat_grads_inplace(reduced, max_grad_norm=max_grad_norm)

        assign_flat_grads(self.meta, reduced)

        t_opt0 = time.perf_counter()
        self.optimizer.step()
        optim_ms = (time.perf_counter() - t_opt0) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "grad_norm": grad_norm,
            "comm_ms": comm_ms,
            "optim_ms": optim_ms,
            "total_ms": total_ms,
        }

    def step(self, max_grad_norm: float = 0.0) -> float:
        return float(self.step_with_stats(max_grad_norm=max_grad_norm)["grad_norm"])

    def state_dict(self) -> Dict[str, object]:
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
