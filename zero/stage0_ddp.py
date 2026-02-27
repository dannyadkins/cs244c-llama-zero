from __future__ import annotations

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

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        flat_grads = flatten_grads_fp32(self.meta)
        reduced = self.collectives.allreduce(flat_grads, average=True)
        assign_flat_grads(self.meta, reduced)
        self.optimizer.step()

    def state_dict(self) -> Dict[str, object]:
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
