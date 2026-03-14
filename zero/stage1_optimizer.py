from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from collectives import CollectiveOps, LocalCollectives, SendRecvCollectives
from profiler import MemoryTracker
from .common import (
    ShardSpec,
    assign_flat_params,
    build_flat_param_metadata,
    bytes_to_mb,
    compute_shard_spec,
    flatten_param_shard_fp32,
    flatten_grads_fp32,
    get_rank_world_size,
    grads_num_bytes,
    params_num_bytes,
    tensors_num_bytes,
)


@dataclass
class ZeROStage1Optimizer:
    """Stage 1: partition optimizer states, allreduce full gradients, allgather updated params."""

    model: nn.Module
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    collectives: Optional[CollectiveOps] = None
    memory_tracker: Optional[MemoryTracker] = None
    memory_trace_active: bool = False
    memory_state_timeline: Optional[List[Dict[str, object]]] = None

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

    def prepare_forward(self) -> Dict[str, float]:
        return {"prepare_comm_ms": 0.0}

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

    def _clip_flat_grads_inplace(self, flat_grads: torch.Tensor, max_grad_norm: float) -> float:
        grad_norm = float(flat_grads.norm(2).item())
        if max_grad_norm > 0 and grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            flat_grads.mul_(scale)
        return grad_norm

    def _record_memory_event(self, label: str) -> None:
        if self.memory_tracker is None or not self.memory_trace_active:
            return
        self.memory_tracker.record(label)
        if self.memory_state_timeline is None:
            return
        self.memory_state_timeline.append({"label": label, **self.live_model_state_breakdown_mb()})

    def step_with_stats(self, max_grad_norm: float = 0.0) -> Dict[str, float]:
        t0 = time.perf_counter()
        flat_grads = flatten_grads_fp32(self.meta)

        self._record_memory_event("measured_step_stage1_pre_allreduce")
        t_comm0 = time.perf_counter()
        synced_grads = self.collectives.allreduce_inplace(flat_grads, average=True)
        comm_ms = (time.perf_counter() - t_comm0) * 1000.0
        self._record_memory_event("measured_step_stage1_post_allreduce")
        grad_norm = self._clip_flat_grads_inplace(synced_grads, max_grad_norm=max_grad_norm)

        local_params = flatten_param_shard_fp32(self.meta, self.shard.shard_start, self.shard.shard_end)
        local_grads = synced_grads[self.shard.shard_start : self.shard.shard_end]

        t_opt0 = time.perf_counter()
        updated_local = self._adamw_update(local_params=local_params, local_grads=local_grads)
        optim_ms = (time.perf_counter() - t_opt0) * 1000.0
        del local_params
        del local_grads
        del synced_grads
        del flat_grads

        self._record_memory_event("measured_step_stage1_pre_allgather")
        t_comm1 = time.perf_counter()
        full_updated = self.collectives.allgather(updated_local)
        comm_ms += (time.perf_counter() - t_comm1) * 1000.0
        del updated_local
        assign_flat_params(self.meta, full_updated[: self.meta.total_numel])
        del full_updated
        self._record_memory_event("measured_step_stage1_post_allgather")

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "grad_norm": grad_norm,
            "comm_ms": comm_ms,
            "optim_ms": optim_ms,
            "total_ms": total_ms,
        }

    def step(self, max_grad_norm: float = 0.0) -> float:
        return float(self.step_with_stats(max_grad_norm=max_grad_norm)["grad_norm"])

    def memory_state_breakdown_mb(self) -> Dict[str, float]:
        params_mb = bytes_to_mb(params_num_bytes(self.meta.params))
        grads_mb = bytes_to_mb(grads_num_bytes(self.meta.params))
        optimizer_mb = bytes_to_mb(tensors_num_bytes([self.exp_avg, self.exp_avg_sq]))
        return {
            "params_mb": params_mb,
            "grads_mb": grads_mb,
            "optimizer_mb": optimizer_mb,
            "total_mb": params_mb + grads_mb + optimizer_mb,
        }

    def live_model_state_breakdown_mb(self) -> Dict[str, float]:
        return self.memory_state_breakdown_mb()

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
