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
    get_rank_world_size,
    grads_num_bytes,
    model_param_dtype,
    params_num_bytes,
    tensor_num_bytes,
    tensors_num_bytes,
    flatten_param_shard_fp32,
)


@dataclass
class ZeROStage2Optimizer:
    """Stage 2: shard optimizer state and gradients, keep params replicated."""

    model: nn.Module
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_bucket_numel: int = 16 * 1024 * 1024
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
        self.param_dtype = model_param_dtype(self.meta)
        self.shard: ShardSpec = compute_shard_spec(self.meta.total_numel, rank=rank, world_size=world_size)
        self.local_master_param_shard = flatten_param_shard_fp32(
            self.meta,
            self.shard.shard_start,
            self.shard.shard_end,
        )
        self.exp_avg = torch.zeros_like(self.local_master_param_shard)
        self.exp_avg_sq = torch.zeros_like(self.local_master_param_shard)
        self.grad_shard: Optional[torch.Tensor] = None
        self.flat_grad_buffer: Optional[torch.Tensor] = None
        self._flat_grad_buffer_dirty = False
        self.step_count = 0
        self._backward_comm_ms = 0.0
        self._post_step_allgather_ms = 0.0
        self._backward_reduce_scatter_calls = 0
        self._post_step_allgather_calls = 0
        self._backward_reduce_scatter_bytes = 0
        self._post_step_allgather_bytes = 0
        self._register_grad_hooks()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
        self._flush_flat_grad_buffer()

    def prepare_forward(self) -> Dict[str, float]:
        return {"prepare_comm_ms": 0.0}

    def zero_grad(self) -> None:
        self._backward_comm_ms = 0.0
        self._post_step_allgather_ms = 0.0
        self._backward_reduce_scatter_calls = 0
        self._post_step_allgather_calls = 0
        self._backward_reduce_scatter_bytes = 0
        self._post_step_allgather_bytes = 0
        self.grad_shard = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.flat_grad_buffer = torch.zeros(self.meta.total_numel, dtype=self.param_dtype, device=self.meta.device)
        self._flat_grad_buffer_dirty = False
        for p in self.meta.params:
            p.grad = None

    def _register_grad_hooks(self) -> None:
        for param_idx, param in enumerate(self.meta.params):

            def _hook(
                grad_param: torch.Tensor,
                *,
                _param_idx: int = param_idx,
            ) -> None:
                del grad_param
                param_ref = self.meta.params[_param_idx]
                if param_ref.grad is None:
                    return
                if self.flat_grad_buffer is None:
                    raise RuntimeError("stage2 flat_grad_buffer must be initialized in zero_grad before backward")

                start = self.meta.offsets[_param_idx]
                end = start + param_ref.numel()
                grad = param_ref.grad.detach().view(-1)
                if grad.dtype != self.param_dtype:
                    grad = grad.to(self.param_dtype)
                self.flat_grad_buffer[start:end].add_(grad)
                self._flat_grad_buffer_dirty = True
                param_ref.grad = None

            param.register_post_accumulate_grad_hook(_hook)

    def _flush_flat_grad_buffer(self) -> None:
        if self.flat_grad_buffer is None or not self._flat_grad_buffer_dirty:
            return

        self._record_memory_event("measured_step_stage2_pre_reduce_scatter")
        t_comm0 = time.perf_counter()
        reduced_shard = self.collectives.reduce_scatter(self.flat_grad_buffer)
        rs_ms = (time.perf_counter() - t_comm0) * 1000.0
        self._backward_comm_ms += rs_ms
        self._backward_reduce_scatter_calls += 1
        self._backward_reduce_scatter_bytes += tensor_num_bytes(self.flat_grad_buffer)
        self._record_memory_event("measured_step_stage2_post_reduce_scatter")

        local_grads = reduced_shard[: self.shard.shard_numel].to(torch.float32)
        if self.world_size > 1:
            local_grads.div_(self.world_size)
        if self.grad_shard is None:
            self.grad_shard = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.grad_shard.add_(local_grads)
        self.flat_grad_buffer.zero_()
        self._flat_grad_buffer_dirty = False
        del local_grads
        del reduced_shard

    def _adamw_update(self, local_grads: torch.Tensor) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas

        self.exp_avg.mul_(beta1).add_(local_grads, alpha=1.0 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(local_grads, local_grads, value=1.0 - beta2)

        bias_correction1 = 1.0 - (beta1**self.step_count)
        bias_correction2 = 1.0 - (beta2**self.step_count)

        if self.weight_decay != 0.0:
            self.local_master_param_shard.mul_(1.0 - (self.lr * self.weight_decay))

        denom = self.exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
        step_size = self.lr / bias_correction1
        self.local_master_param_shard.addcdiv_(self.exp_avg, denom, value=-step_size)

    def _global_grad_norm(self, local_grads: torch.Tensor) -> float:
        sum_sq = torch.sum(local_grads * local_grads)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_sq)
        return float(torch.sqrt(sum_sq).item())

    def _clip_local_grads_inplace(self, local_grads: torch.Tensor, max_grad_norm: float) -> float:
        grad_norm = self._global_grad_norm(local_grads)
        if max_grad_norm > 0 and grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            local_grads.mul_(scale)
        return grad_norm

    def _record_memory_event(self, label: str) -> None:
        if self.memory_tracker is None or not self.memory_trace_active:
            return
        self.memory_tracker.record(label)
        if self.memory_state_timeline is None:
            return
        self.memory_state_timeline.append(
            {
                "label": label,
                **self.live_model_state_breakdown_mb(),
                **self._debug_memory_components_mb(),
            }
        )

    def _debug_memory_components_mb(self) -> Dict[str, float]:
        logical_grad_bytes = self.shard.shard_numel * torch.tensor([], dtype=self.param_dtype).element_size()
        return {
            "logical_params_mb": bytes_to_mb(params_num_bytes(self.meta.params)),
            "logical_grads_mb": bytes_to_mb(logical_grad_bytes),
            "logical_optimizer_mb": bytes_to_mb(tensors_num_bytes([self.local_master_param_shard, self.exp_avg, self.exp_avg_sq])),
            "live_full_grads_mb": bytes_to_mb(grads_num_bytes(self.meta.params)),
            "live_flat_grad_buffer_mb": bytes_to_mb(tensor_num_bytes(self.flat_grad_buffer)),
            "live_grad_shard_mb": bytes_to_mb(tensor_num_bytes(self.grad_shard)),
            "comm_temp_mb": 0.0,
        }

    def debug_memory_components_mb(self) -> Dict[str, float]:
        return self._debug_memory_components_mb()

    def step_with_stats(self, max_grad_norm: float = 0.0) -> Dict[str, float]:
        t0 = time.perf_counter()
        if any(param.grad is not None for param in self.meta.params):
            raise RuntimeError("Stage 2 should drain param.grad into the contiguous flat buffer during backward")
        self._flush_flat_grad_buffer()
        if self.grad_shard is None:
            raise RuntimeError("Stage 2 grad_shard must be initialized before optimizer step")

        grad_norm = self._clip_local_grads_inplace(self.grad_shard, max_grad_norm=max_grad_norm)

        t_opt0 = time.perf_counter()
        self._adamw_update(local_grads=self.grad_shard)
        optim_ms = (time.perf_counter() - t_opt0) * 1000.0

        self._record_memory_event("measured_step_stage2_pre_allgather")
        t_comm1 = time.perf_counter()
        updated_local = self.local_master_param_shard.to(dtype=self.param_dtype)
        full_updated = self.collectives.allgather(updated_local)
        allgather_ms = (time.perf_counter() - t_comm1) * 1000.0
        self._post_step_allgather_ms += allgather_ms
        self._post_step_allgather_calls += 1
        self._post_step_allgather_bytes += tensor_num_bytes(updated_local)
        assign_flat_params(self.meta, full_updated[: self.meta.total_numel])
        self._record_memory_event("measured_step_stage2_post_allgather")

        del full_updated
        del updated_local
        self.grad_shard = None
        self.flat_grad_buffer = None
        self._flat_grad_buffer_dirty = False
        self._record_memory_event("measured_step_stage2_post_step_free_grad_shard")

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "grad_norm": grad_norm,
            "comm_ms": self._backward_comm_ms + self._post_step_allgather_ms,
            "communication_backward_reduce_scatter_ms": self._backward_comm_ms,
            "communication_backward_reduce_scatter_calls": float(self._backward_reduce_scatter_calls),
            "communication_backward_reduce_scatter_bytes": float(self._backward_reduce_scatter_bytes),
            "communication_post_allgather_ms": self._post_step_allgather_ms,
            "communication_post_allgather_calls": float(self._post_step_allgather_calls),
            "communication_post_allgather_bytes": float(self._post_step_allgather_bytes),
            "optim_ms": optim_ms,
            "total_ms": total_ms,
        }

    def step(self, max_grad_norm: float = 0.0) -> float:
        return float(self.step_with_stats(max_grad_norm=max_grad_norm)["grad_norm"])

    def memory_state_breakdown_mb(self) -> Dict[str, float]:
        logical_grad_bytes = self.shard.shard_numel * torch.tensor([], dtype=self.param_dtype).element_size()
        params_mb = bytes_to_mb(params_num_bytes(self.meta.params))
        grads_mb = bytes_to_mb(logical_grad_bytes)
        optimizer_mb = bytes_to_mb(tensors_num_bytes([self.local_master_param_shard, self.exp_avg, self.exp_avg_sq]))
        return {
            "params_mb": params_mb,
            "grads_mb": grads_mb,
            "optimizer_mb": optimizer_mb,
            "total_mb": params_mb + grads_mb + optimizer_mb,
        }

    def live_model_state_breakdown_mb(self) -> Dict[str, float]:
        params_mb = bytes_to_mb(params_num_bytes(self.meta.params))
        grads_mb = bytes_to_mb(
            grads_num_bytes(self.meta.params)
            + tensor_num_bytes(self.flat_grad_buffer)
            + tensor_num_bytes(self.grad_shard)
        )
        optimizer_mb = bytes_to_mb(tensors_num_bytes([self.local_master_param_shard, self.exp_avg, self.exp_avg_sq]))
        return {
            "params_mb": params_mb,
            "grads_mb": grads_mb,
            "optimizer_mb": optimizer_mb,
            "total_mb": params_mb + grads_mb + optimizer_mb,
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "step_count": self.step_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "shard_start": self.shard.shard_start,
            "shard_end": self.shard.shard_end,
            "local_master_param_shard": self.local_master_param_shard.detach().cpu(),
            "exp_avg": self.exp_avg.detach().cpu(),
            "exp_avg_sq": self.exp_avg_sq.detach().cpu(),
            "grad_shard": None if self.grad_shard is None else self.grad_shard.detach().cpu(),
            "hparams": {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            },
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.step_count = int(state["step_count"])
        self.local_master_param_shard = state["local_master_param_shard"].to(device=self.meta.device, dtype=torch.float32)
        self.exp_avg = state["exp_avg"].to(device=self.meta.device, dtype=torch.float32)
        self.exp_avg_sq = state["exp_avg_sq"].to(device=self.meta.device, dtype=torch.float32)
        grad_shard = state.get("grad_shard")
        if grad_shard is None:
            self.grad_shard = None
        else:
            self.grad_shard = grad_shard.to(device=self.meta.device, dtype=torch.float32)
        full_updated = self.collectives.allgather(self.local_master_param_shard.to(dtype=self.param_dtype))
        assign_flat_params(self.meta, full_updated[: self.meta.total_numel])
