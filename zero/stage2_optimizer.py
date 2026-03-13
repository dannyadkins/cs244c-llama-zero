from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from collectives import CollectiveOps, LocalCollectives, SendRecvCollectives
from profiler import MemoryTracker
from .common import (
    ShardSpec,
    assign_flat_params,
    build_flat_param_metadata,
    bytes_to_mb,
    compute_shard_spec,
    flatten_params_fp32,
    get_rank_world_size,
    grads_num_bytes,
    params_num_bytes,
    tensors_num_bytes,
)


@dataclass
class _Stage2GradPartition:
    index: int
    param_idx: int
    start: int
    end: int
    rank_piece_numels: List[int]
    packed_chunk_numel: int
    local_piece_numel: int
    local_shard_offset: int

    @property
    def numel(self) -> int:
        return self.end - self.start


@dataclass
class ZeROStage2Optimizer:
    """Stage 2: reduce-scatter gradients, shard optimizer states, allgather updated params."""

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
        self.grad_shard: Optional[torch.Tensor] = None
        self._backward_comm_ms = 0.0
        self._live_comm_buffer_num_bytes = 0
        self._logical_grad_shard_num_bytes = self.shard.shard_numel * torch.tensor([], dtype=torch.float32).element_size()
        self._grad_partitions = self._build_grad_partitions()
        self._grad_hook_handles = self._register_grad_hooks()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
        self._flush_pending_buckets(force=True)

    def prepare_forward(self) -> Dict[str, float]:
        return {"prepare_comm_ms": 0.0}

    def zero_grad(self) -> None:
        self._backward_comm_ms = 0.0
        self._live_comm_buffer_num_bytes = 0
        self.grad_shard = None
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

    def _global_grad_norm(self, local_grads: torch.Tensor) -> float:
        sum_sq = torch.sum(local_grads * local_grads)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_sq)
        return float(torch.sqrt(sum_sq).item())

    def _clip_local_grads_inplace(self, local_grads: torch.Tensor, max_grad_norm: float) -> float:
        grad_norm = self._global_grad_norm(local_grads)
        if max_grad_norm > 0 and grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            local_grads.mul_(scale)
        return grad_norm

    def _ensure_grad_shard(self) -> torch.Tensor:
        if self.grad_shard is None:
            self.grad_shard = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        return self.grad_shard

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

    def _build_grad_partitions(self) -> List[_Stage2GradPartition]:
        partitions: List[_Stage2GradPartition] = []
        for param_idx, (param, start) in enumerate(zip(self.meta.params, self.meta.offsets)):
            end = start + param.numel()
            partitions.append(self._make_partition(index=len(partitions), param_idx=param_idx, start=start, end=end))
        return partitions

    def _make_partition(self, index: int, param_idx: int, start: int, end: int) -> _Stage2GradPartition:
        rank_piece_numels: List[int] = []
        for rank in range(self.world_size):
            shard_start = rank * self.shard.chunk_size
            shard_end = min(shard_start + self.shard.chunk_size, self.meta.total_numel)
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            rank_piece_numels.append(max(0, overlap_end - overlap_start))

        local_overlap_start = max(start, self.shard.shard_start)
        local_overlap_end = min(end, self.shard.shard_end)
        local_piece_numel = max(0, local_overlap_end - local_overlap_start)
        return _Stage2GradPartition(
            index=index,
            param_idx=param_idx,
            start=start,
            end=end,
            rank_piece_numels=rank_piece_numels,
            packed_chunk_numel=max(1, max(rank_piece_numels) if rank_piece_numels else 0),
            local_piece_numel=local_piece_numel,
            local_shard_offset=max(0, local_overlap_start - self.shard.shard_start),
        )

    def _register_grad_hooks(self):
        handles = []
        for partition in self._grad_partitions:
            param = self.meta.params[partition.param_idx]

            def _hook(
                grad_param: torch.Tensor,
                *,
                _partition_idx: int = partition.index,
                _param_idx: int = partition.param_idx,
            ) -> None:
                del grad_param
                if self.meta.params[_param_idx].grad is None:
                    return
                self._flush_partition(_partition_idx)

            handles.append(param.register_post_accumulate_grad_hook(_hook))
        return handles

    def _pack_partition_inputs(self, partition: _Stage2GradPartition) -> torch.Tensor:
        packed = torch.zeros(
            self.world_size * partition.packed_chunk_numel,
            dtype=torch.float32,
            device=self.meta.device,
        )
        chunk_views = [
            packed[rank * partition.packed_chunk_numel : (rank + 1) * partition.packed_chunk_numel]
            for rank in range(self.world_size)
        ]
        write_offsets = [0 for _ in range(self.world_size)]

        param = self.meta.params[partition.param_idx]
        grad = param.grad
        if grad is None:
            raise RuntimeError(f"partition {partition.index} was marked ready without a gradient tensor")

        flat_grad = grad.detach().view(-1)
        if flat_grad.dtype != torch.float32:
            flat_grad = flat_grad.to(torch.float32)

        first_rank = partition.start // self.shard.chunk_size
        last_rank = (partition.end - 1) // self.shard.chunk_size
        for rank in range(first_rank, last_rank + 1):
            shard_start = rank * self.shard.chunk_size
            shard_end = min(shard_start + self.shard.chunk_size, self.meta.total_numel)
            overlap_start = max(partition.start, shard_start)
            overlap_end = min(partition.end, shard_end)
            if overlap_end <= overlap_start:
                continue
            piece_len = overlap_end - overlap_start
            dst_start = write_offsets[rank]
            dst_end = dst_start + piece_len
            src_start = overlap_start - partition.start
            src_end = src_start + piece_len
            chunk_views[rank][dst_start:dst_end].copy_(flat_grad[src_start:src_end])
            write_offsets[rank] = dst_end

        for rank, piece_len in enumerate(partition.rank_piece_numels):
            if write_offsets[rank] != piece_len:
                raise RuntimeError(
                    f"partition {partition.index} packed wrong numel for rank {rank}: {write_offsets[rank]} != {piece_len}"
                )
        return packed

    def _free_partition_grad(self, partition: _Stage2GradPartition) -> None:
        self.meta.params[partition.param_idx].grad = None

    def _flush_partition(self, partition_idx: int) -> None:
        partition = self._grad_partitions[partition_idx]
        if self.meta.params[partition.param_idx].grad is None:
            return

        self._record_memory_event(f"measured_step_stage2_bucket{partition.index}_ready")
        packed = self._pack_partition_inputs(partition)
        self._free_partition_grad(partition)
        self._live_comm_buffer_num_bytes = tensors_num_bytes([packed])
        self._record_memory_event(f"measured_step_stage2_bucket{partition.index}_pre_reduce_scatter")

        t_comm0 = time.perf_counter()
        reduced_shard = self.collectives.reduce_scatter(packed)
        reduced_shard = reduced_shard / self.world_size
        self._backward_comm_ms += (time.perf_counter() - t_comm0) * 1000.0
        self._live_comm_buffer_num_bytes = tensors_num_bytes([packed, reduced_shard])

        if partition.local_piece_numel > 0:
            if reduced_shard.numel() < partition.local_piece_numel:
                raise ValueError(
                    f"partition {partition.index} reduce_scatter returned too few elements: "
                    f"{reduced_shard.numel()} < {partition.local_piece_numel}"
                )
            start = partition.local_shard_offset
            end = start + partition.local_piece_numel
            grad_shard = self._ensure_grad_shard()
            grad_shard[start:end].add_(reduced_shard[: partition.local_piece_numel])

        self._record_memory_event(f"measured_step_stage2_bucket{partition.index}_post_reduce_scatter")
        del packed
        del reduced_shard
        self._live_comm_buffer_num_bytes = 0
        self._record_memory_event(f"measured_step_stage2_bucket{partition.index}_post_free")

    def _flush_pending_buckets(self, force: bool = False) -> None:
        del force
        for partition_idx, partition in enumerate(self._grad_partitions):
            if self.meta.params[partition.param_idx].grad is None:
                continue
            self._flush_partition(partition_idx)

    def _debug_memory_components_mb(self) -> Dict[str, float]:
        live_full_grads_num_bytes = grads_num_bytes(self.meta.params)
        live_grad_shard_num_bytes = tensors_num_bytes([self.grad_shard])
        comm_temp_mb = bytes_to_mb(self._live_comm_buffer_num_bytes)
        return {
            "logical_params_mb": bytes_to_mb(params_num_bytes(self.meta.params)),
            "logical_grads_mb": bytes_to_mb(self._logical_grad_shard_num_bytes),
            "logical_optimizer_mb": bytes_to_mb(tensors_num_bytes([self.exp_avg, self.exp_avg_sq])),
            "live_full_grads_mb": bytes_to_mb(live_full_grads_num_bytes),
            "live_grad_shard_mb": bytes_to_mb(live_grad_shard_num_bytes),
            "comm_temp_mb": comm_temp_mb,
        }

    def debug_memory_components_mb(self) -> Dict[str, float]:
        return self._debug_memory_components_mb()

    def step_with_stats(self, max_grad_norm: float = 0.0) -> Dict[str, float]:
        t0 = time.perf_counter()
        self._flush_pending_buckets(force=True)
        flat_params = flatten_params_fp32(self.meta)
        if any(param.grad is not None for param in self.meta.params):
            raise RuntimeError("Stage 2 step should not retain full param.grad tensors after backward partitioning")
        grad_shard = self._ensure_grad_shard()
        grad_norm = self._clip_local_grads_inplace(grad_shard, max_grad_norm=max_grad_norm)

        local_params = flat_params[self.shard.shard_start : self.shard.shard_end].clone()
        t_opt0 = time.perf_counter()
        updated_local = self._adamw_update(local_params=local_params, local_grads=grad_shard)
        optim_ms = (time.perf_counter() - t_opt0) * 1000.0

        self._record_memory_event("measured_step_stage2_pre_allgather")
        t_comm1 = time.perf_counter()
        full_updated = self.collectives.allgather(updated_local)
        comm_ms = self._backward_comm_ms + ((time.perf_counter() - t_comm1) * 1000.0)
        assign_flat_params(self.meta, full_updated[: self.meta.total_numel])
        self._record_memory_event("measured_step_stage2_post_allgather")
        self.grad_shard = None
        self._record_memory_event("measured_step_stage2_post_step_free_grad_shard")

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
        grads_mb = bytes_to_mb(self._logical_grad_shard_num_bytes)
        optimizer_mb = bytes_to_mb(tensors_num_bytes([self.exp_avg, self.exp_avg_sq]))
        return {
            "params_mb": params_mb,
            "grads_mb": grads_mb,
            "optimizer_mb": optimizer_mb,
            "total_mb": params_mb + grads_mb + optimizer_mb,
        }

    def live_model_state_breakdown_mb(self) -> Dict[str, float]:
        params_mb = bytes_to_mb(params_num_bytes(self.meta.params))
        grads_mb = bytes_to_mb(grads_num_bytes(self.meta.params) + tensors_num_bytes([self.grad_shard]))
        optimizer_mb = bytes_to_mb(tensors_num_bytes([self.exp_avg, self.exp_avg_sq]))
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
        self.exp_avg = state["exp_avg"].to(device=self.meta.device, dtype=torch.float32)
        self.exp_avg_sq = state["exp_avg_sq"].to(device=self.meta.device, dtype=torch.float32)
        grad_shard = state.get("grad_shard")
        if grad_shard is None:
            self.grad_shard = None
        else:
            self.grad_shard = grad_shard.to(device=self.meta.device, dtype=torch.float32)
