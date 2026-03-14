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
    flatten_param_shard_fp32,
    get_rank_world_size,
    grads_num_bytes,
    params_num_bytes,
    tensors_num_bytes,
)


@dataclass
class _Stage2GradBucket:
    index: int
    param_indices: List[int]
    start: int
    end: int
    rank_piece_numels: List[int]
    packed_chunk_numel: int
    local_piece_numel: int
    local_bucket_offset: int
    local_shard_offset: int

    @property
    def numel(self) -> int:
        return self.end - self.start

    @property
    def packed_numel(self) -> int:
        return len(self.rank_piece_numels) * self.packed_chunk_numel

    @property
    def prefer_allreduce(self) -> bool:
        # Ring all-reduce is cheaper than the current padded reduce-scatter path
        # when the packed input grows beyond 2x the logical bucket size.
        return self.packed_numel > (2 * self.numel)


@dataclass
class ZeROStage2Optimizer:
    """Stage 2: bucketed gradient reduce-scatter, shard optimizer state, allgather updated params."""

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
        self.shard: ShardSpec = compute_shard_spec(self.meta.total_numel, rank=rank, world_size=world_size)

        self.step_count = 0
        self.exp_avg = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.exp_avg_sq = torch.zeros(self.shard.shard_numel, dtype=torch.float32, device=self.meta.device)
        self.grad_bucket_numel = max(1, int(self.grad_bucket_numel))
        self.grad_shard: Optional[torch.Tensor] = None
        self._backward_comm_ms = 0.0
        self._backward_reduce_scatter_ms = 0.0
        self._post_step_allgather_ms = 0.0
        self._backward_reduce_scatter_calls = 0
        self._post_step_allgather_calls = 0
        self._backward_reduce_scatter_bytes = 0
        self._post_step_allgather_bytes = 0
        self._live_comm_buffer_num_bytes = 0
        self._logical_grad_shard_num_bytes = self.shard.shard_numel * torch.tensor([], dtype=torch.float32).element_size()
        self._grad_buckets = self._build_grad_buckets()
        self._param_to_bucket_idx = self._build_param_to_bucket_idx()
        self._bucket_ready_counts = [0 for _ in self._grad_buckets]
        self._param_bucket_ready = [False for _ in self.meta.params]
        self._grad_hook_handles = self._register_grad_hooks()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
        self._flush_pending_buckets(force=True)

    def prepare_forward(self) -> Dict[str, float]:
        return {"prepare_comm_ms": 0.0}

    def zero_grad(self) -> None:
        self._backward_comm_ms = 0.0
        self._backward_reduce_scatter_ms = 0.0
        self._post_step_allgather_ms = 0.0
        self._backward_reduce_scatter_calls = 0
        self._post_step_allgather_calls = 0
        self._backward_reduce_scatter_bytes = 0
        self._post_step_allgather_bytes = 0
        self._live_comm_buffer_num_bytes = 0
        self.grad_shard = None
        self._bucket_ready_counts = [0 for _ in self._grad_buckets]
        self._param_bucket_ready = [False for _ in self.meta.params]
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

    def _build_grad_buckets(self) -> List[_Stage2GradBucket]:
        buckets: List[_Stage2GradBucket] = []
        bucket_param_indices: List[int] = []
        bucket_start = 0
        bucket_numel = 0

        for param_idx, (param, start) in enumerate(zip(self.meta.params, self.meta.offsets)):
            if not bucket_param_indices:
                bucket_param_indices = [param_idx]
                bucket_start = start
                bucket_numel = param.numel()
                continue

            if bucket_numel + param.numel() > self.grad_bucket_numel:
                bucket_end = self.meta.offsets[bucket_param_indices[-1]] + self.meta.params[bucket_param_indices[-1]].numel()
                buckets.append(
                    self._make_bucket(
                        index=len(buckets),
                        param_indices=bucket_param_indices,
                        start=bucket_start,
                        end=bucket_end,
                    )
                )
                bucket_param_indices = [param_idx]
                bucket_start = start
                bucket_numel = param.numel()
                continue

            bucket_param_indices.append(param_idx)
            bucket_numel += param.numel()

        if bucket_param_indices:
            bucket_end = self.meta.offsets[bucket_param_indices[-1]] + self.meta.params[bucket_param_indices[-1]].numel()
            buckets.append(
                self._make_bucket(
                    index=len(buckets),
                    param_indices=bucket_param_indices,
                    start=bucket_start,
                    end=bucket_end,
                )
            )
        return buckets

    def _make_bucket(self, index: int, param_indices: List[int], start: int, end: int) -> _Stage2GradBucket:
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
        return _Stage2GradBucket(
            index=index,
            param_indices=list(param_indices),
            start=start,
            end=end,
            rank_piece_numels=rank_piece_numels,
            packed_chunk_numel=max(1, max(rank_piece_numels) if rank_piece_numels else 0),
            local_piece_numel=local_piece_numel,
            local_bucket_offset=max(0, local_overlap_start - start),
            local_shard_offset=max(0, local_overlap_start - self.shard.shard_start),
        )

    def _build_param_to_bucket_idx(self) -> List[int]:
        mapping = [0 for _ in self.meta.params]
        for bucket in self._grad_buckets:
            for param_idx in bucket.param_indices:
                mapping[param_idx] = bucket.index
        return mapping

    def _register_grad_hooks(self):
        handles = []
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
                bucket_idx = self._param_to_bucket_idx[_param_idx]
                if not self._param_bucket_ready[_param_idx]:
                    self._param_bucket_ready[_param_idx] = True
                    self._bucket_ready_counts[bucket_idx] += 1

                bucket = self._grad_buckets[bucket_idx]
                if self._bucket_ready_counts[bucket_idx] >= len(bucket.param_indices):
                    self._flush_bucket(bucket_idx)

            handles.append(param.register_post_accumulate_grad_hook(_hook))
        return handles

    def _pack_bucket_inputs(self, bucket: _Stage2GradBucket) -> torch.Tensor:
        packed = torch.zeros(
            self.world_size * bucket.packed_chunk_numel,
            dtype=torch.float32,
            device=self.meta.device,
        )
        chunk_views = [
            packed[rank * bucket.packed_chunk_numel : (rank + 1) * bucket.packed_chunk_numel]
            for rank in range(self.world_size)
        ]

        for param_idx in bucket.param_indices:
            param = self.meta.params[param_idx]
            grad = param.grad
            if grad is None:
                continue

            flat_grad = grad.detach().view(-1)
            if flat_grad.dtype != torch.float32:
                flat_grad = flat_grad.to(torch.float32)

            param_start = self.meta.offsets[param_idx]
            param_end = param_start + param.numel()
            first_rank = param_start // self.shard.chunk_size
            last_rank = (param_end - 1) // self.shard.chunk_size
            for rank in range(first_rank, last_rank + 1):
                shard_start = rank * self.shard.chunk_size
                shard_end = min(shard_start + self.shard.chunk_size, self.meta.total_numel)
                overlap_start = max(param_start, shard_start)
                overlap_end = min(param_end, shard_end)
                if overlap_end <= overlap_start:
                    continue
                bucket_rank_start = max(bucket.start, shard_start)
                piece_len = overlap_end - overlap_start
                dst_start = overlap_start - bucket_rank_start
                dst_end = dst_start + piece_len
                src_start = overlap_start - param_start
                src_end = src_start + piece_len
                chunk_views[rank][dst_start:dst_end].copy_(flat_grad[src_start:src_end])
        return packed

    def _pack_bucket_dense(self, bucket: _Stage2GradBucket) -> torch.Tensor:
        packed = torch.zeros(bucket.numel, dtype=torch.float32, device=self.meta.device)
        for param_idx in bucket.param_indices:
            param = self.meta.params[param_idx]
            grad = param.grad
            if grad is None:
                continue

            flat_grad = grad.detach().view(-1)
            if flat_grad.dtype != torch.float32:
                flat_grad = flat_grad.to(torch.float32)

            param_start = self.meta.offsets[param_idx]
            dst_start = param_start - bucket.start
            dst_end = dst_start + param.numel()
            packed[dst_start:dst_end].copy_(flat_grad)
        return packed

    def _reset_bucket_ready(self, bucket_idx: int) -> None:
        bucket = self._grad_buckets[bucket_idx]
        self._bucket_ready_counts[bucket_idx] = 0
        for param_idx in bucket.param_indices:
            self._param_bucket_ready[param_idx] = False

    def _free_bucket_grads(self, bucket: _Stage2GradBucket) -> None:
        for param_idx in bucket.param_indices:
            self.meta.params[param_idx].grad = None

    def _flush_bucket(self, bucket_idx: int) -> None:
        bucket = self._grad_buckets[bucket_idx]
        if not any(self.meta.params[param_idx].grad is not None for param_idx in bucket.param_indices):
            self._reset_bucket_ready(bucket_idx)
            return

        self._record_memory_event(f"measured_step_stage2_bucket{bucket.index}_ready")
        if bucket.prefer_allreduce:
            packed = self._pack_bucket_dense(bucket)
        else:
            packed = self._pack_bucket_inputs(bucket)
        self._free_bucket_grads(bucket)
        self._live_comm_buffer_num_bytes = tensors_num_bytes([packed])
        self._record_memory_event(f"measured_step_stage2_bucket{bucket.index}_pre_reduce_scatter")

        self._backward_reduce_scatter_calls += 1
        self._backward_reduce_scatter_bytes += self._live_comm_buffer_num_bytes
        t_comm0 = time.perf_counter()
        if bucket.prefer_allreduce:
            reduced_shard = self.collectives.allreduce_inplace(packed, average=True)
        else:
            reduced_shard = self.collectives.reduce_scatter(packed)
            reduced_shard = reduced_shard / self.world_size
        rs_ms = (time.perf_counter() - t_comm0) * 1000.0
        self._backward_reduce_scatter_ms += rs_ms
        self._backward_comm_ms += rs_ms
        if reduced_shard.data_ptr() == packed.data_ptr():
            self._live_comm_buffer_num_bytes = tensors_num_bytes([packed])
        else:
            self._live_comm_buffer_num_bytes = tensors_num_bytes([packed, reduced_shard])

        if bucket.local_piece_numel > 0:
            if bucket.prefer_allreduce:
                if reduced_shard.numel() < (bucket.local_bucket_offset + bucket.local_piece_numel):
                    raise ValueError(
                        f"bucket {bucket.index} allreduce returned too few elements: "
                        f"{reduced_shard.numel()} < {bucket.local_bucket_offset + bucket.local_piece_numel}"
                    )
                src_start = bucket.local_bucket_offset
                src_end = src_start + bucket.local_piece_numel
            else:
                if reduced_shard.numel() < bucket.local_piece_numel:
                    raise ValueError(
                        f"bucket {bucket.index} reduce_scatter returned too few elements: "
                        f"{reduced_shard.numel()} < {bucket.local_piece_numel}"
                    )
                src_start = 0
                src_end = bucket.local_piece_numel
            start = bucket.local_shard_offset
            end = start + bucket.local_piece_numel
            grad_shard = self._ensure_grad_shard()
            grad_shard[start:end].add_(reduced_shard[src_start:src_end])

        self._record_memory_event(f"measured_step_stage2_bucket{bucket.index}_post_reduce_scatter")
        if reduced_shard.data_ptr() != packed.data_ptr():
            del reduced_shard
        del packed
        self._live_comm_buffer_num_bytes = 0
        self._record_memory_event(f"measured_step_stage2_bucket{bucket.index}_post_free")
        self._reset_bucket_ready(bucket_idx)

    def _flush_pending_buckets(self, force: bool = False) -> None:
        del force
        for bucket_idx, bucket in enumerate(self._grad_buckets):
            if not any(self.meta.params[param_idx].grad is not None for param_idx in bucket.param_indices):
                continue
            self._flush_bucket(bucket_idx)

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
        if any(param.grad is not None for param in self.meta.params):
            raise RuntimeError("Stage 2 step should not retain full param.grad tensors after backward partitioning")
        grad_shard = self._ensure_grad_shard()
        grad_norm = self._clip_local_grads_inplace(grad_shard, max_grad_norm=max_grad_norm)

        local_params = flatten_param_shard_fp32(self.meta, self.shard.shard_start, self.shard.shard_end)
        t_opt0 = time.perf_counter()
        updated_local = self._adamw_update(local_params=local_params, local_grads=grad_shard)
        optim_ms = (time.perf_counter() - t_opt0) * 1000.0
        del local_params

        self._record_memory_event("measured_step_stage2_pre_allgather")
        t_comm1 = time.perf_counter()
        full_updated = self.collectives.allgather(updated_local)
        allgather_ms = (time.perf_counter() - t_comm1) * 1000.0
        del updated_local
        allgather_bytes = int(self.meta.total_numel * torch.tensor([], dtype=torch.float32).element_size())
        comm_ms = self._backward_comm_ms + allgather_ms
        self._post_step_allgather_ms += allgather_ms
        self._post_step_allgather_calls += 1
        self._post_step_allgather_bytes += allgather_bytes
        assign_flat_params(self.meta, full_updated[: self.meta.total_numel])
        del full_updated
        self._record_memory_event("measured_step_stage2_post_allgather")
        self.grad_shard = None
        self._record_memory_event("measured_step_stage2_post_step_free_grad_shard")

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "grad_norm": grad_norm,
            "comm_ms": comm_ms,
            "communication_backward_reduce_scatter_ms": self._backward_reduce_scatter_ms,
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
