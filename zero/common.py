from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class FlatParamMetadata:
    params: List[nn.Parameter]
    offsets: List[int]
    shapes: List[torch.Size]
    total_numel: int
    device: torch.device


@dataclass
class ShardSpec:
    rank: int
    world_size: int
    shard_start: int
    shard_end: int
    shard_numel: int
    chunk_size: int


def tensor_num_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def params_num_bytes(params: Iterable[nn.Parameter]) -> int:
    return sum(tensor_num_bytes(param.data) for param in unique_trainable_params(list(params)))


def grads_num_bytes(params: Iterable[nn.Parameter]) -> int:
    return sum(tensor_num_bytes(param.grad) for param in unique_trainable_params(list(params)))


def tensors_num_bytes(tensors: Iterable[torch.Tensor | None]) -> int:
    return sum(tensor_num_bytes(tensor) for tensor in tensors)


def get_rank_world_size() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def unique_trainable_params(params: List[nn.Parameter]) -> List[nn.Parameter]:
    unique: List[nn.Parameter] = []
    seen = set()
    for p in params:
        if not p.requires_grad:
            continue
        ident = id(p)
        if ident in seen:
            continue
        seen.add(ident)
        unique.append(p)
    return unique


def build_flat_param_metadata_from_params(params: List[nn.Parameter]) -> FlatParamMetadata:
    params = unique_trainable_params(params)
    if not params:
        raise ValueError("Model has no trainable parameters")

    offsets: List[int] = []
    shapes: List[torch.Size] = []
    cursor = 0
    for p in params:
        offsets.append(cursor)
        shapes.append(torch.Size(p.shape))
        cursor += p.numel()

    return FlatParamMetadata(
        params=params,
        offsets=offsets,
        shapes=shapes,
        total_numel=cursor,
        device=params[0].device,
    )


def build_flat_param_metadata(model: nn.Module) -> FlatParamMetadata:
    return build_flat_param_metadata_from_params(list(model.parameters()))


def compute_shard_spec(total_numel: int, rank: int, world_size: int) -> ShardSpec:
    chunk_size = max(1, math.ceil(total_numel / world_size))
    shard_start = rank * chunk_size
    shard_end = min(shard_start + chunk_size, total_numel)
    shard_numel = max(0, shard_end - shard_start)
    return ShardSpec(
        rank=rank,
        world_size=world_size,
        shard_start=shard_start,
        shard_end=shard_end,
        shard_numel=shard_numel,
        chunk_size=chunk_size,
    )


def _flatten_tensor_fp32(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().view(-1)
    if flat.dtype == torch.float32:
        return flat
    return flat.to(torch.float32)


def flatten_params_fp32(meta: FlatParamMetadata) -> torch.Tensor:
    return torch.cat([_flatten_tensor_fp32(p.data) for p in meta.params], dim=0)


def flatten_param_shard_fp32(meta: FlatParamMetadata, shard_start: int, shard_end: int) -> torch.Tensor:
    shard_numel = max(0, shard_end - shard_start)
    if shard_numel == 0:
        return torch.empty(0, dtype=torch.float32, device=meta.device)

    shard = torch.empty(shard_numel, dtype=torch.float32, device=meta.device)
    for p, start in zip(meta.params, meta.offsets):
        param_end = start + p.numel()
        overlap_start = max(start, shard_start)
        overlap_end = min(param_end, shard_end)
        if overlap_end <= overlap_start:
            continue

        flat = p.data.detach().view(-1)
        src = flat[overlap_start - start : overlap_end - start]
        if src.dtype != torch.float32:
            src = src.to(torch.float32)
        shard[overlap_start - shard_start : overlap_end - shard_start].copy_(src)
    return shard


def flatten_grads_fp32(meta: FlatParamMetadata) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for p in meta.params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        else:
            parts.append(_flatten_tensor_fp32(p.grad))
    return torch.cat(parts, dim=0)


def assign_flat_params(meta: FlatParamMetadata, flat_params: torch.Tensor) -> None:
    if flat_params.numel() < meta.total_numel:
        raise ValueError(f"flat_params too small: {flat_params.numel()} < {meta.total_numel}")

    for p, start, shape in zip(meta.params, meta.offsets, meta.shapes):
        numel = math.prod(shape)
        chunk = flat_params[start : start + numel].view(shape)
        if chunk.dtype != p.dtype or chunk.device != p.device:
            chunk = chunk.to(dtype=p.dtype, device=p.device)

        if p.data.shape == shape and p.data.dtype == chunk.dtype and p.data.device == chunk.device:
            p.data.copy_(chunk)
        else:
            p.data = chunk.clone()


def assign_flat_grads(meta: FlatParamMetadata, flat_grads: torch.Tensor) -> None:
    if flat_grads.numel() < meta.total_numel:
        raise ValueError(f"flat_grads too small: {flat_grads.numel()} < {meta.total_numel}")

    for p, start, shape in zip(meta.params, meta.offsets, meta.shapes):
        numel = math.prod(shape)
        chunk = flat_grads[start : start + numel].view(shape).to(dtype=p.dtype, device=p.device)
        if p.grad is None or p.grad.shape != shape:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
