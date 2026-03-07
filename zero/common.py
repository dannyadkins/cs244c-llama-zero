from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

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


def flatten_params_fp32(meta: FlatParamMetadata) -> torch.Tensor:
    return torch.cat([p.data.detach().view(-1).to(torch.float32) for p in meta.params], dim=0)


def flatten_grads_fp32(meta: FlatParamMetadata) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for p in meta.params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        else:
            parts.append(p.grad.detach().view(-1).to(torch.float32))
    return torch.cat(parts, dim=0)


def assign_flat_params(meta: FlatParamMetadata, flat_params: torch.Tensor) -> None:
    if flat_params.numel() < meta.total_numel:
        raise ValueError(f"flat_params too small: {flat_params.numel()} < {meta.total_numel}")

    for p, start, shape in zip(meta.params, meta.offsets, meta.shapes):
        numel = math.prod(shape)
        chunk = flat_params[start : start + numel].view(shape)
        p.data = chunk.to(dtype=p.dtype, device=p.device).clone()


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
