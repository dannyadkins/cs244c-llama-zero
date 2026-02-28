from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist


def _require_initialized() -> Tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before calling ring collectives")
    return dist.get_rank(), dist.get_world_size()


def _validate_op(op: str) -> None:
    if op != "sum":
        raise ValueError(f"Only op='sum' is supported, got '{op}'")


def _flatten_and_pad(tensor: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, int, int]:
    flat = tensor.contiguous().view(-1)
    original_numel = flat.numel()
    chunk_size = max(1, math.ceil(original_numel / world_size))
    padded_numel = chunk_size * world_size

    if padded_numel == original_numel:
        return flat.clone(), original_numel, chunk_size

    padded = torch.zeros(padded_numel, dtype=flat.dtype, device=flat.device)
    padded[:original_numel] = flat
    return padded, original_numel, chunk_size


def _chunks(flat: torch.Tensor, chunk_size: int, world_size: int) -> List[torch.Tensor]:
    return [flat[i * chunk_size : (i + 1) * chunk_size] for i in range(world_size)]


def _exchange_with_send_recv(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_rank: int,
    recv_rank: int,
    tag: int,
) -> None:
    """Ring exchange built from explicit point-to-point ops."""

    recv_req = dist.irecv(recv_buf, src=recv_rank, tag=tag)
    send_req = dist.isend(send_buf, dst=send_rank, tag=tag)
    send_req.wait()
    recv_req.wait()


def _ring_reduce_scatter_inplace(
    chunk_views: Sequence[torch.Tensor],
    rank: int,
    world_size: int,
    tag_base: int,
) -> None:
    right = (rank + 1) % world_size
    left = (rank - 1 + world_size) % world_size

    # Shifted indexing makes rank r finish with reduced chunk r.
    for step in range(world_size - 1):
        send_idx = (rank - step - 1) % world_size
        recv_idx = (rank - step - 2) % world_size

        send_buf = chunk_views[send_idx].clone()
        recv_buf = torch.empty_like(chunk_views[recv_idx])
        _exchange_with_send_recv(
            send_buf=send_buf,
            recv_buf=recv_buf,
            send_rank=right,
            recv_rank=left,
            tag=tag_base + step,
        )
        chunk_views[recv_idx].add_(recv_buf)


def _ring_allgather_inplace(
    chunk_views: Sequence[torch.Tensor],
    rank: int,
    world_size: int,
    tag_base: int,
) -> None:
    right = (rank + 1) % world_size
    left = (rank - 1 + world_size) % world_size

    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        send_buf = chunk_views[send_idx].clone()
        recv_buf = torch.empty_like(chunk_views[recv_idx])
        _exchange_with_send_recv(
            send_buf=send_buf,
            recv_buf=recv_buf,
            send_rank=right,
            recv_rank=left,
            tag=tag_base + step,
        )
        chunk_views[recv_idx].copy_(recv_buf)


def _equal_size_allgather(local_flat: torch.Tensor, rank: int, world_size: int, tag_base: int) -> torch.Tensor:
    local_numel = local_flat.numel()
    gathered = torch.empty(local_numel * world_size, dtype=local_flat.dtype, device=local_flat.device)
    gathered_chunks = _chunks(gathered, local_numel, world_size)
    gathered_chunks[rank].copy_(local_flat)
    _ring_allgather_inplace(gathered_chunks, rank=rank, world_size=world_size, tag_base=tag_base)
    return gathered


def ring_allgather(local_shard: torch.Tensor, tag_base: int = 30_000) -> torch.Tensor:
    """All-gather local shards into a single flattened tensor on every rank.

    Supports variable-size local shards by padding to the max shard length internally,
    then trimming each received shard with the true sizes.
    """

    rank, world_size = _require_initialized()
    local_flat = local_shard.contiguous().view(-1)

    if world_size == 1:
        return local_flat.clone()

    # Gather per-rank lengths for trimming. The main payload still uses ring send/recv.
    size_tensor = torch.tensor([local_flat.numel()], device=local_flat.device, dtype=torch.long)
    size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor)
    sizes = [int(x.item()) for x in size_list]

    max_size = max(sizes)
    if max_size == 0:
        return torch.empty(0, dtype=local_flat.dtype, device=local_flat.device)

    if local_flat.numel() < max_size:
        padded_local = torch.zeros(max_size, dtype=local_flat.dtype, device=local_flat.device)
        padded_local[: local_flat.numel()] = local_flat
    else:
        padded_local = local_flat

    gathered_padded = _equal_size_allgather(padded_local, rank=rank, world_size=world_size, tag_base=tag_base)

    if len(set(sizes)) == 1:
        # Equal-size case: return dense concatenation directly.
        return gathered_padded

    # Variable-size case: trim each rank chunk to its true length.
    trimmed_parts = []
    for r in range(world_size):
        start = r * max_size
        end = start + sizes[r]
        trimmed_parts.append(gathered_padded[start:end])
    return torch.cat(trimmed_parts, dim=0)


def ring_reduce_scatter(
    tensor: torch.Tensor,
    op: str = "sum",
    tag_base: int = 10_000,
) -> torch.Tensor:
    """Reduce-scatter on a full input tensor.

    Returns this rank's reduced shard as a flattened tensor.
    Uneven input sizes are handled by zero-padding internally and trimming the local shard.
    """

    _validate_op(op)
    rank, world_size = _require_initialized()

    if world_size == 1:
        return tensor.contiguous().view(-1).clone()

    flat, original_numel, chunk_size = _flatten_and_pad(tensor, world_size=world_size)
    chunk_views = _chunks(flat, chunk_size=chunk_size, world_size=world_size)

    _ring_reduce_scatter_inplace(
        chunk_views,
        rank=rank,
        world_size=world_size,
        tag_base=tag_base,
    )

    start = rank * chunk_size
    end = min(start + chunk_size, original_numel)
    local_len = max(0, end - start)
    return chunk_views[rank][:local_len].clone()


def ring_allreduce(
    tensor: torch.Tensor,
    op: str = "sum",
    average: bool = False,
    tag_base: int = 20_000,
) -> torch.Tensor:
    """Ring allreduce implemented from send/recv primitives.

    Returns a tensor with the same shape and dtype as the input.
    """

    _validate_op(op)
    rank, world_size = _require_initialized()

    if world_size == 1:
        out = tensor.clone()
        if average:
            out /= 1.0
        return out

    flat, original_numel, chunk_size = _flatten_and_pad(tensor, world_size=world_size)
    chunk_views = _chunks(flat, chunk_size=chunk_size, world_size=world_size)

    _ring_reduce_scatter_inplace(
        chunk_views,
        rank=rank,
        world_size=world_size,
        tag_base=tag_base,
    )
    _ring_allgather_inplace(
        chunk_views,
        rank=rank,
        world_size=world_size,
        tag_base=tag_base + 1_000,
    )

    if average:
        flat /= world_size

    return flat[:original_numel].view_as(tensor).clone()
