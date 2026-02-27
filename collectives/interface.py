from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ._ring import ring_allgather, ring_allreduce, ring_reduce_scatter


@dataclass
class CollectiveOps:
    """Black-box communication interface consumed by ZeRO wrappers."""

    def allreduce(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class LocalCollectives(CollectiveOps):
    """Single-process fallback used when distributed is not initialized."""

    def allreduce(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        out = tensor.clone()
        if average:
            out /= 1.0
        return out

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(-1).clone()

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        return local_shard.contiguous().view(-1).clone()


@dataclass
class SendRecvCollectives(CollectiveOps):
    """Custom collectives implemented from send/recv primitives."""

    def allreduce(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        return ring_allreduce(tensor=tensor, average=average)

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        return ring_reduce_scatter(tensor=tensor)

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        return ring_allgather(local_shard=local_shard)


@dataclass
class TorchCollectives(CollectiveOps):
    """Reference implementation using built-in distributed collectives."""

    def allreduce(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        out = tensor.clone()
        dist.all_reduce(out)
        if average:
            out /= dist.get_world_size()
        return out

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        flat = tensor.contiguous().view(-1)
        if flat.numel() % world_size != 0:
            raise ValueError(
                "TorchCollectives.reduce_scatter requires numel divisible by world_size; "
                "use SendRecvCollectives for uneven sizes"
            )
        chunk = flat.numel() // world_size
        inputs = [flat[i * chunk : (i + 1) * chunk].clone() for i in range(world_size)]
        out = torch.empty(chunk, dtype=flat.dtype, device=flat.device)

        try:
            dist.reduce_scatter(out, inputs)
            return out
        except RuntimeError as exc:
            # CPU/Gloo does not support reduce_scatter in many environments.
            if "does not support reduce_scatter" not in str(exc):
                raise

        reduced = flat.clone()
        dist.all_reduce(reduced)
        rank = dist.get_rank()
        start = rank * chunk
        end = start + chunk
        return reduced[start:end].contiguous()

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        local_flat = local_shard.contiguous().view(-1)

        objects = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(objects, local_flat)
        return torch.cat([x.view(-1) for x in objects], dim=0)
