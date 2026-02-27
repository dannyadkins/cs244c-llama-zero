from __future__ import annotations

from dataclasses import dataclass
import os
import time

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
        _maybe_simulate_collective_delay(
            num_bytes=out.numel() * out.element_size(),
            world_size=dist.get_world_size(),
        )
        if average:
            out /= dist.get_world_size()
        return out

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        flat = tensor.contiguous().view(-1)
        rank = dist.get_rank()

        if flat.numel() == 0:
            return flat.clone()

        divisible = flat.numel() % world_size == 0
        if divisible:
            chunk = flat.numel() // world_size
            inputs = [flat[i * chunk : (i + 1) * chunk].clone() for i in range(world_size)]
            out = torch.empty(chunk, dtype=flat.dtype, device=flat.device)

            try:
                dist.reduce_scatter(out, inputs)
                _maybe_simulate_collective_delay(
                    num_bytes=out.numel() * out.element_size(),
                    world_size=world_size,
                )
                return out
            except RuntimeError as exc:
                # CPU/Gloo does not support reduce_scatter in many environments.
                if "does not support reduce_scatter" not in str(exc):
                    raise

        # Uneven-size and unsupported-backend fallback: allreduce full tensor,
        # then return this rank's ceil-sized shard.
        reduced = flat.clone()
        dist.all_reduce(reduced)
        _maybe_simulate_collective_delay(
            num_bytes=reduced.numel() * reduced.element_size(),
            world_size=world_size,
        )

        chunk = (flat.numel() + world_size - 1) // world_size
        start = rank * chunk
        end = min(start + chunk, flat.numel())
        return reduced[start:end].contiguous()

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        local_flat = local_shard.contiguous().view(-1)
        world_size = dist.get_world_size()

        size_tensor = torch.tensor([local_flat.numel()], device=local_flat.device, dtype=torch.long)
        size_list = [torch.zeros_like(size_tensor) for _ in range(world_size)]
        dist.all_gather(size_list, size_tensor)
        sizes = [int(x.item()) for x in size_list]

        max_size = max(sizes)
        if max_size == 0:
            return torch.empty(0, dtype=local_flat.dtype, device=local_flat.device)

        if local_flat.numel() < max_size:
            padded = torch.zeros(max_size, dtype=local_flat.dtype, device=local_flat.device)
            padded[: local_flat.numel()] = local_flat
        else:
            padded = local_flat

        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        _maybe_simulate_collective_delay(
            num_bytes=padded.numel() * padded.element_size(),
            world_size=world_size,
        )

        if len(set(sizes)) == 1:
            return torch.cat(gathered, dim=0)

        trimmed = [gathered[i][: sizes[i]] for i in range(world_size)]
        return torch.cat(trimmed, dim=0)


def _maybe_simulate_collective_delay(num_bytes: int, world_size: int) -> None:
    bw_env = os.environ.get("ZERO_SIM_BW_GBPS", "").strip()
    lat_env = os.environ.get("ZERO_SIM_LATENCY_MS", "").strip()
    if not bw_env and not lat_env:
        return

    bandwidth_gbps = float(bw_env) if bw_env else 0.0
    latency_ms = float(lat_env) if lat_env else 0.0

    delay_s = 0.0
    if bandwidth_gbps > 0:
        delay_s += ((world_size - 1) * num_bytes) / (bandwidth_gbps * 1e9)
    if latency_ms > 0:
        delay_s += latency_ms / 1000.0

    if delay_s > 0:
        time.sleep(delay_s)
