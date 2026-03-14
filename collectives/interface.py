from __future__ import annotations

from dataclasses import dataclass
import os
import time

import torch
import torch.distributed as dist

from ._ring import ring_allgather, ring_allreduce, ring_reduce_scatter


def _validate_send_recv_tensor_backend(tensor: torch.Tensor) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if tensor.device.type != "cuda":
        return

    backend = dist.get_backend()
    if backend == "nccl":
        raise RuntimeError(
            "SendRecvCollectives uses point-to-point isend/irecv ring ops. "
            "This code path hangs on CUDA tensors with an NCCL process group in this repo. "
            "Use --collective-impl torch for multi-GPU CUDA runs, or run the ring collectives on CPU/Gloo."
        )


@dataclass
class CollectiveOps:
    """Black-box communication interface consumed by ZeRO wrappers."""

    def allreduce(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def allreduce_inplace(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        reduced = self.allreduce(tensor, average=average)
        if reduced.data_ptr() != tensor.data_ptr():
            tensor.copy_(reduced)
        return tensor

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

    def allreduce_inplace(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        if average:
            tensor /= 1.0
        return tensor

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(-1).clone()

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        return local_shard.contiguous().view(-1).clone()


@dataclass
class SendRecvCollectives(CollectiveOps):
    """Custom collectives implemented from send/recv primitives."""

    def allreduce(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        _validate_send_recv_tensor_backend(tensor)
        return ring_allreduce(tensor=tensor, average=average)

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        _validate_send_recv_tensor_backend(tensor)
        return ring_reduce_scatter(tensor=tensor)

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        _validate_send_recv_tensor_backend(local_shard)
        return ring_allgather(local_shard=local_shard)


@dataclass
class TorchCollectives(CollectiveOps):
    """Reference implementation using built-in distributed collectives."""

    @staticmethod
    def _supports_fast_collectives(tensor: torch.Tensor) -> bool:
        return tensor.device.type == "cuda" and hasattr(dist, "all_gather_into_tensor") and hasattr(dist, "reduce_scatter_tensor")

    @staticmethod
    def _maybe_synchronize(tensor: torch.Tensor) -> None:
        if tensor.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(tensor.device)

    def allreduce(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        out = tensor.clone()
        dist.all_reduce(out)
        _maybe_simulate_collective_delay(
            num_bytes=out.numel() * out.element_size(),
            world_size=dist.get_world_size(),
        )
        self._maybe_synchronize(out)
        if average:
            out /= dist.get_world_size()
        return out

    def allreduce_inplace(self, tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
        dist.all_reduce(tensor)
        _maybe_simulate_collective_delay(
            num_bytes=tensor.numel() * tensor.element_size(),
            world_size=dist.get_world_size(),
        )
        self._maybe_synchronize(tensor)
        if average:
            tensor /= dist.get_world_size()
        return tensor

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        flat = tensor.contiguous().view(-1)
        rank = dist.get_rank()

        if flat.numel() == 0:
            return flat.clone()

        chunk = (flat.numel() + world_size - 1) // world_size
        padded_numel = chunk * world_size

        if self._supports_fast_collectives(flat):
            if padded_numel != flat.numel():
                padded = torch.zeros(padded_numel, dtype=flat.dtype, device=flat.device)
                padded[: flat.numel()] = flat
            else:
                padded = flat

            out = torch.empty(chunk, dtype=flat.dtype, device=flat.device)
            try:
                dist.reduce_scatter_tensor(out, padded)
                _maybe_simulate_collective_delay(
                    num_bytes=out.numel() * out.element_size(),
                    world_size=world_size,
                )
                self._maybe_synchronize(out)
                start = rank * chunk
                end = min(start + chunk, flat.numel())
                return out[: max(0, end - start)].contiguous()
            except RuntimeError:
                pass

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
                self._maybe_synchronize(out)
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
        self._maybe_synchronize(reduced)

        start = rank * chunk
        end = min(start + chunk, flat.numel())
        return reduced[start:end].contiguous()

    def allgather(self, local_shard: torch.Tensor) -> torch.Tensor:
        local_flat = local_shard.contiguous().view(-1)
        world_size = dist.get_world_size()

        size_tensor = torch.tensor([local_flat.numel()], device=local_flat.device, dtype=torch.long)
        if self._supports_fast_collectives(local_flat):
            sizes_tensor = torch.empty(world_size, dtype=torch.long, device=local_flat.device)
            dist.all_gather_into_tensor(sizes_tensor, size_tensor)
            sizes = [int(x.item()) for x in sizes_tensor]
        else:
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

        if self._supports_fast_collectives(local_flat):
            gathered_tensor = torch.empty(world_size * max_size, dtype=padded.dtype, device=padded.device)
            dist.all_gather_into_tensor(gathered_tensor, padded)
            gathered = [gathered_tensor[i * max_size : (i + 1) * max_size] for i in range(world_size)]
        else:
            gathered = [torch.empty_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
        _maybe_simulate_collective_delay(
            num_bytes=padded.numel() * padded.element_size(),
            world_size=world_size,
        )
        self._maybe_synchronize(padded)

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
