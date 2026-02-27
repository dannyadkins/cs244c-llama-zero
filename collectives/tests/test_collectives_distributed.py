from __future__ import annotations

import socket
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from collectives import ring_allgather, ring_allreduce, ring_reduce_scatter


pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_close(a: torch.Tensor, b: torch.Tensor, msg: str) -> None:
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5, msg=msg)


def _reference_reduce_scatter_even(x: torch.Tensor, world_size: int) -> torch.Tensor:
    chunk = x.numel() // world_size
    inputs = [x[i * chunk : (i + 1) * chunk].clone() for i in range(world_size)]
    out = torch.empty(chunk, dtype=x.dtype)

    try:
        dist.reduce_scatter(out, inputs)
        return out
    except RuntimeError as exc:
        if "does not support reduce_scatter" not in str(exc):
            raise

    reduced = x.clone()
    dist.all_reduce(reduced)
    rank = dist.get_rank()
    start = rank * chunk
    end = start + chunk
    return reduced[start:end].contiguous()


def _worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        # 1) allreduce: even + uneven sizes
        for numel in [16, 257, 4096]:
            x = torch.arange(numel, dtype=torch.float32) + rank * 0.25
            custom = ring_allreduce(x)

            ref = x.clone()
            dist.all_reduce(ref)
            _assert_close(custom, ref, msg=f"allreduce mismatch numel={numel} rank={rank}")

        # 2) reduce-scatter: divisible sizes match reference exactly
        divisible_sizes = [world_size * 32, world_size * 341]
        for numel in divisible_sizes:
            x = torch.arange(numel, dtype=torch.float32) + rank
            custom = ring_reduce_scatter(x)
            ref = _reference_reduce_scatter_even(x=x, world_size=world_size)

            _assert_close(custom, ref, msg=f"reduce_scatter mismatch numel={numel} rank={rank}")

        # 3) reduce-scatter: uneven size fallback check against all_reduce + manual split
        numel = 65
        x = torch.arange(numel, dtype=torch.float32) + rank
        custom = ring_reduce_scatter(x)

        ref_full = x.clone()
        dist.all_reduce(ref_full)
        chunk = (numel + world_size - 1) // world_size
        start = rank * chunk
        end = min(start + chunk, numel)
        ref = ref_full[start:end]
        _assert_close(custom, ref, msg=f"uneven reduce_scatter mismatch rank={rank}")

        # 4) allgather: fixed-size shards matches dist.all_gather
        for local_size in [8, 127]:
            local = torch.arange(local_size, dtype=torch.float32) + rank * 10
            custom = ring_allgather(local)

            gathered: List[torch.Tensor] = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            ref = torch.cat(gathered, dim=0)
            _assert_close(custom, ref, msg=f"allgather mismatch local_size={local_size} rank={rank}")

        # 5) allgather: variable-size shards
        variable_size = 5 + rank
        local = torch.arange(variable_size, dtype=torch.float32) + rank * 100
        custom = ring_allgather(local)

        objects: List[torch.Tensor | None] = [None for _ in range(world_size)]
        dist.all_gather_object(objects, local)
        ref = torch.cat([obj.view(-1) for obj in objects], dim=0)
        _assert_close(custom, ref, msg=f"variable allgather mismatch rank={rank}")

    finally:
        dist.barrier()
        dist.destroy_process_group()


def _run(world_size: int) -> None:
    port = _find_free_port()
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)


def test_ring_collectives_world2() -> None:
    _run(world_size=2)


def test_ring_collectives_world3() -> None:
    _run(world_size=3)
