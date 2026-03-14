from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark distributed all_reduce throughput")
    parser.add_argument("--numel", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tensor = torch.ones(args.numel, dtype=dtype, device=device)

    for _ in range(args.warmup):
        dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed_s = []
    for _ in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_s.append(time.perf_counter() - t0)

    mean_s = sum(elapsed_s) / len(elapsed_s)
    payload_bytes = tensor.numel() * tensor.element_size()
    alg_bytes = 2.0 * ((world_size - 1) / world_size) * payload_bytes
    alg_gbps = (alg_bytes * 8.0) / (mean_s * 1e9)

    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": backend,
                    "world_size": world_size,
                    "numel": args.numel,
                    "dtype": args.dtype,
                    "mean_ms": mean_s * 1000.0,
                    "payload_mb": payload_bytes / 1e6,
                    "effective_allreduce_gbps": alg_gbps,
                }
            )
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
