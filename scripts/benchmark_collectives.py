from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List
import sys

import torch
import torch.distributed as dist

# Ensure project root is importable when launched as `torchrun scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collectives import SendRecvCollectives, TorchCollectives
from profiler import MemoryTracker, TimerRegistry


@dataclass
class OpResult:
    op: str
    impl: str
    numel: int
    dtype: str
    bytes_sent_per_rank: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark custom send/recv collectives")
    parser.add_argument("--ops", nargs="+", default=["allreduce", "reduce_scatter", "allgather"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1 << 12, 1 << 16, 1 << 20])
    parser.add_argument("--impl", choices=["ring", "torch", "both"], default="both")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def bytes_per_rank(op: str, numel: int, world_size: int, element_size: int) -> int:
    chunk = max(1, (numel + world_size - 1) // world_size)
    if op == "allreduce":
        return int(2 * (world_size - 1) * chunk * element_size)
    if op == "reduce_scatter":
        return int((world_size - 1) * chunk * element_size)
    if op == "allgather":
        return int((world_size - 1) * numel * element_size)
    raise ValueError(f"unknown op: {op}")


def init_distributed(device_choice: str) -> torch.device:
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and device_choice != "cpu" else "gloo"
        dist.init_process_group(backend=backend)

    if device_choice == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    if device_choice == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    # auto
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def run_op(
    op: str,
    impl_name: str,
    numel: int,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
    device: torch.device,
) -> OpResult:
    world_size = dist.get_world_size()

    impl = SendRecvCollectives() if impl_name == "ring" else TorchCollectives()
    timer = TimerRegistry(device=device).timer(name=f"{impl_name}:{op}:{numel}")

    torch.manual_seed(1234 + dist.get_rank())
    if op == "allgather":
        # allgather takes per-rank local shard size.
        x = torch.randn(numel, dtype=dtype, device=device)
    else:
        x = torch.randn(numel, dtype=dtype, device=device)

    def call_once() -> torch.Tensor:
        if op == "allreduce":
            return impl.allreduce(x)
        if op == "reduce_scatter":
            return impl.reduce_scatter(x)
        if op == "allgather":
            return impl.allgather(x)
        raise ValueError(f"Unsupported op: {op}")

    # Correctness spot-check against torch collectives when benchmarking ring.
    if impl_name == "ring":
        ref_impl = TorchCollectives()
        if op == "allreduce":
            ref = ref_impl.allreduce(x)
        elif op == "reduce_scatter":
            ref = ref_impl.reduce_scatter(x)
        else:
            ref = ref_impl.allgather(x)

        got = call_once()
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

    for _ in range(warmup):
        _ = call_once()

    dist.barrier()
    for _ in range(iters):
        timer.start()
        _ = call_once()
        elapsed = timer.stop()
        del elapsed

    summary = timer.summary()
    return OpResult(
        op=op,
        impl=impl_name,
        numel=numel,
        dtype=str(dtype).replace("torch.", ""),
        bytes_sent_per_rank=bytes_per_rank(
            op=op,
            numel=numel,
            world_size=world_size,
            element_size=torch.tensor([], dtype=dtype).element_size(),
        ),
        mean_ms=summary.mean_ms,
        p50_ms=summary.p50_ms,
        p95_ms=summary.p95_ms,
    )


def main() -> None:
    args = parse_args()
    device = init_distributed(args.device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.dtype in {"float16", "bfloat16"} and device.type == "cpu":
        raise ValueError(f"dtype {args.dtype} is not supported for reliable CPU benchmarking")

    dtype = dtype_from_name(args.dtype)

    memory = MemoryTracker(device=device)
    memory.record("start")

    impls: List[str]
    if args.impl == "both":
        impls = ["ring", "torch"]
    else:
        impls = [args.impl]

    all_results: List[OpResult] = []
    t0 = time.perf_counter()

    for op in args.ops:
        for numel in args.sizes:
            for impl_name in impls:
                result = run_op(
                    op=op,
                    impl_name=impl_name,
                    numel=numel,
                    dtype=dtype,
                    iters=args.iters,
                    warmup=args.warmup,
                    device=device,
                )
                all_results.append(result)

    memory.record("end")
    elapsed_s = time.perf_counter() - t0

    gathered: List[Dict[str, object]] = [None for _ in range(world_size)]
    local_payload: Dict[str, object] = {
        "rank": rank,
        "device": str(device),
        "results": [asdict(x) for x in all_results],
        "memory": memory.as_dicts(),
    }
    dist.all_gather_object(gathered, local_payload)

    if rank == 0:
        report = {
            "world_size": world_size,
            "backend": dist.get_backend(),
            "elapsed_s": elapsed_s,
            "args": vars(args),
            "per_rank": gathered,
        }

        report_json = json.dumps(report, indent=2)
        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report_json)
            print(f"[benchmark] wrote {out_path}")
        else:
            print(report_json)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
