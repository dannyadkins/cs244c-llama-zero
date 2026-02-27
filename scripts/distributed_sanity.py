from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    x = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(x)

    expected = world_size * (world_size + 1) / 2.0
    if abs(float(x.item()) - expected) > 1e-6:
        raise RuntimeError(f"all_reduce failed on rank {rank}: got {x.item()}, expected {expected}")

    if rank == 0:
        print(f"[sanity] backend={backend} world_size={world_size} all_reduce_sum={x.item():.1f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
