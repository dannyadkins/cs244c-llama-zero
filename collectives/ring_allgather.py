from __future__ import annotations

import torch

from ._ring import ring_allgather

__all__ = ["ring_allgather"]


def allgather(local_shard: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper used by Week 2 optimizer wrappers."""
    return ring_allgather(local_shard=local_shard)
