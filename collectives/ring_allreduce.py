from __future__ import annotations

import torch

from ._ring import ring_allreduce

__all__ = ["ring_allreduce"]


def allreduce(tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
    """Compatibility wrapper used by Week 2 optimizer wrappers."""
    return ring_allreduce(tensor=tensor, average=average)
