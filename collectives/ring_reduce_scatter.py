from __future__ import annotations

import torch

from ._ring import ring_reduce_scatter

__all__ = ["ring_reduce_scatter"]


def reduce_scatter(tensor: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper used by Week 2 optimizer wrappers."""
    return ring_reduce_scatter(tensor=tensor)
