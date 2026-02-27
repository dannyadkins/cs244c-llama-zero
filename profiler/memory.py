from __future__ import annotations

import resource
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class MemorySnapshot:
    label: str
    timestamp_s: float
    host_maxrss_mb: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    cuda_max_allocated_mb: float
    cuda_max_reserved_mb: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def _host_maxrss_mb() -> float:
    # macOS reports bytes, Linux reports kilobytes.
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def take_memory_snapshot(label: str, device: Optional[torch.device] = None) -> MemorySnapshot:
    use_cuda = (
        device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    )

    if use_cuda:
        allocated = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
    else:
        allocated = reserved = max_allocated = max_reserved = 0.0

    return MemorySnapshot(
        label=label,
        timestamp_s=time.time(),
        host_maxrss_mb=_host_maxrss_mb(),
        cuda_allocated_mb=allocated,
        cuda_reserved_mb=reserved,
        cuda_max_allocated_mb=max_allocated,
        cuda_max_reserved_mb=max_reserved,
    )


@dataclass
class MemoryTracker:
    device: Optional[torch.device] = None
    snapshots: List[MemorySnapshot] = None

    def __post_init__(self) -> None:
        if self.snapshots is None:
            self.snapshots = []

    def record(self, label: str) -> MemorySnapshot:
        snapshot = take_memory_snapshot(label=label, device=self.device)
        self.snapshots.append(snapshot)
        return snapshot

    def as_dicts(self) -> List[Dict[str, float | str]]:
        return [x.to_dict() for x in self.snapshots]
