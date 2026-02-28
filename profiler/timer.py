from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class TimerSummary:
    name: str
    count: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass
class NamedTimer:
    """Named timer with CUDA-event timing on GPU and wall-clock fallback on CPU."""

    name: str
    device: Optional[torch.device] = None
    _samples_ms: List[float] = field(default_factory=list)
    _start_wall: Optional[float] = None
    _start_event: Optional[torch.cuda.Event] = None
    _end_event: Optional[torch.cuda.Event] = None

    def start(self) -> None:
        use_cuda_events = (
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        if use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_wall = time.perf_counter()

    def stop(self) -> float:
        if self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = float(self._start_event.elapsed_time(self._end_event))
            self._start_event = None
            self._end_event = None
        elif self._start_wall is not None:
            elapsed_ms = (time.perf_counter() - self._start_wall) * 1000.0
            self._start_wall = None
        else:
            raise RuntimeError("Timer.stop() called before Timer.start()")

        self._samples_ms.append(elapsed_ms)
        return elapsed_ms

    def summary(self) -> TimerSummary:
        if not self._samples_ms:
            raise RuntimeError(f"Timer '{self.name}' has no samples")

        samples = sorted(self._samples_ms)

        def percentile(p: float) -> float:
            idx = int(round((len(samples) - 1) * p))
            return float(samples[idx])

        total = float(sum(samples))
        return TimerSummary(
            name=self.name,
            count=len(samples),
            total_ms=total,
            mean_ms=float(statistics.mean(samples)),
            p50_ms=percentile(0.50),
            p95_ms=percentile(0.95),
            min_ms=float(samples[0]),
            max_ms=float(samples[-1]),
        )


@dataclass
class TimerRegistry:
    """Container for multiple named timers within one process."""

    device: Optional[torch.device] = None
    _timers: Dict[str, NamedTimer] = field(default_factory=dict)

    def timer(self, name: str) -> NamedTimer:
        if name not in self._timers:
            self._timers[name] = NamedTimer(name=name, device=self.device)
        return self._timers[name]

    def summarize(self) -> Dict[str, TimerSummary]:
        return {name: timer.summary() for name, timer in self._timers.items() if timer._samples_ms}
