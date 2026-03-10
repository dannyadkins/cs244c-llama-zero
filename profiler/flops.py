from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Dict, Generic, List, Optional, TypeVar

import torch

if TYPE_CHECKING:
    from model.config import ModelConfig


T = TypeVar("T")


@dataclass
class FlopSnapshot:
    label: str
    total_flops: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def flops_to_tflops_per_second(total_flops: float, step_time_s: float) -> float:
    if total_flops < 0.0 or step_time_s <= 0.0:
        raise ValueError("total_flops must be >= 0 and step_time_s must be > 0")
    return float(total_flops) / 1e12 / float(step_time_s)


def estimate_transformer_train_flops(
    cfg: "ModelConfig",
    batch_size: int,
    seq_len: int,
    grad_accum_steps: int = 1,
    world_size: int = 1,
    stage: int = 0,
) -> float:
    if batch_size <= 0 or seq_len <= 0 or grad_accum_steps <= 0 or world_size <= 0:
        raise ValueError("batch_size, seq_len, grad_accum_steps, and world_size must be positive")

    tokens = float(batch_size * seq_len)
    hidden = float(cfg.dim)
    layers = float(cfg.n_layers)
    ffn = float(cfg.ffn_dim)
    heads = float(cfg.n_heads)
    head_dim = float(cfg.head_dim)
    kv_dim = float(cfg.n_kv_heads * cfg.head_dim)
    vocab = float(cfg.vocab_size)

    # Dense projection and attention kernels dominate runtime for these models.
    attn_proj_flops = 2.0 * tokens * ((2.0 * hidden * hidden) + (2.0 * hidden * kv_dim))
    attn_score_flops = 2.0 * float(batch_size) * heads * float(seq_len) * float(seq_len) * head_dim
    attn_apply_flops = attn_score_flops
    mlp_flops = 2.0 * tokens * hidden * ffn * 3.0
    norm_and_pointwise_flops = tokens * ((8.0 * hidden) + (4.0 * ffn))
    lm_head_flops = 2.0 * tokens * hidden * vocab

    forward_flops = layers * (
        attn_proj_flops + attn_score_flops + attn_apply_flops + mlp_flops + norm_and_pointwise_flops
    ) + lm_head_flops

    training_multiplier = 4.0 if stage == 3 else 3.0
    return forward_flops * training_multiplier * float(grad_accum_steps * world_size)


@dataclass
class FlopTracker(Generic[T]):
    device: Optional[torch.device] = None
    snapshots: List[FlopSnapshot] = None

    def __post_init__(self) -> None:
        if self.snapshots is None:
            self.snapshots = []

    def _activities(self) -> List[torch.profiler.ProfilerActivity]:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return activities

    def measure(self, label: str, fn: Callable[[], T]) -> tuple[T, float | None]:
        with torch.profiler.profile(
            activities=self._activities(),
            with_flops=True,
            acc_events=True,
        ) as prof:
            result = fn()
            if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        total_flops = float(sum(float(getattr(evt, "flops", 0.0) or 0.0) for evt in prof.key_averages()))
        if total_flops <= 0.0:
            return result, None

        snapshot = FlopSnapshot(label=label, total_flops=total_flops)
        self.snapshots.append(snapshot)
        return result, total_flops

    def as_dicts(self) -> List[Dict[str, float | str]]:
        return [snapshot.to_dict() for snapshot in self.snapshots]