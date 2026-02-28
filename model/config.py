from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


def _find_multiple(value: int, multiple_of: int) -> int:
    if value % multiple_of == 0:
        return value
    return value + multiple_of - (value % multiple_of)


def _default_ffn_dim(hidden_dim: int) -> int:
    # LLaMA family uses a SwiGLU hidden width close to 8/3 * d, rounded up.
    return _find_multiple(int((8 * hidden_dim) / 3), 256)


@dataclass
class ModelConfig:
    name: str
    vocab_size: int
    max_seq_len: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: Optional[int] = None
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.ffn_dim is None:
            self.ffn_dim = _default_ffn_dim(self.dim)

        if self.dim <= 0 or self.n_layers <= 0:
            raise ValueError("dim and n_layers must be positive")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("n_heads and n_kv_heads must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def with_vocab_size(self, vocab_size: int) -> "ModelConfig":
        return ModelConfig(**{**self.to_dict(), "vocab_size": vocab_size})

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_dim": self.ffn_dim,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "dropout": self.dropout,
            "tie_word_embeddings": self.tie_word_embeddings,
            "initializer_range": self.initializer_range,
        }


def build_tiny_config(vocab_size: int = 128_256, max_seq_len: int = 2048) -> ModelConfig:
    # ~46M params with tied embeddings at vocab_size=128_256.
    return ModelConfig(
        name="tiny",
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=320,
        n_layers=4,
        n_heads=10,
        n_kv_heads=2,
    )


def build_small_config(vocab_size: int = 128_256, max_seq_len: int = 2048) -> ModelConfig:
    # ~318M params with tied embeddings at vocab_size=128_256.
    return ModelConfig(
        name="small",
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=1152,
        n_layers=12,
        n_heads=18,
        n_kv_heads=6,
    )


def build_medium_config(vocab_size: int = 128_256, max_seq_len: int = 2048) -> ModelConfig:
    # ~1.34B params with tied embeddings at vocab_size=128_256.
    return ModelConfig(
        name="medium",
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=8,
    )


def build_config(size: str, vocab_size: int = 128_256, max_seq_len: int = 2048) -> ModelConfig:
    size = size.lower().strip()
    builders = {
        "tiny": build_tiny_config,
        "small": build_small_config,
        "medium": build_medium_config,
    }
    if size not in builders:
        raise ValueError(f"Unknown model size '{size}'. Expected one of {sorted(builders)}")
    return builders[size](vocab_size=vocab_size, max_seq_len=max_seq_len)


def estimate_num_parameters(cfg: ModelConfig) -> int:
    # Embedding table (also used as LM head if tied).
    total = cfg.vocab_size * cfg.dim

    # Transformer blocks.
    q_proj = cfg.dim * cfg.dim
    kv_proj = cfg.dim * (cfg.n_kv_heads * cfg.head_dim)
    o_proj = cfg.dim * cfg.dim
    attn_params = q_proj + kv_proj + kv_proj + o_proj

    mlp_params = cfg.dim * cfg.ffn_dim * 3

    norms = 2 * cfg.dim
    per_block = attn_params + mlp_params + norms
    total += per_block * cfg.n_layers

    # Final RMSNorm.
    total += cfg.dim

    if not cfg.tie_word_embeddings:
        total += cfg.vocab_size * cfg.dim

    return total


def human_readable_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)
