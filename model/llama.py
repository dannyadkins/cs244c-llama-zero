from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 500_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=dtype)
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_len_cached = seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_needs_refresh = (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        )
        if cache_needs_refresh:
            self._build_cache(seq_len=seq_len, device=device, dtype=dtype)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=cfg.rope_theta)

    def _repeat_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return tensor
        repeat_factor = self.n_heads // self.n_kv_heads
        return tensor.repeat_interleave(repeat_factor, dim=1)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len=seq_len, device=q.device, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_dropout = self.dropout if self.training else 0.0

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=attn_dropout,
                is_causal=attention_mask is None,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is None:
                causal = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal, float("-inf"))
            else:
                scores = scores + attention_mask
            probs = F.softmax(scores, dim=-1)
            probs = F.dropout(probs, p=attn_dropout, training=self.training)
            y = torch.matmul(probs, v)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(gated))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.post_attn_norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


@dataclass
class CausalLMOutput:
    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor] = None


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
        self._zero3_executor = None
        self.loss_chunk_size = 0
        self.activation_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_loss_chunk_size(self, chunk_size: int) -> None:
        if chunk_size < 0:
            raise ValueError(f"loss chunk size must be non-negative, got {chunk_size}")
        self.loss_chunk_size = int(chunk_size)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = bool(enabled)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids should have shape [batch, seq], got {tuple(input_ids.shape)}")

        x = self._call_parametrized_module("tok_embeddings", self.tok_embeddings, input_ids)
        x = self.dropout(x)

        for layer_idx in range(len(self.layers)):
            if self._should_checkpoint_activations(x):
                x = activation_checkpoint(
                    lambda hidden_states, _layer_idx=layer_idx, _attention_mask=attention_mask: self._call_layer(
                        _layer_idx,
                        hidden_states,
                        _attention_mask,
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._call_layer(layer_idx, x, attention_mask)

        x = self._call_parametrized_module("norm", self.norm, x)
        logits = None
        if labels is None or return_logits or not self._should_chunk_training_loss(input_ids):
            logits = self._call_parametrized_module("lm_head", self.lm_head, x)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels should have same shape as input_ids; got {tuple(labels.shape)} vs {tuple(input_ids.shape)}"
                )
            if logits is not None:
                loss = self._full_cross_entropy(logits=logits, labels=labels)
            else:
                loss = self._chunked_cross_entropy(hidden_states=x, labels=labels)
            if not return_logits:
                logits = None

        return CausalLMOutput(logits=logits, loss=loss)

    def _call_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        return self._call_parametrized_module(f"layers.{layer_idx}", layer, hidden_states, attention_mask)

    def _should_checkpoint_activations(self, hidden_states: torch.Tensor) -> bool:
        return (
            self.activation_checkpointing
            and self.training
            and self._zero3_executor is None
            and hidden_states.requires_grad
        )

    def _should_chunk_training_loss(self, input_ids: torch.Tensor) -> bool:
        return (
            self.loss_chunk_size > 0
            and input_ids.size(1) > 1
            and self._zero3_executor is None
        )

    def _full_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.size(1) <= 1:
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[:, :-1, :].transpose(1, 2),
            labels[:, 1:],
            ignore_index=-100,
        )

    def _chunked_cross_entropy(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.size(1)
        if seq_len <= 1:
            return hidden_states.sum() * 0.0

        total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=labels.device, dtype=torch.long)
        for start in range(0, seq_len - 1, self.loss_chunk_size):
            end = min(start + self.loss_chunk_size, seq_len - 1)
            hidden_chunk = hidden_states[:, start:end, :]
            label_chunk = labels[:, start + 1 : end + 1]
            logits_chunk = self._call_parametrized_module("lm_head", self.lm_head, hidden_chunk)
            total_loss = total_loss + F.cross_entropy(
                logits_chunk.transpose(1, 2),
                label_chunk,
                ignore_index=-100,
                reduction="sum",
            )
            total_tokens = total_tokens + label_chunk.ne(-100).sum()

        return total_loss / total_tokens.clamp_min(1).to(dtype=total_loss.dtype)

    def _call_parametrized_module(self, module_name: str, module: nn.Module, *args: Any) -> torch.Tensor:
        executor = self._zero3_executor
        if executor is None:
            return module(*args)
        return executor.call_module(module_name, module, *args)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            input_slice = input_ids[:, -self.cfg.max_seq_len :]
            out = self(input_slice)
            next_logits = out.logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < values[:, [-1]]] = float("-inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
