import torch

from model.config import ModelConfig
from model.llama import LlamaForCausalLM, RotaryEmbedding, apply_rotary_pos_emb


def tiny_test_config() -> ModelConfig:
    return ModelConfig(
        name="unit_test",
        vocab_size=128,
        max_seq_len=16,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        dropout=0.0,
    )


def test_forward_shape_and_backward() -> None:
    cfg = tiny_test_config()
    model = LlamaForCausalLM(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), dtype=torch.long)
    out = model(input_ids=x, labels=x)

    assert out.logits.shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)

    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_causal_mask_blocks_future_information() -> None:
    cfg = tiny_test_config()
    model = LlamaForCausalLM(cfg)
    model.eval()

    seq_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    seq_b = seq_a.clone()
    seq_b[0, -1] = 99

    logits_a = model(seq_a).logits
    logits_b = model(seq_b).logits

    # Changing the final token cannot affect earlier-position logits in a causal LM.
    torch.testing.assert_close(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-6, rtol=1e-5)


def test_rotary_embedding_preserves_vector_norm() -> None:
    head_dim = 32
    rotary = RotaryEmbedding(dim=head_dim, base=500_000.0)

    q = torch.randn(2, 4, 10, head_dim)
    k = torch.randn(2, 4, 10, head_dim)

    cos, sin = rotary(seq_len=10, device=q.device, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    torch.testing.assert_close(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5, rtol=1e-4)
