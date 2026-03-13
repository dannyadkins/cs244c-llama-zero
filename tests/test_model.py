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


def _grad_map(model: LlamaForCausalLM) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grads[name] = param.grad.detach().clone()
    return grads


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


def test_chunked_training_loss_matches_golden_value() -> None:
    cfg = tiny_test_config()
    torch.manual_seed(123)
    model = LlamaForCausalLM(cfg)
    model.set_loss_chunk_size(4)

    input_ids = torch.tensor(
        [
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3],
            [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5],
        ],
        dtype=torch.long,
    )

    out = model(input_ids=input_ids, labels=input_ids, return_logits=False)

    assert out.logits is None
    assert out.loss is not None
    torch.testing.assert_close(out.loss.detach(), torch.tensor(4.901787757873535), atol=1e-6, rtol=0.0)


def test_chunked_training_loss_matches_full_loss_and_grads() -> None:
    cfg = tiny_test_config()
    torch.manual_seed(123)
    model_full = LlamaForCausalLM(cfg)
    model_chunked = LlamaForCausalLM(cfg)
    model_chunked.load_state_dict(model_full.state_dict())
    model_chunked.set_loss_chunk_size(4)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), dtype=torch.long)

    out_full = model_full(input_ids=x, labels=x, return_logits=False)
    out_chunked = model_chunked(input_ids=x, labels=x, return_logits=False)

    assert out_full.logits is None
    assert out_chunked.logits is None
    assert out_full.loss is not None
    assert out_chunked.loss is not None
    torch.testing.assert_close(out_full.loss, out_chunked.loss, atol=1e-5, rtol=1e-5)

    out_full.loss.backward()
    out_chunked.loss.backward()

    grads_full = _grad_map(model_full)
    grads_chunked = _grad_map(model_chunked)
    assert grads_full.keys() == grads_chunked.keys()
    for name in grads_full:
        torch.testing.assert_close(grads_full[name], grads_chunked[name], atol=1e-5, rtol=1e-5)


def test_activation_checkpointing_matches_full_forward_and_grads() -> None:
    cfg = tiny_test_config()
    torch.manual_seed(456)
    model_base = LlamaForCausalLM(cfg)
    model_ckpt = LlamaForCausalLM(cfg)
    model_ckpt.load_state_dict(model_base.state_dict())
    model_base.train()
    model_ckpt.train()
    model_ckpt.set_activation_checkpointing(True)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), dtype=torch.long)

    out_base = model_base(input_ids=x, labels=x)
    out_ckpt = model_ckpt(input_ids=x, labels=x)

    assert out_base.loss is not None
    assert out_ckpt.loss is not None
    assert out_base.logits is not None
    assert out_ckpt.logits is not None
    torch.testing.assert_close(out_base.logits, out_ckpt.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_base.loss, out_ckpt.loss, atol=1e-5, rtol=1e-5)

    out_base.loss.backward()
    out_ckpt.loss.backward()

    grads_base = _grad_map(model_base)
    grads_ckpt = _grad_map(model_ckpt)
    assert grads_base.keys() == grads_ckpt.keys()
    for name in grads_base:
        torch.testing.assert_close(grads_base[name], grads_ckpt[name], atol=1e-5, rtol=1e-5)


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
