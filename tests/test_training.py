import torch

from model.config import ModelConfig
from model.llama import LlamaForCausalLM


def test_model_learns_on_deterministic_next_token_task() -> None:
    torch.manual_seed(7)

    cfg = ModelConfig(
        name="overfit_test",
        vocab_size=64,
        max_seq_len=16,
        dim=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        dropout=0.0,
    )
    model = LlamaForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    # Fixed batch with strong local token patterns. This should overfit quickly.
    batch = torch.stack(
        [
            torch.arange(0, cfg.max_seq_len + 1) % cfg.vocab_size,
            torch.arange(3, 3 + cfg.max_seq_len + 1) % cfg.vocab_size,
            torch.arange(9, 9 + cfg.max_seq_len + 1) % cfg.vocab_size,
            torch.arange(15, 15 + cfg.max_seq_len + 1) % cfg.vocab_size,
        ],
        dim=0,
    ).long()

    input_ids = batch[:, :-1]
    labels = input_ids.clone()

    losses = []
    model.train()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0] * 0.7, f"loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
