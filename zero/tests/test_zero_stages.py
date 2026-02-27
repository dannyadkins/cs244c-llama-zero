from __future__ import annotations

import socket
from typing import Callable, Type

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.config import ModelConfig
from model.llama import LlamaForCausalLM
from zero import ZeROStage0DDP, ZeROStage1Optimizer, ZeROStage2Optimizer


pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _test_config() -> ModelConfig:
    return ModelConfig(
        name="zero_test",
        vocab_size=97,
        max_seq_len=8,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=96,
        dropout=0.0,
    )


def _local_batch(step: int, rank: int, batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    base = torch.arange(batch_size * seq_len, dtype=torch.long).view(batch_size, seq_len)
    return (base + (step * 13) + (rank * 7)) % vocab_size


def _assert_models_close(model_a: LlamaForCausalLM, model_b: LlamaForCausalLM, msg: str) -> None:
    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(p_a, p_b, atol=2e-5, rtol=2e-5, msg=f"{msg} :: {name_a}")


def _run_stage_worker(rank: int, world_size: int, port: int, stage: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cfg = _test_config()
        torch.manual_seed(1234)

        model_zero = LlamaForCausalLM(cfg)
        for p in model_zero.parameters():
            dist.broadcast(p.data, src=0)

        model_ref = LlamaForCausalLM(cfg)
        model_ref.load_state_dict(model_zero.state_dict())

        lr = 1e-3
        betas = (0.9, 0.95)
        eps = 1e-8
        wd = 0.05

        if stage == 0:
            engine = ZeROStage0DDP(model=model_zero, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        elif stage == 1:
            engine = ZeROStage1Optimizer(model=model_zero, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        elif stage == 2:
            engine = ZeROStage2Optimizer(model=model_zero, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        else:
            raise ValueError(f"unknown stage: {stage}")

        opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)

        batch_size = 3
        for step in range(3):
            local_ids = _local_batch(
                step=step,
                rank=rank,
                batch_size=batch_size,
                seq_len=cfg.max_seq_len,
                vocab_size=cfg.vocab_size,
            )

            gathered = [torch.empty_like(local_ids) for _ in range(world_size)]
            dist.all_gather(gathered, local_ids)
            global_ids = torch.cat(gathered, dim=0)

            # ZeRO stage step using local micro-batch.
            engine.zero_grad()
            loss_zero = model_zero(input_ids=local_ids, labels=local_ids).loss
            assert loss_zero is not None
            engine.backward(loss_zero)
            engine.step()

            # Reference step using the full effective global batch.
            opt_ref.zero_grad(set_to_none=True)
            loss_ref = model_ref(input_ids=global_ids, labels=global_ids).loss
            assert loss_ref is not None
            loss_ref.backward()
            opt_ref.step()

            _assert_models_close(model_zero, model_ref, msg=f"stage={stage} step={step} rank={rank}")

    finally:
        dist.barrier()
        dist.destroy_process_group()


def _spawn_stage(stage: int, world_size: int = 2) -> None:
    port = _find_free_port()
    mp.spawn(_run_stage_worker, args=(world_size, port, stage), nprocs=world_size, join=True)


def test_stage0_matches_reference_global_batch() -> None:
    _spawn_stage(stage=0)


def test_stage1_matches_reference_global_batch() -> None:
    _spawn_stage(stage=1)


def test_stage2_matches_reference_global_batch() -> None:
    _spawn_stage(stage=2)
