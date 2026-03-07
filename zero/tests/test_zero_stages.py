from __future__ import annotations

import socket
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.config import ModelConfig
from model.llama import LlamaForCausalLM
from zero import ZeROStage0DDP, ZeROStage1Optimizer, ZeROStage2Optimizer, ZeROStage3Optimizer


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


def _full_param_context(engine):
    return engine.summon_full_params() if hasattr(engine, "summon_full_params") else nullcontext()


def _run_stage_worker(rank: int, world_size: int, port: int, stage: int, max_grad_norm: float) -> None:
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
        elif stage == 3:
            engine = ZeROStage3Optimizer(model=model_zero, lr=lr, betas=betas, eps=eps, weight_decay=wd)
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
            engine.prepare_forward()
            loss_zero = model_zero(input_ids=local_ids, labels=local_ids).loss
            assert loss_zero is not None
            engine.backward(loss_zero)
            engine.step(max_grad_norm=max_grad_norm)

            # Reference step using the full effective global batch.
            opt_ref.zero_grad(set_to_none=True)
            loss_ref = model_ref(input_ids=global_ids, labels=global_ids).loss
            assert loss_ref is not None
            loss_ref.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_ref.parameters(), max_norm=max_grad_norm)
            opt_ref.step()

            with _full_param_context(engine):
                _assert_models_close(model_zero, model_ref, msg=f"stage={stage} step={step} rank={rank}")

    finally:
        dist.barrier()
        dist.destroy_process_group()


def _spawn_stage(stage: int, world_size: int, max_grad_norm: float = 0.0) -> None:
    port = _find_free_port()
    mp.spawn(_run_stage_worker, args=(world_size, port, stage, max_grad_norm), nprocs=world_size, join=True)


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
@pytest.mark.parametrize("world_size", [2, 3])
def test_stages_match_reference_global_batch(stage: int, world_size: int) -> None:
    _spawn_stage(stage=stage, world_size=world_size)


@pytest.mark.parametrize("stage", [2, 3])
def test_stage_sharded_with_grad_clipping_matches_reference_world3(stage: int) -> None:
    _spawn_stage(stage=stage, world_size=3, max_grad_norm=0.5)


def _stage3_materialization_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cfg = _test_config()
        model = LlamaForCausalLM(cfg)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        engine = ZeROStage3Optimizer(model=model, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        assert any(param.numel() == 0 for param in model.parameters()), "stage3 should shard params between calls"

        with engine.summon_full_params():
            assert all(param.numel() > 0 for param in model.parameters())

        assert any(param.numel() == 0 for param in model.parameters()), "params should re-shard after summon context"
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_stage3_summon_full_params_round_trip_world2() -> None:
    port = _find_free_port()
    mp.spawn(_stage3_materialization_worker, args=(2, port), nprocs=2, join=True)


def _dropout_config() -> ModelConfig:
    return ModelConfig(
        name="zero_stage3_dropout",
        vocab_size=97,
        max_seq_len=8,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=96,
        dropout=0.2,
    )


def _stage3_dropout_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cfg = _dropout_config()
        torch.manual_seed(4321)

        model_stage0 = LlamaForCausalLM(cfg)
        for param in model_stage0.parameters():
            dist.broadcast(param.data, src=0)

        model_stage3 = LlamaForCausalLM(cfg)
        model_stage3.load_state_dict(model_stage0.state_dict())

        kwargs = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01}
        engine0 = ZeROStage0DDP(model=model_stage0, **kwargs)
        engine3 = ZeROStage3Optimizer(model=model_stage3, **kwargs)

        batch_size = 3
        for step in range(2):
            local_ids = _local_batch(
                step=step,
                rank=rank,
                batch_size=batch_size,
                seq_len=cfg.max_seq_len,
                vocab_size=cfg.vocab_size,
            )

            seed = 10_000 + (step * 31) + rank

            model_stage0.train()
            engine0.zero_grad()
            engine0.prepare_forward()
            torch.manual_seed(seed)
            loss0 = model_stage0(input_ids=local_ids, labels=local_ids).loss
            assert loss0 is not None
            engine0.backward(loss0)
            engine0.step()

            model_stage3.train()
            engine3.zero_grad()
            engine3.prepare_forward()
            torch.manual_seed(seed)
            loss3 = model_stage3(input_ids=local_ids, labels=local_ids).loss
            assert loss3 is not None
            engine3.backward(loss3)
            engine3.step()

            with engine3.summon_full_params():
                _assert_models_close(model_stage3, model_stage0, msg=f"dropout stage3 parity step={step} rank={rank}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_stage3_matches_stage0_with_dropout_world2() -> None:
    port = _find_free_port()
    mp.spawn(_stage3_dropout_worker, args=(2, port), nprocs=2, join=True)


def _stage3_checkpoint_restore_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cfg = _test_config()
        torch.manual_seed(2024)

        kwargs = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01}

        model_src = LlamaForCausalLM(cfg)
        for param in model_src.parameters():
            dist.broadcast(param.data, src=0)
        engine_src = ZeROStage3Optimizer(model=model_src, **kwargs)

        for step in range(2):
            local_ids = _local_batch(
                step=step,
                rank=rank,
                batch_size=3,
                seq_len=cfg.max_seq_len,
                vocab_size=cfg.vocab_size,
            )
            engine_src.zero_grad()
            engine_src.prepare_forward()
            loss = model_src(input_ids=local_ids, labels=local_ids).loss
            assert loss is not None
            engine_src.backward(loss)
            engine_src.step(max_grad_norm=0.25)

        saved_engine_state = engine_src.state_dict()

        torch.manual_seed(9999)
        model_restored = LlamaForCausalLM(cfg)
        engine_restored = ZeROStage3Optimizer(model=model_restored, **kwargs)
        engine_restored.load_state_dict(saved_engine_state)

        assert engine_restored.step_count == engine_src.step_count

        with engine_src.summon_full_params(), engine_restored.summon_full_params():
            _assert_models_close(model_restored, model_src, msg=f"stage3 restore rank={rank}")

        next_ids = _local_batch(
            step=7,
            rank=rank,
            batch_size=3,
            seq_len=cfg.max_seq_len,
            vocab_size=cfg.vocab_size,
        )
        for model, engine in ((model_src, engine_src), (model_restored, engine_restored)):
            engine.zero_grad()
            engine.prepare_forward()
            loss = model(input_ids=next_ids, labels=next_ids).loss
            assert loss is not None
            engine.backward(loss)
            engine.step(max_grad_norm=0.25)

        with engine_src.summon_full_params(), engine_restored.summon_full_params():
            _assert_models_close(model_restored, model_src, msg=f"stage3 restore continue rank={rank}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_stage3_checkpoint_restore_world2() -> None:
    port = _find_free_port()
    mp.spawn(_stage3_checkpoint_restore_worker, args=(2, port), nprocs=2, join=True)
