from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
import torch.distributed as dist

from collectives import LocalCollectives, SendRecvCollectives, TorchCollectives
from model import LlamaForCausalLM, build_config, estimate_num_parameters, human_readable_count
from profiler import MemoryTracker, TimerRegistry
from train import batch_from_chunk, make_data_loader, set_seed
from zero import ZeROStage0DDP, ZeROStage1Optimizer, ZeROStage2Optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 2: ZeRO stages 0-2 training")

    parser.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--collective-impl", type=str, default="ring", choices=["ring", "torch"])

    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)

    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["fineweb", "synthetic"])
    parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--fineweb-subset", type=str, default="sample-10BT")
    parser.add_argument("--fineweb-split", type=str, default="train")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--synthetic-vocab-size", type=int, default=8_192)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--profile-json", type=str, default="")
    parser.add_argument("--profile-memory-interval", type=int, default=0)
    parser.add_argument("--profile-rank0-only", action="store_true")

    return parser.parse_args()


def init_distributed() -> Tuple[int, int, torch.device]:
    using_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not using_torchrun:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    if dtype_name == "float32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def broadcast_model(model: LlamaForCausalLM) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def pick_collectives(world_size: int, impl: str):
    if world_size == 1:
        return LocalCollectives()
    if impl == "ring":
        return SendRecvCollectives()
    return TorchCollectives()


def build_engine(args: argparse.Namespace, model: LlamaForCausalLM, collectives):
    kwargs = {
        "model": model,
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "collectives": collectives,
    }
    if args.zero_stage == 0:
        return ZeROStage0DDP(**kwargs)
    if args.zero_stage == 1:
        return ZeROStage1Optimizer(**kwargs)
    if args.zero_stage == 2:
        return ZeROStage2Optimizer(**kwargs)
    raise ValueError(f"Unsupported stage: {args.zero_stage}")


def global_mean_scalar(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return float(tensor.item() / world_size)


def save_checkpoint(
    save_dir: str,
    step: int,
    rank: int,
    model: LlamaForCausalLM,
    engine,
    args: argparse.Namespace,
    extra: Dict[str, object],
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"zero_stage{args.zero_stage}_step{step:06d}_rank{rank:03d}.pt")
    payload = {
        "step": step,
        "rank": rank,
        "model_state_dict": model.state_dict(),
        "engine_state_dict": engine.state_dict(),
        "train_args": vars(args),
        "extra": extra,
    }
    torch.save(payload, path)
    return path


def train(args: argparse.Namespace) -> None:
    rank, world_size, device = init_distributed()
    set_seed(args.seed)

    base_vocab_size = args.synthetic_vocab_size
    cfg = build_config(size=args.model_size, vocab_size=base_vocab_size, max_seq_len=args.seq_len)

    try:
        loader, detected_vocab_size, metadata = make_data_loader(
            args,
            vocab_size_for_model=base_vocab_size,
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:
        if not args.allow_synthetic_fallback:
            raise
        if rank == 0:
            print(f"[warn] data init failed ({exc}); falling back to synthetic")
        args.data_mode = "synthetic"
        loader, detected_vocab_size, metadata = make_data_loader(
            args,
            vocab_size_for_model=base_vocab_size,
            rank=rank,
            world_size=world_size,
        )

    if detected_vocab_size != cfg.vocab_size:
        cfg = cfg.with_vocab_size(detected_vocab_size)

    model = LlamaForCausalLM(cfg).to(device)
    broadcast_model(model)

    collectives = pick_collectives(world_size=world_size, impl=args.collective_impl)
    engine = build_engine(args=args, model=model, collectives=collectives)

    if rank == 0:
        n_params = estimate_num_parameters(cfg)
        print(
            f"[init] stage={args.zero_stage} impl={args.collective_impl} "
            f"world_size={world_size} model={cfg.name} params={human_readable_count(n_params)} ({n_params:,}) "
            f"vocab={cfg.vocab_size:,} seq_len={cfg.max_seq_len} device={device} dtype={args.dtype}"
        )
        print(f"[init] data_mode={args.data_mode} metadata={json.dumps(metadata)}")

    global_tokens_per_step = args.batch_size * args.grad_accum_steps * args.seq_len * world_size
    data_iter: Iterator[torch.Tensor] = iter(loader)

    timers = TimerRegistry(device=device)
    memory = MemoryTracker(device=device)
    enable_profile_output = bool(args.profile_json)
    should_record_memory = args.profile_memory_interval > 0
    rank_profile_enabled = enable_profile_output and (rank == 0 or not args.profile_rank0_only)
    step_profiles = []

    if should_record_memory and rank_profile_enabled:
        memory.record("start")

    recent_losses = []
    t_train_start = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        iter_timer = timers.timer("iteration")
        iter_timer.start()

        model.train()
        engine.zero_grad()
        fwd_bwd_timer = timers.timer("forward_backward")
        fwd_bwd_timer.start()

        micro_losses = []
        for _micro in range(args.grad_accum_steps):
            try:
                chunk = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                chunk = next(data_iter)

            batch = batch_from_chunk(chunk)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            with autocast_context(device, args.dtype):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
                if loss is None:
                    raise RuntimeError("Model did not return loss even though labels were provided")
                scaled_loss = loss / args.grad_accum_steps

            micro_losses.append(float(loss.detach().float().item()))
            engine.backward(scaled_loss)

        fwd_bwd_ms = fwd_bwd_timer.stop()

        step_timer = timers.timer("optimizer_step")
        step_timer.start()
        step_stats = engine.step_with_stats(max_grad_norm=args.max_grad_norm)
        _ = step_timer.stop()
        grad_norm_value = float(step_stats["grad_norm"])

        local_step_loss = float(sum(micro_losses) / len(micro_losses))
        step_loss = global_mean_scalar(local_step_loss, device=device, world_size=world_size)

        recent_losses.append(step_loss)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        iter_ms = iter_timer.stop()
        step_time = iter_ms / 1000.0
        tokens_per_second = global_tokens_per_step / max(step_time, 1e-8)

        if rank == 0 and (step % args.log_interval == 0 or step == 1 or step == args.max_steps):
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(
                f"[step {step:05d}] loss={step_loss:.4f} avg100={avg_loss:.4f} "
                f"tokens/s={tokens_per_second:,.0f} grad_norm={grad_norm_value:.3f} "
                f"fb_ms={fwd_bwd_ms:.2f} comm_ms={step_stats['comm_ms']:.2f} opt_ms={step_stats['optim_ms']:.2f}"
            )

        if rank_profile_enabled:
            step_profiles.append(
                {
                    "step": step,
                    "loss": step_loss,
                    "tokens_per_second": tokens_per_second,
                    "grad_norm": grad_norm_value,
                    "forward_backward_ms": fwd_bwd_ms,
                    "communication_ms": float(step_stats["comm_ms"]),
                    "optimizer_ms": float(step_stats["optim_ms"]),
                    "iteration_ms": iter_ms,
                }
            )

        if should_record_memory and rank_profile_enabled:
            if step == 1 or step == args.max_steps or step % args.profile_memory_interval == 0:
                memory.record(f"step_{step}")

        if args.checkpoint_interval > 0 and (step % args.checkpoint_interval == 0 or step == args.max_steps):
            path = save_checkpoint(
                save_dir=args.save_dir,
                step=step,
                rank=rank,
                model=model,
                engine=engine,
                args=args,
                extra={"model_config": cfg.to_dict(), "step_loss": step_loss},
            )
            if rank == 0:
                print(f"[checkpoint] wrote {path}")

    elapsed = time.perf_counter() - t_train_start
    if rank == 0:
        print(f"[done] trained {args.max_steps} steps in {elapsed:.1f}s")

    if rank_profile_enabled:
        summaries = timers.summarize()
        timer_payload = {name: asdict(summary) for name, summary in summaries.items()}
        profile_payload = {
            "rank": rank,
            "world_size": world_size,
            "stage": args.zero_stage,
            "collective_impl": args.collective_impl,
            "args": vars(args),
            "timers": timer_payload,
            "memory": memory.as_dicts() if should_record_memory else [],
            "steps": step_profiles,
        }

        profile_path = Path(args.profile_json)
        if world_size > 1 and not args.profile_rank0_only:
            profile_path = profile_path.with_suffix(f".rank{rank:03d}.json")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile_payload, indent=2))
        if rank == 0 or not args.profile_rank0_only:
            print(f"[profile] wrote {profile_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
