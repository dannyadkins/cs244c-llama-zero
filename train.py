from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from torch.utils.data import DataLoader

from data import (
    PackedFineWebDataset,
    SyntheticPatternDataset,
    load_fineweb_stream,
    load_llama_tokenizer,
)
from model import (
    LlamaForCausalLM,
    build_config,
    estimate_num_parameters,
    human_readable_count,
)


@dataclass
class TrainBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 Person A: single-GPU LLaMA training baseline")

    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--data-mode", type=str, default="fineweb", choices=["fineweb", "synthetic"])
    parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--fineweb-subset", type=str, default="sample-10BT")
    parser.add_argument("--fineweb-split", type=str, default="train")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--synthetic-vocab-size", type=int, default=8_192)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def make_data_loader(
    args: argparse.Namespace,
    vocab_size_for_model: int,
) -> Tuple[DataLoader, int, Dict[str, str]]:
    metadata: Dict[str, str] = {}

    if args.data_mode == "fineweb":
        bundle = load_llama_tokenizer(args.tokenizer_name)
        stream = load_fineweb_stream(
            subset=args.fineweb_subset,
            split=args.fineweb_split,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        )
        dataset = PackedFineWebDataset(
            text_iterable=stream,
            tokenizer=bundle.tokenizer,
            seq_len=args.seq_len,
            eos_token_id=bundle.eos_token_id,
        )
        vocab_size = bundle.vocab_size
        metadata["tokenizer"] = args.tokenizer_name
        metadata["dataset"] = f"HuggingFaceFW/fineweb-edu:{args.fineweb_subset}/{args.fineweb_split}"
    else:
        vocab_size = vocab_size_for_model
        dataset = SyntheticPatternDataset(seq_len=args.seq_len, vocab_size=vocab_size)
        metadata["tokenizer"] = "synthetic"
        metadata["dataset"] = "synthetic_pattern"

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, vocab_size, metadata


def batch_from_chunk(chunk: torch.Tensor) -> TrainBatch:
    # Input is a packed sequence of length (seq_len + 1).
    input_ids = chunk[:, :-1].contiguous()
    # Model follows standard causal LM convention: labels == input_ids,
    # and shifting is handled inside model.forward().
    labels = input_ids.clone()
    return TrainBatch(input_ids=input_ids, labels=labels)


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    if dtype_name == "float32":
        return nullcontext()
    target = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=target)


def save_checkpoint(
    save_dir: str,
    step: int,
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    extra: Dict[str, object],
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{step:06d}.pt")
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_args": vars(args),
        "extra": extra,
    }
    torch.save(payload, path)
    return path


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)

    base_vocab_size = args.synthetic_vocab_size
    cfg = build_config(size=args.model_size, vocab_size=base_vocab_size, max_seq_len=args.seq_len)

    try:
        loader, detected_vocab_size, metadata = make_data_loader(args, vocab_size_for_model=base_vocab_size)
    except Exception as exc:
        if not args.allow_synthetic_fallback:
            raise
        print(f"[warn] failed to initialize fineweb pipeline ({exc}); falling back to synthetic data")
        args.data_mode = "synthetic"
        loader, detected_vocab_size, metadata = make_data_loader(args, vocab_size_for_model=base_vocab_size)

    if detected_vocab_size != cfg.vocab_size:
        cfg = cfg.with_vocab_size(detected_vocab_size)

    model = LlamaForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    use_grad_scaler = device.type == "cuda" and args.dtype == "float16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    n_params = estimate_num_parameters(cfg)
    print(
        f"[init] model={cfg.name} params={human_readable_count(n_params)} ({n_params:,}) "
        f"vocab={cfg.vocab_size:,} seq_len={cfg.max_seq_len} device={device.type} dtype={args.dtype}"
    )
    print(f"[init] data_mode={args.data_mode} metadata={json.dumps(metadata)}")

    token_budget_per_step = args.batch_size * args.grad_accum_steps * args.seq_len
    data_iter: Iterator[torch.Tensor] = iter(loader)

    recent_losses = []
    t_train_start = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t_step_start = time.perf_counter()

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

            micro_losses.append(loss.detach().float().item())
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if args.max_grad_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            grad_norm_value = float(grad_norm.item())
        else:
            grad_norm_value = math.nan

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step_loss = float(sum(micro_losses) / len(micro_losses))
        recent_losses.append(step_loss)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        step_time = time.perf_counter() - t_step_start
        tokens_per_second = token_budget_per_step / max(step_time, 1e-8)

        if step % args.log_interval == 0 or step == 1 or step == args.max_steps:
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(
                f"[step {step:05d}] loss={step_loss:.4f} avg100={avg_loss:.4f} "
                f"tokens/s={tokens_per_second:,.0f} grad_norm={grad_norm_value:.3f}"
            )

        if args.checkpoint_interval > 0 and (step % args.checkpoint_interval == 0 or step == args.max_steps):
            ckpt_path = save_checkpoint(
                save_dir=args.save_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                args=args,
                extra={
                    "model_config": cfg.to_dict(),
                    "step_loss": step_loss,
                },
            )
            print(f"[checkpoint] wrote {ckpt_path}")

    elapsed = time.perf_counter() - t_train_start
    print(f"[done] trained {args.max_steps} steps in {elapsed:.1f}s")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
