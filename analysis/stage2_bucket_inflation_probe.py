from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from model import LlamaForCausalLM, build_config
from zero.common import build_flat_param_metadata, compute_shard_spec


@dataclass
class BucketRow:
    index: int
    bucket_mb: float
    packed_mb: float
    inflation: float
    shard_piece_mbs: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify Stage-2 reduce-scatter bucket padding inflation")
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--grad-bucket-mb", type=float, default=64.0)
    parser.add_argument("--print-buckets", action="store_true")
    return parser.parse_args()


def _build_stage2_bucket_ranges(total_numel: int, param_numels: List[int], offsets: List[int], bucket_numel: int) -> List[tuple[int, int]]:
    ranges: List[tuple[int, int]] = []
    bucket_param_indices: List[int] = []
    bucket_start = 0
    bucket_size = 0

    for param_idx, (param_numel, start) in enumerate(zip(param_numels, offsets)):
        if not bucket_param_indices:
            bucket_param_indices = [param_idx]
            bucket_start = start
            bucket_size = param_numel
            continue

        if bucket_size + param_numel > bucket_numel:
            last_idx = bucket_param_indices[-1]
            bucket_end = offsets[last_idx] + param_numels[last_idx]
            ranges.append((bucket_start, bucket_end))
            bucket_param_indices = [param_idx]
            bucket_start = start
            bucket_size = param_numel
            continue

        bucket_param_indices.append(param_idx)
        bucket_size += param_numel

    if bucket_param_indices:
        last_idx = bucket_param_indices[-1]
        bucket_end = offsets[last_idx] + param_numels[last_idx]
        ranges.append((bucket_start, bucket_end))

    if sum(end - start for start, end in ranges) != total_numel:
        raise RuntimeError("bucket ranges do not cover the flattened parameter space exactly")
    return ranges


def main() -> None:
    args = parse_args()
    model = LlamaForCausalLM(build_config(size=args.model_size, vocab_size=args.vocab_size, max_seq_len=args.seq_len))
    meta = build_flat_param_metadata(model)
    shard = compute_shard_spec(meta.total_numel, rank=0, world_size=args.world_size)
    target_bucket_numel = max(1, int((args.grad_bucket_mb * 1024.0 * 1024.0) / 4.0))
    param_numels = [param.numel() for param in meta.params]
    ranges = _build_stage2_bucket_ranges(
        total_numel=meta.total_numel,
        param_numels=param_numels,
        offsets=meta.offsets,
        bucket_numel=target_bucket_numel,
    )

    rows: List[BucketRow] = []
    fully_single_shard = 0
    sum_bucket_numel = 0
    sum_packed_numel = 0
    for bucket_idx, (start, end) in enumerate(ranges):
        bucket_numel = end - start
        shard_piece_numels: List[int] = []
        for rank in range(args.world_size):
            shard_start = rank * shard.chunk_size
            shard_end = min(shard_start + shard.chunk_size, meta.total_numel)
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            shard_piece_numels.append(max(0, overlap_end - overlap_start))

        non_zero_pieces = sum(1 for value in shard_piece_numels if value > 0)
        if non_zero_pieces == 1:
            fully_single_shard += 1

        packed_chunk_numel = max(1, max(shard_piece_numels) if shard_piece_numels else 0)
        packed_numel = args.world_size * packed_chunk_numel
        sum_bucket_numel += bucket_numel
        sum_packed_numel += packed_numel
        rows.append(
            BucketRow(
                index=bucket_idx,
                bucket_mb=bucket_numel * 4.0 / (1024.0 * 1024.0),
                packed_mb=packed_numel * 4.0 / (1024.0 * 1024.0),
                inflation=packed_numel / bucket_numel,
                shard_piece_mbs=[value * 4.0 / (1024.0 * 1024.0) for value in shard_piece_numels],
            )
        )

    print(f"model={args.model_size} vocab_size={args.vocab_size} seq_len={args.seq_len} world_size={args.world_size}")
    print(f"model_mb={meta.total_numel * 4.0 / (1024.0 * 1024.0):.2f}")
    print(f"rank_shard_mb={shard.chunk_size * 4.0 / (1024.0 * 1024.0):.2f}")
    print(f"target_bucket_mb={args.grad_bucket_mb:.2f}")
    print(f"bucket_count={len(rows)}")
    print(f"single_shard_buckets={fully_single_shard}/{len(rows)}")
    print(f"logical_bucketed_mb={sum_bucket_numel * 4.0 / (1024.0 * 1024.0):.2f}")
    print(f"packed_reduce_scatter_input_mb={sum_packed_numel * 4.0 / (1024.0 * 1024.0):.2f}")
    print(f"overall_inflation={sum_packed_numel / sum_bucket_numel:.4f}x")

    if not args.print_buckets:
        return

    print("")
    print("| Bucket | Bucket MB | Packed MB | Inflation | Rank pieces MB |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        shard_pieces = ", ".join(f"{value:.2f}" for value in row.shard_piece_mbs)
        print(
            f"| {row.index} | {row.bucket_mb:.2f} | {row.packed_mb:.2f} | {row.inflation:.3f}x | {shard_pieces} |"
        )


if __name__ == "__main__":
    main()
