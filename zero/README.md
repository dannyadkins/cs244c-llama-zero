# ZeRO Implementations

This directory contains the repository's ZeRO stage implementations and shared helpers.

## Stages

- `stage0_ddp.py`: fully replicated model with gradient all-reduce and local optimizer state
- `stage1_optimizer.py`: optimizer-state sharding plus parameter all-gather
- `stage2_optimizer.py`: optimizer-state sharding plus gradient reduce-scatter and parameter all-gather
- `stage3_optimizer.py`: module-wise parameter sharding with lazy materialization, backward recomputation, and sharded optimizer state

## Design Principles

- explicit communication through the collectives interface
- correctness-first implementations that are easy to audit
- deterministic comparison against reference AdamW behavior
- shared metadata utilities in `common.py` so stage transitions are easy to reason about

## Entry Point

All stages are exercised through [`train_zero.py`](/home/thomason/github/cs244c-llama-zero/train_zero.py):

```bash
python3 -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=<master-addr> \
  --master_port=<master-port> \
  train_zero.py \
  --zero-stage 3 \
  --collective-impl torch \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 50
```

## Testing

Core correctness coverage lives in [`zero/tests/test_zero_stages.py`](/home/thomason/github/cs244c-llama-zero/zero/tests/test_zero_stages.py). The tests compare stage behavior against a reference optimizer path and include gradient clipping parity checks.

## Current Scope

The implementations focus on correctness, measurement, and experimental clarity. They do not yet aim to reproduce every production optimization such as aggressive overlap, prefetch scheduling, or kernel fusion.
