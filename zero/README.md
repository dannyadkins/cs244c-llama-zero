# ZeRO Optimizers (Weeks 2-3)

Implemented stages:

- `stage0_ddp.py`: full replication + gradient allreduce + local AdamW step
- `stage1_optimizer.py`: optimizer-state sharding + gradient allreduce + parameter allgather
- `stage2_optimizer.py`: optimizer-state sharding + gradient reduce-scatter + parameter allgather
- `stage3_optimizer.py`: correctness-first stage-3 communication pattern with forward/backward rematerialization accounting + gradient reduce-scatter + sharded optimizer state

Design goals:

- Clear flat-parameter metadata in `common.py`
- Explicit communication boundaries through `CollectiveOps`
- Minimal hidden magic so stage transitions are easy to audit
- Deterministic correctness checks against global-batch AdamW reference updates
- Synchronized-gradient clipping semantics via `step(max_grad_norm=...)`

Usage entrypoint:

- `train_zero.py` for multi-process stage 0/1/2/3 training

Example:

```bash
torchrun --standalone --nproc_per_node=2 train_zero.py \
  --zero-stage 3 \
  --collective-impl ring \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 50
```

Test coverage:

- `zero/tests/test_zero_stages.py` validates stages 0/1/2/3 against reference optimizer
- coverage includes `world_size` 2 and 3
- includes a stage-2 gradient clipping parity check

Note:

- Stage 3 here is intentionally correctness-first and explicit: no overlap/prefetch optimizations yet.
- It is designed to make communication and update semantics auditable before adding performance overlap.
