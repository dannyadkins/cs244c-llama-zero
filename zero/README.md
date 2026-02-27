# ZeRO Optimizers (Week 2)

Implemented stages:

- `stage0_ddp.py`: full replication + gradient allreduce + local AdamW step
- `stage1_optimizer.py`: optimizer-state sharding + gradient allreduce + parameter allgather
- `stage2_optimizer.py`: optimizer-state sharding + gradient reduce-scatter + parameter allgather

Design goals:

- Clear flat-parameter metadata in `common.py`
- Explicit communication boundaries through `CollectiveOps`
- Minimal hidden magic so stage transitions are easy to audit

Usage entrypoint:

- `train_zero.py` for multi-process stage 0/1/2 training

Example:

```bash
torchrun --standalone --nproc_per_node=2 train_zero.py \
  --zero-stage 2 \
  --collective-impl ring \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 50
```
