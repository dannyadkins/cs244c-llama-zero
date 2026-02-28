# Experiments

`harness.py` runs ZeRO stage sweeps and writes structured JSON + per-stage logs.

## Week 2 Baseline Sweep

```bash
python experiments/harness.py \
  --name week2_baseline \
  --stages 0 1 2 \
  --nproc-per-node 2 \
  --steps 50 \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4
```

Outputs:

- `experiments/results/<name>/summary.json`
- `experiments/results/<name>/stage0.log`
- `experiments/results/<name>/stage1.log`
- `experiments/results/<name>/stage2.log`

The harness is idempotent with `--skip-existing`.
