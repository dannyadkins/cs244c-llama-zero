# Analysis

Generate plots directly from `experiments/results/<run>/summary.json`.

## Throughput vs Bandwidth

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot throughput \
  --output analysis/figures/week3_throughput_vs_bandwidth.png
```

## Communication vs Bandwidth

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot comm \
  --output analysis/figures/week3_comm_vs_bandwidth.png
```

## Loss Curves

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot loss \
  --output analysis/figures/week3_loss_curves.png
```

Optional filter:

- `--model-size tiny|small|medium`
