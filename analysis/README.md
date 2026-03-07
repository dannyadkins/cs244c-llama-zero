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

## Measured Peak Memory

Use `--profile-memory-interval 1` (or higher) when collecting the run so the harness writes memory snapshots into each profile JSON.

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot memory \
  --bandwidth-gbps-filter 0 \
  --output analysis/figures/week3_peak_memory.png
```

## Theoretical State Memory

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot theory-memory \
  --bandwidth-gbps-filter 0 \
  --output analysis/figures/week3_theory_memory.png
```

Optional filters:

- `--model-size tiny|small|medium`
- `--bandwidth-gbps-filter <value>`
