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

## Measured State Memory

Use `--profile-memory-interval 1` (or higher) when collecting the run so the harness writes memory snapshots into each profile JSON. The training profile now also exports a measured ZeRO state-memory breakdown (`params`, `grads`, `optimizer`) from the engine internals, which is what `--plot memory` renders.

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot memory \
  --bandwidth-gbps-filter 0 \
  --output analysis/figures/week3_measured_state_memory.png
```

## Measured Peak Memory

Use `--profile-memory-interval 1` (or higher) when collecting the run so the harness writes memory snapshots into each profile JSON.

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot peak-memory \
  --bandwidth-gbps-filter 0 \
  --output analysis/figures/week3_peak_memory.png
```

Optional filters:

- `--model-size tiny|small|medium`
- `--bandwidth-gbps-filter <value>`
