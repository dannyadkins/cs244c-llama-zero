# Analysis

The analysis tools consume `experiments/results/<run-name>/summary.json` and turn completed sweeps into figures and markdown summaries.

## Main Entry Points

- [`visualize.py`](/home/thomason/github/cs244c-llama-zero/analysis/visualize.py): plot throughput, communication, loss, and memory views
- [`bandwidth_report.py`](/home/thomason/github/cs244c-llama-zero/analysis/bandwidth_report.py): generate a markdown report from one run directory

## Common Plots

Throughput vs bandwidth:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/<run-name> \
  --plot throughput \
  --output experiments/results/<run-name>/throughput.png
```

Communication vs bandwidth:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/<run-name> \
  --plot comm \
  --output experiments/results/<run-name>/comm.png
```

TFLOPs vs bandwidth:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/<run-name> \
  --plot tflops \
  --output experiments/results/<run-name>/tflops.png
```

Measured state memory:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/<run-name> \
  --plot memory \
  --bandwidth-gbps-filter 0 \
  --output experiments/results/<run-name>/memory.png
```

Peak memory:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/<run-name> \
  --plot peak-memory \
  --bandwidth-gbps-filter 0 \
  --output experiments/results/<run-name>/peak_memory.png
```

Loss curves:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/<run-name> \
  --plot loss \
  --output experiments/results/<run-name>/loss.png
```

## Plot Requirements

Memory-oriented plots are most useful when the underlying run was collected with:

```bash
--profile-memory-interval 1
```

That enables step-level memory snapshots and measured ZeRO state-memory breakdowns in the profile JSON.

## Markdown Summary

Generate a report directly from the run directory:

```bash
python3 analysis/bandwidth_report.py \
  --run-dir experiments/results/<run-name> \
  --output experiments/results/<run-name>/bandwidth_report.md
```

The generated report summarizes:

- methodology
- per-stage workload
- bandwidth-by-bandwidth ranking
- throughput and communication tables
- fit-to-memory tuning details when present
