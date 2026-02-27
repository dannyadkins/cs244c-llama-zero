# Experiments

`harness.py` runs ZeRO experiment matrices and writes per-case logs, profiles, and summary JSON.

## Quick Start

```bash
python3 experiments/harness.py --config experiments/configs/week3_smoke_matrix.json
```

Outputs are written under `experiments/results/<name>/`:

- `summary.json`: run-level metadata + all case results
- `cases/<case_id>.json`: idempotent per-case result payload
- `logs/<case_id>.log`: stdout/stderr from `torchrun`
- `profiles/<case_id>.json`: per-step profiler output from rank 0 (default)

Each case JSON also includes a theoretical state-memory breakdown (`params_mb`, `grads_mb`, `optimizer_mb`, `total_mb`) computed from model size and ZeRO stage.

## CLI Matrix Example

```bash
python3 experiments/harness.py \
  --name week3_medium_bandwidth \
  --stages 0 1 2 3 \
  --model-sizes medium \
  --bandwidth-gbps 0 1 2.5 5 10 25 50 \
  --bandwidth-mode simulated \
  --nproc-per-node 2 \
  --steps 100
```

## Config-File Format

Harness config files are JSON with two top-level objects:

- `defaults`: scalar CLI overrides
- `matrix`: sweep axes (`stages`, `model_sizes`, `bandwidth_gbps`)

Example files:

- `experiments/configs/week3_smoke_matrix.json`
- `experiments/configs/week3_medium_bandwidth.json`

## Bandwidth Modes

- `simulated` (default): injects delay inside collective ops via
  - `ZERO_SIM_BW_GBPS`
  - `ZERO_SIM_LATENCY_MS`
- `tc`: calls `./infra/throttle.sh apply/delete` around each case
- `none`: no bandwidth manipulation

## Idempotency

Use `--skip-existing` to reuse completed per-case JSON records and rerun only missing/failed cases.
