# Experiments

`harness.py` runs ZeRO experiment matrices and writes per-case logs, profiles, and summary JSON.
It launches distributed jobs through the current Python interpreter (`python -m torch.distributed.run`) with explicit master address/port assignment, so it does not depend on a separate `torchrun` binary being on `PATH`.

## Quick Start

```bash
python3 experiments/harness.py --config experiments/configs/week3_smoke_matrix.json
```

Outputs are written under `experiments/results/<name>/`:

- `summary.json`: run-level metadata + all case results
- `cases/<case_id>.json`: idempotent per-case result payload
- `logs/<case_id>.log`: stdout/stderr from the launched distributed job
- `profiles/<case_id>.json`: per-step profiler output from rank 0 (default)

Each case JSON also includes a theoretical state-memory breakdown (`params_mb`, `grads_mb`, `optimizer_mb`, `total_mb`) computed from model size and ZeRO stage.
If the training command records memory snapshots (`--profile-memory-interval > 0`), the harness also extracts peak host/CUDA memory fields into each case result in `summary.json`.

## CLI Matrix Example

```bash
python3 experiments/harness.py \
  --name week3_medium_bandwidth \
  --stages 0 1 2 3 \
  --model-sizes medium \
  --bandwidth-gbps 0 1 2.5 5 10 25 50 \
  --bandwidth-mode simulated \
  --nproc-per-node 2 \
  --master-addr 127.0.0.1 \
  --master-port-base 29500 \
  --case-timeout-s 1800 \
  --profile-memory-interval 1 \
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

## Launch Notes

- Single-node runs default to `--master-addr 127.0.0.1`, which avoids brittle hostname-based rendezvous.
- `--master-port-base` allocates one port per case in the sweep (`base + case_index`).
- `--case-timeout-s` marks hung cases as failed instead of blocking the full matrix indefinitely.
