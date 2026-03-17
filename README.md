# LLaMA ZeRO Research Harness

This repository is a from-scratch PyTorch implementation of LLaMA-style training plus a research harness for studying how ZeRO stages 0-3 trade memory savings against communication cost.

The codebase is organized for reproducible systems experiments:

- a baseline single-process trainer in [`train.py`](/home/thomason/github/cs244c-llama-zero/train.py)
- distributed ZeRO training in [`train_zero.py`](/home/thomason/github/cs244c-llama-zero/train_zero.py)
- config-driven experiment orchestration in [`experiments/harness.py`](/home/thomason/github/cs244c-llama-zero/experiments/harness.py)
- plotting and markdown report generation in [`analysis/`](/home/thomason/github/cs244c-llama-zero/analysis)

## Research Focus

The project is built to answer three questions:

1. How much model-state memory does each ZeRO stage save in practice?
2. How much communication overhead does each stage introduce as bandwidth drops?
3. When stages are allowed to use their memory savings for larger microbatches, how does the throughput ranking change?

## What Is Implemented

- LLaMA-style language model components in pure PyTorch: RMSNorm, RoPE, grouped-query attention, and SwiGLU
- ZeRO stages 0, 1, 2, and 3 with explicit communication boundaries
- synthetic and FineWeb-based training data paths
- a bandwidth-sensitive experiment harness with local, single-node, and remote workflows
- measurement plumbing for throughput, TFLOPs, communication time, and memory
- plotting and markdown reporting utilities for completed runs

## Repository Map

- [`model/`](/home/thomason/github/cs244c-llama-zero/model): model configs and LLaMA-style modules
- [`data/`](/home/thomason/github/cs244c-llama-zero/data): tokenization and dataset utilities
- [`collectives/`](/home/thomason/github/cs244c-llama-zero/collectives): communication interfaces and ring collectives
- [`zero/`](/home/thomason/github/cs244c-llama-zero/zero): ZeRO stage implementations
- [`profiler/`](/home/thomason/github/cs244c-llama-zero/profiler): step timing, FLOP estimation, and memory tracking
- [`experiments/`](/home/thomason/github/cs244c-llama-zero/experiments): experiment harness, configs, and remote runners
- [`analysis/`](/home/thomason/github/cs244c-llama-zero/analysis): plots and report generation
- [`report/`](/home/thomason/github/cs244c-llama-zero/report): research notes and synthesized findings
- [`tests/`](/home/thomason/github/cs244c-llama-zero/tests): unit and integration coverage

## Code Logic

The core control flow is straightforward:

1. [`train.py`](/home/thomason/github/cs244c-llama-zero/train.py) builds a model, loads synthetic or streamed text data, and runs a standard single-process training loop.
2. [`train_zero.py`](/home/thomason/github/cs244c-llama-zero/train_zero.py) initializes distributed state, selects a ZeRO engine, and records step-level metrics such as `tokens/s`, `comm_ms`, and optional memory traces.
3. [`experiments/harness.py`](/home/thomason/github/cs244c-llama-zero/experiments/harness.py) enumerates a matrix over stages, model sizes, and bandwidth settings, launches `torch.distributed.run`, and writes one result record per case plus a run-level `summary.json`.
4. [`analysis/visualize.py`](/home/thomason/github/cs244c-llama-zero/analysis/visualize.py) reads `summary.json` and produces throughput, communication, loss, and memory figures.
5. [`analysis/bandwidth_report.py`](/home/thomason/github/cs244c-llama-zero/analysis/bandwidth_report.py) turns a completed run directory into a markdown summary suitable for a report appendix or experiment log.

## Setup

Install Python dependencies from the repository root:

```bash
./infra/setup.sh
```

Or install them directly:

```bash
python3 -m pip install -r requirements.txt
```

## Tests

Run the full suite:

```bash
pytest -q
```

Focused distributed coverage:

```bash
pytest -q zero/tests/test_zero_stages.py
pytest -q collectives/tests/test_collectives_distributed.py
```

## Quick Start

Single-process baseline:

```bash
python3 train.py \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --max-steps 100
```

Two-process ZeRO smoke test:

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
  --max-steps 20 \
  --profile-json experiments/results/stage3_smoke/profile.json \
  --profile-rank0-only
```

For single-node runs, `<master-addr>` is typically loopback on the launch host and `<master-port>` should be any free rendezvous port on that machine.

## Running Experiments

Config-driven bandwidth sweep:

```bash
python3 experiments/harness.py \
  --config experiments/configs/remote_4gpu_small_bandwidth_socket.json
```

Equivalent CLI-driven sweep:

```bash
python3 experiments/harness.py \
  --name small_gpu_bw_sweep \
  --results-dir experiments/results \
  --stages 0 1 2 3 \
  --model-sizes small \
  --bandwidth-gbps 0 1 2 5 10 \
  --nproc-per-node 2 \
  --steps 30 \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --collective-impl torch \
  --data-mode synthetic \
  --dtype bfloat16 \
  --profile-memory-interval 1 \
  --bandwidth-mode socket \
  --socket-interface lo
```

Fit-to-memory workflow:

```bash
python3 experiments/run_fit_memory_bandwidth.py \
  --config experiments/configs/remote_4gpu_small_fit_memory_socket.json
```

Remote execution and result collection:

```bash
python3 experiments/run_remote_bandwidth_sweep.py \
  --host <remote-host> \
  --port <ssh-port> \
  --config experiments/configs/remote_4gpu_small_bandwidth_socket.json \
  --overwrite-local
```

The remote runners archive the current repository state, execute in a clean remote workspace, pull the finished run directory back into `experiments/results/remote/`, and optionally generate plots and a markdown report.

More operational detail lives in [`experiments/README.md`](/home/thomason/github/cs244c-llama-zero/experiments/README.md).

## Result Artifacts

Each harness run writes a self-contained directory under `experiments/results/<run-name>/`:

- `summary.json`: aggregate metadata plus the result record for every case
- `cases/*.json`: one machine-readable result per launched case
- `logs/*.log`: captured stdout and stderr from the training job
- `profiles/*.json`: rank-0 profile data and memory snapshots when enabled

Key metrics include:

- `mean_tokens_per_s`: end-to-end training throughput
- `mean_tflops_per_s`: training throughput derived from profiler or estimator output
- `mean_comm_ms`: mean communication time attributed per step
- `measured_state_memory_mb`: measured parameter, gradient, and optimizer memory when available
- peak host or CUDA memory fields: observed allocation behavior during the run

## Analysis And Reporting

Generate a throughput plot:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot throughput \
  --output experiments/results/small_gpu_bw_sweep/throughput.png
```

Generate a communication plot:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot comm \
  --output experiments/results/small_gpu_bw_sweep/comm.png
```

Generate a markdown summary:

```bash
python3 analysis/bandwidth_report.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --output experiments/results/small_gpu_bw_sweep/bandwidth_report.md
```

Additional plotting notes are in [`analysis/README.md`](/home/thomason/github/cs244c-llama-zero/analysis/README.md).

## Notes On Bandwidth Control

The repository supports multiple bandwidth modes:

- `none`: no throttling
- `simulated`: artificial delay inside the collective layer
- `socket`: socket-level shaping via `LD_PRELOAD` for NCCL `NET/Socket`
- `tc`: Linux traffic-control shaping on a chosen interface

For interface-level validation, use [`scripts/validate_bandwidth.py`](/home/thomason/github/cs244c-llama-zero/scripts/validate_bandwidth.py) with a user-selected bind address and service port appropriate for your test environment.

## Reproducibility

This repository is intended to be reusable across clusters and personal workstations. Documentation examples therefore use placeholders for network addresses, rendezvous ports, remote hosts, and filesystem roots rather than lab-specific infrastructure details.
