# Experiments

`harness.py` is the main experiment runner for this repo. It launches ZeRO stage sweeps through `python -m torch.distributed.run`, writes per-case logs and profiles, and produces a machine-readable `summary.json` for plotting and evaluation.

This file is the authoritative experiment runbook. It replaces the old short project-status note and is intended to be enough to execute, validate, and interpret the full evaluation workflow.

## Goal

The project goal is simple:

- reproduce ZeRO stages 0, 1, 2, and 3 in this codebase
- validate them on real GPU hardware
- evaluate the tradeoff between memory savings and communication overhead

The evaluation should let you answer three questions clearly:

- how state memory changes from stage 0 to stage 3
- how communication cost changes from stage 0 to stage 3
- where each stage becomes impractical as bandwidth drops

## Current State

As of March 9, 2026, the repo is in a good state for single-host 2-GPU CUDA evaluation and for preparing real multi-host validation.

Already validated:

- ZeRO stages 0, 1, 2, and 3 are implemented
- stage 3 uses real module-wise sharding and rematerialization
- local CPU and Gloo correctness tests pass
- single-host 2-GPU CUDA runs work with `--collective-impl torch`
- the harness and plotting pipeline produce JSON summaries and figures
- measured state-memory instrumentation is wired end to end
- profiler-based TFLOPs logging is wired through training, harness parsing, and plotting

Not yet proven on real hardware:

- real multi-host NCCL runs across separate machines
- real `tc` shaping on the target setup
- final bandwidth crossover results with real throttling instead of simulated delay

Operational constraints:

- for CUDA runs, use `--collective-impl torch`
- do not use the custom `ring` backend for multi-GPU CUDA runs in this repo
- the CUDA ring send/recv path is not a valid final baseline here
- overlap metrics exist, but there is still no overlap on/off switch, so overlap is not yet a supported evaluation axis

## Outputs

Every harness run writes to `experiments/results/<name>/`:

- `summary.json`: run-level metadata and all case results
- `cases/<case_id>.json`: one result record per case
- `logs/<case_id>.log`: stdout and stderr from the launched distributed job
- `profiles/<case_id>.json`: rank-0 profile output by default

Each case result can include:

- final loss and number of logged steps
- mean tokens/s
- mean TFLOPs/s from profiler-measured operator FLOPs
- mean forward/backward, communication, and optimizer time
- measured peak host and CUDA memory
- measured state-memory breakdown when available
- theoretical ZeRO state-memory breakdown for params, grads, and optimizer state

## Metrics

The main metrics in this repo are:

- `tokens/s`: end-to-end training throughput per full optimization step
- `tflops`: profiler-derived training TFLOPs/s, measured from torch profiler operator FLOPs and normalized by iteration time
- `comm_ms`: average communication time per step
- measured state memory: state memory attributed to params, grads, and optimizer state
- peak memory: observed peak allocated or reserved memory during the run

Bandwidth in the standard local sweep is usually simulated. In that mode the harness sets `ZERO_SIM_BW_GBPS` and `ZERO_SIM_LATENCY_MS`, and the collective wrappers inject delay to emulate slower communication. A bandwidth value of `0` means no injected slowdown.

## Recommended Evaluation Order

Run the evaluation in this order.

### 1. Single-Host GPU Validation

Goal:

- prove stages 0 to 3 work under CUDA and NCCL before doing any expensive sweep

Run the same command four times with `--zero-stage 0`, `1`, `2`, and `3`:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_zero.py \
  --zero-stage 0 \
  --collective-impl torch \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 10 \
  --dtype bfloat16 \
  --profile-json experiments/results/zero_cuda_smoke.json \
  --profile-rank0-only \
  --profile-memory-interval 1
```

Pass criteria:

- no hang
- no NCCL error
- loss stays finite
- throughput is non-zero
- stage 3 uses less measured state memory than stage 0

If this fails:

- stop there
- retry with `--dtype float32`
- compare behavior against the same command with `--collective-impl torch` if you accidentally used another backend

### 2. First Real 2-GPU Sweep

Goal:

- get the first complete local GPU dataset for the project

Example baseline sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python experiments/harness.py \
  --name small_gpu_bw_sweep \
  --results-dir experiments/results \
  --stages 0 1 2 3 \
  --model-sizes small \
  --bandwidth-gbps 0 1 2 5 10 25 \
  --nproc-per-node 2 \
  --steps 30 \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --collective-impl torch \
  --data-mode synthetic \
  --dtype bfloat16 \
  --profile-memory-interval 1 \
  --bandwidth-mode simulated \
  --case-timeout-s 1800
```

Expected pattern:

- memory ordering should be `stage 0 > stage 1 > stage 2 > stage 3`
- lower bandwidth should hurt stage 3 the most
- communication time should rise as bandwidth drops

If the pattern is missing, debug before doing more runs.

### 3. Plot the Core Figures

Generate the most important figures from a completed run directory:

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot throughput \
  --output experiments/results/small_gpu_bw_sweep/throughput.png
```

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot tflops \
  --output experiments/results/small_gpu_bw_sweep/tflops.png
```

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot comm \
  --output experiments/results/small_gpu_bw_sweep/comm.png
```

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot memory \
  --bandwidth-gbps-filter 0 \
  --output experiments/results/small_gpu_bw_sweep/memory.png
```

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot peak-memory \
  --bandwidth-gbps-filter 0 \
  --output experiments/results/small_gpu_bw_sweep/peak_memory.png
```

Optional:

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot loss \
  --output experiments/results/small_gpu_bw_sweep/loss.png
```

If the figures do not look sensible, fix the issue before moving on.

### 4. Validate Real Multi-Host Launch

Goal:

- prove the same code path works across separate machines, not just on one host

Pass criteria:

- distributed sanity passes across hosts
- tiny stage 0 passes across hosts
- tiny stage 3 passes across hosts
- no rendezvous, interface, firewall, or NCCL timeout issue

Recommended first topology:

- 2 hosts
- 2 GPUs per host
- `nnodes=2`
- `nproc_per_node=2`

Host 0 sanity command:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=<host0-ip> \
  --master_port=29500 \
  scripts/distributed_sanity.py
```

Then tiny stage 0:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=<host0-ip> \
  --master_port=29501 \
  train_zero.py \
  --zero-stage 0 \
  --collective-impl torch \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 3 \
  --dtype bfloat16 \
  --profile-memory-interval 1
```

Repeat with `--zero-stage 3`, and run matching commands on host 1 with `--node_rank=1`.

Do not start expensive multi-host sweeps until all three checks pass.

### 5. Validate Real Bandwidth Throttling

Goal:

- prove whether `tc` actually changes throughput on the target setup

Server host:

```bash
python3 scripts/validate_bandwidth.py server --bind 0.0.0.0 --port 5201
```

Client host:

```bash
python3 scripts/validate_bandwidth.py validate \
  --target-host <server-ip> \
  --port 5201 \
  --device eth0 \
  --rate 10gbit \
  --duration-s 5 \
  --json-output /tmp/tc_validation.json
```

Decision rule:

- if shaped throughput clearly drops, `tc` is usable
- if it does not, use `--bandwidth-mode simulated` and state that clearly in the evaluation

Environment warning:

- local container runs are not enough to validate `tc`
- some container setups route local traffic over `lo`
- some container setups do not expose `CAP_NET_ADMIN`, so `tc` fails even as root

### 6. Run the Final Experiment Set

Only do this after multi-host launch and bandwidth validation are complete.

Priority order:

1. memory by stage at full bandwidth
2. throughput by stage at full bandwidth
3. throughput versus bandwidth sweep
4. torch collectives as the CUDA baseline
5. custom collectives only on CPU or after the CUDA ring path is actually fixed

Example host-0 command:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python experiments/harness.py \
  --name real_multihost_small_bw_sweep \
  --results-dir experiments/results \
  --stages 0 1 2 3 \
  --model-sizes small \
  --bandwidth-gbps 0 2 10 \
  --nproc-per-node 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <host0-ip> \
  --steps 20 \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --collective-impl torch \
  --data-mode synthetic \
  --dtype bfloat16 \
  --profile-memory-interval 1 \
  --bandwidth-mode simulated \
  --case-timeout-s 1800
```

Run the same command on host 1 with `--node-rank 1`. Only switch from simulated bandwidth to `tc` after `scripts/validate_bandwidth.py` proves real shaping works.

## Current Useful Local Artifacts

The repo already contains completed local 2-GPU results under `experiments/results/`.

Useful examples:

- `experiments/results/repro2gpu_suite/`
- `experiments/results/small_gpu_bw_sweep/`
- `experiments/results/small_gpu_bw_sweep_fine/`
- `experiments/results/small_gpu_bw_sweep_fine_profiled/`

Those are useful references for expected JSON structure and expected plot shapes.

## Quick Start

Config-driven run:

```bash
python3 experiments/harness.py --config experiments/configs/week3_smoke_matrix.json
```

Custom CLI sweep:

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

## Config Format

Harness config files are JSON with two top-level objects:

- `defaults`: scalar CLI overrides
- `matrix`: sweep axes such as `stages`, `model_sizes`, and `bandwidth_gbps`

Example files:

- `experiments/configs/week3_smoke_matrix.json`
- `experiments/configs/week3_medium_bandwidth.json`

## Bandwidth Modes

- `simulated`: injects delay inside collective calls via `ZERO_SIM_BW_GBPS` and `ZERO_SIM_LATENCY_MS`
- `tc`: calls `infra/throttle.sh apply` and `infra/throttle.sh delete` around each case
- `none`: runs without bandwidth manipulation

Use `simulated` for local single-host evaluation unless you have explicitly validated `tc` on a real multi-host path.

## Idempotency And Launch Notes

- use `--skip-existing` to reuse completed case JSON and rerun only missing or failed cases
- single-node runs default to `--master-addr 127.0.0.1`, which avoids brittle hostname-based rendezvous
- `--master-port-base` allocates one port per case as `base + case_index`
- `--case-timeout-s` prevents a single hung run from blocking the entire matrix

## What Not To Do

- do not run expensive multi-host sweeps before real tiny multi-host smoke tests pass
- do not trust `tc` results until `iperf3` confirms a real throughput drop
- do not use the CUDA `ring` backend for final experiments in the current repo state
- do not present overlap as an evaluated experiment axis until a real overlap toggle exists

## Done Means

The evaluation work is done when all of this is true:

- stages 0 to 3 are validated on real GPU hardware
- real multi-host launch works across separate machines
- bandwidth manipulation is validated with either real `tc` or a justified simulated fallback
- final figures exist for memory, throughput, TFLOPs, and communication sensitivity
- you can clearly explain where each ZeRO stage wins and where it becomes too communication-heavy
