# ZeRO From Scratch (Weeks 1-3)

This repository now contains the Week 1 foundation, Week 2 ZeRO stages 0-2 integration, and Week 3 Stage 3 + experiment harness/analysis tooling.

For the current experiment status, runbook, and remaining execution plan, read `experiments/README.md`.

## Implemented Scope

### Week 1 Person A

- LLaMA-style architecture in pure PyTorch (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`)
- `tiny`, `small`, `medium` model configs
- Single-device training script: `train.py`
- FineWeb-Edu streaming pipeline + synthetic fallback
- Baseline correctness tests for model/data/training

### Week 1 Person B

- Ring collectives from point-to-point `send/recv`
- Distributed correctness tests vs torch collective behavior
- Profiling utilities: timers, memory snapshots, overlap metrics
- Infra scripts for setup, launch, and `tc` throttling

### Week 2 (ZeRO 0-2)

- `zero/stage0_ddp.py`: replicated model + gradient allreduce + local AdamW
- `zero/stage1_optimizer.py`: sharded optimizer state + gradient allreduce + param allgather
- `zero/stage2_optimizer.py`: sharded optimizer state + gradient reduce-scatter + param allgather
- `train_zero.py`: distributed training entrypoint with profiling output

### Week 3 (ZeRO 3 + Harness)

- `zero/stage3_optimizer.py`: module-wise Stage 3 parameter sharding/materialization with backward recomputation
- `train_zero.py --zero-stage 3` integration
- `experiments/harness.py`: idempotent matrix runner for stage/model/bandwidth sweeps
- `experiments/run_remote_bandwidth_sweep.py`: ship the current repo to an isolated remote workspace, run a sweep, sync results back, and generate plots + a markdown report
- `experiments/run_fit_memory_bandwidth.py`: tune the largest per-stage microbatch under one GPU-memory budget, then sweep bandwidth with those tuned workloads
- `experiments/run_remote_fit_memory_bandwidth.py`: remote wrapper for the fit-to-memory workflow
- Linux socket-transport bandwidth shaping via `LD_PRELOAD` + forced NCCL `NET/Socket` on loopback, using a shared token bucket across all ranks in a case
- Legacy collective-delay simulation mode for quick debugging (`ZERO_SIM_BW_GBPS`, `ZERO_SIM_LATENCY_MS`)
- Optional `tc` throttling mode integration in harness
- Per-case measured peak memory extraction + theoretical state-memory breakdown (params/grads/optimizer by stage)
- `analysis/visualize.py`: plots for throughput/communication vs bandwidth, loss curves, and measured/theoretical memory
- `analysis/bandwidth_report.py`: markdown summary of best stage by bandwidth plus throughput/communication tables
- `scripts/benchmark_allreduce.py`: raw allreduce throughput benchmark for validating shaped transports

## Repository Layout

- `model/`: architecture + configs
- `data/`: streaming/tokenization/dataset utilities
- `collectives/`: ring collectives + communication interface + distributed tests
- `zero/`: ZeRO stage implementations and shared utilities
- `profiler/`: timing/memory/overlap instrumentation
- `infra/`: setup/launch/throttle scripts
- `scripts/`: distributed sanity + collective benchmarking
- `experiments/`: harness + configs + result artifacts
- `analysis/`: figure generation tools
- `report/`: writeup scaffold
- `tests/`: unit/integration tests

## Setup

```bash
cd /Users/danieladkins/cs244c-llama-zero
./infra/setup.sh
```

## Test Everything

```bash
pytest -q
```

Distributed-focused tests:

```bash
pytest -q collectives/tests/test_collectives_distributed.py
pytest -q zero/tests/test_zero_stages.py
```

## Baseline Training (Single GPU / CPU)

```bash
python3 train.py \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --max-steps 100
```

## ZeRO Training (Stages 0-3)

```bash
python3 -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_zero.py \
  --zero-stage 3 \
  --collective-impl ring \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 50 \
  --profile-json experiments/results/stage3_profile.json \
  --profile-rank0-only
```

## Week 3 Harness

Run a config-driven matrix:

```bash
python3 experiments/harness.py --config experiments/configs/remote_4gpu_bandwidth_smoke.json
```

Run a custom sweep directly from CLI:

```bash
python3 experiments/harness.py \
  --name week3_medium_bandwidth \
  --stages 0 1 2 3 \
  --model-sizes medium \
  --bandwidth-gbps 0 1 2.5 5 10 25 50 \
  --bandwidth-mode socket \
  --socket-interface lo \
  --nproc-per-node 2 \
  --steps 100
```

Optional real throttling mode (Linux root/sudo required):

```bash
python3 experiments/harness.py \
  --name week3_tc_trial \
  --stages 2 3 \
  --model-sizes tiny \
  --bandwidth-gbps 5 10 \
  --bandwidth-mode tc \
  --tc-interface eth0
```

Run the current branch on a remote GPU machine in an isolated workspace and pull the finished run back locally:

```bash
python3 experiments/run_remote_bandwidth_sweep.py \
  --host 184.144.213.79 \
  --port 40787 \
  --config experiments/configs/remote_4gpu_small_bandwidth_socket.json \
  --overwrite-local
```

Run the fit-to-memory experiment, which first tunes the largest per-stage microbatch under a fixed GPU-memory budget and then reruns the bandwidth sweep with those tuned microbatches:

```bash
python3 experiments/run_remote_fit_memory_bandwidth.py \
  --host 184.144.213.79 \
  --port 40787 \
  --config experiments/configs/remote_4gpu_small_fit_memory_socket.json \
  --overwrite-local
```

Use the simulated fast config only for coarse debugging when socket shaping is unavailable:

```bash
python3 experiments/run_remote_bandwidth_sweep.py \
  --host 184.144.213.79 \
  --port 40787 \
  --config experiments/configs/remote_4gpu_small_bandwidth_fast.json \
  --overwrite-local
```

## Visualization

Throughput vs bandwidth:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot throughput \
  --output analysis/figures/week3_throughput_vs_bandwidth.png
```

Communication vs bandwidth:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot comm \
  --output analysis/figures/week3_comm_vs_bandwidth.png
```

Loss curves:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot loss \
  --output analysis/figures/week3_loss_curves.png
```

Measured peak memory by stage:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot memory \
  --bandwidth-gbps-filter 0 \
  --output analysis/figures/week3_peak_memory.png
```

Theoretical state memory by stage:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/week3_medium_bandwidth \
  --plot theory-memory \
  --bandwidth-gbps-filter 0 \
  --output analysis/figures/week3_theory_memory.png
```

## Bandwidth Control (`tc`)

```bash
./infra/throttle.sh apply eth0 10gbit 1mb 10ms
./infra/throttle.sh status eth0
./infra/throttle.sh delete eth0
```

Validate applied throttling with `iperf3` before collecting data.

For a reproducible `tc` + `iperf3` check, run an iperf server on the peer host:

```bash
python3 scripts/validate_bandwidth.py server --bind 0.0.0.0 --port 5201
```

Then run the validation from the client host:

```bash
python3 scripts/validate_bandwidth.py validate \
  --target-host <peer-ip> \
  --port 5201 \
  --device eth0 \
  --rate 10gbit \
  --duration-s 5 \
  --json-output /tmp/tc_validation.json
```

This script records baseline throughput, applies `tc`, measures shaped throughput, clears `tc`, and fails if the shaped bandwidth does not drop enough. Use two hosts or two VMs connected through the interface you plan to shape; traffic to the local machine's own IP will typically route through `lo` and is not a meaningful `eth0` validation. In containers, root may still be insufficient if the runtime does not grant `CAP_NET_ADMIN`.

## Communication Interface Contract

ZeRO wrappers treat communication as a black box:

- `allreduce(tensor, average=True)`
- `reduce_scatter(tensor)`
- `allgather(local_shard)`

Implementations:

- `SendRecvCollectives`: custom ring transport
- `TorchCollectives`: torch distributed baseline
- `LocalCollectives`: single-process fallback
