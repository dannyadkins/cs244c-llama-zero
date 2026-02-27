# ZeRO From Scratch (Weeks 1-3)

This repository now contains the Week 1 foundation, Week 2 ZeRO stages 0-2 integration, and Week 3 Stage 3 + experiment harness/analysis tooling.

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

- `zero/stage3_optimizer.py`: correctness-first Stage 3 communication pattern
- `train_zero.py --zero-stage 3` integration
- `experiments/harness.py`: idempotent matrix runner for stage/model/bandwidth sweeps
- Simulated bandwidth mode via env-driven collective delay (`ZERO_SIM_BW_GBPS`, `ZERO_SIM_LATENCY_MS`)
- Optional `tc` throttling mode integration in harness
- Per-case theoretical state-memory breakdown (params/grads/optimizer by stage)
- `analysis/visualize.py`: plots for throughput/communication vs bandwidth and loss curves

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
torchrun --standalone --nproc_per_node=2 train_zero.py \
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
python3 experiments/harness.py --config experiments/configs/week3_smoke_matrix.json
```

Run a custom sweep directly from CLI:

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

## Bandwidth Control (`tc`)

```bash
./infra/throttle.sh apply eth0 10gbit 1mb 10ms
./infra/throttle.sh status eth0
./infra/throttle.sh delete eth0
```

Validate applied throttling with `iperf3` before collecting data.

## Communication Interface Contract

ZeRO wrappers treat communication as a black box:

- `allreduce(tensor, average=True)`
- `reduce_scatter(tensor)`
- `allgather(local_shard)`

Implementations:

- `SendRecvCollectives`: custom ring transport
- `TorchCollectives`: torch distributed baseline
- `LocalCollectives`: single-process fallback
