# ZeRO From Scratch (Weeks 1-2)

This repository now includes:

- Week 1 Person A baseline (model, data, single-device training correctness)
- Week 1 Person B foundation (custom collectives, profiling, infra)
- Week 2 ZeRO stages 0-2 implementation and distributed training entrypoint

## Implemented Scope

### Week 1 Person A

- LLaMA-style architecture in pure PyTorch (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`)
- `tiny`, `small`, `medium` model configs
- Single-device training script: `train.py`
- FineWeb-Edu streaming pipeline + synthetic fallback
- Correctness tests for model/data/training

### Week 1 Person B

- Ring collectives from point-to-point primitives
- Distributed correctness tests vs torch reference behavior
- Profiling utilities: timer, memory tracker, overlap metric
- Infra scripts for setup, launch, and Linux `tc` throttling

### Week 2 (ZeRO 0-2)

- `zero/stage0_ddp.py`: allreduce gradients + replicated optimizer
- `zero/stage1_optimizer.py`: partition optimizer states + allreduce gradients + allgather params
- `zero/stage2_optimizer.py`: partition optimizer states + reduce-scatter gradients + allgather params
- `train_zero.py`: distributed training for ZeRO stages 0/1/2
- Distributed tests validating stage updates against full global-batch AdamW reference

## Repository Layout

- `model/`: architecture + configs
- `data/`: streaming/tokenization/dataset utilities
- `collectives/`: ring collectives + communication interface + distributed tests
- `zero/`: ZeRO stage implementations and shared utilities
- `profiler/`: timing/memory/overlap instrumentation
- `infra/`: setup/launch/throttle scripts
- `scripts/`: distributed sanity + collective benchmarking
- `tests/`: unit tests for model/data/training/profiler
- `experiments/`: configs/results scaffolding
- `analysis/`: figure scaffolding
- `report/`: writeup scaffolding

## Setup

```bash
cd /Users/danieladkins/cs244c-llama-zero
./infra/setup.sh
```

## Run Tests

```bash
pytest -q
```

Key distributed tests:

```bash
pytest -q collectives/tests/test_collectives_distributed.py
pytest -q zero/tests/test_zero_stages.py
```

## Week 1 Baseline Training

Synthetic sanity:

```bash
python train.py \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --max-steps 100
```

FineWeb-Edu streaming:

```bash
python train.py \
  --data-mode fineweb \
  --tokenizer-name meta-llama/Llama-3.1-8B \
  --fineweb-subset sample-10BT \
  --model-size tiny \
  --seq-len 512 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --max-steps 500
```

If tokenizer access is gated, use `--allow-synthetic-fallback` or `--tokenizer-name gpt2`.

## Week 2 ZeRO Training

Run stage 0/1/2 with 2 processes on one node:

```bash
torchrun --standalone --nproc_per_node=2 train_zero.py \
  --zero-stage 0 \
  --collective-impl ring \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 50
```

```bash
torchrun --standalone --nproc_per_node=2 train_zero.py \
  --zero-stage 1 \
  --collective-impl ring \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 50
```

```bash
torchrun --standalone --nproc_per_node=2 train_zero.py \
  --zero-stage 2 \
  --collective-impl ring \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 50
```

## Distributed Sanity and Benchmarking

Sanity check:

```bash
./infra/launch.sh --nproc-per-node 2 --script scripts/distributed_sanity.py
```

Collective benchmark to JSON:

```bash
torchrun --standalone --nproc_per_node=2 scripts/benchmark_collectives.py \
  --ops allreduce reduce_scatter allgather \
  --sizes 4096 65536 1048576 \
  --impl both \
  --iters 20 \
  --warmup 5 \
  --output-json experiments/results/week1_collectives.json
```

## Week 2 Harness and Plotting

Run a staged ZeRO sweep and save logs + summary JSON:

```bash
python experiments/harness.py \
  --name week2_baseline \
  --stages 0 1 2 \
  --nproc-per-node 2 \
  --steps 50 \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4
```

Plot loss curves from the run:

```bash
python analysis/visualize.py \
  --run-dir experiments/results/week2_baseline \
  --output analysis/figures/week2_loss_curves.png
```

## Bandwidth Throttling (Linux)

```bash
./infra/throttle.sh apply eth0 10gbit 1mb 10ms
./infra/throttle.sh status eth0
./infra/throttle.sh delete eth0
```

Validate with `iperf3` before and after throttling.

## Week 2 Interface Contract

ZeRO wrappers use communication as a black-box API:

- `allreduce(tensor, average=True)`
- `reduce_scatter(tensor)`
- `allgather(local_shard)`

Implementations:

- `SendRecvCollectives`: custom ring transport
- `TorchCollectives`: built-in torch baseline
- `LocalCollectives`: single-process fallback
