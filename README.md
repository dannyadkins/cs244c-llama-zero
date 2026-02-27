# ZeRO From Scratch (Week 1 Complete)

This repo now includes both Week 1 tracks:

- **Person A**: LLaMA-style model + single-device training + FineWeb-Edu pipeline
- **Person B**: send/recv collectives + distributed correctness tests + profiling + launch/throttle infra

## Week 1 Deliverables

### Person A (training correctness baseline)

- LLaMA-style architecture (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`)
- Config presets for `tiny`, `small`, `medium`
- Single-device training entrypoint (`train.py`)
- FineWeb-Edu streaming and deterministic synthetic fallback
- Unit tests proving shape correctness, causality, gradient flow, and learning signal

### Person B (collectives + instrumentation)

- Ring collectives from `torch.distributed.send/recv`:
  - `allreduce`
  - `reduce_scatter`
  - `allgather`
- Distributed tests vs built-in torch collectives (`gloo`, multi-process)
- Profiling utilities:
  - CUDA-event timer with CPU fallback
  - memory snapshot/tracker
  - overlap efficiency utility
- Infra scripts for setup, `torchrun` launch, and Linux `tc` throttling

## Repository Layout

- `model/`
  - `config.py`
  - `llama.py`
- `data/`
  - `fineweb.py`
- `collectives/`
  - `_ring.py`
  - `ring_allreduce.py`
  - `ring_reduce_scatter.py`
  - `ring_allgather.py`
  - `interface.py`
  - `tests/test_collectives_distributed.py`
- `profiler/`
  - `timer.py`
  - `memory.py`
  - `overlap.py`
- `scripts/`
  - `distributed_sanity.py`
  - `benchmark_collectives.py`
- `infra/`
  - `setup.sh`
  - `launch.sh`
  - `throttle.sh`
- `tests/`
  - model/data/training/profiler unit tests

## Setup

```bash
cd /Users/danieladkins/cs244c-llama-zero
./infra/setup.sh
```

## Person A: Single-Device Training

### Offline sanity run (synthetic data)

```bash
python train.py \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --max-steps 100 \
  --log-interval 10
```

### FineWeb-Edu streaming run

```bash
python train.py \
  --data-mode fineweb \
  --tokenizer-name meta-llama/Llama-3.1-8B \
  --fineweb-subset sample-10BT \
  --model-size tiny \
  --seq-len 512 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --max-steps 500 \
  --log-interval 10
```

Notes:

- `meta-llama/Llama-3.1-8B` is gated on HuggingFace. If unavailable, use:
  - `--allow-synthetic-fallback`
  - or `--tokenizer-name gpt2`

## Person B: Distributed Collectives + Profiling

### Run all tests

```bash
pytest -q
```

### Run distributed collective correctness tests only

```bash
pytest -q collectives/tests/test_collectives_distributed.py
```

### Torchrun sanity check

```bash
./infra/launch.sh --nproc-per-node 2 --script scripts/distributed_sanity.py
```

### Benchmark ring vs torch collectives (JSON output)

```bash
torchrun --standalone --nproc_per_node=2 scripts/benchmark_collectives.py \
  --ops allreduce reduce_scatter allgather \
  --sizes 4096 65536 1048576 \
  --impl both \
  --iters 20 \
  --warmup 5 \
  --output-json experiments/results/week1_collectives.json
```

## Bandwidth Throttling (Linux)

```bash
# Apply egress cap
./infra/throttle.sh apply eth0 10gbit 1mb 10ms

# Inspect qdisc
./infra/throttle.sh status eth0

# Remove throttle
./infra/throttle.sh delete eth0
```

Validate with `iperf3` before and after throttling to confirm the cap is effective.

## Week 2 Integration Contract

Person A should treat communication as a black-box interface:

- `CollectiveOps.allreduce(tensor)`
- `CollectiveOps.reduce_scatter(tensor)`
- `CollectiveOps.allgather(local_shard)`

Use:

- `SendRecvCollectives` for custom implementation
- `TorchCollectives` as reference baseline

This keeps ZeRO stage wrappers decoupled from transport internals.
