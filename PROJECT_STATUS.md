# Project Status

Updated: March 7, 2026

This is the short version of where we are and what happens next.

## Where We Are

The codebase is in a good local state.

What already works:
- ZeRO stages 0, 1, 2, 3 are implemented.
- Stage 3 is real module-wise sharding/materialization.
- Local CPU/Gloo correctness tests pass.
- Stage 3 parity, dropout/recompute, and checkpoint/restore are tested.
- The harness runs sweeps and writes `summary.json`.
- The plotting code generates throughput, comm, measured-memory, and theoretical-memory figures.

What is still not proven:
- CUDA/NCCL
- multi-node training
- real `tc` throttling on the target cluster/cloud
- final bandwidth crossover results on a real GPU run

Simple summary:
- The implementation is mostly done.
- The validation and final experiments are not done.

## What Not To Do

Do not do these yet:
- big multi-node runs before single-node GPU passes
- report polishing before real figures exist
- fancy optimizations before the baseline GPU sweep works
- spending days on `tc` if simulated bandwidth already works

## What To Do Next

Do these in this exact order.

### 1. Validate stages 0-3 on one GPU machine

Goal:
- prove the code works on CUDA/NCCL before touching multi-node

Run this command 4 times, once with `--zero-stage 0`, `1`, `2`, and `3`.

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_zero.py \
  --zero-stage 0 \
  --collective-impl ring \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 10 \
  --dtype bfloat16 \
  --profile-json /tmp/zero_cuda_smoke.json \
  --profile-rank0-only \
  --profile-memory-interval 1
```

Pass criteria:
- no hang
- no NCCL error
- loss is finite
- throughput is non-zero
- stage 3 uses less memory than stage 0

If this fails:
- stop here
- try `--dtype float32`
- compare `--collective-impl ring` vs `--collective-impl torch`

### 2. Run the first real GPU sweep

Goal:
- get the first real dataset for the project

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python experiments/harness.py \
  --name small_gpu_bw_sweep \
  --results-dir /tmp/zero_harness \
  --stages 0 1 2 3 \
  --model-sizes small \
  --bandwidth-gbps 0 1 2 5 10 25 \
  --nproc-per-node 2 \
  --steps 30 \
  --seq-len 128 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --collective-impl ring \
  --data-mode synthetic \
  --dtype bfloat16 \
  --profile-memory-interval 1 \
  --case-timeout-s 1800
```

This should create:
- `/tmp/zero_harness/small_gpu_bw_sweep/summary.json`
- logs
- profiles

Expected pattern:
- memory: `stage 0 > 1 > 2 > 3`
- low bandwidth hurts stage 3 the most
- comm time goes up as bandwidth goes down

If that pattern is missing, debug before doing more runs.

### 3. Generate the four important figures

Run:

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir /tmp/zero_harness/small_gpu_bw_sweep \
  --plot throughput \
  --output /tmp/zero_harness/small_gpu_bw_sweep/throughput.png
```

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir /tmp/zero_harness/small_gpu_bw_sweep \
  --plot comm \
  --output /tmp/zero_harness/small_gpu_bw_sweep/comm.png
```

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir /tmp/zero_harness/small_gpu_bw_sweep \
  --plot memory \
  --bandwidth-gbps-filter 0 \
  --output /tmp/zero_harness/small_gpu_bw_sweep/memory.png
```

```bash
.venv/bin/python analysis/visualize.py \
  --run-dir /tmp/zero_harness/small_gpu_bw_sweep \
  --plot theory-memory \
  --bandwidth-gbps-filter 0 \
  --output /tmp/zero_harness/small_gpu_bw_sweep/theory_memory.png
```

If these look sensible, move on.
If they do not, fix the issue now.

### 4. Validate multi-node before any expensive run

Goal:
- prove cross-node launch works

Need to verify:
- simple distributed smoke works across nodes
- tiny stage-0 run works across nodes
- tiny stage-3 run works across nodes
- no port / interface / NCCL timeout issue

Do not start full experiments until a trivial cross-node run passes.

### 5. Validate bandwidth throttling

Goal:
- prove whether `tc` actually changes bandwidth on the real setup

Process:
1. run `iperf3` with no shaping
2. apply `tc`
3. run `iperf3` again
4. confirm throughput actually dropped

Decision:
- if `tc` works, use it
- if `tc` is unreliable, use simulated bandwidth and say so clearly in the report

### 6. Run the final experiment set

Priority order:
1. memory by stage
2. throughput at full bandwidth
3. throughput vs bandwidth sweep
4. custom collectives vs torch/NCCL
5. scaling over more GPUs/nodes if time remains

## What “Done” Means

We are done when all of this is true:
- stages 0-3 are validated on GPU
- multi-node works
- bandwidth manipulation is validated
- final figures exist
- we can clearly answer:
  - how memory changes from stage 0 to 3
  - how communication changes from stage 0 to 3
  - where each stage becomes impractical as bandwidth drops

## Team Split

Person A:
- own training correctness on GPU
- compare losses across stages
- debug stage-specific training issues

Person B:
- own multi-node bring-up
- own `tc` and `iperf3`
- own harness runs and plot generation

Shared:
- interpret the results
- decide whether final bandwidth results use real `tc` or simulated bandwidth
- write the evaluation section

## Immediate Next Action

Do this now:
1. Run single-node GPU validation for stages 0-3.
2. If that passes, run `small_gpu_bw_sweep`.
3. Generate the four figures.

That is the shortest path to a real result.
