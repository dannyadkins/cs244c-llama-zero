# Experiments

[`harness.py`](/home/thomason/github/cs244c-llama-zero/experiments/harness.py) is the main experiment runner. It launches ZeRO training jobs, records per-case outputs, and writes a run-level `summary.json` that downstream analysis tools consume.

## What The Harness Does

For each case in a stage/model/bandwidth matrix, the harness:

1. builds a `torch.distributed.run` command for [`train_zero.py`](/home/thomason/github/cs244c-llama-zero/train_zero.py)
2. configures the requested bandwidth mode
3. captures logs and optional profiles
4. parses step-level metrics from stdout
5. writes case JSON plus an aggregated `summary.json`

The harness is designed to be idempotent and scriptable, which makes it suitable for parameter sweeps, regression checks, and paper-quality figure generation.

## Result Layout

Each run lives under `experiments/results/<run-name>/`:

- `summary.json`: run arguments plus all parsed case results
- `cases/<case-id>.json`: one serialized case result
- `logs/<case-id>.log`: stdout and stderr from the launched job
- `profiles/<case-id>.json`: optional profile output

Remote runners sync the same structure back under `experiments/results/remote/<remote-label>/<run-name>/` and may also write:

- `bandwidth_report.md`
- `remote_run_metadata.json`
- plot images such as `throughput.png`, `comm.png`, `memory.png`, and `loss.png`

## Recommended Workflow

### 1. Local smoke test

Validate that the distributed path works before launching a sweep:

```bash
python3 -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=<master-addr> \
  --master_port=<master-port> \
  scripts/distributed_sanity.py
```

Then run a short ZeRO check:

```bash
python3 -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=<master-addr> \
  --master_port=<master-port> \
  train_zero.py \
  --zero-stage 0 \
  --collective-impl torch \
  --data-mode synthetic \
  --model-size tiny \
  --seq-len 128 \
  --batch-size 4 \
  --max-steps 10
```

Repeat with `--zero-stage 1`, `2`, and `3` before trusting a larger matrix.

### 2. Fixed-workload sweep

Use a checked-in config:

```bash
python3 experiments/harness.py \
  --config experiments/configs/remote_4gpu_small_bandwidth_socket.json
```

Or specify the matrix directly:

```bash
python3 experiments/harness.py \
  --name small_gpu_bw_sweep \
  --stages 0 1 2 3 \
  --model-sizes small \
  --bandwidth-gbps 0 1 2 5 10 \
  --nproc-per-node 2 \
  --steps 30 \
  --seq-len 128 \
  --batch-size 4 \
  --collective-impl torch \
  --data-mode synthetic \
  --dtype bfloat16 \
  --bandwidth-mode socket \
  --socket-interface lo \
  --profile-memory-interval 1 \
  --metrics-warmup-steps 2
```

This is the cleanest comparison when every stage should process the same workload.

### 3. Fit-to-memory sweep

Use the tuning workflow when you want each stage to exploit its memory savings:

```bash
python3 experiments/run_fit_memory_bandwidth.py \
  --config experiments/configs/remote_4gpu_small_fit_memory_socket.json
```

This workflow first finds the largest stable microbatch for each stage under a shared memory criterion, then reruns the bandwidth sweep with those tuned workloads.

### 4. Remote execution

Run the exact repository state on a remote machine and collect the finished artifacts back locally:

```bash
python3 experiments/run_remote_bandwidth_sweep.py \
  --host <remote-host> \
  --port <ssh-port> \
  --config experiments/configs/remote_4gpu_small_bandwidth_socket.json \
  --overwrite-local
```

Fit-to-memory has a parallel wrapper:

```bash
python3 experiments/run_remote_fit_memory_bandwidth.py \
  --host <remote-host> \
  --port <ssh-port> \
  --config experiments/configs/remote_4gpu_small_fit_memory_socket.json \
  --overwrite-local
```

## Config Files

Experiment configs live in [`experiments/configs/`](/home/thomason/github/cs244c-llama-zero/experiments/configs). They define:

- default harness arguments
- the stage/model/bandwidth matrix
- reusable sweep presets for smoke tests, bandwidth studies, fit-to-memory studies, and scaling runs

## Bandwidth Modes

The harness supports four modes:

- `none`: no throttling
- `simulated`: injects delay in the collective path for quick debugging
- `socket`: forces NCCL onto `NET/Socket` and applies socket-level shaping
- `tc`: applies Linux `tc` shaping to a network interface

Use `socket` for single-host multi-GPU experiments when you want real communication to pass through a shaped socket transport. Use `tc` only when you control the target interface and have the required privileges.

## Interpreting Metrics

Important fields in `summary.json`:

- `mean_tokens_per_s`: end-to-end training throughput
- `mean_tflops_per_s`: throughput in TFLOPs/s
- `mean_comm_ms`: communication time attributed per step
- `mean_fb_ms`: forward/backward step time attribution
- `peak_cuda_*` and `peak_host_rss_mb`: observed peak memory
- `measured_state_memory_mb`: measured parameter, gradient, and optimizer memory
- `theoretical_memory_mb`: analytical memory estimate for the selected stage

## Typical Validation Sequence

1. Run `scripts/distributed_sanity.py`.
2. Run a short ZeRO smoke test for each stage.
3. Launch one fixed-workload sweep.
4. Plot throughput, communication, and memory.
5. If the patterns are sensible, move on to fit-to-memory or remote runs.

## After The Run

Use the analysis tools directly on a completed run directory:

```bash
python3 analysis/visualize.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --plot throughput \
  --output experiments/results/small_gpu_bw_sweep/throughput.png
```

```bash
python3 analysis/bandwidth_report.py \
  --run-dir experiments/results/small_gpu_bw_sweep \
  --output experiments/results/small_gpu_bw_sweep/bandwidth_report.md
```
