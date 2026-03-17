# Textbook ZeRO Rerun Report

## Scope

This report records the first bandwidth results after the ZeRO stage implementations were cleaned up to match the paper more closely.

It covers two questions:

1. Fixed workload: with the same microbatch on every stage, what is the throughput ordering?
2. Fit-to-memory at very low bandwidth: if stages 2 and 3 are each tuned to their own OOM boundary, is there a stage-2 / stage-3 crossover?

The fixed-workload section is complete below. The low-bandwidth fit-to-memory section is being filled from the current remote rerun:

- `experiments/results/remote/<cluster-a>/remote_4gpu_small_fit_memory_socket_stage23_currentpaper_lowbw_oom1024`

## What Changed

These runs should not be compared directly against the older stage-2 / stage-3 bandwidth conclusions, because the implementation changed in ways that matter for both communication volume and memory.

Main changes:

- `train_zero.py` and `train.py` now instantiate runtime model parameters in the requested training dtype, so `--dtype bfloat16` actually means bf16 model weights on GPU instead of fp32 weights plus autocast.
- `zero/stage0_ddp.py` and `zero/stage1_optimizer.py` now keep fp32 master weights and optimizer state, but communicate gradients in the runtime model dtype.
- `zero/stage2_optimizer.py` was rewritten to use a textbook-style flat-gradient reduce-scatter path instead of the earlier bucketed path that could fall back to allreduce and distort the communication comparison.
- `zero/stage3_optimizer.py` now materializes parameter shards and communicates them in the runtime model dtype while keeping optimizer state in fp32.

The practical effect is:

- stage 2 now behaves much closer to theoretical ZeRO-2
- stage 3 no longer pays unnecessary fp32 parameter traffic
- the old “stage 2 is pathologically slow” result is obsolete

## Fixed-Workload Rerun

Run:

- `experiments/results/remote/<cluster-a>/remote_4gpu_small_bandwidth_socket_currentpaper`

Method:

- model: `small`
- sequence length: `128`
- per-GPU microbatch: `4`
- grad accumulation: `1`
- world size: `4`
- bandwidth mode: socket throttling over forced NCCL `NET/Socket`
- steps: `12`
- metrics warmup skipped: `2`

Every stage processes the same `2048` tokens per step, so this isolates communication and recomputation overhead rather than memory-enabled scaling.

### Fixed-Workload Throughput

| stage | unlimited | 1 Gbps | 2 Gbps | 5 Gbps | 10 Gbps |
| --- | --- | --- | --- | --- | --- |
| 0 | 5193.7 | 1871.7 | 2474.2 | 3370.0 | 4368.9 |
| 1 | 3823.2 | 1297.7 | 1721.7 | 2368.5 | 3082.7 |
| 2 | 5155.2 | 1890.1 | 2456.5 | 3406.2 | 4363.8 |
| 3 | 3519.2 | 1189.8 | 1580.6 | 2155.7 | 2828.1 |

### Fixed-Workload Communication Time

| stage | unlimited | 1 Gbps | 2 Gbps | 5 Gbps | 10 Gbps |
| --- | --- | --- | --- | --- | --- |
| 0 | 315.8 | 1014.0 | 745.9 | 525.8 | 392.2 |
| 1 | 472.0 | 1519.3 | 1122.9 | 799.6 | 598.2 |
| 2 | 331.7 | 1017.1 | 768.3 | 538.0 | 404.9 |
| 3 | 522.2 | 1657.7 | 1235.9 | 891.8 | 664.3 |

### Interpretation

This rerun is much more consistent with the textbook ZeRO story.

- stages 0 and 2 are nearly tied on fixed workload
- stage 1 is slower than stage 0 because it adds the post-step parameter allgather without reducing the fixed workload
- stage 3 is slowest because it combines more communication with rematerialization overhead

The small stage-0 / stage-2 winner flips at a few bandwidth points should not be overinterpreted. The clean conclusion is that, on a matched workload, stages 0 and 2 are in the same regime and both clearly beat stages 1 and 3.

One important measurement note:

- `mean_fb_ms` and `mean_comm_ms` are attribution views, not additive wall-clock components
- for stages 2 and 3, communication happens inside backward, so some time is intentionally visible in both metrics

## Low-Bandwidth Fit-to-Memory Rerun

Status:

- superseded by a transport-emulation bug fix
- the socket shaper previously interpreted configured `Gbps` as `GB/s`, so all low-bandwidth points in this section were about `8x` less throttled than labeled
- the stage-2 / stage-3 qualitative pattern may still hold, but these exact numeric bandwidth conclusions should not be treated as authoritative until rerun

Run:

- `experiments/results/remote/<cluster-a>/remote_4gpu_small_pairwise_crossover_socket_currentpaper_fixedbatches`

Method:

- model: `small`
- sequence length: `256`
- stages: `2`, `3`
- runtime dtype: `bfloat16`
- bandwidth mode: socket throttling over forced NCCL `NET/Socket`
- fixed per-GPU microbatches taken from the authoritative OOM-boundary fit:
  - stage 2: `96`
  - stage 3: `588`
- measured bandwidth points: `unlimited`, `0.5`, `0.0707107`, `0.01 Gbps`

The important setup fact is that this is not a matched-workload comparison. It is a practical “use the memory you saved” comparison:

- stage 2 processes `4 * 96 * 256 = 98,304` tokens per step
- stage 3 processes `4 * 588 * 256 = 602,112` tokens per step
- stage 3 gets a `588 / 96 = 6.125x` larger per-GPU microbatch

### Low-Bandwidth Throughput

| bandwidth | stage 2 tok/s | stage 3 tok/s | stage 3 / stage 2 |
| --- | --- | --- | --- |
| unlimited | 132056.2 | 149598.2 | 1.133x |
| 0.5 Gbps | 50227.5 | 103031.2 | 2.051x |
| 0.0707107 Gbps | 11622.8 | 37225.5 | 3.203x |
| 0.01 Gbps | 1818.0 | 6764.0 | 3.721x |

### Low-Bandwidth Communication Time

| bandwidth | stage 2 comm ms | stage 3 comm ms | stage 3 / stage 2 |
| --- | --- | --- | --- |
| unlimited | 520.1 | 3884.4 | 7.469x |
| 0.5 Gbps | 1732.0 | 5701.7 | 3.292x |
| 0.0707107 Gbps | 8221.6 | 16032.0 | 1.950x |
| 0.01 Gbps | 53788.3 | 88872.8 | 1.652x |

### Interpretation

This interpretation is currently provisional because of the socket-shaper unit bug above.

No stage-2 / stage-3 crossover was found in the searched bandwidth range.

- stage 3 wins at every sampled point, including `0.01 Gbps`
- the stage-3 advantage grows as bandwidth drops: `1.133x -> 3.721x`
- stage 3 does pay more communication per step at every point, but the penalty is far smaller than its `6.125x` token-per-step advantage from fitting a much larger batch

The most important scientific point is that “lower bandwidth hurts stage 3 more” is true in per-step communication cost, but that does not imply lower throughput in this experiment. Throughput is tokens per step divided by step time. As bandwidth gets very small, both stages become communication-dominated, and the comparison tends toward:

- stage-3 token advantage per step: `6.125x`
- divided by the stage-3 communication-time penalty per step

At `0.01 Gbps`, that penalty is only `1.652x`, so stage 3 still comes out ahead by `3.721x`.

So the clean conclusion is:

- fixed workload: stages 0 and 2 are the efficient choices
- fit to memory: stage 3 wins decisively over stage 2 for this workload, and no bandwidth-only crossover appeared even at `0.01 Gbps`
