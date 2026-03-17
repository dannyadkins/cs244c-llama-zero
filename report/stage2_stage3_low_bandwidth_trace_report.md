# Stage 2 vs Stage 3 Low-Bandwidth Trace Report

## Why this rerun was necessary

The earlier low-bandwidth stage-2 vs stage-3 conclusions were confounded by three measurement problems:

1. The socket shaper bandwidth env var was interpreted as bytes/s instead of bits/s.
2. Collective timing only measured CUDA enqueue time unless `ZERO_COLLECTIVE_CUDA_SYNC=1`.
3. The generic experiment harness still defaulted to the repo's `ring` collectives backend, which is the wrong backend for CUDA + NCCL experiments in this repo.

This rerun fixes all three:

- `infra/socket_shaper.c` now interprets `ZERO_SOCKET_SHAPER_BW_GBPS` correctly and emits per-process shaping stats.
- `experiments/harness.py` now sets `ZERO_COLLECTIVE_CUDA_SYNC=1`.
- `train_zero.py` and `experiments/harness.py` now default to `--collective-impl torch`.

## Experimental setup

- Machine: remote 4x RTX 4090 server
- Transport: NCCL forced onto `NET/Socket` over `lo`
- Bandwidth shaping: `LD_PRELOAD` socket shaper with real token-bucket throttling
- Model: `small`
- Sequence length: `256`
- Steps: `2` with `1` warmup step, so metrics are from the second step
- Fixed max-fit microbatches from the current code:
  - stage 2: `96`
  - stage 3: `588`
- Result artifacts:
  - stage 2: `experiments/results/remote/<cluster-b>/manual_trace_stage2_u_0p1_syncstats_torch`
  - stage 3: `experiments/results/remote/<cluster-b>/manual_trace_stage3_u_0p1_syncstats_torch`

## Top-line results

| bandwidth | stage | batch / GPU | tok/s | comm ms | fb ms |
| --- | --- | ---: | ---: | ---: | ---: |
| unlimited | 2 | 96 | 112,493 | 620.1 | 653.2 |
| unlimited | 3 | 588 | 153,296 | 3,740.6 | 3,918.3 |
| 0.1 Gbps | 2 | 96 | 2,252 | 43,368.2 | 21,994.0 |
| 0.1 Gbps | 3 | 588 | 8,373 | 71,733.9 | 71,903.9 |

At `0.1 Gbps`, stage 3 is still faster in throughput:

- throughput ratio: `8373 / 2252 = 3.72x`
- communication ratio: `71733.9 / 43368.2 = 1.65x`
- batch ratio: `588 / 96 = 6.125x`

That ratio pattern is the core result. Stage 3 really does communicate more, but not enough more to erase its much larger fitted batch.

Latency/throughput figure:

- `report/figures/stage2_stage3_step_latency.png`

## What the traces show

### Stage 2

Profile: `experiments/results/remote/<cluster-b>/manual_trace_stage2_u_0p1_syncstats_torch/profiles/s2_msmall_bw0.1gbps_np4_sl256_bs96_ga1_seed1337.json`

Per measured step, stage 2 performs only two large collectives:

- one backward `reduce_scatter` of the flat grad buffer
- one post-step `allgather` of the updated param shard

Aggregate trace over 2 traced steps:

- `reduce_scatter`: `43.93 s`
- `allgather`: `43.21 s`
- total trace time: `87.14 s`
- total traced calls: `4`

### Stage 3

Profile: `experiments/results/remote/<cluster-b>/manual_trace_stage3_u_0p1_syncstats_torch/profiles/s3_msmall_bw0.1gbps_np4_sl256_bs588_ga1_seed1337.json`

Stage 3 performs many more collectives at module granularity:

- forward allgathers for embeddings, each transformer block, and final norm
- backward allgathers for recomputation
- backward reduce-scatters for grads

Aggregate trace over 2 traced steps:

- `allgather`: `93.69 s`
- `reduce_scatter`: `50.23 s`
- total trace time: `143.92 s`
- total traced calls: `90`

That is `45` collective calls per step for stage 3 versus `2` per step for stage 2.

## What the transport shaper shows

The shaper stats are the cleanest check that the trace is measuring real transport cost instead of an artifact.

Stage-2 shaper stats directory:

- `experiments/results/remote/<cluster-b>/manual_trace_stage2_u_0p1_syncstats_torch/shaper_stats`

Stage-3 shaper stats directory:

- `experiments/results/remote/<cluster-b>/manual_trace_stage3_u_0p1_syncstats_torch/shaper_stats`

Summed over the 4 rank processes at `0.1 Gbps`:

| stage | shaped bytes | injected sleep |
| --- | ---: | ---: |
| 2 | `5.41 GB` | `428.86 s` |
| 3 | `7.90 GB` | `623.53 s` |

Ratios:

- shaped-byte ratio: `7.90 / 5.41 = 1.46x`
- injected-sleep ratio: `623.53 / 428.86 = 1.45x`

This is the important scientific correction: at low bandwidth, stage 3 is not paying some mysterious runaway communication penalty. It is paying about `1.45x` the throttled transport work of stage 2.

## Interpretation

The data now supports a clean explanation:

1. At high bandwidth, stage 3 is hurt heavily by many collective launches and orchestration overhead.
   - unlimited-bandwidth comm ratio: `6.03x`
2. At low bandwidth, those fixed overheads wash out.
   - corrected `0.1 Gbps` comm ratio: `1.65x`
   - raw throttled-byte ratio from the shaper: `1.46x`
3. Stage 3 still wins in `tok/s` because its per-step token count is much larger.
   - stage 2: `4 * 96 * 256 = 98,304` tokens/step
   - stage 3: `4 * 588 * 256 = 602,112` tokens/step
   - token ratio: `6.125x`

So the earlier skepticism was valid in one sense: stage 3 should communicate more, and now the corrected experiment clearly shows that it does. But the corrected data also shows why there is still no crossover in the max-fit, bandwidth-only setting:

- stage 3 comm cost grows by about `1.5-1.65x`
- stage 3 token throughput opportunity grows by `6.125x`

That is why the low-bandwidth throughput ratio trends toward something like:

`throughput ratio ~= token ratio / communication ratio`

and why the measured `3.72x` stage-3 advantage at `0.1 Gbps` is not paradoxical once the measurement path is corrected.
