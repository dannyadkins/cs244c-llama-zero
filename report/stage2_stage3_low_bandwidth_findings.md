# Stage 2 vs Stage 3 at Low Bandwidth

## Question

Why did ZeRO Stage 3 outperform ZeRO Stage 2 in the low-bandwidth sweep even though the microbatch size was the same?

## Runs Used

- Main communication-shape sweep:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_bandwidth_socket_stage2_stage3_commshape_steps6`
- One-case retry for the missing `stage2 @ 0.5 Gbps` point:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_bandwidth_socket_stage2_stage3_commshape_s2_bw0p5_retry`

The main run used:

- model: `small`
- GPUs: `4`
- bandwidth mode: `socket`
- bandwidths: `0`, `0.5`, `1`, `2`, `5 Gbps`
- steps: `6`
- warmup steps ignored in analysis: `2`

## Method

We instrumented per-step communication phase timing in both implementations:

- Stage 2:
  - backward reduce-scatter
  - post-step parameter all-gather
- Stage 3:
  - forward parameter all-gather
  - backward parameter all-gather
  - backward reduce-scatter

We then compared iteration time, communication time, per-phase timing, and effective communicated tensor sizes.

## Main Findings

### 1. The original overlap explanation does not match this codebase

Stage 3 in this repo does not prefetch the next layer asynchronously while the current layer computes.

The Stage 3 path in `zero/stage3_optimizer.py` performs synchronous:

- all-gather before a module forward
- all-gather before backward recomputation
- reduce-scatter after the backward pass for that module

`zero/README.md` already reflects this correctly: Stage 3 is "correctness-first" and does not yet implement overlap/prefetch optimization.

Conclusion:

- The measured Stage 3 win is not evidence that our implementation hides latency better.
- The win must come from a different cause.

### 2. Stage 2 is paying a large reduce-scatter padding penalty

This is the dominant reason Stage 3 wins in the low-bandwidth runs.

For the `small` model with `4` ranks and `64 MB` Stage 2 buckets:

- total model size in fp32 flat form: `684.11 MB`
- per-rank shard size: `171.03 MB`
- Stage 2 bucket count: `13`
- buckets fully inside a single rank shard: `11 / 13`

Because the current Stage 2 implementation packs each bucket into a `world_size * max(rank_piece)` tensor before calling `reduce_scatter`, most buckets get padded heavily. Measured from the actual bucket layout:

- logical flattened gradient size: `684.11 MB`
- actual packed Stage 2 reduce-scatter input per step: `2628.41 MB`
- inflation: `3.84x`

This is a performance pathology of the current Stage 2 implementation. It is not a ZeRO-2 algorithm requirement.

Reproduce the bucket inflation summary with:

```bash
./.venv/bin/python analysis/stage2_bucket_inflation_probe.py --model-size small --vocab-size 8192 --seq-len 128 --world-size 4 --grad-bucket-mb 64 --print-buckets
```

### 3. The Stage 2 post-step all-gather is real, but it is not the main bottleneck here

At `0.5 Gbps`:

- Stage 2 backward reduce-scatter: `5845.4 ms`
- Stage 2 post-step all-gather: `1526.5 ms`

So the blocking post-step all-gather is only about `21%` of Stage 2 communication in this setting. The much larger problem is the inflated Stage 2 reduce-scatter path.

This means the simple story

- "Stage 2 loses because of one blocking all-gather wall"

is incomplete for this repo.

### 4. Stage 3 actually spends more time in all-gather than Stage 2 at low bandwidth and still wins

At `0.5 Gbps`:

- Stage 2 total all-gather time: `1526.5 ms`
- Stage 3 total all-gather time: `3309.3 ms`

Stage 3 still wins because:

- Stage 2 reduce-scatter: `5845.4 ms`
- Stage 3 reduce-scatter: `1642.1 ms`

So Stage 3 is not winning because it somehow avoids all-gather cost. It wins because Stage 2's current bucketed reduce-scatter path is much more expensive than it should be.

### 5. The low-bandwidth inversion is mostly an implementation artifact, not a fundamental Stage 3 advantage

At `0.5 Gbps`:

- Stage 2 iteration: `7427.7 ms`
- Stage 3 iteration: `5014.6 ms`

If we estimate a counterfactual Stage 2 where reduce-scatter scaled with the logical full-gradient size instead of the current packed `3.84x` padded input, we get:

- estimated Stage 2 reduce-scatter: about `1521.4 ms`
- estimated Stage 2 total communication: about `3047.9 ms`

This is well below the measured Stage 3 communication time:

- Stage 3 total communication: `4951.5 ms`

This is only an estimate, not a direct measurement, but it strongly suggests:

- with an efficient Stage 2 reduce-scatter implementation, Stage 2 would likely beat the current Stage 3 implementation in this low-bandwidth setting

That is closer to the original intuition.

## Bandwidth Summary

| BW (Gbps) | Stage 2 iter (ms) | Stage 3 iter (ms) | S2 / S3 iter | Stage 2 comm (ms) | Stage 3 comm (ms) | Stage 2 RS (ms) | Stage 3 RS (ms) | Stage 2 post-AG (ms) | Stage 3 total AG (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2261.1 | 1113.7 | 2.030x | 2187.2 | 1053.1 | 1726.5 | 343.3 | 460.7 | 709.8 |
| 0.5 | 7427.7 | 5014.6 | 1.481x | 7371.9 | 4951.5 | 5845.4 | 1642.1 | 1526.5 | 3309.3 |
| 1 | 5434.9 | 3348.6 | 1.623x | 5360.1 | 3281.7 | 4248.7 | 1086.0 | 1111.4 | 2195.7 |
| 2 | 3667.0 | 2505.8 | 1.463x | 3610.4 | 2444.6 | 2862.1 | 800.8 | 748.4 | 1643.9 |
| 5 | 2600.4 | 1831.4 | 1.420x | 2546.5 | 1769.1 | 2020.4 | 572.6 | 526.1 | 1196.5 |

## Interesting Non-Bug Observation

Stage 3 reports `720.11 MB` per forward all-gather even though the unique model parameter size is `684.11 MB`.

This is expected for this model:

- token embeddings and `lm_head` share weights
- Stage 3 uses that shared `36 MB` handle twice per training step:
  - once for `tok_embeddings`
  - once for `lm_head`

So the Stage 3 forward materialization volume is:

- `684.11 MB` unique parameters
- plus one extra `36 MB` reuse of the tied embedding handle
- total: `720.11 MB`

This is not a bug.

## What Is Actually Wrong vs What Is Just Interesting

Actually wrong:

- Attributing the result to Stage 3 overlap/prefetch in this repo is incorrect.
- The Stage 2 bucketed reduce-scatter path has a severe padding inefficiency for this model/bucket-size/world-size combination.

Interesting but not wrong:

- Stage 3 performs more all-gather than Stage 2 in this implementation and still wins.
- Tied embeddings cause Stage 3 to rematerialize the same handle twice per step.

## Validation After Fixing Stage 2

We applied a targeted Stage 2 performance fix in `zero/stage2_optimizer.py`:

- when a bucket is badly imbalanced across parameter shards, Stage 2 now avoids the padded bucket packing path and uses a dense bucket synchronization path instead

This keeps the same math and sharded optimizer update, but removes the `3.84x` padded communication blow-up that dominated the original low-bandwidth result.

We then reran a single remote GPU validation case on the same host:

- run: `experiments/results/remote/184.144.213.79/remote_4gpu_small_bandwidth_socket_stage2_fixcheck_bw0p5`
- model: `small`
- GPUs: `4`
- bandwidth: `0.5 Gbps`
- steps: `6`

Patched Stage 2 vs original Stage 3 at `0.5 Gbps`:

- patched Stage 2 iteration: `4658.4 ms`
- original Stage 3 iteration: `5014.6 ms`
- patched Stage 2 communication: `4603.2 ms`
- original Stage 3 communication: `4951.5 ms`

Patched Stage 2 communication breakdown at `0.5 Gbps`:

- backward sync path: `3067.2 ms`
- post-step all-gather: `1536.0 ms`

So after removing the Stage 2 padding pathology, Stage 2 no longer loses this low-bandwidth point. That supports the diagnosis that the original Stage 3 advantage was mostly a Stage 2 implementation artifact.

## Follow-up: Controlled 1 Gbps Recheck

After rebuilding the full sweep by replacing only Stage 2, the merged summary still appeared to show Stage 2 slightly behind Stage 3 at `1 Gbps`.

That was not a clean current-code comparison, because:

- Stage 2 came from the refreshed post-fix run
- Stage 3 was copied from the older full sweep

To test the `1 Gbps` point directly, we ran a fresh remote head-to-head with the current code for both stages:

- run: `experiments/results/remote/184.144.213.79/remote_4gpu_small_bandwidth_socket_stage2_stage3_commshape_bw1_postfix`
- model: `small`
- GPUs: `4`
- bandwidth: `1 Gbps`
- steps: `12`

Measured result:

- Stage 2 iteration: `3075.8 ms`
- Stage 3 iteration: `3347.1 ms`
- Stage 2 communication: `3020.7 ms`
- Stage 3 communication: `3285.0 ms`

Phase breakdown:

- Stage 2:
  - backward sync path: `2010.7 ms` for `684.1 MB`
  - post-step all-gather: `1009.9 ms` for `684.1 MB`
- Stage 3:
  - forward all-gather: `1096.4 ms` for `720.1 MB`
  - backward all-gather: `1100.2 ms` for `720.1 MB`
  - backward reduce-scatter: `1088.4 ms` for `720.1 MB`

Interpretation:

- Under a clean current-code rerun, Stage 2 is not behind at `1 Gbps`.
- The earlier merged full-sweep crossover at `1 Gbps` should be treated as a mixed-run artifact, not a robust algorithmic conclusion.
- At this point, the evidence supports a simpler picture: once the Stage 2 padding bug is fixed, Stage 2 generally behaves as expected against the current non-overlapped Stage 3 implementation.

## Bottom Line

The clean interpretation is:

- The current Stage 3 win at low bandwidth is mostly not a fundamental ZeRO-3 advantage.
- It is mostly a Stage 2 implementation artifact caused by padded reduce-scatter buckets.
- The blocking Stage 2 post-step all-gather matters, but it is secondary to the reduce-scatter inflation in this repo.
- For fair Stage 2 vs Stage 3 conclusions, we should not present this result as "Stage 3 hides latency better" until Stage 2 uses a more communication-efficient reduce-scatter path and Stage 3 has real overlap/prefetch support.
