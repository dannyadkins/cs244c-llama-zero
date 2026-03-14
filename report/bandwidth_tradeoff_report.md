# Bandwidth Tradeoff Report

## Scope

This report consolidates the bandwidth and fit-to-memory evidence that already exists in the repo, without rerunning the full experiment matrix.

The goal is to separate two different questions:

1. Fixed workload: with the same microbatch on every stage, which stage wins as bandwidth falls?
2. Fit-to-memory: if each stage is allowed to spend its memory savings on a larger microbatch, does that change the winner?

## Data Used

### Fixed-workload bandwidth sweeps

- Main 4-stage socket sweep:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_bandwidth_socket`
- Corrected stage-2/stage-3 rerun after stage-2 bucketing and shared-link shaping fixes:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_bandwidth_socket_stage23_bucketed_shared`
- Stage-1 tail rerun:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_bandwidth_socket_stage1_tail_shared`
- Stage-2 vs stage-3 low-bandwidth diagnosis and targeted fix validation:
  - `report/stage2_stage3_low_bandwidth_findings.md`

### Fit-to-memory data

- Completed stage-0 vs stage-3 fit-to-memory probe:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_fit_memory_socket_probe_sl256`
- Completed fixed current-code stage-1 fit-to-memory rerun:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_fit_memory_socket_stage1_fixed`
- Completed fixed current-code stage-2 fit-to-memory rerun:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_fit_memory_socket_stage2_fixed`
- Earlier current-code stage-2 vs stage-3 fit-to-memory bandwidth sweep:
  - `experiments/results/remote/184.144.213.79/remote_4gpu_small_fit_memory_socket_stage23_current`

## Method

### Fixed workload

- model: `small`
- sequence length: `128`
- per-GPU microbatch: `4`
- GPUs: `4`
- bandwidth mode: socket shaping over forced NCCL `NET/Socket`

This isolates communication tradeoffs because the actual training workload is the same for every stage.

### Fit-to-memory

- model: `small`
- sequence length: `256`
- GPUs: `4`
- memory budget: about `21.67 GB` per GPU
- tuning metric: measured `peak_cuda_max_reserved_mb`

Each stage is allowed to grow per-GPU microbatch until it reaches the same memory budget. This measures the practical throughput advantage from using ZeRO memory savings to process more tokens per step.

## Fixed-Workload Findings

From the original fixed-workload socket sweep:

| Stage | Unlimited tokens/s | 1 Gbps tokens/s | 2 Gbps tokens/s | 5 Gbps tokens/s | 10 Gbps tokens/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 2812.3 | 1094.8 | 1380.5 | 1886.5 | 2388.9 |
| 1 | 2028.5 | 752.8 | 957.9 | 1294.2 | 1683.9 |
| 2 | 1214.3 | 452.7 | 558.1 | 799.6 | 985.3 |
| 3 | 1788.5 | 702.0 | 911.5 | 1194.3 | 1531.5 |

Main takeaway:

- with the same microbatch on every stage, stage 0 is best across the tested bandwidth range
- stage 1 is second
- stage 3 is slower than stage 1 because its extra communication and rematerialization cost are not being repaid by any larger workload

Important correction:

- the original extreme stage-2 weakness at low bandwidth was partly an implementation bug
- after fixing the padded stage-2 reduce-scatter path, the stage-2 vs stage-3 low-bandwidth inversion largely disappears
- see `report/stage2_stage3_low_bandwidth_findings.md`

So the clean interpretation is:

- fixed-workload plots are mainly measuring communication overhead
- under that lens, low stages win because higher stages are not allowed to use their memory advantage

## Fit-to-Memory Findings

### What we know for sure

From the completed fit-to-memory probe and fixed reruns:

- stage 0 selected microbatch: `85`
- stage 1 selected microbatch after the temp-buffer fix: `84`
- stage 2 selected microbatch after the temp-buffer fix: `84`
- stage 3 selected microbatch in the existing probe: `256`
- later stage-3 tuning in the current-code sweep reached `368`

Measured throughput from that probe:

| Stage | Microbatch / GPU | Unlimited tokens/s | 10 Gbps tokens/s |
| --- | --- | --- | --- |
| 0 | 85 | 85,259.5 | 74,260.0 |
| 3 | 256 | 101,237.0 | 89,969.5 |

This already shows the practical regime shift:

- once stage 3 is allowed to spend its memory savings on a larger microbatch, it beats stage 0 at high bandwidth
- in the measured probe, it still beats stage 0 at `10 Gbps`

More importantly, the earlier stage-1/stage-2 tuning result was partly broken.

Before the stage-1 and stage-2 temp-buffer fixes:

| Stage | Selected microbatch / GPU | Peak reserved MB |
| --- | --- | --- |
| 0 | 85 | 20818 |
| 1 | 76 | 21362 |
| 2 | 80 | 20804 |

After the fixes:

| Stage | Selected microbatch / GPU | Peak reserved MB |
| --- | --- | --- |
| 0 | 85 | 20818 |
| 1 | 84 | 21026 |
| 2 | 84 | 21026 |

So the old result that "stage 0 fits a meaningfully larger microbatch than stages 1 and 2" was not trustworthy. The fixes removed a real implementation pathology in stages 1 and 2.

From the earlier current-code stage-2 vs stage-3 bandwidth sweep:

- stage 2 selected microbatch: `80`
- stage 3 selected microbatch: `368`

Measured throughput from that run:

| Stage | Microbatch / GPU | Unlimited tokens/s | 1 Gbps tokens/s | 2 Gbps tokens/s | 5 Gbps tokens/s | 10 Gbps tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 80 | 60,370.5 | 23,938.8 | 32,343.8 | 45,844.8 | 52,790.0 |
| 3 | 368 | 114,678.5 | 71,763.2 | 89,891.8 | 102,185.5 | 106,520.2 |

This is the clearest practical result in the repo right now:

- once stage 3 is allowed to use its memory savings, it is much faster than stage 2 at every tested bandwidth
- the extra communication does increase `comm_ms`, but not enough to offset the much larger microbatch
- stage 3 remains about `3.0x` faster than stage 2 even at `1 Gbps`

### Why stage 0 can still edge out stages 1 and 2 by one sample

At the matched fixed point `bs=84`, the live measured peak is already lower for stages 1 and 2 than for stage 0:

| Stage | Microbatch / GPU | Peak allocated MB | Peak reserved MB | Live model state MB |
| --- | --- | --- | --- | --- |
| 0 | 84 | 19791.1 | 21016 | 2736.4 |
| 1 | 84 | 18768.2 | 21026 | 1710.3 |
| 2 | 84 | 18768.2 | 21026 | 1197.2 |

This is the key scientific point:

- stages 1 and 2 really do save model-state memory
- the saved live memory at `bs=84` is about `1.0 GB` for stage 1 and about `1.5 GB` for stage 2 versus stage 0
- but the fit-to-memory tuner uses `peak_cuda_max_reserved_mb`, not allocated memory
- CUDA's caching allocator is holding onto nearly the same reserved footprint in all three cases, so the selection ends up `85 / 84 / 84` instead of giving a visibly larger batch to stages 1 and 2

The old stage-1 result was therefore a real bug. The remaining `85 vs 84` gap is not the same kind of bug; it is allocator behavior under a reserved-memory budget.

### What we learned from the fixed stage-1 and stage-2 reruns

Measured unlimited-bandwidth throughput from the fixed reruns:

| Stage | Selected microbatch / GPU | Unlimited tokens/s | Mean comm ms | Mean fb ms |
| --- | --- | --- | --- | --- |
| 1 | 84 | 65,136.0 | 937.6 | 374.4 |
| 2 | 84 | 64,419.0 | 1176.9 | 1011.3 |

This changes the interpretation of stages 1 and 2:

- stage 1 is no longer fitting a smaller batch than stage 0 for any meaningful reason
- stage 2 is also no longer artificially pinned at `80`
- however, neither stage 1 nor stage 2 gets a dramatic fit-to-memory win because their memory savings mostly reduce the intercept, while stage-0/1/2 still share a similar activation-dominated slope

### What we do not have yet

We still do not have a full current-code all-4-stage fit-to-memory bandwidth sweep on disk after the stage-1 and stage-2 fixes.

That means:

- we can now say stage 3 beats stage 2 across the full tested bandwidth range in the fit-to-memory regime
- we can already say stage 3 beats stage 0 in the existing stage-0/stage-3 probe
- we can now say the old stage-1 and stage-2 fit limits were partly implementation artifacts and have been corrected
- the remaining gap is a full post-fix fit-to-memory bandwidth sweep if we want one single canonical all-4-stage artifact

## Scientific Interpretation

The existing evidence supports a simple two-regime story.

### Regime 1: fixed workload

If every stage trains the same microbatch:

- higher stages mostly add communication and rematerialization overhead
- stage 0 wins

This is the right regime for isolating communication tradeoffs.

### Regime 2: fit to a common memory budget

If every stage is allowed to scale microbatch until it fills the same GPU-memory budget:

- stage 3 gains a large practical advantage because it can process far more tokens per step
- stages 1 and 2 do gain real model-state headroom, but not enough to change the outcome much under a reserved-memory budget
- stage 3 already beats stage 0 in the measured probe even at `10 Gbps`
- stage 3 beats stage 2 across the full current-code bandwidth sweep from unlimited down to `1 Gbps`

This is the right regime for measuring practical training throughput rather than pure communication cost.

## Current Best Conclusion

Using only the data already on disk:

- fixed workload: stage 0 is the winner
- fit-to-memory: stage 3 is the winner against stage 2 in the completed current-code sweep
- fit-to-memory: stage 3 also beats stage 0 in the existing probe
- stages 1 and 2 now reach rough parity with stage 0 on tuned microbatch after the implementation fixes
- stage 1 is still not a serious throughput contender on this workload because its extra communication is not repaid by a larger enough batch

So the real crossover is not "stage 3 beats stage 0 when bandwidth falls enough" inside the fixed-workload plot.

The real crossover is:

- if you ignore the memory advantage, stage 0 wins
- if you let later stages actually use their memory savings for larger parallel work, stage 3 becomes worthwhile and in the measured runs it becomes decisively faster than stage 2 and faster than stage 0 where measured

## Missing Piece

If we want one final all-4-stage fit-to-memory figure as a single canonical run directory, the only missing work is:

- rerun the fit-to-memory bandwidth sweep for stages 1 and 2 after the temp-buffer fixes
- optionally rerun stage 0 and stage 3 in the same sweep for one perfectly uniform artifact

We do not need to rerun the fixed-workload experiment again.
