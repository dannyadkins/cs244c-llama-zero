# Stage 3 Scaling Findings

This note captures the measured behavior behind the Stage 3 GPU-scaling workflow.

## Setup

- host: single machine with `16x RTX 4090`
- model: `medium`
- dtype: `bfloat16`
- sequence length: `1024`
- microbatch per GPU: `8`
- ZeRO stage: `3`
- collective implementation: `torch`
- activation checkpointing: enabled

## Main Result

Using `2 GPUs` as the first distributed baseline, the measured Stage 3 throughput shows a small superlinear region before communication dominates:

| GPUs | total TFLOPs | TFLOPs / GPU | perfect-linear TFLOPs from 2-GPU base | gain vs linear |
| --- | --- | --- | --- | --- |
| 2 | 50.16 | 25.08 | 50.16 | 1.000x |
| 4 | 108.46 | 27.12 | 100.32 | 1.081x |
| 8 | 206.10 | 25.76 | 200.64 | 1.027x |
| 16 | 321.05 | 20.07 | 401.28 | 0.800x |

The strongest superlinear point in this run is `4 GPUs`, at `1.081x` of perfect-linear relative to the `2-GPU` baseline.

## Interpretation

- A `1-GPU` baseline is misleading for this plot because Stage 3 at `world_size=1` has no real distributed communication. The first meaningful comparison point is the first distributed run.
- The early superlinear region is consistent with the paper's qualitative story: larger data-parallel degree improves usable aggregate throughput before communication catches up.
- On this hardware, the superlinear region is modest and disappears by `16 GPUs`.

## Bottleneck Evidence

A traced `16-GPU` run shows that Stage 3 becomes communication-bound at the top end:

- aggregate traced `allgather` time: about `5766 ms` over `3` steps
- aggregate traced `reduce_scatter` time: about `4681 ms` over `3` steps
- the largest individual costs are backward `reduce_scatter` calls on transformer layers, at roughly `185-198 ms` each

This machine has no GPU peer access between any pair of GPUs, which is a major caveat when comparing against the original ZeRO paper setups.
