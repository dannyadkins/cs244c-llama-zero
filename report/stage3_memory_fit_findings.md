# Stage 3 Memory-Fit Progress

## What Changed

- The Stage 3 scaling sweep now supports a predictive tuner that seeds each GPU count from prior fits instead of restarting from a blind OOM search.
- The remote runner now excludes local result artifacts from the upload bundle and forces unbuffered remote Python output so long runs are easier to monitor.
- The scaling analysis now carries the tuning-time peak memory into the report so fit-to-memory runs do not show `NA` for the peak-memory column.

## Current Remote Findings

Host:

- `remote 16-GPU RTX 4090 host`
- `16x RTX 4090`

Workload:

- `Stage 3`
- `model=medium`
- `seq_len=1024`
- `dtype=bfloat16`
- `--activation-checkpointing`

Measured successful short-fit boundaries so far:

| GPUs | successful short-fit batches | first short-fit failure |
| --- | --- | --- |
| 1 | `16, 18, 19` | `20` |
| 2 | `26, 38, 39, 44, 45, 46` | `47, 48, 50, 53` |
| 4 | `60, 62` | `63, 64` |
| 8 | `68` | `69, 70, 72` |
| 16 | `69, 71, 72` | `73, 77` |

Longer benchmark results collected so far:

| GPUs | batch/GPU | status | total TFLOPs | note |
| --- | --- | --- | --- | --- |
| 2 | `46` | success | `125.09` | targeted memory-fit benchmark |
| 4 | `62` | failed | `NA` | short-fit boundary was too optimistic for the longer run |
| 8 | `68` | failed | `NA` | short-fit boundary was too optimistic for the longer run |
| 16 | `72` | failed | `NA` | fragmented/OOM during longer run |
| 16 | `69` | success | `1024.66` | stable rerun after backing off from `72` |

## Main Takeaways

- The predictive path is materially faster than the old blind sweep. Example: `2 GPUs` converged from `46 fit -> 48 fail -> 47 fail`, instead of walking the whole range.
- A short measured-step fit is still slightly optimistic for longer `8-step` benchmarks at higher GPU counts.
- On this host, the stable benchmark-safe batch is lower than the raw measured-step boundary once `world_size >= 4`.

## Remaining Work

- Rerun the long `4 GPU` benchmark at a safer batch (`60` instead of `62`).
- Rerun the long `8 GPU` benchmark at a safer batch (below `68`).
- Rebuild the final memory-fit scaling summary/figure once those stable points are in place.
