# Stage 3 Superlinear Fixed-Batch Run

This is the completed Stage 3 run that already showed real superlinear behavior.

Source run:

- `stage3_fixed_seq1024_bs8_long`
- single host with `16x RTX 4090`
- `Stage 3`, `model=medium`, `seq_len=1024`, `dtype=bfloat16`
- fixed `batch/GPU=8`
- linear reference baseline: `2 GPUs`

Machine-readable data:

- `report/stage3_superlinear_fixed_batch_data.json`

Key results:

| GPUs | total TFLOPs | perfect-linear TFLOPs from 2-GPU base | gain vs linear |
| --- | --- | --- | --- |
| 2 | `50.16` | `50.16` | `1.000x` |
| 4 | `108.46` | `100.32` | `1.081x` |
| 8 | `206.10` | `200.64` | `1.027x` |
| 16 | `321.05` | `401.28` | `0.800x` |

The strongest superlinear point is `4 GPUs`, at `1.081x` of perfect-linear relative to the `2 GPU` baseline.

For the interpretation and 16-GPU collective bottleneck evidence, see `report/stage3_scaling_findings.md`.
