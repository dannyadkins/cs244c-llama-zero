# ZeRO From Scratch (Week 1 - Person A)

This repository contains a clean Week 1 baseline for the ZeRO project:

- LLaMA-style model in PyTorch (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`)
- Size configs for `tiny` (~46M), `small` (~318M), `medium` (~1.34B)
- Single-device training loop with AdamW
- FineWeb-Edu streaming pipeline
- Correctness-oriented tests

## Project Layout

- `model/config.py`: model configs + parameter estimation
- `model/llama.py`: full model implementation
- `data/fineweb.py`: tokenizer + streaming/packing datasets
- `train.py`: training entrypoint (single GPU baseline)
- `tests/`: unit/invariant/optimization tests

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

## Train (Synthetic sanity run)

This is fully offline and should show loss decreasing quickly:

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

## Train (FineWeb-Edu + LLaMA tokenizer)

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

- `meta-llama/Llama-3.1-8B` may require a HuggingFace token with accepted license terms.
- If external dataset/tokenizer access fails, use `--allow-synthetic-fallback`.

## Correctness Checks Implemented

- Shape and backward-pass checks
- Causal masking invariant (future tokens do not affect past logits)
- Rotary embedding norm preservation
- Deterministic overfit test proving optimizer + gradient flow works
- Config/parameter-count sanity checks for tiny/small/medium scale ordering

## Week 1 Deliverable Mapping

- Model and training loop: complete (`model/`, `train.py`)
- Data loading from FineWeb-Edu streaming: complete (`data/fineweb.py`)
- Single-GPU correctness baseline: complete (tests + synthetic train command)
- Ready extension points for Week 2 ZeRO wrappers: model forward API and data pipeline are stable and modular
