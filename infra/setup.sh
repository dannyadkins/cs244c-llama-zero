#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[setup] repository root: $ROOT_DIR"
python3 --version

python3 -m pip install -r requirements.txt

echo "[setup] validating single-process imports"
python3 - <<'PY'
import torch
from collectives import ring_allreduce
from model import build_tiny_config, LlamaForCausalLM
print('[setup] torch', torch.__version__)
cfg = build_tiny_config(vocab_size=1024, max_seq_len=64)
model = LlamaForCausalLM(cfg)
print('[setup] model params', sum(p.numel() for p in model.parameters()))
print('[setup] imports ok')
PY

echo "[setup] done"
