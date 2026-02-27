#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SCRIPT="scripts/distributed_sanity.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes) NNODES="$2"; shift 2 ;;
    --node-rank) NODE_RANK="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --script) SCRIPT="$2"; shift 2 ;;
    --help|-h)
      cat <<'EOF'
Usage: infra/launch.sh [options] [-- script args]

Options:
  --nnodes N
  --node-rank R
  --nproc-per-node N
  --master-addr HOST
  --master-port PORT
  --script PATH               (default: scripts/distributed_sanity.py)

All additional args after '--' are passed to the script.
EOF
      exit 0
      ;;
    --) shift; break ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "[launch] nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"
echo "[launch] script=$SCRIPT"

torchrun \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$SCRIPT" "$@"
