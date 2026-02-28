#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  infra/throttle.sh apply <device> <rate> [burst] [latency]
  infra/throttle.sh delete <device>
  infra/throttle.sh status <device>

Examples:
  infra/throttle.sh apply eth0 10gbit 1mb 10ms
  infra/throttle.sh delete eth0
  infra/throttle.sh status eth0

Notes:
- Uses Linux tc tbf qdisc on egress traffic.
- Most cloud setups require sudo/root for tc commands.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CMD="$1"
shift

run_tc() {
  if [[ "$(id -u)" -eq 0 ]]; then
    tc "$@"
  else
    sudo tc "$@"
  fi
}

case "$CMD" in
  apply)
    DEV="${1:-}"
    RATE="${2:-}"
    BURST="${3:-1mb}"
    LATENCY="${4:-10ms}"
    if [[ -z "$DEV" || -z "$RATE" ]]; then
      usage
      exit 1
    fi
    echo "[throttle] apply dev=$DEV rate=$RATE burst=$BURST latency=$LATENCY"
    run_tc qdisc replace dev "$DEV" root tbf rate "$RATE" burst "$BURST" latency "$LATENCY"
    ;;
  delete)
    DEV="${1:-}"
    if [[ -z "$DEV" ]]; then
      usage
      exit 1
    fi
    echo "[throttle] delete dev=$DEV"
    run_tc qdisc del dev "$DEV" root || true
    ;;
  status)
    DEV="${1:-}"
    if [[ -z "$DEV" ]]; then
      usage
      exit 1
    fi
    run_tc qdisc show dev "$DEV"
    ;;
  *)
    usage
    exit 1
    ;;
esac
