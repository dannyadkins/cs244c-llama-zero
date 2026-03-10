from __future__ import annotations

import argparse
import json
import socket
import signal
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THROTTLE_SCRIPT = PROJECT_ROOT / "infra" / "throttle.sh"
_ACTIVE_TC_DEVICE: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tc shaping with iperf3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Check TCP reachability to a host and port")
    check.add_argument("--target-host", type=str, required=True)
    check.add_argument("--port", type=int, default=5201)
    check.add_argument("--connect-timeout-s", type=float, default=5.0)
    check.add_argument("--json-output", type=str, default="")

    server = subparsers.add_parser("server", help="Run an iperf3 server")
    server.add_argument("--bind", type=str, default="0.0.0.0")
    server.add_argument("--port", type=int, default=5201)
    server.add_argument("--one-off", action="store_true", help="Exit after one client run")

    validate = subparsers.add_parser("validate", help="Measure baseline and shaped throughput")
    validate.add_argument("--target-host", type=str, required=True)
    validate.add_argument("--port", type=int, default=5201)
    validate.add_argument("--device", type=str, required=True)
    validate.add_argument("--rate", type=str, required=True, help="tc rate, e.g. 10gbit or 500mbit")
    validate.add_argument("--burst", type=str, default="1mb")
    validate.add_argument("--latency", type=str, default="10ms")
    validate.add_argument("--duration-s", type=int, default=5)
    validate.add_argument(
        "--connect-timeout-s",
        type=float,
        default=5.0,
        help="Maximum time allowed for the TCP preflight connectivity check.",
    )
    validate.add_argument(
        "--iperf-timeout-s",
        type=int,
        default=20,
        help="Maximum wall-clock time allowed for each iperf3 client run.",
    )
    validate.add_argument("--baseline-runs", type=int, default=1)
    validate.add_argument("--shaped-runs", type=int, default=1)
    validate.add_argument(
        "--max-shaped-ratio",
        type=float,
        default=0.8,
        help="Fail validation if shaped throughput exceeds this fraction of baseline throughput.",
    )
    validate.add_argument(
        "--allow-route-mismatch",
        action="store_true",
        help="Proceed even if the route to the target does not use the selected device.",
    )
    validate.add_argument("--json-output", type=str, default="")

    return parser.parse_args()


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"required command not found: {name}")


def _run(cmd: list[str], *, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _write_json_output(output_path_str: str, payload: dict[str, object]) -> None:
    if not output_path_str:
        return
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _signal_cleanup_and_exit(signum: int, _frame: object) -> None:
    global _ACTIVE_TC_DEVICE
    if _ACTIVE_TC_DEVICE is not None:
        _clear_tc(_ACTIVE_TC_DEVICE)
        _ACTIVE_TC_DEVICE = None
    raise SystemExit(128 + signum)


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_cleanup_and_exit)
    signal.signal(signal.SIGTERM, _signal_cleanup_and_exit)


def _route_device(target_host: str) -> str:
    proc = _run(["ip", "route", "get", target_host])
    tokens = proc.stdout.strip().split()
    if "dev" not in tokens:
        raise RuntimeError(f"could not determine route device for {target_host!r}: {proc.stdout!r}")
    return tokens[tokens.index("dev") + 1]


def _tcp_connect_check(target_host: str, port: int, timeout_s: float) -> dict[str, object]:
    route_device = _route_device(target_host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((target_host, port))
        return {
            "target_host": target_host,
            "port": port,
            "timeout_s": timeout_s,
            "routed_device": route_device,
            "connectable": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "target_host": target_host,
            "port": port,
            "timeout_s": timeout_s,
            "routed_device": route_device,
            "connectable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sock.close()


def _iperf_bits_per_second(target_host: str, port: int, duration_s: int, timeout_s: int) -> tuple[float, dict[str, object]]:
    cmd = [
        "iperf3",
        "-c",
        target_host,
        "-p",
        str(port),
        "-t",
        str(duration_s),
        "-J",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"iperf3 client timed out after {timeout_s}s while connecting to {target_host}:{port}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(f"iperf3 client failed for {target_host}:{port}: {detail}") from exc

    payload = json.loads(proc.stdout)
    end = payload.get("end", {})
    summary = end.get("sum_received") or end.get("sum_sent")
    if not isinstance(summary, dict) or "bits_per_second" not in summary:
        raise RuntimeError(f"iperf3 JSON did not contain a summary throughput: {payload}")
    return float(summary["bits_per_second"]), payload


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _run_iperf_series(target_host: str, port: int, duration_s: int, count: int, timeout_s: int) -> list[float]:
    results: list[float] = []
    for _ in range(count):
        bps, _ = _iperf_bits_per_second(
            target_host=target_host,
            port=port,
            duration_s=duration_s,
            timeout_s=timeout_s,
        )
        results.append(bps)
    return results


def _apply_tc(device: str, rate: str, burst: str, latency: str) -> None:
    try:
        _run([str(THROTTLE_SCRIPT), "apply", device, rate, burst, latency])
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(
            "failed to apply tc shaping. This usually means the process lacks CAP_NET_ADMIN, "
            f"the device is invalid, or the environment blocks qdisc changes: {detail}"
        ) from exc


def _clear_tc(device: str) -> None:
    subprocess.run(
        [str(THROTTLE_SCRIPT), "delete", device],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def run_server(args: argparse.Namespace) -> int:
    _require_binary("iperf3")
    cmd = ["iperf3", "-s", "-B", args.bind, "-p", str(args.port)]
    if args.one_off:
        cmd.append("-1")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return 0


def run_check(args: argparse.Namespace) -> int:
    _require_binary("ip")
    result = _tcp_connect_check(
        target_host=args.target_host,
        port=args.port,
        timeout_s=args.connect_timeout_s,
    )
    _write_json_output(args.json_output, result)
    print(json.dumps(result, indent=2))
    return 0 if bool(result["connectable"]) else 2


def run_validate(args: argparse.Namespace) -> int:
    global _ACTIVE_TC_DEVICE
    _require_binary("iperf3")
    _require_binary("ip")
    if not THROTTLE_SCRIPT.exists():
        raise FileNotFoundError(f"missing throttle helper: {THROTTLE_SCRIPT}")

    _install_signal_handlers()

    routed_device = _route_device(args.target_host)
    if routed_device != args.device and not args.allow_route_mismatch:
        raise RuntimeError(
            "selected device does not match route to target host: "
            f"target_host={args.target_host} routed_device={routed_device} requested_device={args.device}. "
            "Use the actual egress device, or pass --allow-route-mismatch if this is intentional."
        )

    preflight = _tcp_connect_check(
        target_host=args.target_host,
        port=args.port,
        timeout_s=args.connect_timeout_s,
    )
    if not bool(preflight["connectable"]):
        raise RuntimeError(
            "TCP preflight connection check failed before iperf3 baseline measurement: "
            f"target_host={args.target_host} port={args.port} device={routed_device} error={preflight['error']}"
        )

    baseline_runs = _run_iperf_series(
        target_host=args.target_host,
        port=args.port,
        duration_s=args.duration_s,
        count=args.baseline_runs,
        timeout_s=args.iperf_timeout_s,
    )

    try:
        _ACTIVE_TC_DEVICE = args.device
        _apply_tc(device=args.device, rate=args.rate, burst=args.burst, latency=args.latency)
        shaped_runs = _run_iperf_series(
            target_host=args.target_host,
            port=args.port,
            duration_s=args.duration_s,
            count=args.shaped_runs,
            timeout_s=args.iperf_timeout_s,
        )
    finally:
        _clear_tc(args.device)
        _ACTIVE_TC_DEVICE = None

    baseline_bps = _mean(baseline_runs)
    shaped_bps = _mean(shaped_runs)
    shaped_ratio = shaped_bps / baseline_bps if baseline_bps > 0 else float("inf")
    passed = shaped_ratio <= args.max_shaped_ratio

    summary = {
        "target_host": args.target_host,
        "port": args.port,
        "requested_device": args.device,
        "routed_device": routed_device,
        "rate": args.rate,
        "burst": args.burst,
        "latency": args.latency,
        "duration_s": args.duration_s,
        "baseline_runs_bps": baseline_runs,
        "shaped_runs_bps": shaped_runs,
        "baseline_mean_gbps": baseline_bps / 1e9,
        "shaped_mean_gbps": shaped_bps / 1e9,
        "shaped_ratio": shaped_ratio,
        "max_shaped_ratio": args.max_shaped_ratio,
        "passed": passed,
    }

    text = json.dumps(summary, indent=2)
    _write_json_output(args.json_output, summary)
    print(text)
    return 0 if passed else 2


def main() -> int:
    args = parse_args()
    try:
        if args.command == "check":
            return run_check(args)
        if args.command == "server":
            return run_server(args)
        if args.command == "validate":
            return run_validate(args)
        raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        _write_json_output(
            getattr(args, "json_output", ""),
            {"passed": False, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)