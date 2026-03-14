from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 vs Stage-3 communication-shape probe")
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--model-size", default="", help="Optional model filter, e.g. small")
    parser.add_argument("--start-step", type=int, default=2, help="Ignore initial steps before this number")
    parser.add_argument("--bandwidth-gbps", nargs="+", type=float, default=None, help="Optional exact bandwidth filter")
    parser.add_argument("--output-csv", default="", help="Optional CSV output path")
    return parser.parse_args()


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _read_bandwidth(case_id: str) -> float:
    match = re.search(r"bw(?P<raw>[^_]+)", case_id)
    if not match:
        return 0.0
    raw = match.group("raw")
    if raw == "unlimited":
        return 0.0
    if raw.endswith("gbps"):
        raw = raw[:-4]
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _to_str_ms(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value:.{precision}f}"


def _to_str_ratio(value: float | None, precision: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.{precision}f}%"


def _to_str_mb(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value / (1024.0 * 1024.0):.{precision}f}"


def _to_str_calls(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value:.{precision}f}"


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "No data."
    sep = "| " + " | ".join(headers) + " |"
    hdr = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([sep, hdr] + body)


def _summarize_case(payload: Dict[str, object], start_step: int) -> Dict[str, float | int | None] | None:
    stage = int(payload.get("stage", 0))
    model = str(payload.get("model_config", {}).get("name", "unknown"))
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return None

    filtered = [row for row in steps if isinstance(row, dict) and int(row.get("step", 0)) >= start_step]
    if not filtered:
        return None

    def s(key: str) -> List[float]:
        return [float(row.get(key, 0.0)) for row in filtered]

    summary: Dict[str, float | int | None] = {
        "stage": stage,
        "model": model,
        "samples": len(filtered),
        "iteration_ms": _mean(s("iteration_ms")),
        "forward_backward_ms": _mean(s("forward_backward_ms")),
        "optimizer_ms": _mean(s("optimizer_ms")),
        "communication_ms": _mean(s("communication_ms")),
        "tokens_per_second": _mean(s("tokens_per_second")),
        "backward_reduce_scatter_ms": _mean(s("communication_backward_reduce_scatter_ms")),
        "backward_reduce_scatter_calls": _mean(s("communication_backward_reduce_scatter_calls")),
        "backward_reduce_scatter_bytes": _mean(s("communication_backward_reduce_scatter_bytes")),
        "post_allgather_ms": _mean(s("communication_post_allgather_ms")),
        "post_allgather_calls": _mean(s("communication_post_allgather_calls")),
        "post_allgather_bytes": _mean(s("communication_post_allgather_bytes")),
        "forward_allgather_ms": _mean(s("communication_forward_allgather_ms")),
        "forward_allgather_calls": _mean(s("communication_forward_allgather_calls")),
        "forward_allgather_bytes": _mean(s("communication_forward_allgather_bytes")),
        "backward_allgather_ms": _mean(s("communication_backward_allgather_ms")),
        "backward_allgather_calls": _mean(s("communication_backward_allgather_calls")),
        "backward_allgather_bytes": _mean(s("communication_backward_allgather_bytes")),
    }
    if stage == 2:
        for key in [
            "forward_allgather_ms",
            "forward_allgather_calls",
            "forward_allgather_bytes",
            "backward_allgather_ms",
            "backward_allgather_calls",
            "backward_allgather_bytes",
        ]:
            summary[key] = 0.0
    return summary


def _load_profiles(run_dir: Path, model_filter: str, start_step: int, bandwidth_filter: set[float] | None) -> Dict[Tuple[str, float, int], Dict[str, object]]:
    profile_dir = run_dir / "profiles"
    if not profile_dir.exists():
        raise FileNotFoundError(f"missing profiles directory: {profile_dir}")

    out: Dict[Tuple[str, float, int], Dict[str, object]] = {}

    for profile_path in sorted(profile_dir.glob("*.json")):
        payload = json.loads(profile_path.read_text())
        stage = int(payload.get("stage", 0))
        if stage not in {2, 3}:
            continue

        model = str(payload.get("model_config", {}).get("name", "unknown"))
        if model_filter and model != model_filter:
            continue

        bandwidth = _read_bandwidth(profile_path.stem)
        if bandwidth_filter is not None and bandwidth not in bandwidth_filter:
            continue

        summary = _summarize_case(payload, start_step=start_step)
        if summary is None:
            continue

        summary["profile_path"] = str(profile_path)
        out[(model, bandwidth, stage)] = summary

    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    bandwidth_filter = None if args.bandwidth_gbps is None else {float(v) for v in args.bandwidth_gbps}
    cases = _load_profiles(run_dir, model_filter=args.model_size, start_step=args.start_step, bandwidth_filter=bandwidth_filter)
    if not cases:
        raise RuntimeError("no matching profiles found")

    stage2_rows: List[List[str]] = []
    stage3_rows: List[List[str]] = []
    compare_rows: List[List[str]] = []
    by_model_bandwidth: Dict[Tuple[str, float], Dict[int, Dict[str, object]]] = defaultdict(dict)

    for (model, bandwidth, stage), summary in sorted(cases.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        it = float(summary["iteration_ms"]) if summary["iteration_ms"] is not None else None
        comm_ms = float(summary["communication_ms"]) if summary["communication_ms"] is not None else None
        comm_share = None if it in (None, 0.0) else (comm_ms or 0.0) / it

        if stage == 2:
            rs_ms = float(summary["backward_reduce_scatter_ms"])
            rs_calls = float(summary["backward_reduce_scatter_calls"])
            rs_bytes = float(summary["backward_reduce_scatter_bytes"])
            ag_ms = float(summary["post_allgather_ms"])
            ag_calls = float(summary["post_allgather_calls"])
            ag_bytes = float(summary["post_allgather_bytes"])
            stage2_rows.append(
                [
                    model,
                    f"{bandwidth:g}",
                    str(summary["samples"]),
                    _to_str_ms(it),
                    _to_str_ms(comm_ms),
                    _to_str_ratio(comm_share),
                    _to_str_ms(float(summary["forward_backward_ms"])),
                    _to_str_ms(float(summary["optimizer_ms"])),
                    _to_str_ms(rs_ms),
                    _to_str_calls(rs_calls),
                    _to_str_mb(rs_bytes),
                    _to_str_ms(None if rs_calls in (None, 0.0) else rs_ms / rs_calls),
                    _to_str_ms(ag_ms),
                    _to_str_calls(ag_calls),
                    _to_str_mb(ag_bytes),
                    _to_str_ms(None if ag_calls in (None, 0.0) else ag_ms / ag_calls),
                ]
            )
        else:
            fa_ms = float(summary["forward_allgather_ms"])
            fa_calls = float(summary["forward_allgather_calls"])
            fa_bytes = float(summary["forward_allgather_bytes"])
            ba_ms = float(summary["backward_allgather_ms"])
            ba_calls = float(summary["backward_allgather_calls"])
            ba_bytes = float(summary["backward_allgather_bytes"])
            rs_ms = float(summary["backward_reduce_scatter_ms"])
            rs_calls = float(summary["backward_reduce_scatter_calls"])
            rs_bytes = float(summary["backward_reduce_scatter_bytes"])
            stage3_rows.append(
                [
                    model,
                    f"{bandwidth:g}",
                    str(summary["samples"]),
                    _to_str_ms(it),
                    _to_str_ms(comm_ms),
                    _to_str_ratio(comm_share),
                    _to_str_ms(float(summary["forward_backward_ms"])),
                    _to_str_ms(float(summary["optimizer_ms"])),
                    _to_str_ms(fa_ms),
                    _to_str_calls(fa_calls),
                    _to_str_mb(fa_bytes),
                    _to_str_ms(None if fa_calls in (None, 0.0) else fa_ms / fa_calls),
                    _to_str_ms(ba_ms),
                    _to_str_calls(ba_calls),
                    _to_str_mb(ba_bytes),
                    _to_str_ms(None if ba_calls in (None, 0.0) else ba_ms / ba_calls),
                    _to_str_ms(rs_ms),
                    _to_str_calls(rs_calls),
                    _to_str_mb(rs_bytes),
                    _to_str_ms(None if rs_calls in (None, 0.0) else rs_ms / rs_calls),
                ]
            )

        by_model_bandwidth[(model, bandwidth)][int(stage)] = summary

    for (model, bandwidth), stage_map in sorted(by_model_bandwidth.items(), key=lambda item: (item[0][0], item[0][1])):
        s2 = stage_map.get(2)
        s3 = stage_map.get(3)
        if not (s2 and s3):
            continue
        i2 = float(s2["iteration_ms"]) if s2["iteration_ms"] is not None else None
        i3 = float(s3["iteration_ms"]) if s3["iteration_ms"] is not None else None
        c2 = float(s2["communication_ms"]) if s2["communication_ms"] is not None else None
        c3 = float(s3["communication_ms"]) if s3["communication_ms"] is not None else None
        speedup = None if i2 in (None, 0.0) else (i2 / i3) if i3 not in (None, 0.0) else None
        comm_ratio = None if c2 in (None, 0.0) or c3 in (None, 0.0) else c2 / c3

        compare_rows.append(
            [
                model,
                f"{bandwidth:g}",
                _to_str_ms(i2),
                _to_str_ms(i3),
                _to_str_ratio(speedup, precision=1),
                _to_str_ratio(comm_ratio, precision=1),
                _to_str_ms(float(s2["post_allgather_ms"])),
                _to_str_ms(float(s3["backward_allgather_ms"])),
                _to_str_ms(float(s2["backward_reduce_scatter_ms"])),
                _to_str_ms(float(s3["backward_reduce_scatter_ms"])),
            ]
        )

    print("\n## Stage 2 breakdown")
    print(
        _markdown_table(
            headers=[
                "Model",
                "BW",
                "Steps",
                "Iter(ms)",
                "Comm(ms)",
                "Comm%",
                "FB(ms)",
                "Opt(ms)",
                "RS(ms)",
                "RS calls",
                "RS MB/step",
                "RS ms/call",
                "PostAG(ms)",
                "PostAG calls",
                "PostAG MB/step",
                "PostAG ms/call",
            ],
            rows=stage2_rows,
        )
    )

    print("\n## Stage 3 breakdown")
    print(
        _markdown_table(
            headers=[
                "Model",
                "BW",
                "Steps",
                "Iter(ms)",
                "Comm(ms)",
                "Comm%",
                "FB(ms)",
                "Opt(ms)",
                "FwdAG(ms)",
                "FwdAG calls",
                "FwdAG MB/step",
                "FwdAG ms/call",
                "BwdAG(ms)",
                "BwdAG calls",
                "BwdAG MB/step",
                "BwdAG ms/call",
                "RS(ms)",
                "RS calls",
                "RS MB/step",
                "RS ms/call",
            ],
            rows=stage3_rows,
        )
    )

    print("\n## Stage 2 vs 3 (same bandwidth)")
    print(
        _markdown_table(
            headers=[
                "Model",
                "BW",
                "S2 Iter(ms)",
                "S3 Iter(ms)",
                "Speedup",
                "Comm Ratio",
                "S2 Post-allgather(ms)",
                "S3 Backward AG(ms)",
                "S2 RS(ms)",
                "S3 RS(ms)",
            ],
            rows=compare_rows,
        )
    )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "model",
                    "bandwidth_gbps",
                    "stage",
                    "samples",
                    "iteration_ms",
                    "forward_backward_ms",
                    "optimizer_ms",
                    "communication_ms",
                    "communication_ratio",
                    "forward_allgather_ms",
                    "forward_allgather_calls",
                    "forward_allgather_bytes",
                    "backward_allgather_ms",
                    "backward_allgather_calls",
                    "backward_allgather_bytes",
                    "backward_reduce_scatter_ms",
                    "backward_reduce_scatter_calls",
                    "backward_reduce_scatter_bytes",
                    "post_allgather_ms",
                    "post_allgather_calls",
                    "post_allgather_bytes",
                ]
            )
            for (model, bandwidth, stage), summary in sorted(cases.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
                writer.writerow(
                    [
                        model,
                        bandwidth,
                        stage,
                        summary["samples"],
                        summary["iteration_ms"],
                        summary["forward_backward_ms"],
                        summary["optimizer_ms"],
                        summary["communication_ms"],
                        None
                        if summary["iteration_ms"] in (None, 0.0)
                        else (float(summary["communication_ms"]) / float(summary["iteration_ms"])),
                        summary["forward_allgather_ms"],
                        summary["forward_allgather_calls"],
                        summary["forward_allgather_bytes"],
                        summary["backward_allgather_ms"],
                        summary["backward_allgather_calls"],
                        summary["backward_allgather_bytes"],
                        summary["backward_reduce_scatter_ms"],
                        summary["backward_reduce_scatter_calls"],
                        summary["backward_reduce_scatter_bytes"],
                        summary["post_allgather_ms"],
                        summary["post_allgather_calls"],
                        summary["post_allgather_bytes"],
                    ]
                )
        print(f"\nWrote CSV: {output_path}")


if __name__ == "__main__":
    main()
