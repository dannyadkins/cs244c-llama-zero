from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Point:
    bandwidth_gbps: float
    step_time_s: float
    tokens_per_s: float
    stage: int
    batch_size: int
    seq_len: int
    grad_accum_steps: int
    world_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stage-2 vs stage-3 step latency from bandwidth runs")
    parser.add_argument("--stage2-run", action="append", required=True, help="Run directory containing summary.json for stage 2")
    parser.add_argument("--stage3-run", action="append", required=True, help="Run directory containing summary.json for stage 3")
    parser.add_argument(
        "--output",
        default="report/figures/stage2_stage3_step_latency.png",
        help="Output figure path",
    )
    return parser.parse_args()


def _load_run_points(run_dir: Path) -> List[Point]:
    payload = json.loads((run_dir / "summary.json").read_text())
    world_size = int(payload["args"]["nproc_per_node"])
    out: List[Point] = []
    for result in payload["results"]:
        if int(result.get("return_code", 1)) != 0:
            continue
        tokens_per_s = float(result["mean_tokens_per_s"])
        config = result["config"]
        batch_size = int(config["batch_size"])
        seq_len = int(config["seq_len"])
        grad_accum_steps = int(config["grad_accum_steps"])
        tokens_per_step = batch_size * seq_len * grad_accum_steps * world_size
        out.append(
            Point(
                bandwidth_gbps=float(config["bandwidth_gbps"]),
                step_time_s=tokens_per_step / tokens_per_s,
                tokens_per_s=tokens_per_s,
                stage=int(config["stage"]),
                batch_size=batch_size,
                seq_len=seq_len,
                grad_accum_steps=grad_accum_steps,
                world_size=world_size,
            )
        )
    out.sort(key=lambda point: (point.bandwidth_gbps <= 0, point.bandwidth_gbps))
    return out


def _load_points(run_dirs: List[str]) -> List[Point]:
    by_bandwidth: dict[float, Point] = {}
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        for point in _load_run_points(run_dir):
            by_bandwidth[point.bandwidth_gbps] = point
    return sorted(by_bandwidth.values(), key=lambda point: (point.bandwidth_gbps <= 0, -point.bandwidth_gbps if point.bandwidth_gbps > 0 else 0.0))


def _fit_bandwidth_model(points: Iterable[Point]) -> tuple[float, float]:
    points = list(points)
    unlimited = [point for point in points if point.bandwidth_gbps <= 0]
    finite = [point for point in points if point.bandwidth_gbps > 0]
    if len(unlimited) != 1:
        raise ValueError("expected exactly one unlimited-bandwidth point")
    if not finite:
        raise ValueError("expected at least one finite-bandwidth point")

    a = unlimited[0].step_time_s
    if len(finite) == 1:
        point = finite[0]
        b = (point.step_time_s - a) * point.bandwidth_gbps
        return a, b

    xs = [1.0 / point.bandwidth_gbps for point in finite]
    ys = [point.step_time_s - a for point in finite]
    denom = sum(x * x for x in xs)
    if denom <= 0:
        raise ValueError("degenerate finite-bandwidth points")
    b = sum(x * y for x, y in zip(xs, ys)) / denom
    return a, b


def _ratio_stage3_over_stage2(a2: float, b2: float, a3: float, b3: float, token_ratio: float, bandwidth_gbps: float) -> float:
    t2 = a2 + b2 / bandwidth_gbps
    t3 = a3 + b3 / bandwidth_gbps
    return token_ratio * t2 / t3


def main() -> None:
    args = parse_args()
    stage2_points = _load_points(args.stage2_run)
    stage3_points = _load_points(args.stage3_run)

    if not stage2_points or not stage3_points:
        raise RuntimeError("missing successful points")

    a2, b2 = _fit_bandwidth_model(stage2_points)
    a3, b3 = _fit_bandwidth_model(stage3_points)

    token_ratio = (
        stage3_points[0].batch_size * stage3_points[0].seq_len * stage3_points[0].grad_accum_steps * stage3_points[0].world_size
    ) / (
        stage2_points[0].batch_size * stage2_points[0].seq_len * stage2_points[0].grad_accum_steps * stage2_points[0].world_size
    )

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for visualization") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    finite_points = sorted(
        {point.bandwidth_gbps for point in stage2_points + stage3_points if point.bandwidth_gbps > 0},
        reverse=True,
    )
    x_min = min(finite_points)
    x_max = max(finite_points)
    curve_x = np.logspace(np.log10(x_min), np.log10(max(x_max, x_min * 1.01)), 200)

    fig, (ax_latency, ax_ratio) = plt.subplots(2, 1, figsize=(10.6, 7.6), sharex=True)

    colors = {2: "#2458A6", 3: "#B04A1F"}

    for points, stage, a, b in [
        (stage2_points, 2, a2, b2),
        (stage3_points, 3, a3, b3),
    ]:
        finite = [point for point in points if point.bandwidth_gbps > 0]
        xs = [point.bandwidth_gbps for point in finite]
        ys = [point.step_time_s for point in finite]
        ax_latency.plot(xs, ys, "o", color=colors[stage], label=f"stage {stage} observed")
        ax_latency.plot(curve_x, a + b / curve_x, "-", color=colors[stage], linewidth=2)
        ax_latency.axhline(a, color=colors[stage], linestyle="--", alpha=0.7)
        ax_latency.text(x_max * 1.04, a, f"stage {stage} inf-bw = {a:.2f}s", color=colors[stage], va="center", fontsize=9)

    ratio_curve = [(a3 + b3 / x) / (a2 + b2 / x) for x in curve_x]
    throughput_ratio_curve = [_ratio_stage3_over_stage2(a2, b2, a3, b3, token_ratio, x) for x in curve_x]
    ax_ratio.plot(curve_x, ratio_curve, color="#5A5A5A", linewidth=2, label="step latency ratio (stage 3 / stage 2)")
    ax_ratio.plot(curve_x, throughput_ratio_curve, color="#2F8F6A", linewidth=2, linestyle="--", label="throughput ratio (stage 3 / stage 2)")
    ax_ratio.axhline(1.0, color="black", linestyle=":", alpha=0.7)

    ax_latency.set_xscale("log")
    ax_ratio.set_xscale("log")
    ax_latency.set_ylabel("Step Latency (s)")
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_xlabel("Bandwidth (Gbps, log scale)")
    ax_latency.set_title("Stage 2 vs Stage 3: Step Latency and Throughput Ratios")
    ax_latency.grid(True, alpha=0.3)
    ax_ratio.grid(True, alpha=0.3)
    ax_latency.legend(loc="upper left")
    ax_ratio.legend(loc="upper left")

    latency_ratio_inf = a3 / a2
    latency_ratio_low = b3 / b2
    throughput_ratio_inf = token_ratio * a2 / a3
    throughput_ratio_low = token_ratio * b2 / b3

    subtitle = (
        f"Latency ratio: {latency_ratio_inf:.2f}x at inf-bw -> {latency_ratio_low:.2f}x at low-bw.  "
        f"Throughput ratio: {throughput_ratio_inf:.2f}x -> {throughput_ratio_low:.2f}x."
    )
    fig.subplots_adjust(bottom=0.14, hspace=0.18, right=0.86)
    fig.text(0.5, 0.035, subtitle, ha="center", fontsize=9)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")
    print(f"stage2 model: step_time = {a2:.6f} + {b2:.6f} / B")
    print(f"stage3 model: step_time = {a3:.6f} + {b3:.6f} / B")
    print(f"latency ratio inf-bw = {latency_ratio_inf:.6f}")
    print(f"latency ratio low-bw = {latency_ratio_low:.6f}")
    print(f"throughput ratio inf-bw = {throughput_ratio_inf:.6f}")
    print(f"throughput ratio low-bw = {throughput_ratio_low:.6f}")
    if latency_ratio_inf > 1.0 and latency_ratio_low > 1.0:
        print("No step-latency crossover: stage 3 is slower per step at both high and low bandwidth.")


if __name__ == "__main__":
    main()
