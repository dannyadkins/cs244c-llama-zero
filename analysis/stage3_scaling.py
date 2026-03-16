from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Figure-3-style Stage 3 GPU-scaling outputs")
    parser.add_argument("--run-dir", type=str, required=True, help="experiments/results/<name>")
    parser.add_argument("--figure-output", type=str, default="", help="Optional output path for the figure")
    parser.add_argument("--report-output", type=str, default="", help="Optional output path for the markdown report")
    parser.add_argument(
        "--baseline-gpu-count",
        type=int,
        default=0,
        help="GPU count used for the perfect-linear reference line. 0 means: use the smallest distributed point (>1) if present, else the smallest point.",
    )
    return parser.parse_args()


def load_summary(run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json under {run_dir}")
    payload = json.loads(summary_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary.json under {run_dir} must contain a JSON object")
    return payload


def scaling_points(summary: Dict[str, object]) -> List[Dict[str, object]]:
    raw_points = summary.get("scaling_points", [])
    if not isinstance(raw_points, list):
        raise ValueError("summary scaling_points must be a list")

    out: List[Dict[str, object]] = []
    for raw in raw_points:
        if not isinstance(raw, dict):
            continue
        if raw.get("mean_tflops_per_s") is None or raw.get("per_gpu_tflops_per_s") is None:
            continue
        out.append(
            {
                "gpu_count": int(raw["gpu_count"]),
                "selected_batch_size": None if raw.get("selected_batch_size") is None else int(raw["selected_batch_size"]),
                "selected_peak_memory_mb": None
                if raw.get("selected_peak_memory_mb") is None
                else float(raw["selected_peak_memory_mb"]),
                "global_tokens_per_step": None
                if raw.get("global_tokens_per_step") is None
                else int(raw["global_tokens_per_step"]),
                "mean_tokens_per_s": None if raw.get("mean_tokens_per_s") is None else float(raw["mean_tokens_per_s"]),
                "mean_tflops_per_s": float(raw["mean_tflops_per_s"]),
                "per_gpu_tflops_per_s": float(raw["per_gpu_tflops_per_s"]),
                "perfect_linear_tflops_per_s": None
                if raw.get("perfect_linear_tflops_per_s") is None
                else float(raw["perfect_linear_tflops_per_s"]),
                "speedup_vs_base": None if raw.get("speedup_vs_base") is None else float(raw["speedup_vs_base"]),
                "scaling_efficiency_vs_base": None
                if raw.get("scaling_efficiency_vs_base") is None
                else float(raw["scaling_efficiency_vs_base"]),
                "superlinear_gain_vs_perfect_linear": None
                if raw.get("superlinear_gain_vs_perfect_linear") is None
                else float(raw["superlinear_gain_vs_perfect_linear"]),
                "peak_cuda_max_allocated_mb": None
                if raw.get("peak_cuda_max_allocated_mb") is None
                else float(raw["peak_cuda_max_allocated_mb"]),
                "peak_memory_mb": None
                if raw.get("selected_peak_memory_mb") is None and raw.get("peak_cuda_max_allocated_mb") is None
                else float(
                    raw["selected_peak_memory_mb"]
                    if raw.get("selected_peak_memory_mb") is not None
                    else raw["peak_cuda_max_allocated_mb"]
                ),
            }
        )
    out.sort(key=lambda item: item["gpu_count"])
    if not out:
        raise RuntimeError("no successful scaling points were found in summary.json")
    return out


def choose_baseline_gpu_count(points: List[Dict[str, object]], requested_gpu_count: int) -> int:
    available = {int(point["gpu_count"]) for point in points}
    if requested_gpu_count > 0:
        if requested_gpu_count not in available:
            raise ValueError(f"requested baseline GPU count {requested_gpu_count} is not present in the scaling points")
        return requested_gpu_count

    distributed_points = sorted(gpu_count for gpu_count in available if gpu_count > 1)
    if distributed_points:
        return distributed_points[0]
    return min(available)


def annotate_points(points: List[Dict[str, object]], baseline_gpu_count: int) -> List[Dict[str, object]]:
    point_by_gpu = {int(point["gpu_count"]): dict(point) for point in points}
    if baseline_gpu_count not in point_by_gpu:
        raise ValueError(f"baseline GPU count {baseline_gpu_count} is not present in the scaling points")

    baseline = point_by_gpu[baseline_gpu_count]
    baseline_tflops = float(baseline["mean_tflops_per_s"])
    annotated: List[Dict[str, object]] = []
    for gpu_count in sorted(point_by_gpu):
        if gpu_count < baseline_gpu_count:
            continue
        point = point_by_gpu[gpu_count]
        perfect_linear_tflops = baseline_tflops * (float(gpu_count) / float(baseline_gpu_count))
        speedup = float(point["mean_tflops_per_s"]) / baseline_tflops
        ideal_speedup = float(gpu_count) / float(baseline_gpu_count)
        point["perfect_linear_tflops_per_s"] = perfect_linear_tflops
        point["speedup_vs_base"] = speedup
        point["scaling_efficiency_vs_base"] = speedup / ideal_speedup if ideal_speedup > 0 else None
        point["superlinear_gain_vs_perfect_linear"] = (
            float(point["mean_tflops_per_s"]) / perfect_linear_tflops if perfect_linear_tflops > 0 else None
        )
        annotated.append(point)
    return annotated


def plot_figure(points: List[Dict[str, object]], output: Path, baseline_gpu_count: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    xs = list(range(len(points)))
    labels = [str(point["gpu_count"]) for point in points]
    per_gpu = [float(point["per_gpu_tflops_per_s"]) for point in points]
    observed = [float(point["mean_tflops_per_s"]) for point in points]
    perfect_linear = [
        float(point["perfect_linear_tflops_per_s"])
        if point["perfect_linear_tflops_per_s"] is not None
        else float(point["mean_tflops_per_s"])
        for point in points
    ]

    fig, ax_total = plt.subplots(figsize=(9.2, 5.6))
    ax_gpu = ax_total.twinx()

    ax_gpu.bar(xs, per_gpu, width=0.46, color="#d0d0d0", edgecolor="#a8a8a8", label="Performance/GPU (TFLOPs)")
    ax_total.plot(xs, observed, color="#1b9e77", marker="o", linewidth=2.4, label="Observed Performance (TFLOPs)")
    ax_total.plot(
        xs,
        perfect_linear,
        color="#377eb8",
        marker="o",
        linewidth=2.0,
        label="Perfect Linear Scalability (TFLOPs)",
    )

    ax_total.set_xticks(xs)
    ax_total.set_xticklabels(labels)
    ax_total.set_xlabel("Number of GPUs")
    ax_total.set_ylabel("Total Performance (TFLOPs)")
    ax_gpu.set_ylabel("Performance/GPU (TFLOPs)")
    ax_total.grid(axis="y", linestyle="--", alpha=0.3)
    ax_total.set_title(f"Stage 3 Scaling (baseline {baseline_gpu_count} GPU)")

    handles_total, labels_total = ax_total.get_legend_handles_labels()
    handles_gpu, labels_gpu = ax_gpu.get_legend_handles_labels()
    fig.legend(
        handles_gpu + handles_total,
        labels_gpu + labels_total,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def render_report(summary: Dict[str, object], points: List[Dict[str, object]], baseline_gpu_count: int) -> str:
    lines: List[str] = []
    lines.append("# Stage 3 GPU Scaling")
    lines.append("")
    lines.append("Methodology:")
    scaling_mode = str(summary.get("scaling_mode", "fit_to_oom"))
    lines.append(
        "- Adapted from Figure 3 of the ZeRO paper: fixed model, sweep GPU count, and measure total TFLOPs plus TFLOPs/GPU."
    )
    if scaling_mode == "fixed_batch":
        lines.append(
            f"- This run uses one fixed per-GPU microbatch across all GPU counts: `{summary.get('fixed_batch_size', 'unknown')}`."
        )
    else:
        lines.append(
            "- For each GPU count, this run tunes the largest ZeRO Stage 3 microbatch that fits, then benchmarks that tuned workload."
        )
    lines.append(
        "- This differs from the original paper in scale: the paper reports a 60B model from 64 to 400 GPUs, while this run uses one 16-GPU host and this repo's preset model sizes."
    )
    lines.append("")
    run_args = summary.get("args", {})
    lines.append("Configuration:")
    lines.append(f"- model_size: `{run_args.get('model_size', 'unknown')}`")
    lines.append(f"- seq_len: `{run_args.get('seq_len', 'unknown')}`")
    lines.append(f"- dtype: `{run_args.get('dtype', 'unknown')}`")
    lines.append(f"- tflops_mode: `{run_args.get('tflops_mode', 'unknown')}`")
    lines.append(f"- steps: `{run_args.get('steps', 'unknown')}`")
    lines.append(f"- baseline_gpu_count_for_linear_reference: `{baseline_gpu_count}`")
    lines.append("")
    lines.append("| GPUs | batch/GPU | global tokens/step | total TFLOPs | TFLOPs/GPU | perfect linear TFLOPs | speedup vs base | efficiency | peak alloc MB |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for point in points:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(point["gpu_count"]),
                    str(point["selected_batch_size"]),
                    str(point["global_tokens_per_step"]),
                    _format_float(point["mean_tflops_per_s"]),
                    _format_float(point["per_gpu_tflops_per_s"]),
                    _format_float(point["perfect_linear_tflops_per_s"]),
                    _format_float(point["speedup_vs_base"]),
                    _format_float(point["scaling_efficiency_vs_base"], digits=3),
                    _format_float(point["peak_memory_mb"]),
                ]
            )
            + " |"
        )

    base_point = next(point for point in points if int(point["gpu_count"]) == baseline_gpu_count)
    best_point = max(points, key=lambda item: float(item["mean_tflops_per_s"]))
    lines.append("")
    lines.append("Observations:")
    lines.append(
        f"- Base point: `{base_point['gpu_count']}` GPU at `{_format_float(base_point['mean_tflops_per_s'])}` TFLOPs total and `{_format_float(base_point['per_gpu_tflops_per_s'])}` TFLOPs/GPU."
    )
    lines.append(
        f"- Best total throughput: `{best_point['gpu_count']}` GPUs at `{_format_float(best_point['mean_tflops_per_s'])}` TFLOPs."
    )
    superlinear_points = [
        point for point in points if point["superlinear_gain_vs_perfect_linear"] is not None and point["superlinear_gain_vs_perfect_linear"] > 1.01
    ]
    if superlinear_points:
        strongest = max(superlinear_points, key=lambda item: float(item["superlinear_gain_vs_perfect_linear"]))
        lines.append(
            f"- Strongest superlinear gain vs the base-point linear projection: `{strongest['gpu_count']}` GPUs at `{_format_float(strongest['superlinear_gain_vs_perfect_linear'], digits=3)}x` of perfect-linear."
        )
    else:
        lines.append("- No superlinear gain above the base-point linear projection was observed in this run.")

    skipped: List[str] = []
    raw_per_gpu_count = summary.get("per_gpu_count", {})
    if isinstance(raw_per_gpu_count, dict):
        for gpu_key, payload in raw_per_gpu_count.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("fit_status") == "no_fit":
                skipped.append(str(gpu_key))
    if skipped:
        lines.append(f"- GPU counts skipped for lack of fit at microbatch 1: `{', '.join(sorted(skipped, key=int))}`.")

    lines.append("")
    lines.append("Reference:")
    lines.append("- ZeRO paper, Figure 3: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = load_summary(run_dir)
    raw_points = scaling_points(summary)
    baseline_gpu_count = choose_baseline_gpu_count(raw_points, int(args.baseline_gpu_count))
    points = annotate_points(raw_points, baseline_gpu_count)

    if args.figure_output:
        plot_figure(points, Path(args.figure_output), baseline_gpu_count)
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(summary, points, baseline_gpu_count))


if __name__ == "__main__":
    main()
