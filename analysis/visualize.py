from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


STEP_RE = re.compile(r"\[step\s+(\d+)\]\s+loss=([0-9.]+)")


@dataclass
class CaseView:
    stage: int
    model_size: str
    bandwidth_gbps: float
    log_path: Path
    mean_tokens_per_s: float | None
    mean_comm_ms: float | None
    mean_fb_ms: float | None
    mean_opt_ms: float | None
    peak_host_rss_mb: float | None
    peak_cuda_allocated_mb: float | None
    peak_cuda_reserved_mb: float | None
    peak_cuda_max_allocated_mb: float | None
    peak_cuda_max_reserved_mb: float | None
    final_loss: float | None
    return_code: int
    measured_state_memory_mb: Dict[str, float] | None
    theoretical_memory_mb: Dict[str, float] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Week 2/3 harness outputs")
    parser.add_argument("--run-dir", type=str, required=True, help="experiments/results/<name>")
    parser.add_argument(
        "--plot",
        type=str,
        default="throughput",
        choices=["loss", "throughput", "comm", "stage-throughput", "stage-comm", "memory", "peak-memory"],
        help="Plot type: per-step loss, bandwidth sweep throughput/comm, stage comparison throughput/comm, measured state memory, or measured peak memory",
    )
    parser.add_argument("--output", type=str, default="analysis/figures/week3_plot.png")
    parser.add_argument("--model-size", type=str, default="", help="Optional filter for one model size")
    parser.add_argument(
        "--bandwidth-gbps-filter",
        type=float,
        default=None,
        help="Optional exact-match bandwidth filter for memory-oriented plots",
    )
    return parser.parse_args()


def _result_stage(result: Dict[str, object]) -> int:
    if "stage" in result:
        return int(result["stage"])
    config = result.get("config", {})
    return int(config.get("stage", -1))


def _result_model_size(result: Dict[str, object]) -> str:
    config = result.get("config", {})
    return str(config.get("model_size", "unknown"))


def _result_bandwidth(result: Dict[str, object]) -> float:
    config = result.get("config", {})
    return float(config.get("bandwidth_gbps", 0.0))


def _result_log_path(result: Dict[str, object]) -> Path:
    if "log_path" in result:
        return Path(str(result["log_path"]))
    # Week-2 fallback format.
    stage = _result_stage(result)
    return Path(f"stage{stage}.log")


def parse_summary(summary_path: Path) -> List[CaseView]:
    summary = json.loads(summary_path.read_text())
    out: List[CaseView] = []
    run_dir = summary_path.parent

    for result in summary.get("results", []):
        raw_log_path = _result_log_path(result)
        if not raw_log_path.is_absolute() and not raw_log_path.exists():
            raw_log_path = run_dir / raw_log_path

        out.append(
            CaseView(
                stage=_result_stage(result),
                model_size=_result_model_size(result),
                bandwidth_gbps=_result_bandwidth(result),
                log_path=raw_log_path,
                mean_tokens_per_s=_as_optional_float(result.get("mean_tokens_per_s")),
                mean_comm_ms=_as_optional_float(result.get("mean_comm_ms")),
                mean_fb_ms=_as_optional_float(result.get("mean_fb_ms")),
                mean_opt_ms=_as_optional_float(result.get("mean_opt_ms")),
                peak_host_rss_mb=_as_optional_float(result.get("peak_host_rss_mb")),
                peak_cuda_allocated_mb=_as_optional_float(result.get("peak_cuda_allocated_mb")),
                peak_cuda_reserved_mb=_as_optional_float(result.get("peak_cuda_reserved_mb")),
                peak_cuda_max_allocated_mb=_as_optional_float(result.get("peak_cuda_max_allocated_mb")),
                peak_cuda_max_reserved_mb=_as_optional_float(result.get("peak_cuda_max_reserved_mb")),
                final_loss=_as_optional_float(result.get("final_loss")),
                return_code=int(result.get("return_code", 0)),
                measured_state_memory_mb=_as_optional_memory(result.get("measured_state_memory_mb")),
                theoretical_memory_mb=_as_optional_memory(result.get("theoretical_memory_mb")),
            )
        )
    return out


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_optional_memory(value: object) -> Dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    return {str(key): float(val) for key, val in value.items()}


def _filter_cases(cases: Iterable[CaseView], model_size: str) -> List[CaseView]:
    if not model_size:
        return list(cases)
    return [c for c in cases if c.model_size == model_size]


def _filter_bandwidth(cases: Iterable[CaseView], bandwidth_gbps: float | None) -> List[CaseView]:
    if bandwidth_gbps is None:
        return list(cases)
    return [c for c in cases if abs(c.bandwidth_gbps - bandwidth_gbps) < 1e-9]


def _representative_cases_by_stage(cases: Iterable[CaseView]) -> List[CaseView]:
    def sort_key(case: CaseView) -> tuple[int, float]:
        # Prefer the unlimited/no-simulation baseline when present; otherwise use the fastest link.
        return (1 if case.bandwidth_gbps > 0 else 0, -case.bandwidth_gbps)

    best: Dict[int, CaseView] = {}
    for case in cases:
        current = best.get(case.stage)
        if current is None or sort_key(case) < sort_key(current):
            best[case.stage] = case
    return [best[stage] for stage in sorted(best)]


def _case_peak_memory_mb(case: CaseView) -> float | None:
    cuda_candidates = [
        case.peak_cuda_allocated_mb,
        case.peak_cuda_max_allocated_mb,
        case.peak_cuda_max_reserved_mb,
        case.peak_cuda_reserved_mb,
    ]
    positive_cuda = [value for value in cuda_candidates if value is not None and value > 0.0]
    if positive_cuda:
        return positive_cuda[0]
    return case.peak_host_rss_mb


def parse_loss_log(path: Path) -> tuple[list[int], list[float]]:
    steps: List[int] = []
    losses: List[float] = []
    if not path.exists():
        return steps, losses
    for line in path.read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return steps, losses


def plot_loss(cases: List[CaseView], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    plt.figure(figsize=(9, 5))
    plotted = 0
    for case in sorted(cases, key=lambda c: (c.stage, c.bandwidth_gbps)):
        steps, losses = parse_loss_log(case.log_path)
        if not steps:
            continue
        bw_label = "unlimited" if case.bandwidth_gbps <= 0 else f"{case.bandwidth_gbps:g}Gbps"
        label = f"stage {case.stage}, bw {bw_label}, model {case.model_size}"
        plt.plot(steps, losses, label=label, linewidth=1.8)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No parsable loss data found in logs for the selected cases")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("ZeRO Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)


def plot_bandwidth_metric(cases: List[CaseView], metric: str, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    key = "mean_tokens_per_s" if metric == "throughput" else "mean_comm_ms"
    ylabel = "Tokens / s" if metric == "throughput" else "Mean communication ms / step"
    title = "Throughput vs Bandwidth" if metric == "throughput" else "Communication Cost vs Bandwidth"

    by_stage: Dict[int, List[CaseView]] = {}
    for case in cases:
        by_stage.setdefault(case.stage, []).append(case)

    plt.figure(figsize=(8, 5))
    plotted = 0
    for stage in sorted(by_stage):
        stage_cases = sorted(by_stage[stage], key=lambda c: c.bandwidth_gbps)
        xs: List[float] = []
        ys: List[float] = []
        for case in stage_cases:
            if case.return_code != 0:
                continue
            val = getattr(case, key)
            if val is None:
                continue
            xs.append(case.bandwidth_gbps)
            ys.append(val)
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"stage {stage}")
        plotted += 1

    if plotted == 0:
        raise RuntimeError(f"No data points available for metric '{metric}'")

    plt.xlabel("Bandwidth limit (Gbps, 0 means unlimited/no simulation)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)


def plot_stage_metric(cases: List[CaseView], metric: str, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    metric_key = "mean_tokens_per_s" if metric == "stage-throughput" else "mean_comm_ms"
    ylabel = "Tokens / s" if metric == "stage-throughput" else "Mean communication ms / step"
    title = "Throughput by ZeRO Stage" if metric == "stage-throughput" else "Communication by ZeRO Stage"
    color = "#1E6D5A" if metric == "stage-throughput" else "#B05D2B"

    selected = _representative_cases_by_stage(case for case in cases if case.return_code == 0)
    if not selected:
        raise RuntimeError("No successful cases available for stage comparison plotting")

    points = []
    for case in selected:
        value = getattr(case, metric_key)
        if value is None:
            continue
        points.append((case.stage, value))

    if not points:
        raise RuntimeError(f"No metric data available for plot '{metric}'")

    points.sort(key=lambda item: item[0])
    stages = [stage for stage, _value in points]
    values = [value for _stage, value in points]

    plt.figure(figsize=(7, 4.5))
    plt.bar(stages, values, color=color)
    plt.xticks(stages, [f"stage {stage}" for stage in stages])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)


def plot_measured_state_memory(cases: List[CaseView], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    selected = _representative_cases_by_stage(case for case in cases if case.return_code == 0 and case.measured_state_memory_mb is not None)
    if not selected:
        raise RuntimeError("No measured state memory data available in the selected cases")

    stages = [case.stage for case in selected]
    params = [case.measured_state_memory_mb["params_mb"] for case in selected]
    grads = [case.measured_state_memory_mb["grads_mb"] for case in selected]
    optimizer = [case.measured_state_memory_mb["optimizer_mb"] for case in selected]

    plt.figure(figsize=(8, 5))
    bars_params = plt.bar(stages, params, label="params", color="#315A9A")
    bars_grads = plt.bar(stages, grads, bottom=params, label="grads", color="#4E8A3A")
    bottoms = [p + g for p, g in zip(params, grads)]
    bars_optimizer = plt.bar(stages, optimizer, bottom=bottoms, label="optimizer", color="#C56B1A")
    plt.xticks(stages, [f"stage {stage}" for stage in stages])
    plt.ylabel("State memory (MB)")
    plt.title("Measured ZeRO State Memory by Stage")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    for bars, values, base_values in [
        (bars_params, params, [0.0 for _ in params]),
        (bars_grads, grads, params),
        (bars_optimizer, optimizer, bottoms),
    ]:
        for bar, value, base in zip(bars, values, base_values):
            if value < 80.0:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            y = base + value / 2.0
            plt.text(x, y, f"{value:.0f}", ha="center", va="center", color="white", fontsize=8)

    totals = [p + g + o for p, g, o in zip(params, grads, optimizer)]
    for stage, total in zip(stages, totals):
        plt.text(stage, total + max(totals) * 0.015, f"{total:.0f} MB", ha="center", va="bottom", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)


def plot_peak_memory(cases: List[CaseView], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    selected = _representative_cases_by_stage(case for case in cases if case.return_code == 0)
    if not selected:
        raise RuntimeError("No successful cases available for memory plotting")

    points = [(case.stage, _case_peak_memory_mb(case)) for case in selected]
    points = [(stage, value) for stage, value in points if value is not None]
    if not points:
        raise RuntimeError("No peak memory data available in the selected cases")

    points.sort(key=lambda item: item[0])
    stages = [stage for stage, _value in points]
    values = [value for _stage, value in points]

    plt.figure(figsize=(7, 4.5))
    plt.bar(stages, values, color="#2E5EAA")
    plt.xticks(stages, [f"stage {stage}" for stage in stages])
    plt.ylabel("Peak memory (MB)")
    plt.title("Measured Peak Allocated Memory by ZeRO Stage")
    plt.grid(True, axis="y", alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    cases = parse_summary(summary_path=summary_path)
    cases = _filter_cases(cases, model_size=args.model_size)
    cases = _filter_bandwidth(cases, bandwidth_gbps=args.bandwidth_gbps_filter)
    out = Path(args.output)

    if args.plot == "loss":
        plot_loss(cases=cases, output=out)
    elif args.plot == "throughput":
        plot_bandwidth_metric(cases=cases, metric="throughput", output=out)
    elif args.plot == "comm":
        plot_bandwidth_metric(cases=cases, metric="comm", output=out)
    elif args.plot == "stage-throughput":
        plot_stage_metric(cases=cases, metric="stage-throughput", output=out)
    elif args.plot == "stage-comm":
        plot_stage_metric(cases=cases, metric="stage-comm", output=out)
    elif args.plot == "memory":
        plot_measured_state_memory(cases=cases, output=out)
    elif args.plot == "peak-memory":
        plot_peak_memory(cases=cases, output=out)
    else:
        raise ValueError(f"Unsupported plot type: {args.plot}")

    print(f"[visualize] wrote {out}")


if __name__ == "__main__":
    main()
