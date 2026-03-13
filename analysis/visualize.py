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
    profile_path: Path | None
    mean_tokens_per_s: float | None
    mean_tflops_per_s: float | None
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
        choices=["loss", "throughput", "tflops", "comm", "stage-throughput", "grouped-stage-throughput", "stage-comm", "memory", "peak-memory", "avg-memory"],
        help="Plot type: per-step loss, bandwidth sweep throughput/TFLOPs/comm, stage comparison throughput/comm, grouped model-vs-stage throughput, measured state memory, peak GPU memory split into model state and non-state residual memory, or average sampled live GPU memory split into model state and non-state memory",
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


def _result_profile_path(result: Dict[str, object]) -> Path | None:
    raw = result.get("profile_path")
    if raw in {None, ""}:
        return None
    return Path(str(raw))


def parse_summary(summary_path: Path) -> List[CaseView]:
    summary = json.loads(summary_path.read_text())
    out: List[CaseView] = []
    run_dir = summary_path.parent

    for result in summary.get("results", []):
        raw_log_path = _result_log_path(result)
        if not raw_log_path.is_absolute() and not raw_log_path.exists():
            raw_log_path = run_dir / raw_log_path
        raw_profile_path = _result_profile_path(result)
        if raw_profile_path is not None and not raw_profile_path.is_absolute() and not raw_profile_path.exists():
            raw_profile_path = run_dir / raw_profile_path

        out.append(
            CaseView(
                stage=_result_stage(result),
                model_size=_result_model_size(result),
                bandwidth_gbps=_result_bandwidth(result),
                log_path=raw_log_path,
                profile_path=raw_profile_path,
                mean_tokens_per_s=_as_optional_float(result.get("mean_tokens_per_s")),
                mean_tflops_per_s=_as_optional_float(result.get("mean_tflops_per_s")),
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
        case.peak_cuda_max_allocated_mb,
        case.peak_cuda_max_reserved_mb,
        case.peak_cuda_allocated_mb,
        case.peak_cuda_reserved_mb,
    ]
    positive_cuda = [value for value in cuda_candidates if value is not None and value > 0.0]
    if positive_cuda:
        return positive_cuda[0]
    return case.peak_host_rss_mb


def _case_peak_breakdown_mb(case: CaseView) -> tuple[float, float] | None:
    peak_mb = _case_peak_memory_mb(case)
    if peak_mb is None:
        return None

    logical_state = _case_logical_state_breakdown(case)
    state_total_mb = 0.0 if logical_state is None else float(logical_state.get("total_mb", 0.0))

    state_component_mb = min(max(state_total_mb, 0.0), peak_mb)
    residual_component_mb = max(peak_mb - state_component_mb, 0.0)
    return state_component_mb, residual_component_mb


def _profile_memory_snapshots(case: CaseView) -> List[Dict[str, object]]:
    if case.profile_path is None or not case.profile_path.exists():
        return []
    payload = json.loads(case.profile_path.read_text())
    raw_snapshots = payload.get("memory", [])
    if not isinstance(raw_snapshots, list):
        return []
    return [snapshot for snapshot in raw_snapshots if isinstance(snapshot, dict)]


def _profile_payload(case: CaseView) -> Dict[str, object] | None:
    if case.profile_path is None or not case.profile_path.exists():
        return None
    payload = json.loads(case.profile_path.read_text())
    if not isinstance(payload, dict):
        return None
    return payload


def _measured_step_memory_snapshots(case: CaseView) -> List[Dict[str, object]]:
    snapshots = _profile_memory_snapshots(case)
    filtered = [snapshot for snapshot in snapshots if str(snapshot.get("label", "")).startswith("measured_step_")]
    if filtered:
        return filtered
    return snapshots


def _measured_step_state_map(case: CaseView) -> Dict[str, float]:
    payload = _profile_payload(case)
    if payload is None:
        return {}
    raw_timeline = payload.get("measured_step_state_timeline", [])
    if not isinstance(raw_timeline, list):
        return {}

    state_map: Dict[str, float] = {}
    for item in raw_timeline:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        total_mb = item.get("total_mb")
        if label and total_mb is not None:
            state_map[label] = float(total_mb)

    # Backfill older stage-3 profiles that omitted the always-resident local
    # parameter shards from the live-state timeline.
    if case.stage == 3 and state_map and case.measured_state_memory_mb is not None:
        params_mb = float(case.measured_state_memory_mb.get("params_mb", 0.0))
        total_mb = float(case.measured_state_memory_mb.get("total_mb", 0.0))
        observed_floor = min(state_map.values())
        if abs((total_mb - observed_floor) - params_mb) <= max(1.0, params_mb * 0.05):
            state_map = {label: value + params_mb for label, value in state_map.items()}
    return state_map


def _measured_step_state_timeline(case: CaseView) -> List[Dict[str, float | str]]:
    payload = _profile_payload(case)
    if payload is None:
        return []
    raw_timeline = payload.get("measured_step_state_timeline", [])
    if not isinstance(raw_timeline, list):
        return []
    out: List[Dict[str, float | str]] = []
    for item in raw_timeline:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        total_mb = item.get("total_mb")
        if not label or total_mb is None:
            continue
        out.append(
            {
                "label": label,
                "params_mb": float(item.get("params_mb", 0.0)),
                "grads_mb": float(item.get("grads_mb", 0.0)),
                "optimizer_mb": float(item.get("optimizer_mb", 0.0)),
                "total_mb": float(total_mb),
            }
        )
    return out


def _case_peak_state_breakdown(case: CaseView) -> Dict[str, float] | None:
    timeline = _measured_step_state_timeline(case)
    if timeline:
        best_item = max(timeline, key=lambda item: float(item["total_mb"]))
        return {
            "params_mb": float(best_item["params_mb"]),
            "grads_mb": float(best_item["grads_mb"]),
            "optimizer_mb": float(best_item["optimizer_mb"]),
            "total_mb": float(best_item["total_mb"]),
        }
    return case.measured_state_memory_mb


def _case_post_backward_state_breakdown(case: CaseView) -> Dict[str, float] | None:
    timeline = _measured_step_state_timeline(case)
    if timeline:
        for item in timeline:
            label = str(item["label"])
            if label.endswith("_post_backward"):
                return {
                    "params_mb": float(item["params_mb"]),
                    "grads_mb": float(item["grads_mb"]),
                    "optimizer_mb": float(item["optimizer_mb"]),
                    "total_mb": float(item["total_mb"]),
                }
    return case.measured_state_memory_mb


def _case_logical_state_breakdown(case: CaseView) -> Dict[str, float] | None:
    if case.measured_state_memory_mb is not None:
        return case.measured_state_memory_mb
    return _case_post_backward_state_breakdown(case)


def _case_state_breakdown_at_peak_total(case: CaseView) -> Dict[str, float] | None:
    peak_snapshots = _measured_step_memory_snapshots(case)
    if peak_snapshots:
        max_peak_seen = max(float(snapshot.get("cuda_max_allocated_mb", 0.0)) for snapshot in peak_snapshots)
        peak_label = None
        for snapshot in peak_snapshots:
            if abs(float(snapshot.get("cuda_max_allocated_mb", 0.0)) - max_peak_seen) < 1e-6:
                peak_label = str(snapshot.get("label", ""))
                break
        if peak_label:
            for item in _measured_step_state_timeline(case):
                if str(item["label"]) == peak_label:
                    return {
                        "params_mb": float(item["params_mb"]),
                        "grads_mb": float(item["grads_mb"]),
                        "optimizer_mb": float(item["optimizer_mb"]),
                        "total_mb": float(item["total_mb"]),
                    }
    return _case_peak_state_breakdown(case)


def _training_memory_snapshots(case: CaseView) -> List[Dict[str, object]]:
    snapshots = _profile_memory_snapshots(case)
    filtered = [snapshot for snapshot in snapshots if str(snapshot.get("label", "")).startswith("step_")]
    if filtered:
        return filtered
    return snapshots


def _case_average_live_breakdown_mb(case: CaseView) -> tuple[float, float] | None:
    state_map = _measured_step_state_map(case)
    measured_pairs = []
    for snapshot in _measured_step_memory_snapshots(case):
        label = str(snapshot.get("label", ""))
        if label not in state_map or "cuda_allocated_mb" not in snapshot:
            continue
        allocated_mb = float(snapshot["cuda_allocated_mb"])
        state_mb = min(max(state_map[label], 0.0), allocated_mb)
        timestamp_s = snapshot.get("timestamp_s")
        if timestamp_s is None:
            measured_pairs.append((None, allocated_mb, state_mb))
        else:
            measured_pairs.append((float(timestamp_s), allocated_mb, state_mb))

    if measured_pairs:
        timed_pairs = [pair for pair in measured_pairs if pair[0] is not None]
        if len(timed_pairs) >= 2:
            total_duration_s = timed_pairs[-1][0] - timed_pairs[0][0]
            if total_duration_s > 0.0:
                weighted_state = 0.0
                weighted_non_state = 0.0
                for idx in range(len(timed_pairs) - 1):
                    current_t, current_alloc, current_state = timed_pairs[idx]
                    next_t = timed_pairs[idx + 1][0]
                    dt = max(0.0, next_t - current_t)
                    weighted_state += current_state * dt
                    weighted_non_state += max(current_alloc - current_state, 0.0) * dt
                avg_state_mb = weighted_state / total_duration_s
                avg_non_state_mb = weighted_non_state / total_duration_s
                return float(avg_state_mb), float(avg_non_state_mb)

        avg_state_mb = float(sum(state for _timestamp, _allocated, state in measured_pairs) / len(measured_pairs))
        avg_non_state_mb = float(
            sum(max(allocated - state, 0.0) for _timestamp, allocated, state in measured_pairs) / len(measured_pairs)
        )
        return avg_state_mb, avg_non_state_mb

    snapshots = _training_memory_snapshots(case)
    allocated_values = [float(snapshot["cuda_allocated_mb"]) for snapshot in snapshots if "cuda_allocated_mb" in snapshot]
    if not allocated_values:
        return None

    avg_allocated_mb = float(sum(allocated_values) / len(allocated_values))
    state_total_mb = 0.0
    if case.measured_state_memory_mb is not None:
        state_total_mb = float(case.measured_state_memory_mb.get("total_mb", 0.0))

    state_component_mb = min(max(state_total_mb, 0.0), avg_allocated_mb)
    non_state_component_mb = max(avg_allocated_mb - state_component_mb, 0.0)
    return state_component_mb, non_state_component_mb


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

    if metric == "throughput":
        key = "mean_tokens_per_s"
        ylabel = "Tokens / s"
        title = "Throughput vs Bandwidth"
    elif metric == "tflops":
        key = "mean_tflops_per_s"
        ylabel = "Approx. training TFLOPs / s"
        title = "Approx. Training TFLOPs vs Bandwidth"
    else:
        key = "mean_comm_ms"
        ylabel = "Mean communication ms / step"
        title = "Communication Cost vs Bandwidth"

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


def plot_grouped_stage_throughput(cases: List[CaseView], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    filtered = [case for case in cases if case.return_code == 0 and case.mean_tokens_per_s is not None]
    if not filtered:
        raise RuntimeError("No successful cases available for grouped throughput plotting")

    models = sorted({case.model_size for case in filtered})
    stages = sorted({case.stage for case in filtered})
    width = 0.8 / max(len(models), 1)
    x_positions = list(range(len(stages)))

    plt.figure(figsize=(8.5, 5))
    for model_index, model_size in enumerate(models):
        heights = []
        for stage in stages:
            candidates = [case for case in filtered if case.model_size == model_size and case.stage == stage]
            if not candidates:
                heights.append(0.0)
                continue
            chosen = _representative_cases_by_stage(candidates)[0]
            heights.append(float(chosen.mean_tokens_per_s))

        offsets = [x + (model_index - (len(models) - 1) / 2.0) * width for x in x_positions]
        plt.bar(offsets, heights, width=width, label=model_size)

    plt.xticks(x_positions, [f"stage {stage}" for stage in stages])
    plt.ylabel("Tokens / s")
    plt.title("Throughput by ZeRO Stage and Model Size")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(title="model")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)


def plot_measured_state_memory(cases: List[CaseView], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    selected = _representative_cases_by_stage(case for case in cases if case.return_code == 0 and _case_logical_state_breakdown(case) is not None)
    if not selected:
        raise RuntimeError("No measured state memory data available in the selected cases")

    stages = [case.stage for case in selected]
    breakdowns = [_case_logical_state_breakdown(case) for case in selected]
    params = [breakdown["params_mb"] for breakdown in breakdowns]
    grads = [breakdown["grads_mb"] for breakdown in breakdowns]
    optimizer = [breakdown["optimizer_mb"] for breakdown in breakdowns]

    plt.figure(figsize=(8, 5))
    bars_params = plt.bar(stages, params, label="params", color="#315A9A")
    bars_grads = plt.bar(stages, grads, bottom=params, label="grads", color="#4E8A3A")
    bottoms = [p + g for p, g in zip(params, grads)]
    bars_optimizer = plt.bar(stages, optimizer, bottom=bottoms, label="optimizer", color="#C56B1A")
    plt.xticks(stages, [f"stage {stage}" for stage in stages])
    plt.ylabel("State memory (MB)")
    plt.title("Logical ZeRO Model State Memory by Stage")
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

    points = []
    for case in selected:
        breakdown = _case_peak_breakdown_mb(case)
        if breakdown is None:
            continue
        points.append((case.stage, breakdown[0], breakdown[1]))

    if not points:
        raise RuntimeError("No peak memory data available in the selected cases")

    points.sort(key=lambda item: item[0])
    stages = [stage for stage, _state, _residual in points]
    state_values = [state for _stage, state, _residual in points]
    residual_values = [residual for _stage, _state, residual in points]
    total_values = [state + residual for state, residual in zip(state_values, residual_values)]

    plt.figure(figsize=(8, 5))
    bars_state = plt.bar(stages, state_values, color="#315A9A", label="logical model state")
    bars_residual = plt.bar(stages, residual_values, bottom=state_values, color="#C56B1A", label="peak transient / residual")
    plt.xticks(stages, [f"stage {stage}" for stage in stages])
    plt.ylabel("Peak memory (MB)")
    plt.title("Measured Step Peak GPU Memory by ZeRO Stage")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    for bars, values, base_values in [
        (bars_state, state_values, [0.0 for _ in state_values]),
        (bars_residual, residual_values, state_values),
    ]:
        for bar, value, base in zip(bars, values, base_values):
            if value < 80.0:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            y = base + value / 2.0
            plt.text(x, y, f"{value:.0f}", ha="center", va="center", color="white", fontsize=8)

    for stage, total in zip(stages, total_values):
        plt.text(stage, total + max(total_values) * 0.015, f"{total:.0f} MB", ha="center", va="bottom", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)


def plot_average_memory(cases: List[CaseView], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    selected = _representative_cases_by_stage(case for case in cases if case.return_code == 0)
    if not selected:
        raise RuntimeError("No successful cases available for average-memory plotting")

    points = []
    for case in selected:
        breakdown = _case_average_live_breakdown_mb(case)
        if breakdown is None:
            continue
        points.append((case.stage, breakdown[0], breakdown[1]))

    if not points:
        raise RuntimeError("No sampled memory data available for average-memory plotting")

    points.sort(key=lambda item: item[0])
    stages = [stage for stage, _state, _non_state in points]
    state_values = [state for _stage, state, _non_state in points]
    non_state_values = [non_state for _stage, _state, non_state in points]
    total_values = [state + non_state for state, non_state in zip(state_values, non_state_values)]

    plt.figure(figsize=(8, 5))
    bars_state = plt.bar(stages, state_values, color="#315A9A", label="model state")
    bars_non_state = plt.bar(stages, non_state_values, bottom=state_values, color="#B05D2B", label="avg live non-state")
    plt.xticks(stages, [f"stage {stage}" for stage in stages])
    plt.ylabel("Average sampled live memory (MB)")
    plt.title("Average Sampled GPU Memory by ZeRO Stage")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    for bars, values, base_values in [
        (bars_state, state_values, [0.0 for _ in state_values]),
        (bars_non_state, non_state_values, state_values),
    ]:
        for bar, value, base in zip(bars, values, base_values):
            if value < 80.0:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            y = base + value / 2.0
            plt.text(x, y, f"{value:.0f}", ha="center", va="center", color="white", fontsize=8)

    for stage, total in zip(stages, total_values):
        plt.text(stage, total + max(total_values) * 0.015, f"{total:.0f} MB", ha="center", va="bottom", fontsize=8)

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
    elif args.plot == "tflops":
        plot_bandwidth_metric(cases=cases, metric="tflops", output=out)
    elif args.plot == "comm":
        plot_bandwidth_metric(cases=cases, metric="comm", output=out)
    elif args.plot == "stage-throughput":
        plot_stage_metric(cases=cases, metric="stage-throughput", output=out)
    elif args.plot == "grouped-stage-throughput":
        plot_grouped_stage_throughput(cases=cases, output=out)
    elif args.plot == "stage-comm":
        plot_stage_metric(cases=cases, metric="stage-comm", output=out)
    elif args.plot == "memory":
        plot_measured_state_memory(cases=cases, output=out)
    elif args.plot == "peak-memory":
        plot_peak_memory(cases=cases, output=out)
    elif args.plot == "avg-memory":
        plot_average_memory(cases=cases, output=out)
    else:
        raise ValueError(f"Unsupported plot type: {args.plot}")

    print(f"[visualize] wrote {out}")


if __name__ == "__main__":
    main()
