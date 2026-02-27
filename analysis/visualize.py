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
    final_loss: float | None
    return_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Week 2/3 harness outputs")
    parser.add_argument("--run-dir", type=str, required=True, help="experiments/results/<name>")
    parser.add_argument(
        "--plot",
        type=str,
        default="throughput",
        choices=["loss", "throughput", "comm"],
        help="Plot type: per-step loss, throughput vs bandwidth, or comm-ms vs bandwidth",
    )
    parser.add_argument("--output", type=str, default="analysis/figures/week3_plot.png")
    parser.add_argument("--model-size", type=str, default="", help="Optional filter for one model size")
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
                final_loss=_as_optional_float(result.get("final_loss")),
                return_code=int(result.get("return_code", 0)),
            )
        )
    return out


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _filter_cases(cases: Iterable[CaseView], model_size: str) -> List[CaseView]:
    if not model_size:
        return list(cases)
    return [c for c in cases if c.model_size == model_size]


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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    cases = parse_summary(summary_path=summary_path)
    cases = _filter_cases(cases, model_size=args.model_size)
    out = Path(args.output)

    if args.plot == "loss":
        plot_loss(cases=cases, output=out)
    elif args.plot == "throughput":
        plot_bandwidth_metric(cases=cases, metric="throughput", output=out)
    else:
        plot_bandwidth_metric(cases=cases, metric="comm", output=out)

    print(f"[visualize] wrote {out}")


if __name__ == "__main__":
    main()
