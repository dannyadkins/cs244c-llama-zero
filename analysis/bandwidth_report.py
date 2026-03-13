from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import visualize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact markdown report for a bandwidth sweep")
    parser.add_argument("--run-dir", type=str, required=True, help="experiments/results/<name>")
    parser.add_argument("--output", type=str, default="", help="Defaults to <run-dir>/bandwidth_report.md")
    return parser.parse_args()


def _format_bandwidth(bandwidth_gbps: float) -> str:
    return "unlimited" if bandwidth_gbps <= 0 else f"{bandwidth_gbps:g} Gbps"


def _format_float(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100.0:.1f}%"


def _successful_cases(cases: Iterable[visualize.CaseView]) -> List[visualize.CaseView]:
    return [case for case in cases if case.return_code == 0]


def _case_lookup(cases: Iterable[visualize.CaseView]) -> Dict[tuple[int, float], visualize.CaseView]:
    lookup: Dict[tuple[int, float], visualize.CaseView] = {}
    for case in cases:
        lookup[(case.stage, case.bandwidth_gbps)] = case
    return lookup


def _baseline_bandwidth(stage_cases: List[visualize.CaseView]) -> float:
    if any(case.bandwidth_gbps <= 0 for case in stage_cases):
        return 0.0
    return max(case.bandwidth_gbps for case in stage_cases)


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _section_for_model(model_size: str, cases: List[visualize.CaseView]) -> str:
    successful = _successful_cases(cases)
    if not successful:
        return f"## Model `{model_size}`\n\nNo successful cases.\n"

    bandwidths = sorted({case.bandwidth_gbps for case in successful})
    stages = sorted({case.stage for case in successful})
    lookup = _case_lookup(successful)

    best_rows: List[List[str]] = []
    for bandwidth in bandwidths:
        candidates = [case for case in successful if abs(case.bandwidth_gbps - bandwidth) < 1e-9]
        if not candidates:
            continue
        best = max(candidates, key=lambda case: float(case.mean_tokens_per_s or 0.0))
        best_rows.append(
            [
                _format_bandwidth(bandwidth),
                str(best.stage),
                _format_float(best.mean_tokens_per_s),
                _format_float(best.mean_comm_ms),
            ]
        )

    throughput_rows: List[List[str]] = []
    slowdown_rows: List[List[str]] = []
    comm_rows: List[List[str]] = []

    for stage in stages:
        stage_cases = [case for case in successful if case.stage == stage]
        baseline_bw = _baseline_bandwidth(stage_cases)
        baseline_case = lookup.get((stage, baseline_bw))
        baseline_tps = None if baseline_case is None else baseline_case.mean_tokens_per_s

        throughput_row = [f"stage {stage}"]
        slowdown_row = [f"stage {stage}"]
        comm_row = [f"stage {stage}"]
        for bandwidth in bandwidths:
            case = lookup.get((stage, bandwidth))
            throughput_row.append(_format_float(None if case is None else case.mean_tokens_per_s))
            if case is None or baseline_tps in {None, 0.0} or case.mean_tokens_per_s is None:
                slowdown_row.append("NA")
            else:
                slowdown_row.append(_format_ratio(case.mean_tokens_per_s / baseline_tps))
            comm_row.append(_format_float(None if case is None else case.mean_comm_ms))
        throughput_rows.append(throughput_row)
        slowdown_rows.append(slowdown_row)
        comm_rows.append(comm_row)

    headers = ["stage"] + [_format_bandwidth(bandwidth) for bandwidth in bandwidths]

    parts = [f"## Model `{model_size}`", ""]
    parts.append("Best stage by bandwidth:")
    parts.append(_markdown_table(["bandwidth", "best stage", "tokens/s", "comm ms"], best_rows))
    parts.append("")
    parts.append("Throughput (tokens/s):")
    parts.append(_markdown_table(headers, throughput_rows))
    parts.append("")
    parts.append("Throughput relative to the stage baseline:")
    parts.append(
        _markdown_table(
            headers,
            slowdown_rows,
        )
    )
    parts.append("")
    parts.append("Communication time (ms / step):")
    parts.append(_markdown_table(headers, comm_rows))

    failures = [case for case in cases if case.return_code != 0]
    if failures:
        parts.append("")
        parts.append("Failed cases:")
        for case in sorted(failures, key=lambda item: (item.stage, item.bandwidth_gbps)):
            parts.append(
                f"- stage {case.stage}, {_format_bandwidth(case.bandwidth_gbps)}: "
                f"return_code={case.return_code} log={case.log_path}"
            )
    parts.append("")
    return "\n".join(parts)


def generate_report_markdown(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json under {run_dir}")

    summary = json.loads(summary_path.read_text())
    cases = visualize.parse_summary(summary_path)
    grouped: Dict[str, List[visualize.CaseView]] = defaultdict(list)
    for case in cases:
        grouped[case.model_size].append(case)

    parts = ["# Bandwidth Sweep Report", ""]
    parts.append(f"Run dir: `{run_dir}`")
    parts.append(f"Run name: `{summary.get('name', run_dir.name)}`")
    parts.append(f"Successful cases: {sum(case.return_code == 0 for case in cases)} / {len(cases)}")
    run_args = summary.get("args", {})
    if isinstance(run_args, dict):
        parts.append(
            "Settings: "
            f"bandwidth_mode={run_args.get('bandwidth_mode', 'unknown')}, "
            f"nproc_per_node={run_args.get('nproc_per_node', 'unknown')}, "
            f"steps={run_args.get('steps', 'unknown')}, "
            f"metrics_warmup_steps={run_args.get('metrics_warmup_steps', 0)}"
        )
    parts.append("")

    for model_size in sorted(grouped):
        parts.append(_section_for_model(model_size=model_size, cases=grouped[model_size]))

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "bandwidth_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_report_markdown(run_dir))
    print(f"[bandwidth-report] wrote {output_path}")


if __name__ == "__main__":
    main()
