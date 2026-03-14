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
    parser = argparse.ArgumentParser(description="Generate a markdown report for a bandwidth sweep")
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


def _load_tuning_summary(run_dir: Path) -> Dict[str, object] | None:
    tuning_path = run_dir / "tuning_summary.json"
    if not tuning_path.exists():
        return None
    payload = json.loads(tuning_path.read_text())
    return payload if isinstance(payload, dict) else None


def _tokens_per_step(case: visualize.CaseView, world_size: int) -> int:
    return int(case.batch_size * case.grad_accum_steps * case.seq_len * world_size)


def _render_methodology(run_args: Dict[str, object], tuning_summary: Dict[str, object] | None) -> List[str]:
    bandwidth_mode = str(run_args.get("bandwidth_mode", "unknown"))
    steps = run_args.get("steps", "unknown")
    metrics_warmup_steps = run_args.get("metrics_warmup_steps", 0)

    lines = ["Method:"]
    if tuning_summary is None:
        lines.append(
            "- fixed workload: every stage uses the same per-GPU microbatch, grad accumulation, and sequence length"
        )
    else:
        fit_mode = str(tuning_summary.get("fit_mode", "memory_budget"))
        memory_budget_mb = tuning_summary.get("memory_budget_mb")
        memory_metric = tuning_summary.get("memory_metric", "peak_cuda_max_reserved_mb")
        if fit_mode == "oom_boundary":
            lines.append(
                "- fit-to-memory: each stage is first tuned at unlimited bandwidth to the largest per-GPU microbatch "
                "that completes without OOM"
            )
            lines.append(
                f"- the selected tuning profile reports `{memory_metric}` for the largest passing microbatch"
            )
        else:
            lines.append(
                "- fit-to-memory: each stage is first tuned at unlimited bandwidth to the largest per-GPU microbatch "
                f"whose measured `{memory_metric}` stays within {_format_float(_as_optional_float(memory_budget_mb))} MB"
            )
        lines.append("- the tuned per-stage microbatch is then frozen and reused across the bandwidth sweep")

    if bandwidth_mode == "socket":
        lines.append("- communication is forced over NCCL NET/Socket and throttled at the socket layer")
    elif bandwidth_mode == "simulated":
        lines.append("- bandwidth is simulated inside the collective interface")
    elif bandwidth_mode == "tc":
        lines.append("- bandwidth is throttled with Linux tc on the configured interface")
    else:
        lines.append("- no bandwidth throttling is applied when bandwidth is unlimited")

    lines.append(
        f"- per-case metrics are averaged over logged steps after skipping {metrics_warmup_steps} warmup step(s); "
        f"nominal step count is {steps}"
    )
    return lines


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _stage_workload_rows(
    stages: List[int],
    successful: List[visualize.CaseView],
    world_size: int,
    tuning_summary: Dict[str, object] | None,
) -> List[List[str]]:
    by_stage: Dict[int, List[visualize.CaseView]] = defaultdict(list)
    for case in successful:
        by_stage[case.stage].append(case)

    memory_budget_mb = None if tuning_summary is None else _as_optional_float(tuning_summary.get("memory_budget_mb"))
    per_stage_tuning = {} if tuning_summary is None else dict(tuning_summary.get("per_stage", {}))

    rows: List[List[str]] = []
    for stage in stages:
        stage_cases = by_stage.get(stage, [])
        if not stage_cases:
            continue
        batch_sizes = sorted({case.batch_size for case in stage_cases})
        representative = min(stage_cases, key=lambda case: (case.bandwidth_gbps > 0, case.bandwidth_gbps))
        row = [
            str(stage),
            ",".join(str(batch_size) for batch_size in batch_sizes),
            str(_tokens_per_step(representative, world_size)),
        ]

        if tuning_summary is not None:
            tuned = per_stage_tuning.get(str(stage), {})
            peak_memory_mb = _as_optional_float(tuned.get("selected_peak_memory_mb"))
            budget_ratio = None
            if peak_memory_mb is not None and memory_budget_mb not in {None, 0.0}:
                budget_ratio = peak_memory_mb / memory_budget_mb
            row.extend(
                [
                    _format_float(peak_memory_mb),
                    _format_ratio(budget_ratio),
                ]
            )
        rows.append(row)
    return rows


def _ranking_rows(successful: List[visualize.CaseView], bandwidths: List[float]) -> List[List[str]]:
    rows: List[List[str]] = []
    for bandwidth in bandwidths:
        candidates = [case for case in successful if abs(case.bandwidth_gbps - bandwidth) < 1e-9]
        ranking = sorted(
            [case for case in candidates if case.mean_tokens_per_s is not None],
            key=lambda case: float(case.mean_tokens_per_s or 0.0),
            reverse=True,
        )
        if not ranking:
            continue
        rows.append(
            [
                _format_bandwidth(bandwidth),
                " > ".join(f"stage {case.stage}" for case in ranking),
                str(ranking[0].stage),
                _format_float(ranking[0].mean_tokens_per_s),
            ]
        )
    return rows


def _transition_lines(successful: List[visualize.CaseView], bandwidths: List[float]) -> List[str]:
    lines: List[str] = []
    previous_best_stage = None
    for bandwidth in bandwidths:
        candidates = [case for case in successful if abs(case.bandwidth_gbps - bandwidth) < 1e-9]
        ranked = sorted(
            [case for case in candidates if case.mean_tokens_per_s is not None],
            key=lambda case: float(case.mean_tokens_per_s or 0.0),
            reverse=True,
        )
        if not ranked:
            continue
        best_stage = ranked[0].stage
        if previous_best_stage != best_stage:
            lines.append(f"- {_format_bandwidth(bandwidth)}: stage {best_stage} is the fastest stage")
        previous_best_stage = best_stage
    return lines


def _section_for_model(
    model_size: str,
    cases: List[visualize.CaseView],
    run_args: Dict[str, object],
    tuning_summary: Dict[str, object] | None,
) -> str:
    successful = _successful_cases(cases)
    if not successful:
        return f"## Model `{model_size}`\n\nNo successful cases.\n"

    bandwidths = sorted({case.bandwidth_gbps for case in successful})
    stages = sorted({case.stage for case in successful})
    lookup = _case_lookup(successful)
    world_size = int(run_args.get("nproc_per_node", 1))

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

    workload_headers = ["stage", "microbatch / gpu", "global tokens / step"]
    if tuning_summary is not None:
        memory_metric = str(tuning_summary.get("memory_metric", "peak_cuda_max_reserved_mb"))
        metric_label = memory_metric.replace("peak_", "").replace("_mb", "").replace("_", " ")
        workload_headers.extend([f"selected {metric_label} MB", "budget used"])

    parts = [f"## Model `{model_size}`", ""]
    representative = min(successful, key=lambda case: (case.bandwidth_gbps > 0, case.bandwidth_gbps))
    parts.append(
        f"Workload: seq_len={representative.seq_len}, grad_accum_steps={representative.grad_accum_steps}, "
        f"world_size={world_size}"
    )
    parts.append("")
    parts.append("Stage workload:")
    parts.append(
        _markdown_table(
            workload_headers,
            _stage_workload_rows(
                stages=stages,
                successful=successful,
                world_size=world_size,
                tuning_summary=tuning_summary,
            ),
        )
    )
    parts.append("")
    parts.append("Stage ranking by bandwidth:")
    parts.append(
        _markdown_table(
            ["bandwidth", "ranking by tokens/s", "winner", "winner tokens/s"],
            _ranking_rows(successful=successful, bandwidths=bandwidths),
        )
    )
    parts.append("")
    parts.append("Winner transitions:")
    parts.extend(_transition_lines(successful=successful, bandwidths=bandwidths))
    parts.append("")
    parts.append("Best stage by bandwidth:")
    parts.append(_markdown_table(["bandwidth", "best stage", "tokens/s", "comm ms"], best_rows))
    parts.append("")
    parts.append("Throughput (tokens/s):")
    parts.append(_markdown_table(headers, throughput_rows))
    parts.append("")
    parts.append("Throughput relative to the stage baseline:")
    parts.append(_markdown_table(headers, slowdown_rows))
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

    run_args = summary.get("args", {})
    if not isinstance(run_args, dict):
        run_args = {}
    tuning_summary = _load_tuning_summary(run_dir)

    experiment_type = "fit-to-memory bandwidth sweep" if tuning_summary is not None else "fixed-workload bandwidth sweep"
    parts = ["# Bandwidth Sweep Report", ""]
    parts.append(f"Run dir: `{run_dir}`")
    parts.append(f"Run name: `{summary.get('name', run_dir.name)}`")
    parts.append(f"Experiment type: {experiment_type}")
    parts.append(f"Successful cases: {sum(case.return_code == 0 for case in cases)} / {len(cases)}")
    parts.append(
        "Settings: "
        f"bandwidth_mode={run_args.get('bandwidth_mode', 'unknown')}, "
        f"nproc_per_node={run_args.get('nproc_per_node', 'unknown')}, "
        f"steps={run_args.get('steps', 'unknown')}, "
        f"metrics_warmup_steps={run_args.get('metrics_warmup_steps', 0)}"
    )
    parts.append("")
    parts.extend(_render_methodology(run_args=run_args, tuning_summary=tuning_summary))
    parts.append("")

    for model_size in sorted(grouped):
        parts.append(
            _section_for_model(
                model_size=model_size,
                cases=grouped[model_size],
                run_args=run_args,
                tuning_summary=tuning_summary,
            )
        )

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
