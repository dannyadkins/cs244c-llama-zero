from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import harness
from experiments.run_fit_memory_bandwidth import (
    TuningTrial,
    _detect_gpu_total_memory_mb,
    _make_case,
    _select_max_batch_size,
    _training_extra_args,
    _trial_from_result,
)


@dataclass(frozen=True)
class PairwisePoint:
    bandwidth_gbps: float
    stage_a_tokens_per_s: float
    stage_b_tokens_per_s: float
    stage_a_comm_ms: float
    stage_b_comm_ms: float
    stage_a_fb_ms: float
    stage_b_fb_ms: float
    ratio_stage_b_over_stage_a: float
    winner_stage: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune two ZeRO stages to OOM boundary, then adaptively search bandwidth points for a crossover"
    )
    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--name", type=str, default="pairwise_crossover_search")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--stage-a", type=int, default=2)
    parser.add_argument("--stage-b", type=int, default=3)
    parser.add_argument(
        "--stage-a-batch-size",
        type=int,
        default=0,
        help="If >0, use this fixed per-GPU microbatch for stage-a and skip tuning for that stage.",
    )
    parser.add_argument(
        "--stage-b-batch-size",
        type=int,
        default=0,
        help="If >0, use this fixed per-GPU microbatch for stage-b and skip tuning for that stage.",
    )

    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port-base", type=int, default=29500)
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)

    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--collective-impl", type=str, default="torch", choices=["ring", "torch"])
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "fineweb"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--stage2-grad-bucket-mb", type=float, default=64.0)
    parser.add_argument("--profile-memory-interval", type=int, default=0)
    parser.add_argument("--metrics-warmup-steps", type=int, default=2)

    parser.add_argument("--bandwidth-mode", type=str, default="socket", choices=["simulated", "none", "tc", "socket"])
    parser.add_argument("--sim-latency-ms", type=float, default=0.0)
    parser.add_argument("--tc-interface", type=str, default="eth0")
    parser.add_argument("--socket-interface", type=str, default="lo")
    parser.add_argument("--socket-shaper-burst-bytes", type=int, default=262_144)
    parser.add_argument("--theory-vocab-size", type=int, default=0)
    parser.add_argument("--extra-args", type=str, default="")

    parser.add_argument("--fit-mode", type=str, default="oom_boundary", choices=["oom_boundary"])
    parser.add_argument(
        "--memory-metric",
        type=str,
        default="peak_cuda_max_allocated_mb",
        choices=[
            "peak_cuda_allocated_mb",
            "peak_cuda_reserved_mb",
            "peak_cuda_max_allocated_mb",
            "peak_cuda_max_reserved_mb",
        ],
    )
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--initial-batch-size", type=int, default=0)
    parser.add_argument("--batch-size-multiple", type=int, default=4)
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--tuning-memory-warmup-steps", type=int, default=2)

    parser.add_argument("--include-unlimited", action="store_true")
    parser.add_argument("--bandwidth-min-gbps", type=float, default=0.01)
    parser.add_argument("--bandwidth-max-gbps", type=float, default=0.5)
    parser.add_argument(
        "--seed-bandwidth-gbps",
        nargs="*",
        type=float,
        default=[],
        help="Optional extra finite seed points. The geometric midpoint of min/max is always included.",
    )
    parser.add_argument(
        "--max-pairwise-bandwidth-points",
        type=int,
        default=6,
        help="Maximum number of finite bandwidth points to evaluate after tuning.",
    )
    parser.add_argument(
        "--bandwidth-tolerance-ratio",
        type=float,
        default=1.25,
        help="Stop refining a bracketing interval once high/low bandwidth is within this ratio.",
    )
    parser.add_argument(
        "--near-parity-log-ratio",
        type=float,
        default=0.05,
        help="If |log(stage_b / stage_a throughput)| is below this threshold, treat the sampled point as near parity.",
    )
    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    preliminary_args, _ = parser.parse_known_args(argv)
    if preliminary_args.config:
        harness._apply_config_to_parser(parser, preliminary_args.config)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_markdown(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _log_ratio(point: PairwisePoint) -> float:
    return math.log(point.ratio_stage_b_over_stage_a)


def _points_close(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _geometric_midpoint(low: float, high: float) -> float:
    return math.sqrt(low * high)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _interval_priority(left: PairwisePoint, right: PairwisePoint) -> Tuple[int, float, float]:
    left_sign = _sign(_log_ratio(left))
    right_sign = _sign(_log_ratio(right))
    span = math.log(right.bandwidth_gbps / left.bandwidth_gbps)
    if left_sign != right_sign:
        return (0, span, 0.0)
    closeness = max(abs(_log_ratio(left)), abs(_log_ratio(right)))
    return (1, closeness, -span)


def choose_next_finite_bandwidth(
    sampled: Dict[float, PairwisePoint],
    *,
    tolerance_ratio: float,
) -> float | None:
    if len(sampled) < 2:
        return None
    points = sorted(sampled.values(), key=lambda item: item.bandwidth_gbps)
    candidates: List[Tuple[Tuple[int, float, float], float]] = []
    for left, right in zip(points, points[1:]):
        if right.bandwidth_gbps / left.bandwidth_gbps <= tolerance_ratio:
            continue
        midpoint = _geometric_midpoint(left.bandwidth_gbps, right.bandwidth_gbps)
        if any(_points_close(midpoint, existing) for existing in sampled):
            continue
        candidates.append((_interval_priority(left, right), midpoint))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _best_flip_bracket(sampled: Dict[float, PairwisePoint]) -> Tuple[PairwisePoint, PairwisePoint] | None:
    points = sorted(sampled.values(), key=lambda item: item.bandwidth_gbps)
    brackets: List[Tuple[float, PairwisePoint, PairwisePoint]] = []
    for left, right in zip(points, points[1:]):
        if _sign(_log_ratio(left)) == _sign(_log_ratio(right)):
            continue
        brackets.append((right.bandwidth_gbps / left.bandwidth_gbps, left, right))
    if not brackets:
        return None
    brackets.sort(key=lambda item: item[0])
    return brackets[0][1], brackets[0][2]


def _nearest_parity_point(sampled: Dict[float, PairwisePoint]) -> PairwisePoint | None:
    if not sampled:
        return None
    return min(sampled.values(), key=lambda item: abs(_log_ratio(item)))


def _seed_finite_bandwidths(min_bw: float, max_bw: float, extras: List[float]) -> List[float]:
    seeds = {float(min_bw), float(max_bw), float(_geometric_midpoint(min_bw, max_bw))}
    for value in extras:
        if value > 0:
            seeds.add(float(value))
    return sorted(seeds, reverse=True)


def _pairwise_point_payload(point: PairwisePoint) -> Dict[str, object]:
    return {
        "bandwidth_gbps": float(point.bandwidth_gbps),
        "stage_a_tokens_per_s": float(point.stage_a_tokens_per_s),
        "stage_b_tokens_per_s": float(point.stage_b_tokens_per_s),
        "stage_a_comm_ms": float(point.stage_a_comm_ms),
        "stage_b_comm_ms": float(point.stage_b_comm_ms),
        "stage_a_fb_ms": float(point.stage_a_fb_ms),
        "stage_b_fb_ms": float(point.stage_b_fb_ms),
        "ratio_stage_b_over_stage_a": float(point.ratio_stage_b_over_stage_a),
        "winner_stage": int(point.winner_stage),
    }


def _render_report(
    *,
    args: argparse.Namespace,
    selected_trials: Dict[int, TuningTrial],
    unlimited_point: PairwisePoint | None,
    finite_points: Dict[float, PairwisePoint],
    bracket: Tuple[PairwisePoint, PairwisePoint] | None,
) -> str:
    stage_a = int(args.stage_a)
    stage_b = int(args.stage_b)
    nearest = _nearest_parity_point(finite_points)

    lines: List[str] = []
    lines.append("# Pairwise Crossover Search")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- stage A: `{stage_a}`")
    lines.append(f"- stage B: `{stage_b}`")
    lines.append(f"- model: `{args.model_size}`")
    lines.append(f"- seq len: `{int(args.seq_len)}`")
    lines.append(f"- steps: `{int(args.steps)}`")
    lines.append(f"- bandwidth range searched: `{float(args.bandwidth_min_gbps):g}` to `{float(args.bandwidth_max_gbps):g} Gbps`")
    lines.append(f"- max finite bandwidth points: `{int(args.max_pairwise_bandwidth_points)}`")
    lines.append("")
    lines.append("## Tuned Microbatches")
    lines.append("")
    lines.append("| stage | selected microbatch / GPU | peak allocated MB | tuned unlimited tokens/s |")
    lines.append("| --- | --- | --- | --- |")
    for stage in (stage_a, stage_b):
        trial = selected_trials[stage]
        peak = "NA" if trial.peak_memory_mb is None else f"{trial.peak_memory_mb:.1f}"
        tps = "NA" if trial.mean_tokens_per_s is None else f"{trial.mean_tokens_per_s:.1f}"
        lines.append(f"| {stage} | {trial.batch_size} | {peak} | {tps} |")

    if unlimited_point is not None:
        lines.append("")
        lines.append("## Unlimited Context")
        lines.append("")
        lines.append(
            f"- stage {stage_b} / stage {stage_a} throughput ratio at unlimited bandwidth: "
            f"`{unlimited_point.ratio_stage_b_over_stage_a:.3f}x`"
        )

    lines.append("")
    lines.append("## Finite Bandwidth Samples")
    lines.append("")
    lines.append("| bandwidth (Gbps) | stage A tok/s | stage B tok/s | B / A | winner |")
    lines.append("| --- | --- | --- | --- | --- |")
    for bandwidth in sorted(finite_points, reverse=True):
        point = finite_points[bandwidth]
        lines.append(
            f"| {bandwidth:g} | {point.stage_a_tokens_per_s:.1f} | {point.stage_b_tokens_per_s:.1f} | "
            f"{point.ratio_stage_b_over_stage_a:.3f}x | stage {point.winner_stage} |"
        )

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if bracket is not None:
        lines.append(
            f"- sampled points bracket a crossover between `{bracket[0].bandwidth_gbps:g}` and "
            f"`{bracket[1].bandwidth_gbps:g} Gbps`"
        )
    elif nearest is not None:
        lines.append(
            f"- no sampled crossover was found in the searched bandwidth range; the closest sampled point to parity "
            f"was `{nearest.bandwidth_gbps:g} Gbps`, where stage {nearest.winner_stage} still led by "
            f"`{abs(_log_ratio(nearest)):.3f}` log-ratio (`{nearest.ratio_stage_b_over_stage_a:.3f}x` B/A)"
        )
    else:
        lines.append("- no finite bandwidth samples were completed")
    lines.append("")
    lines.append("Method note:")
    lines.append("- this is an adaptive pairwise search, not an exhaustive grid")
    lines.append("- if no bracketing sign change appears, the result is 'no crossover found in the searched range', not a proof of impossibility")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if int(args.stage_a) == int(args.stage_b):
        raise ValueError("stage-a and stage-b must be different")
    if float(args.bandwidth_min_gbps) <= 0 or float(args.bandwidth_max_gbps) <= 0:
        raise ValueError("bandwidth-min-gbps and bandwidth-max-gbps must be > 0")
    if float(args.bandwidth_min_gbps) >= float(args.bandwidth_max_gbps):
        raise ValueError("bandwidth-min-gbps must be smaller than bandwidth-max-gbps")

    launch = harness._launch_config_from_args(args)
    run_dir = Path(args.results_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir = run_dir / "tuning"

    gpu_total_memory_mb = _detect_gpu_total_memory_mb(args.nproc_per_node)
    case_counter = itertools.count()
    stages = [int(args.stage_a), int(args.stage_b)]
    selected_trials: Dict[int, TuningTrial] = {}
    fixed_batch_sizes = {
        int(args.stage_a): int(args.stage_a_batch_size),
        int(args.stage_b): int(args.stage_b_batch_size),
    }
    tuning_payload: Dict[str, object] = {
        "experiment_type": "pairwise_crossover_search",
        "name": args.name,
        "model_size": args.model_size,
        "seq_len": int(args.seq_len),
        "grad_accum_steps": int(args.grad_accum_steps),
        "nproc_per_node": int(args.nproc_per_node),
        "gpu_total_memory_mb": float(gpu_total_memory_mb),
        "memory_metric": str(args.memory_metric),
        "stages": stages,
        "per_stage": {},
    }

    for stage in stages:
        fixed_batch_size = fixed_batch_sizes.get(stage, 0)
        if fixed_batch_size > 0:
            selected_trials[stage] = TuningTrial(
                batch_size=int(fixed_batch_size),
                fits=True,
                peak_memory_mb=None,
                return_code=0,
                reason="fixed_batch_size",
                case_id="",
                log_path="",
                profile_path="",
                mean_tokens_per_s=None,
            )
            tuning_payload["per_stage"][str(stage)] = {
                "selected_batch_size": int(fixed_batch_size),
                "selected_peak_memory_mb": None,
                "selected_case_id": "",
                "selected_mean_tokens_per_s": None,
                "trials": [],
                "selection_mode": "fixed_batch_size",
            }
            print(f"[pairwise] stage {stage}: using fixed microbatch {fixed_batch_size}", flush=True)
            continue

        def evaluate(batch_size: int) -> TuningTrial:
            tuning_case = _make_case(
                args,
                stage=stage,
                batch_size=batch_size,
                bandwidth_gbps=0.0,
                steps=int(args.tuning_memory_warmup_steps) + 1,
                metrics_warmup_steps=0,
                bandwidth_mode="none",
                extra_args=_training_extra_args(args.extra_args, args.tuning_memory_warmup_steps),
            )
            result = harness._run_case(
                case=tuning_case,
                run_dir=tuning_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                launch=launch,
                case_index=next(case_counter),
            )
            return _trial_from_result(
                result,
                fit_mode="oom_boundary",
                memory_metric=str(args.memory_metric),
                memory_budget_mb=None,
            )

        best_trial, trials = _select_max_batch_size(
            min_batch_size=int(args.min_batch_size),
            max_batch_size=int(args.max_batch_size),
            batch_size_multiple=max(1, int(args.batch_size_multiple)),
            growth_factor=max(1.1, float(args.growth_factor)),
            initial_batch_size=int(args.initial_batch_size),
            evaluator=evaluate,
        )
        selected_trials[stage] = best_trial
        tuning_payload["per_stage"][str(stage)] = {
            "selected_batch_size": int(best_trial.batch_size),
            "selected_peak_memory_mb": None if best_trial.peak_memory_mb is None else float(best_trial.peak_memory_mb),
            "selected_case_id": best_trial.case_id,
            "selected_mean_tokens_per_s": None
            if best_trial.mean_tokens_per_s is None
            else float(best_trial.mean_tokens_per_s),
            "trials": [asdict(trial) for trial in trials],
            "selection_mode": "tuned",
        }
        print(
            f"[pairwise] stage {stage}: selected microbatch {best_trial.batch_size} "
            f"peak={best_trial.peak_memory_mb if best_trial.peak_memory_mb is not None else 'NA'} MB",
            flush=True,
        )

    tuning_summary_path = run_dir / "tuning_summary.json"
    _write_json(tuning_summary_path, tuning_payload)
    print(f"[pairwise] wrote {tuning_summary_path}", flush=True)
    if args.dry_run:
        print("[pairwise] dry run complete; skipping pairwise search", flush=True)
        return

    unlimited_point: PairwisePoint | None = None
    finite_points: Dict[float, PairwisePoint] = {}
    search_history: List[Dict[str, object]] = []

    def evaluate_pair(bandwidth_gbps: float) -> PairwisePoint:
        stage_results: Dict[int, harness.CaseResult] = {}
        for stage in stages:
            case = _make_case(
                args,
                stage=stage,
                batch_size=selected_trials[stage].batch_size,
                bandwidth_gbps=float(bandwidth_gbps),
                steps=int(args.steps),
                metrics_warmup_steps=int(args.metrics_warmup_steps),
                bandwidth_mode=str(args.bandwidth_mode),
                extra_args=str(args.extra_args),
            )
            result = harness._run_case(
                case=case,
                run_dir=run_dir,
                skip_existing=args.skip_existing,
                dry_run=False,
                launch=launch,
                case_index=next(case_counter),
            )
            stage_results[stage] = result
            if result.return_code != 0 or result.mean_tokens_per_s is None:
                raise RuntimeError(f"pairwise case failed: {result.case_id} (see {result.log_path})")

        point = PairwisePoint(
            bandwidth_gbps=float(bandwidth_gbps),
            stage_a_tokens_per_s=float(stage_results[int(args.stage_a)].mean_tokens_per_s),
            stage_b_tokens_per_s=float(stage_results[int(args.stage_b)].mean_tokens_per_s),
            stage_a_comm_ms=float(stage_results[int(args.stage_a)].mean_comm_ms or 0.0),
            stage_b_comm_ms=float(stage_results[int(args.stage_b)].mean_comm_ms or 0.0),
            stage_a_fb_ms=float(stage_results[int(args.stage_a)].mean_fb_ms or 0.0),
            stage_b_fb_ms=float(stage_results[int(args.stage_b)].mean_fb_ms or 0.0),
            ratio_stage_b_over_stage_a=float(
                stage_results[int(args.stage_b)].mean_tokens_per_s / stage_results[int(args.stage_a)].mean_tokens_per_s
            ),
            winner_stage=int(args.stage_b)
            if float(stage_results[int(args.stage_b)].mean_tokens_per_s) >= float(stage_results[int(args.stage_a)].mean_tokens_per_s)
            else int(args.stage_a),
        )
        payload = _pairwise_point_payload(point)
        payload["stage_a"] = int(args.stage_a)
        payload["stage_b"] = int(args.stage_b)
        search_history.append(payload)
        print(
            f"[pairwise] bw={bandwidth_gbps:g} Gbps: stage {args.stage_a}={point.stage_a_tokens_per_s:.1f} tok/s, "
            f"stage {args.stage_b}={point.stage_b_tokens_per_s:.1f} tok/s, "
            f"ratio={point.ratio_stage_b_over_stage_a:.3f}x",
            flush=True,
        )
        return point

    if args.include_unlimited:
        unlimited_point = evaluate_pair(0.0)

    for bandwidth in _seed_finite_bandwidths(
        min_bw=float(args.bandwidth_min_gbps),
        max_bw=float(args.bandwidth_max_gbps),
        extras=[float(item) for item in args.seed_bandwidth_gbps],
    ):
        if len(finite_points) >= int(args.max_pairwise_bandwidth_points):
            break
        finite_points[bandwidth] = evaluate_pair(bandwidth)

    while len(finite_points) < int(args.max_pairwise_bandwidth_points):
        bracket = _best_flip_bracket(finite_points)
        if bracket is not None and bracket[1].bandwidth_gbps / bracket[0].bandwidth_gbps <= float(
            args.bandwidth_tolerance_ratio
        ):
            break
        nearest = _nearest_parity_point(finite_points)
        if nearest is not None and abs(_log_ratio(nearest)) <= float(args.near_parity_log_ratio):
            break
        next_bandwidth = choose_next_finite_bandwidth(
            finite_points,
            tolerance_ratio=float(args.bandwidth_tolerance_ratio),
        )
        if next_bandwidth is None:
            break
        finite_points[next_bandwidth] = evaluate_pair(next_bandwidth)

    bracket = _best_flip_bracket(finite_points)
    nearest = _nearest_parity_point(finite_points)
    summary = {
        "experiment_type": "pairwise_crossover_search",
        "name": args.name,
        "stage_a": int(args.stage_a),
        "stage_b": int(args.stage_b),
        "model_size": str(args.model_size),
        "seq_len": int(args.seq_len),
        "steps": int(args.steps),
        "nproc_per_node": int(args.nproc_per_node),
        "tuning": {
            str(stage): {
                "selected_batch_size": int(trial.batch_size),
                "selected_peak_memory_mb": None if trial.peak_memory_mb is None else float(trial.peak_memory_mb),
                "selected_mean_tokens_per_s": None
                if trial.mean_tokens_per_s is None
                else float(trial.mean_tokens_per_s),
            }
            for stage, trial in selected_trials.items()
        },
        "include_unlimited": bool(args.include_unlimited),
        "unlimited_point": None if unlimited_point is None else _pairwise_point_payload(unlimited_point),
        "finite_points": [_pairwise_point_payload(finite_points[bw]) for bw in sorted(finite_points)],
        "nearest_parity_point": None if nearest is None else _pairwise_point_payload(nearest),
        "crossover_bracket": None
        if bracket is None
        else {
            "low_bandwidth_gbps": float(bracket[0].bandwidth_gbps),
            "high_bandwidth_gbps": float(bracket[1].bandwidth_gbps),
            "low_ratio_stage_b_over_stage_a": float(bracket[0].ratio_stage_b_over_stage_a),
            "high_ratio_stage_b_over_stage_a": float(bracket[1].ratio_stage_b_over_stage_a),
        },
        "search_history": search_history,
    }
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    report_path = run_dir / "pairwise_report.md"
    _write_markdown(
        report_path,
        _render_report(
            args=args,
            selected_trials=selected_trials,
            unlimited_point=unlimited_point,
            finite_points=finite_points,
            bracket=bracket,
        ),
    )
    print(f"[pairwise] wrote {summary_path}", flush=True)
    print(f"[pairwise] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
