from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import harness


@dataclass
class TuningTrial:
    batch_size: int
    fits: bool
    peak_memory_mb: float | None
    return_code: int
    reason: str
    case_id: str
    log_path: str
    profile_path: str
    mean_tokens_per_s: float | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune per-stage microbatch to a fixed memory budget, then sweep bandwidth")
    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--name", type=str, default="fit_memory_bandwidth")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--stages", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--bandwidth-gbps", nargs="+", type=float, default=[0.0, 1.0, 2.0, 5.0, 10.0])

    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port-base", type=int, default=29500)
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)

    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--collective-impl", type=str, default="torch", choices=["ring", "torch"])
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "fineweb"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument(
        "--stage2-grad-bucket-mb",
        type=float,
        default=64.0,
        help="Approximate fp32 gradient bucket size for ZeRO stage 2 reduce-scatter bucketing.",
    )
    parser.add_argument("--profile-memory-interval", type=int, default=0)
    parser.add_argument("--metrics-warmup-steps", type=int, default=2)

    parser.add_argument("--bandwidth-mode", type=str, default="socket", choices=["simulated", "none", "tc", "socket"])
    parser.add_argument("--sim-latency-ms", type=float, default=0.0)
    parser.add_argument("--tc-interface", type=str, default="eth0")
    parser.add_argument("--socket-interface", type=str, default="lo")
    parser.add_argument("--socket-shaper-burst-bytes", type=int, default=262_144)
    parser.add_argument("--theory-vocab-size", type=int, default=0)
    parser.add_argument("--extra-args", type=str, default="")

    parser.add_argument("--memory-budget-mb", type=float, default=0.0)
    parser.add_argument(
        "--memory-budget-fraction",
        type=float,
        default=0.90,
        help="If --memory-budget-mb is 0, use this fraction of the smallest visible GPU memory.",
    )
    parser.add_argument(
        "--fit-mode",
        type=str,
        default="oom_boundary",
        choices=["oom_boundary", "memory_budget"],
        help="How to decide whether a microbatch size fits during tuning.",
    )
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
    parser.add_argument("--max-batch-size", type=int, default=0, help="0 means auto-cap at 1024")
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        default=0,
        help="Initial microbatch guess for tuning. 0 uses the midpoint of the search interval.",
    )
    parser.add_argument("--batch-size-multiple", type=int, default=1)
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--tuning-memory-warmup-steps", type=int, default=2)
    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    preliminary_args, _ = parser.parse_known_args(argv)
    if preliminary_args.config:
        harness._apply_config_to_parser(parser, preliminary_args.config)
    return parser.parse_args(argv)


def _detect_gpu_total_memory_mb(nproc_per_node: int) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("fit-to-memory bandwidth experiment requires CUDA")
    visible = torch.cuda.device_count()
    if visible < nproc_per_node:
        raise RuntimeError(f"requested nproc_per_node={nproc_per_node}, but only {visible} CUDA device(s) are visible")
    totals = [float(torch.cuda.get_device_properties(idx).total_memory) / (1024.0 * 1024.0) for idx in range(nproc_per_node)]
    return min(totals)


def _memory_budget_mb(args: argparse.Namespace) -> tuple[float, float]:
    if args.memory_budget_mb > 0:
        if torch.cuda.is_available():
            gpu_total_mb = _detect_gpu_total_memory_mb(args.nproc_per_node)
        else:
            gpu_total_mb = float(args.memory_budget_mb)
        return float(args.memory_budget_mb), gpu_total_mb
    gpu_total_mb = _detect_gpu_total_memory_mb(args.nproc_per_node)
    return float(gpu_total_mb * args.memory_budget_fraction), gpu_total_mb


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return max(multiple, int(math.ceil(value / multiple) * multiple))


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return max(multiple, int(math.floor(value / multiple) * multiple))


def _next_exponential_batch_size(current: int, multiple: int, growth_factor: float, max_batch_size: int) -> int:
    grown = max(current + multiple, int(math.ceil(current * max(growth_factor, 1.1))))
    return min(max_batch_size, _round_up_to_multiple(grown, multiple))


def _reason_from_result(
    result: harness.CaseResult,
    *,
    fit_mode: str,
    memory_metric: str,
    memory_budget_mb: float | None,
) -> tuple[bool, float | None, str]:
    if result.return_code != 0:
        log_path = Path(result.log_path)
        if log_path.exists():
            log_text = log_path.read_text(errors="replace").lower()
            if "out of memory" in log_text:
                return False, None, "oom"
        return False, None, f"return_code_{result.return_code}"

    peak_memory_mb = getattr(result, memory_metric)
    peak_memory_value = None if peak_memory_mb is None else float(peak_memory_mb)
    if fit_mode == "oom_boundary":
        return True, peak_memory_value, "ok"

    if peak_memory_value is None:
        return False, None, "missing_memory_metric"
    if memory_budget_mb is None:
        raise ValueError("memory_budget_mb is required when fit_mode=memory_budget")
    if peak_memory_value > memory_budget_mb:
        return False, peak_memory_value, "over_budget"
    return True, peak_memory_value, "ok"


def _trial_from_result(
    result: harness.CaseResult,
    *,
    fit_mode: str,
    memory_metric: str,
    memory_budget_mb: float | None,
) -> TuningTrial:
    fits, peak_memory_mb, reason = _reason_from_result(
        result,
        fit_mode=fit_mode,
        memory_metric=memory_metric,
        memory_budget_mb=memory_budget_mb,
    )
    return TuningTrial(
        batch_size=int(result.config["batch_size"]),
        fits=fits,
        peak_memory_mb=peak_memory_mb,
        return_code=int(result.return_code),
        reason=reason,
        case_id=result.case_id,
        log_path=result.log_path,
        profile_path=result.profile_path,
        mean_tokens_per_s=result.mean_tokens_per_s,
    )


def _initial_batch_size_guess(
    *,
    min_batch_size: int,
    max_batch_size: int,
    batch_size_multiple: int,
    initial_batch_size: int,
) -> int:
    multiple = max(1, batch_size_multiple)
    if initial_batch_size > 0:
        guess = initial_batch_size
    else:
        guess = (min_batch_size + max_batch_size) // 2
    guess = max(min_batch_size, min(max_batch_size, guess))
    guess = _round_down_to_multiple(guess, multiple)
    if guess < min_batch_size:
        guess = _round_up_to_multiple(min_batch_size, multiple)
    return guess


def _select_max_batch_size(
    min_batch_size: int,
    max_batch_size: int,
    batch_size_multiple: int,
    growth_factor: float,
    initial_batch_size: int,
    evaluator: Callable[[int], TuningTrial],
) -> tuple[TuningTrial, List[TuningTrial]]:
    if min_batch_size < 1:
        raise ValueError("min_batch_size must be >= 1")
    if max_batch_size < min_batch_size:
        raise ValueError("max_batch_size must be >= min_batch_size")

    multiple = max(1, batch_size_multiple)
    current = _initial_batch_size_guess(
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_size_multiple=multiple,
        initial_batch_size=initial_batch_size,
    )
    max_batch_size = _round_down_to_multiple(max_batch_size, multiple)
    trials: List[TuningTrial] = []
    trial_by_batch: Dict[int, TuningTrial] = {}

    def evaluate(batch_size: int) -> TuningTrial:
        if batch_size not in trial_by_batch:
            trial_by_batch[batch_size] = evaluator(batch_size)
            trials.append(trial_by_batch[batch_size])
        return trial_by_batch[batch_size]

    trial = evaluate(current)
    last_fit: TuningTrial | None = trial if trial.fits else None
    first_fail_batch: int | None = None if trial.fits else current

    if trial.fits:
        while True:
            if current >= max_batch_size:
                return trial, sorted(trials, key=lambda item: item.batch_size)
            next_batch = _next_exponential_batch_size(
                current=current,
                multiple=multiple,
                growth_factor=growth_factor,
                max_batch_size=max_batch_size,
            )
            if next_batch == current:
                return trial, sorted(trials, key=lambda item: item.batch_size)
            current = next_batch
            trial = evaluate(current)
            if trial.fits:
                last_fit = trial
                continue
            first_fail_batch = current
            break
    else:
        while True:
            if current <= min_batch_size:
                raise RuntimeError(
                    f"no fitting batch size found: minimum tried microbatch {current} failed ({trial.reason})"
                )
            next_batch = max(min_batch_size, int(math.floor(current / max(growth_factor, 1.1))))
            next_batch = _round_down_to_multiple(next_batch, multiple)
            if next_batch >= current:
                next_batch = current - multiple
            next_batch = max(_round_up_to_multiple(min_batch_size, multiple), next_batch)
            current = next_batch
            trial = evaluate(current)
            if trial.fits:
                last_fit = trial
                break
            first_fail_batch = current

    assert last_fit is not None
    assert first_fail_batch is not None
    low = last_fit.batch_size
    high = first_fail_batch
    while high - low > multiple:
        mid = _round_down_to_multiple((low + high) // 2, multiple)
        if mid <= low:
            mid = low + multiple
        if mid >= high:
            break
        trial = evaluate(mid)
        if trial.fits:
            last_fit = trial
            low = mid
        else:
            high = mid

    return last_fit, sorted(trials, key=lambda trial: trial.batch_size)


def _training_extra_args(base_extra_args: str, tuning_memory_warmup_steps: int) -> str:
    measured_step_args = f"--measure-memory-step --memory-warmup-steps {tuning_memory_warmup_steps}"
    if base_extra_args.strip():
        return f"{base_extra_args.strip()} {measured_step_args}"
    return measured_step_args


def _make_case(
    args: argparse.Namespace,
    *,
    stage: int,
    batch_size: int,
    bandwidth_gbps: float,
    steps: int,
    metrics_warmup_steps: int,
    bandwidth_mode: str,
    extra_args: str,
) -> harness.CaseConfig:
    return harness.CaseConfig(
        stage=int(stage),
        model_size=str(args.model_size),
        bandwidth_gbps=float(bandwidth_gbps),
        nproc_per_node=int(args.nproc_per_node),
        steps=int(steps),
        seq_len=int(args.seq_len),
        batch_size=int(batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        collective_impl=str(args.collective_impl),
        data_mode=str(args.data_mode),
        seed=int(args.seed),
        dtype=str(args.dtype),
        max_grad_norm=float(args.max_grad_norm),
        stage2_grad_bucket_mb=float(args.stage2_grad_bucket_mb),
        profile_memory_interval=int(args.profile_memory_interval),
        metrics_warmup_steps=int(metrics_warmup_steps),
        bandwidth_mode=str(bandwidth_mode),
        sim_latency_ms=float(args.sim_latency_ms),
        tc_interface=str(args.tc_interface),
        socket_interface=str(args.socket_interface),
        socket_shaper_burst_bytes=int(args.socket_shaper_burst_bytes),
        theory_vocab_size=int(args.theory_vocab_size),
        extra_args=str(extra_args),
    )


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    launch = harness._launch_config_from_args(args)
    run_dir = Path(args.results_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir = run_dir / "tuning"

    if args.fit_mode == "memory_budget":
        memory_budget_mb, gpu_total_memory_mb = _memory_budget_mb(args)
    else:
        gpu_total_memory_mb = _detect_gpu_total_memory_mb(args.nproc_per_node)
        memory_budget_mb = None
    max_batch_size = int(args.max_batch_size) if int(args.max_batch_size) > 0 else 1024
    batch_size_multiple = max(1, int(args.batch_size_multiple))
    growth_factor = max(1.1, float(args.growth_factor))
    tuning_steps = int(args.tuning_memory_warmup_steps) + 1

    case_counter = itertools.count()
    tuning_payload: Dict[str, object] = {
        "experiment_type": "fit_to_memory_bandwidth",
        "name": args.name,
        "model_size": args.model_size,
        "seq_len": int(args.seq_len),
        "grad_accum_steps": int(args.grad_accum_steps),
        "nproc_per_node": int(args.nproc_per_node),
        "gpu_total_memory_mb": float(gpu_total_memory_mb),
        "fit_mode": str(args.fit_mode),
        "memory_budget_mb": None if memory_budget_mb is None else float(memory_budget_mb),
        "memory_budget_fraction": None if args.fit_mode != "memory_budget" else float(args.memory_budget_fraction),
        "memory_metric": str(args.memory_metric),
        "tuning_memory_warmup_steps": int(args.tuning_memory_warmup_steps),
        "stages": [int(stage) for stage in args.stages],
        "per_stage": {},
    }
    selected_batch_sizes: Dict[int, int] = {}

    for stage in args.stages:
        stage_int = int(stage)

        def evaluate(batch_size: int) -> TuningTrial:
            tuning_case = _make_case(
                args,
                stage=stage_int,
                batch_size=batch_size,
                bandwidth_gbps=0.0,
                steps=tuning_steps,
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
                fit_mode=str(args.fit_mode),
                memory_metric=str(args.memory_metric),
                memory_budget_mb=memory_budget_mb,
            )

        best_trial, trials = _select_max_batch_size(
            min_batch_size=int(args.min_batch_size),
            max_batch_size=max_batch_size,
            batch_size_multiple=batch_size_multiple,
            growth_factor=growth_factor,
            initial_batch_size=int(args.initial_batch_size),
            evaluator=evaluate,
        )
        selected_batch_sizes[stage_int] = best_trial.batch_size
        tuning_payload["per_stage"][str(stage_int)] = {
            "selected_batch_size": int(best_trial.batch_size),
            "selected_peak_memory_mb": None if best_trial.peak_memory_mb is None else float(best_trial.peak_memory_mb),
            "selected_case_id": best_trial.case_id,
            "selected_log_path": best_trial.log_path,
            "selected_profile_path": best_trial.profile_path,
            "selected_mean_tokens_per_s": None
            if best_trial.mean_tokens_per_s is None
            else float(best_trial.mean_tokens_per_s),
            "selected_global_tokens_per_step": int(
                best_trial.batch_size * args.grad_accum_steps * args.seq_len * args.nproc_per_node
            ),
            "trials": [asdict(trial) for trial in trials],
        }
        print(
            f"[fit-memory] stage {stage_int}: selected microbatch {best_trial.batch_size} "
            f"peak={best_trial.peak_memory_mb if best_trial.peak_memory_mb is not None else 'NA'} MB",
            flush=True,
        )

    tuning_summary_path = run_dir / "tuning_summary.json"
    _write_json(tuning_summary_path, tuning_payload)
    print(f"[fit-memory] wrote {tuning_summary_path}", flush=True)

    if args.dry_run:
        print("[fit-memory] dry run complete; skipping final bandwidth sweep", flush=True)
        return

    results: List[harness.CaseResult] = []
    cases: List[harness.CaseConfig] = []
    for stage in args.stages:
        stage_int = int(stage)
        batch_size = selected_batch_sizes[stage_int]
        for bandwidth_gbps in args.bandwidth_gbps:
            cases.append(
                _make_case(
                    args,
                    stage=stage_int,
                    batch_size=batch_size,
                    bandwidth_gbps=float(bandwidth_gbps),
                    steps=int(args.steps),
                    metrics_warmup_steps=int(args.metrics_warmup_steps),
                    bandwidth_mode=str(args.bandwidth_mode),
                    extra_args=str(args.extra_args),
                )
            )

    print(f"[fit-memory] prepared {len(cases)} final sweep case(s)", flush=True)
    for idx, case in enumerate(cases, start=1):
        print(f"[fit-memory] ({idx}/{len(cases)}) {harness._case_id(case)}", flush=True)
        result = harness._run_case(
            case=case,
            run_dir=run_dir,
            skip_existing=args.skip_existing,
            dry_run=False,
            launch=launch,
            case_index=next(case_counter),
        )
        results.append(result)
        if result.return_code != 0:
            print(f"[fit-memory] case failed: {result.case_id} (see {result.log_path})", flush=True)

    summary = harness._build_summary(results=results, args=args)
    summary["experiment_type"] = "fit_to_memory_bandwidth"
    summary["fit_mode"] = str(args.fit_mode)
    summary["memory_budget_mb"] = None if memory_budget_mb is None else float(memory_budget_mb)
    summary["gpu_total_memory_mb"] = float(gpu_total_memory_mb)
    summary["memory_metric"] = str(args.memory_metric)
    summary["fit_memory_tuning"] = {
        stage: {
            "selected_batch_size": stage_payload["selected_batch_size"],
            "selected_peak_memory_mb": stage_payload["selected_peak_memory_mb"],
            "selected_global_tokens_per_step": stage_payload["selected_global_tokens_per_step"],
        }
        for stage, stage_payload in dict(tuning_payload["per_stage"]).items()
    }
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    print(f"[fit-memory] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
