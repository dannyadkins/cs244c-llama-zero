from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import build_config, estimate_num_parameters


STEP_RE = re.compile(
    r"\[step\s+(\d+)\]\s+loss=([0-9.eE+\-]+).*?tokens/s=([0-9,]+).*?grad_norm=([0-9.eE+\-a-zA-Z]+).*?fb_ms=([0-9.eE+\-]+).*?comm_ms=([0-9.eE+\-]+).*?opt_ms=([0-9.eE+\-]+)(?:.*?tflops=([0-9.eE+\-]+))?"
)


@dataclass
class CaseConfig:
    stage: int
    model_size: str
    bandwidth_gbps: float
    nproc_per_node: int
    steps: int
    seq_len: int
    batch_size: int
    grad_accum_steps: int
    collective_impl: str
    data_mode: str
    seed: int
    dtype: str
    max_grad_norm: float
    profile_memory_interval: int
    bandwidth_mode: str
    sim_latency_ms: float
    tc_interface: str
    theory_vocab_size: int
    extra_args: str


@dataclass
class CaseResult:
    case_id: str
    config: Dict[str, object]
    command: str
    return_code: int
    elapsed_s: float
    log_path: str
    profile_path: str
    final_loss: float | None
    logged_steps: int
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
    measured_state_memory_mb: Dict[str, float] | None
    theoretical_param_count: int
    theoretical_memory_mb: Dict[str, float]
    notes: str = ""


@dataclass(frozen=True)
class LaunchConfig:
    nnodes: int
    node_rank: int
    master_addr: str
    master_port_base: int
    case_timeout_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 3 experiment harness for ZeRO stages 0-3")
    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--name", type=str, default="week3_matrix")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--stages", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--model-sizes", nargs="+", default=["tiny"])
    parser.add_argument("--bandwidth-gbps", nargs="+", type=float, default=[0.0])

    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port-base", type=int, default=29500)
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--collective-impl", type=str, default="ring", choices=["ring", "torch"])
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "fineweb"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--max-grad-norm", type=float, default=0.0)

    parser.add_argument("--profile-memory-interval", type=int, default=0)

    parser.add_argument("--bandwidth-mode", type=str, default="simulated", choices=["simulated", "none", "tc"])
    parser.add_argument("--sim-latency-ms", type=float, default=0.0)
    parser.add_argument("--tc-interface", type=str, default="eth0")
    parser.add_argument(
        "--theory-vocab-size",
        type=int,
        default=0,
        help="If >0, use this vocab size for theoretical memory estimates; otherwise inferred from data mode.",
    )

    parser.add_argument("--extra-args", type=str, default="")
    return parser.parse_args()


def _merge_config_file(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    payload = json.loads(Path(args.config).read_text())
    defaults = payload.get("defaults", {})
    matrix = payload.get("matrix", {})

    for key, value in defaults.items():
        if hasattr(args, key):
            setattr(args, key, value)

    if "stages" in matrix:
        args.stages = matrix["stages"]
    if "model_sizes" in matrix:
        args.model_sizes = matrix["model_sizes"]
    if "bandwidth_gbps" in matrix:
        args.bandwidth_gbps = matrix["bandwidth_gbps"]

    return args


def _case_id(case: CaseConfig) -> str:
    bw = "unlimited" if case.bandwidth_gbps <= 0 else f"{case.bandwidth_gbps:g}gbps"
    return (
        f"s{case.stage}_m{case.model_size}_bw{bw}_np{case.nproc_per_node}_"
        f"sl{case.seq_len}_bs{case.batch_size}_ga{case.grad_accum_steps}_seed{case.seed}"
    )


def _launch_config_from_args(args: argparse.Namespace) -> LaunchConfig:
    return LaunchConfig(
        nnodes=int(args.nnodes),
        node_rank=int(args.node_rank),
        master_addr=str(args.master_addr),
        master_port_base=int(args.master_port_base),
        case_timeout_s=float(args.case_timeout_s),
    )


def _build_cases(args: argparse.Namespace) -> List[CaseConfig]:
    cases: List[CaseConfig] = []
    for stage, model_size, bw in itertools.product(args.stages, args.model_sizes, args.bandwidth_gbps):
        cases.append(
            CaseConfig(
                stage=int(stage),
                model_size=str(model_size),
                bandwidth_gbps=float(bw),
                nproc_per_node=int(args.nproc_per_node),
                steps=int(args.steps),
                seq_len=int(args.seq_len),
                batch_size=int(args.batch_size),
                grad_accum_steps=int(args.grad_accum_steps),
                collective_impl=str(args.collective_impl),
                data_mode=str(args.data_mode),
                seed=int(args.seed),
                dtype=str(args.dtype),
                max_grad_norm=float(args.max_grad_norm),
                profile_memory_interval=int(args.profile_memory_interval),
                bandwidth_mode=str(args.bandwidth_mode),
                sim_latency_ms=float(args.sim_latency_ms),
                tc_interface=str(args.tc_interface),
                theory_vocab_size=int(args.theory_vocab_size),
                extra_args=str(args.extra_args),
            )
        )
    return cases


def _theoretical_memory(case: CaseConfig) -> tuple[int, Dict[str, float]]:
    vocab_size = case.theory_vocab_size
    if vocab_size <= 0:
        vocab_size = 8_192 if case.data_mode == "synthetic" else 128_256

    cfg = build_config(size=case.model_size, vocab_size=vocab_size, max_seq_len=case.seq_len)
    num_params = estimate_num_parameters(cfg)

    # In this codebase, model parameters and gradient buffers are fp32.
    param_bytes = num_params * 4.0
    grad_bytes = num_params * 4.0
    opt_bytes = num_params * 8.0  # AdamW exp_avg + exp_avg_sq
    world = max(1, case.nproc_per_node)

    if case.stage == 0:
        stage_param = param_bytes
        stage_grad = grad_bytes
        stage_opt = opt_bytes
    elif case.stage == 1:
        stage_param = param_bytes
        stage_grad = grad_bytes
        stage_opt = opt_bytes / world
    elif case.stage == 2:
        stage_param = param_bytes
        stage_grad = grad_bytes / world
        stage_opt = opt_bytes / world
    elif case.stage == 3:
        # Ideal ZeRO-3 memory model for optimizer/gradient/parameter states.
        stage_param = param_bytes / world
        stage_grad = grad_bytes / world
        stage_opt = opt_bytes / world
    else:
        raise ValueError(f"Unsupported stage: {case.stage}")

    to_mb = 1024.0 * 1024.0
    breakdown = {
        "params_mb": stage_param / to_mb,
        "grads_mb": stage_grad / to_mb,
        "optimizer_mb": stage_opt / to_mb,
        "total_mb": (stage_param + stage_grad + stage_opt) / to_mb,
    }
    return num_params, breakdown


def _parse_step_metrics(log_text: str) -> Dict[str, object]:
    losses: List[float] = []
    tps: List[float] = []
    tflops: List[float] = []
    comm_ms: List[float] = []
    fb_ms: List[float] = []
    opt_ms: List[float] = []

    for line in log_text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        losses.append(float(m.group(2)))
        tps.append(float(m.group(3).replace(",", "")))
        if m.group(8) is not None:
            tflops.append(float(m.group(8)))
        fb_ms.append(float(m.group(5)))
        comm_ms.append(float(m.group(6)))
        opt_ms.append(float(m.group(7)))

    def mean_or_none(xs: List[float]) -> float | None:
        return float(sum(xs) / len(xs)) if xs else None

    return {
        "final_loss": losses[-1] if losses else None,
        "logged_steps": len(losses),
        "mean_tokens_per_s": mean_or_none(tps),
        "mean_tflops_per_s": mean_or_none(tflops),
        "mean_comm_ms": mean_or_none(comm_ms),
        "mean_fb_ms": mean_or_none(fb_ms),
        "mean_opt_ms": mean_or_none(opt_ms),
    }


def _parse_profile_memory(profile_path: Path) -> Dict[str, object]:
    if not profile_path.exists():
        return {
            "peak_host_rss_mb": None,
            "peak_cuda_allocated_mb": None,
            "peak_cuda_reserved_mb": None,
            "peak_cuda_max_allocated_mb": None,
            "peak_cuda_max_reserved_mb": None,
            "measured_state_memory_mb": None,
        }

    payload = json.loads(profile_path.read_text())
    snapshots = payload.get("memory", [])
    state_memory = payload.get("state_memory_breakdown_mb")
    measured_state_memory_mb = None
    if isinstance(state_memory, dict):
        measured_state_memory_mb = {str(key): float(value) for key, value in state_memory.items()}
    if not snapshots:
        return {
            "peak_host_rss_mb": None,
            "peak_cuda_allocated_mb": None,
            "peak_cuda_reserved_mb": None,
            "peak_cuda_max_allocated_mb": None,
            "peak_cuda_max_reserved_mb": None,
            "measured_state_memory_mb": measured_state_memory_mb,
        }

    def peak(key: str) -> float | None:
        values = [float(snapshot[key]) for snapshot in snapshots if key in snapshot]
        return max(values) if values else None

    return {
        "peak_host_rss_mb": peak("host_maxrss_mb"),
        "peak_cuda_allocated_mb": peak("cuda_allocated_mb"),
        "peak_cuda_reserved_mb": peak("cuda_reserved_mb"),
        "peak_cuda_max_allocated_mb": peak("cuda_max_allocated_mb"),
        "peak_cuda_max_reserved_mb": peak("cuda_max_reserved_mb"),
        "measured_state_memory_mb": measured_state_memory_mb,
    }


def _apply_tc(case: CaseConfig) -> None:
    rate = f"{case.bandwidth_gbps:g}gbit"
    subprocess.run(
        [str(PROJECT_ROOT / "infra" / "throttle.sh"), "apply", case.tc_interface, rate, "1mb", "10ms"],
        check=True,
        cwd=PROJECT_ROOT,
    )


def _clear_tc(case: CaseConfig) -> None:
    subprocess.run(
        [str(PROJECT_ROOT / "infra" / "throttle.sh"), "delete", case.tc_interface],
        check=False,
        cwd=PROJECT_ROOT,
    )


def _master_port_for_case(launch: LaunchConfig, case_index: int) -> int:
    return launch.master_port_base + case_index


def _build_launch_env(case: CaseConfig) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("ZERO_SIM_BW_GBPS", None)
    env.pop("ZERO_SIM_LATENCY_MS", None)

    if case.bandwidth_mode == "simulated":
        if case.bandwidth_gbps > 0:
            env["ZERO_SIM_BW_GBPS"] = str(case.bandwidth_gbps)
        else:
            env.pop("ZERO_SIM_BW_GBPS", None)

        if case.sim_latency_ms > 0:
            env["ZERO_SIM_LATENCY_MS"] = str(case.sim_latency_ms)
        else:
            env.pop("ZERO_SIM_LATENCY_MS", None)

    return env


def _build_train_zero_cmd(case: CaseConfig, profile_path: Path, launch: LaunchConfig, case_index: int) -> List[str]:
    master_port = _master_port_for_case(launch=launch, case_index=case_index)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(launch.nnodes),
        "--node_rank",
        str(launch.node_rank),
        "--nproc_per_node",
        str(case.nproc_per_node),
        "--master_addr",
        launch.master_addr,
        "--master_port",
        str(master_port),
        str(PROJECT_ROOT / "train_zero.py"),
        "--zero-stage",
        str(case.stage),
        "--collective-impl",
        case.collective_impl,
        "--data-mode",
        case.data_mode,
        "--model-size",
        case.model_size,
        "--seq-len",
        str(case.seq_len),
        "--batch-size",
        str(case.batch_size),
        "--grad-accum-steps",
        str(case.grad_accum_steps),
        "--max-steps",
        str(case.steps),
        "--seed",
        str(case.seed),
        "--dtype",
        case.dtype,
        "--max-grad-norm",
        str(case.max_grad_norm),
        "--log-interval",
        "1",
        "--checkpoint-interval",
        "0",
        "--profile-json",
        str(profile_path),
        "--profile-rank0-only",
    ]

    if case.profile_memory_interval > 0:
        cmd.extend(["--profile-memory-interval", str(case.profile_memory_interval)])

    if case.extra_args.strip():
        cmd.extend(shlex.split(case.extra_args))

    return cmd


def _combine_process_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    def _to_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    return _to_text(stdout) + "\n" + _to_text(stderr)


def _run_case(
    case: CaseConfig,
    run_dir: Path,
    skip_existing: bool,
    dry_run: bool,
    launch: LaunchConfig,
    case_index: int,
) -> CaseResult:
    case_id = _case_id(case)
    case_dir = run_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    case_result_path = case_dir / f"{case_id}.json"
    if skip_existing and case_result_path.exists():
        payload = json.loads(case_result_path.read_text())
        return CaseResult(**payload)

    logs_dir = run_dir / "logs"
    profiles_dir = run_dir / "profiles"
    logs_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{case_id}.log"
    profile_path = profiles_dir / f"{case_id}.json"
    cmd = _build_train_zero_cmd(case=case, profile_path=profile_path, launch=launch, case_index=case_index)
    env = _build_launch_env(case)
    command_str = " ".join(shlex.quote(x) for x in cmd)

    if dry_run:
        num_params, memory_mb = _theoretical_memory(case)
        result = CaseResult(
            case_id=case_id,
            config=asdict(case),
            command=command_str,
            return_code=0,
            elapsed_s=0.0,
            log_path=str(log_path),
            profile_path=str(profile_path),
            final_loss=None,
            logged_steps=0,
            mean_tokens_per_s=None,
            mean_tflops_per_s=None,
            mean_comm_ms=None,
            mean_fb_ms=None,
            mean_opt_ms=None,
            peak_host_rss_mb=None,
            peak_cuda_allocated_mb=None,
            peak_cuda_reserved_mb=None,
            peak_cuda_max_allocated_mb=None,
            peak_cuda_max_reserved_mb=None,
            measured_state_memory_mb=None,
            theoretical_param_count=num_params,
            theoretical_memory_mb=memory_mb,
            notes="dry_run",
        )
        case_result_path.write_text(json.dumps(asdict(result), indent=2))
        return result

    use_tc = case.bandwidth_mode == "tc" and case.bandwidth_gbps > 0
    if use_tc:
        _apply_tc(case)

    t0 = time.perf_counter()
    notes = ""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
            timeout=launch.case_timeout_s if launch.case_timeout_s > 0 else None,
        )
        log_text = _combine_process_output(completed.stdout, completed.stderr)
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        log_text = _combine_process_output(exc.stdout, exc.stderr)
        log_text += f"\n[harness] timeout after {launch.case_timeout_s:.1f}s\n"
        return_code = 124
        notes = f"timeout_after_s={launch.case_timeout_s:.1f}"
    finally:
        if use_tc:
            _clear_tc(case)

    elapsed_s = time.perf_counter() - t0
    log_path.write_text(log_text)

    metrics = _parse_step_metrics(log_text)
    memory_metrics = _parse_profile_memory(profile_path)
    num_params, memory_mb = _theoretical_memory(case)

    result = CaseResult(
        case_id=case_id,
        config=asdict(case),
        command=command_str,
        return_code=return_code,
        elapsed_s=elapsed_s,
        log_path=str(log_path),
        profile_path=str(profile_path),
        final_loss=metrics["final_loss"],
        logged_steps=int(metrics["logged_steps"]),
        mean_tokens_per_s=metrics["mean_tokens_per_s"],
        mean_tflops_per_s=metrics["mean_tflops_per_s"],
        mean_comm_ms=metrics["mean_comm_ms"],
        mean_fb_ms=metrics["mean_fb_ms"],
        mean_opt_ms=metrics["mean_opt_ms"],
        peak_host_rss_mb=memory_metrics["peak_host_rss_mb"],
        peak_cuda_allocated_mb=memory_metrics["peak_cuda_allocated_mb"],
        peak_cuda_reserved_mb=memory_metrics["peak_cuda_reserved_mb"],
        peak_cuda_max_allocated_mb=memory_metrics["peak_cuda_max_allocated_mb"],
        peak_cuda_max_reserved_mb=memory_metrics["peak_cuda_max_reserved_mb"],
        measured_state_memory_mb=memory_metrics["measured_state_memory_mb"],
        theoretical_param_count=num_params,
        theoretical_memory_mb=memory_mb,
        notes=notes,
    )

    case_result_path.write_text(json.dumps(asdict(result), indent=2))
    return result


def _build_summary(results: List[CaseResult], args: argparse.Namespace) -> Dict[str, object]:
    return {
        "name": args.name,
        "args": vars(args),
        "theoretical_memory_note": (
            "Memory estimates are idealized ZeRO state-memory terms (params, grads, optimizer states), "
            "fp32-based, and do not include activations, temporary buffers, or framework overhead."
        ),
        "profiled_tflops_note": (
            "Training TFLOPs/s is derived from the train_zero TFLOPs mode: estimated cluster step FLOPs by default, "
            "or torch.profiler-derived FLOPs when profile mode is explicitly enabled."
        ),
        "num_cases": len(results),
        "num_failures": sum(1 for r in results if r.return_code != 0),
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    args = _merge_config_file(parse_args())
    launch = _launch_config_from_args(args)

    if shutil.which("tc") is None and args.bandwidth_mode == "tc":
        raise FileNotFoundError("bandwidth mode 'tc' requires the 'tc' command to be installed")

    run_dir = Path(args.results_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = _build_cases(args)
    print(f"[harness] prepared {len(cases)} cases")

    results: List[CaseResult] = []
    for idx, case in enumerate(cases, start=1):
        case_id = _case_id(case)
        print(f"[harness] ({idx}/{len(cases)}) {case_id}")
        result = _run_case(
            case=case,
            run_dir=run_dir,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            launch=launch,
            case_index=idx - 1,
        )
        results.append(result)
        if result.return_code != 0:
            print(f"[harness] case failed: {case_id} (see {result.log_path})")

    summary = _build_summary(results=results, args=args)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[harness] wrote {summary_path}")


if __name__ == "__main__":
    main()
