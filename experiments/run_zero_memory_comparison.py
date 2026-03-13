from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"

DEFAULT_MODEL_SIZE = "small"
DEFAULT_SEQ_LEN = 512
DEFAULT_BATCH_SIZE = 44
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_STEPS = 20
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MEASURE_MEMORY_STEP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ZeRO stage memory comparison sweep")
    parser.add_argument(
        "--name",
        type=str,
        default="memory_comparison",
        help="Optional run name. If omitted, use the project memory-comparison preset name.",
    )
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--model-size", type=str, default=DEFAULT_MODEL_SIZE, choices=["tiny", "small", "medium"])
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["float32", "bfloat16"])
    parser.add_argument("--nproc-per-node", type=int, default=0, help="If omitted, use all visible GPUs.")
    parser.add_argument("--profile-memory-interval", type=int, default=1)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port-base", type=int, default=29500)
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--measure-memory-step",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MEASURE_MEMORY_STEP,
        help="Warm up a few steps, then record one measured step with reset CUDA peak stats.",
    )
    parser.add_argument(
        "--memory-warmup-steps",
        type=int,
        default=None,
        help="If omitted, use steps - 1 when measured-step mode is enabled.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def detect_visible_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        return len(devices)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to detect GPUs when CUDA_VISIBLE_DEVICES is not set"
        ) from exc

    return int(torch.cuda.device_count())


def default_run_name(args: argparse.Namespace) -> str:
    suffix = "measured_step" if args.measure_memory_step else "train"
    return f"{args.model_size}_memory_{suffix}_bs{args.batch_size}_allgpus_project"


def run_command(cmd: list[str]) -> None:
    command_str = " ".join(cmd)
    print(f"[memory-sweep] running: {command_str}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    warmup_steps = args.memory_warmup_steps
    if args.measure_memory_step and warmup_steps is None:
        warmup_steps = max(args.steps - 1, 0)

    nproc_per_node = args.nproc_per_node if args.nproc_per_node > 0 else detect_visible_gpus()
    if nproc_per_node < 1:
        raise SystemExit("No visible GPUs detected. Set CUDA_VISIBLE_DEVICES or pass --nproc-per-node.")

    if nproc_per_node == 1:
        print(
            "[memory-sweep] warning: running with 1 visible GPU; stage-to-stage communication effects will not be representative",
            flush=True,
        )

    run_name = args.name or default_run_name(args)
    run_dir = Path(args.results_dir) / run_name

    print(f"[memory-sweep] python={sys.executable}", flush=True)
    print(f"[memory-sweep] nproc_per_node={nproc_per_node} run_name={run_name}", flush=True)
    print(f"[memory-sweep] output_dir={run_dir}", flush=True)

    steps = args.steps
    extra_train_args: list[str] = []
    if args.measure_memory_step:
        steps = warmup_steps + 1
        extra_train_args.extend([
            "--measure-memory-step",
            "--memory-warmup-steps",
            str(warmup_steps),
        ])

    harness_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "harness.py"),
        "--name",
        run_name,
        "--results-dir",
        str(args.results_dir),
        "--stages",
        "0",
        "1",
        "2",
        "3",
        "--model-sizes",
        args.model_size,
        "--bandwidth-gbps",
        "0",
        "--nproc-per-node",
        str(nproc_per_node),
        "--steps",
        str(steps),
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--collective-impl",
        "torch",
        "--data-mode",
        "synthetic",
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--profile-memory-interval",
        str(args.profile_memory_interval),
        "--bandwidth-mode",
        "none",
        "--master-addr",
        args.master_addr,
        "--master-port-base",
        str(args.master_port_base),
        "--case-timeout-s",
        str(args.case_timeout_s),
    ]

    if extra_train_args:
        harness_cmd.extend(["--extra-args", " ".join(extra_train_args)])

    if args.skip_existing:
        harness_cmd.append("--skip-existing")
    if args.dry_run:
        harness_cmd.append("--dry-run")

    run_command(harness_cmd)

    if args.dry_run:
        print("[memory-sweep] dry run complete; skipping plot generation", flush=True)
        return

    memory_plot_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "analysis" / "visualize.py"),
        "--run-dir",
        str(run_dir),
        "--plot",
        "memory",
        "--model-size",
        args.model_size,
        "--output",
        str(run_dir / "memory.png"),
    ]
    peak_plot_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "analysis" / "visualize.py"),
        "--run-dir",
        str(run_dir),
        "--plot",
        "peak-memory",
        "--model-size",
        args.model_size,
        "--output",
        str(run_dir / "peak_memory.png"),
    ]
    avg_plot_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "analysis" / "visualize.py"),
        "--run-dir",
        str(run_dir),
        "--plot",
        "avg-memory",
        "--model-size",
        args.model_size,
        "--output",
        str(run_dir / "avg_memory.png"),
    ]
    run_command(memory_plot_cmd)
    run_command(peak_plot_cmd)
    run_command(avg_plot_cmd)

    print(f"[memory-sweep] wrote {run_dir / 'memory.png'}", flush=True)
    print(f"[memory-sweep] wrote {run_dir / 'peak_memory.png'}", flush=True)
    print(f"[memory-sweep] wrote {run_dir / 'avg_memory.png'}", flush=True)


if __name__ == "__main__":
    main()
