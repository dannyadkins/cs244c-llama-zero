from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


STEP_RE = re.compile(r"\[step\s+(\d+)\]\s+loss=([0-9.]+)")


@dataclass
class StageRunResult:
    stage: int
    command: str
    return_code: int
    elapsed_s: float
    final_loss: float | None
    num_logged_steps: int
    log_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 2 experiment harness for ZeRO stages 0-2")
    parser.add_argument("--stages", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--collective-impl", type=str, default="ring", choices=["ring", "torch"])
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "fineweb"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--name", type=str, default="week2_baseline")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--extra-args", type=str, default="")
    return parser.parse_args()


def parse_losses(log_text: str) -> List[float]:
    losses: List[float] = []
    for line in log_text.splitlines():
        match = STEP_RE.search(line)
        if match:
            losses.append(float(match.group(2)))
    return losses


def run_stage(args: argparse.Namespace, stage: int, out_dir: Path) -> StageRunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"stage{stage}.log"

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "train_zero.py",
        "--zero-stage",
        str(stage),
        "--collective-impl",
        args.collective_impl,
        "--data-mode",
        args.data_mode,
        "--model-size",
        args.model_size,
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--max-steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--log-interval",
        "1",
        "--checkpoint-interval",
        "0",
    ]

    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))

    t0 = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - t0

    log_text = completed.stdout + "\n" + completed.stderr
    log_path.write_text(log_text)

    losses = parse_losses(log_text)
    return StageRunResult(
        stage=stage,
        command=" ".join(shlex.quote(x) for x in cmd),
        return_code=completed.returncode,
        elapsed_s=elapsed_s,
        final_loss=losses[-1] if losses else None,
        num_logged_steps=len(losses),
        log_path=str(log_path),
    )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_dir)
    run_dir = results_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"[harness] skipping existing run: {summary_path}")
        return

    results: List[StageRunResult] = []
    for stage in args.stages:
        print(f"[harness] running stage {stage}")
        result = run_stage(args=args, stage=stage, out_dir=run_dir)
        results.append(result)
        if result.return_code != 0:
            print(f"[harness] stage {stage} failed. see {result.log_path}")
            break

    payload: Dict[str, object] = {
        "args": vars(args),
        "results": [asdict(x) for x in results],
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"[harness] wrote {summary_path}")


if __name__ == "__main__":
    main()
