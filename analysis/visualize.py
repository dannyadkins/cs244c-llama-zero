from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"\[step\s+(\d+)\]\s+loss=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ZeRO stage loss curves from harness logs")
    parser.add_argument("--run-dir", type=str, required=True, help="experiments/results/<name>")
    parser.add_argument("--output", type=str, default="analysis/figures/week2_loss_curves.png")
    return parser.parse_args()


def parse_loss_log(path: Path):
    steps = []
    losses = []
    if not path.exists():
        return steps, losses
    for line in path.read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return steps, losses


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization: pip install matplotlib") from exc

    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text())
    results = summary.get("results", [])

    plt.figure(figsize=(8, 5))
    for result in results:
        stage = result["stage"]
        log_path = Path(result["log_path"])
        steps, losses = parse_loss_log(log_path)
        if steps:
            plt.plot(steps, losses, label=f"stage {stage}")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("ZeRO Stage Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[visualize] wrote {out}")


if __name__ == "__main__":
    main()
