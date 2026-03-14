from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import bandwidth_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple harness summary.json files into one combined run dir")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories that contain summary.json")
    parser.add_argument("--output-run-dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="", help="Defaults to the output directory name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def _result_key(result: Dict[str, object]) -> Tuple[object, ...]:
    config = result.get("config", {})
    if not isinstance(config, dict):
        config = {}
    return (
        config.get("stage"),
        config.get("model_size"),
        config.get("bandwidth_gbps"),
        config.get("batch_size"),
        config.get("seq_len"),
        config.get("grad_accum_steps"),
    )


def _result_key_str(key: Tuple[object, ...]) -> str:
    stage, model_size, bandwidth_gbps, batch_size, seq_len, grad_accum_steps = key
    return (
        f"stage={stage}|model={model_size}|bw={bandwidth_gbps}|"
        f"batch={batch_size}|seq={seq_len}|ga={grad_accum_steps}"
    )


def _load_summary(run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json under {run_dir}")
    payload = json.loads(summary_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary.json under {run_dir} must contain a JSON object")
    return payload


def main() -> None:
    args = parse_args()
    run_dirs = [Path(item).resolve() for item in args.run_dirs]
    output_run_dir = Path(args.output_run_dir).resolve()
    if output_run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output run dir already exists: {output_run_dir}")
    output_run_dir.mkdir(parents=True, exist_ok=True)

    loaded = [(_load_summary(run_dir), run_dir) for run_dir in run_dirs]
    base_summary = dict(loaded[0][0])
    merged_results: Dict[Tuple[object, ...], Dict[str, object]] = {}
    source_by_key: Dict[Tuple[object, ...], str] = {}

    for summary, run_dir in loaded:
        for raw_result in summary.get("results", []):
            if not isinstance(raw_result, dict):
                continue
            key = _result_key(raw_result)
            merged_results[key] = raw_result
            source_by_key[key] = str(run_dir)

    output_summary = {
        **base_summary,
        "name": args.name or output_run_dir.name,
        "num_cases": len(merged_results),
        "num_failures": sum(1 for result in merged_results.values() if int(result.get("return_code", 0)) != 0),
        "results": [
            merged_results[key]
            for key in sorted(
                merged_results,
                key=lambda item: (
                    int(-1 if item[0] is None else item[0]),
                    float(0.0 if item[2] is None else item[2]),
                    str("" if item[1] is None else item[1]),
                ),
            )
        ],
        "merged_from_run_dirs": [str(run_dir) for run_dir in run_dirs],
        "merged_result_sources": {_result_key_str(key): source for key, source in source_by_key.items()},
    }

    summary_path = output_run_dir / "summary.json"
    summary_path.write_text(json.dumps(output_summary, indent=2) + "\n")
    print(f"[merge-runs] wrote {summary_path}")

    if not args.skip_report:
        report_path = output_run_dir / "bandwidth_report.md"
        report_path.write_text(bandwidth_report.generate_report_markdown(output_run_dir))
        print(f"[merge-runs] wrote {report_path}")


if __name__ == "__main__":
    main()
