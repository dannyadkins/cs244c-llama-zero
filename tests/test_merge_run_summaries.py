from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_summary(run_dir: Path, payload: dict[str, object]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(payload) + "\n")


def test_merge_run_summaries_combines_results_and_prefers_later_inputs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    out = tmp_path / "merged"

    _write_summary(
        run_a,
        {
            "name": "run_a",
            "args": {"bandwidth_mode": "socket", "nproc_per_node": 4, "steps": 4, "metrics_warmup_steps": 1},
            "results": [
                {
                    "config": {
                        "stage": 0,
                        "model_size": "small",
                        "bandwidth_gbps": 0.0,
                        "batch_size": 4,
                        "seq_len": 128,
                        "grad_accum_steps": 1,
                    },
                    "log_path": "logs/s0.log",
                    "profile_path": "",
                    "mean_tokens_per_s": 100.0,
                    "mean_comm_ms": 10.0,
                    "return_code": 0,
                }
            ],
        },
    )
    _write_summary(
        run_b,
        {
            "name": "run_b",
            "args": {"bandwidth_mode": "socket", "nproc_per_node": 4, "steps": 4, "metrics_warmup_steps": 1},
            "results": [
                {
                    "config": {
                        "stage": 0,
                        "model_size": "small",
                        "bandwidth_gbps": 0.0,
                        "batch_size": 4,
                        "seq_len": 128,
                        "grad_accum_steps": 1,
                    },
                    "log_path": "logs/s0_new.log",
                    "profile_path": "",
                    "mean_tokens_per_s": 110.0,
                    "mean_comm_ms": 9.0,
                    "return_code": 0,
                },
                {
                    "config": {
                        "stage": 3,
                        "model_size": "small",
                        "bandwidth_gbps": 0.0,
                        "batch_size": 8,
                        "seq_len": 128,
                        "grad_accum_steps": 1,
                    },
                    "log_path": "logs/s3.log",
                    "profile_path": "",
                    "mean_tokens_per_s": 150.0,
                    "mean_comm_ms": 20.0,
                    "return_code": 0,
                },
            ],
        },
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/merge_run_summaries.py",
            "--run-dirs",
            str(run_a),
            str(run_b),
            "--output-run-dir",
            str(out),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    merged = json.loads((out / "summary.json").read_text())
    assert merged["name"] == "merged"
    assert merged["num_cases"] == 2
    assert merged["merged_from_run_dirs"] == [str(run_a.resolve()), str(run_b.resolve())]
    assert [result["mean_tokens_per_s"] for result in merged["results"]] == [110.0, 150.0]
    assert (out / "bandwidth_report.md").exists()
