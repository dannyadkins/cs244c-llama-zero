from __future__ import annotations

import json
from pathlib import Path

from analysis import bandwidth_report


def test_generate_report_markdown_includes_best_stage_and_relative_throughput(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "name": "bandwidth_demo",
                "args": {
                    "bandwidth_mode": "simulated",
                    "nproc_per_node": 4,
                    "steps": 20,
                    "metrics_warmup_steps": 1,
                },
                "results": [
                    {
                        "config": {"stage": 0, "model_size": "small", "bandwidth_gbps": 0.0},
                        "log_path": "logs/s0.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 100.0,
                        "mean_tflops_per_s": 1.0,
                        "mean_comm_ms": 5.0,
                        "mean_fb_ms": 10.0,
                        "mean_opt_ms": 1.0,
                        "return_code": 0,
                        "final_loss": 8.0,
                    },
                    {
                        "config": {"stage": 0, "model_size": "small", "bandwidth_gbps": 5.0},
                        "log_path": "logs/s0_bw5.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 80.0,
                        "mean_tflops_per_s": 0.8,
                        "mean_comm_ms": 8.0,
                        "mean_fb_ms": 12.0,
                        "mean_opt_ms": 1.0,
                        "return_code": 0,
                        "final_loss": 8.1,
                    },
                    {
                        "config": {"stage": 1, "model_size": "small", "bandwidth_gbps": 0.0},
                        "log_path": "logs/s1.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 120.0,
                        "mean_tflops_per_s": 1.2,
                        "mean_comm_ms": 6.0,
                        "mean_fb_ms": 11.0,
                        "mean_opt_ms": 1.0,
                        "return_code": 0,
                        "final_loss": 7.9,
                    },
                    {
                        "config": {"stage": 1, "model_size": "small", "bandwidth_gbps": 5.0},
                        "log_path": "logs/s1_bw5.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 60.0,
                        "mean_tflops_per_s": 0.6,
                        "mean_comm_ms": 12.0,
                        "mean_fb_ms": 14.0,
                        "mean_opt_ms": 1.0,
                        "return_code": 0,
                        "final_loss": 8.2,
                    },
                ],
            }
        )
    )

    report = bandwidth_report.generate_report_markdown(run_dir)

    assert "# Bandwidth Sweep Report" in report
    assert "Successful cases: 4 / 4" in report
    assert "metrics_warmup_steps=1" in report
    assert "Best stage by bandwidth:" in report
    assert "| unlimited | 1 | 120.0 | 6.0 |" in report
    assert "| 5 Gbps | 0 | 80.0 | 8.0 |" in report
    assert "Throughput relative to the stage baseline:" in report
    assert "| stage 0 | 100.0% | 80.0% |" in report
    assert "| stage 1 | 100.0% | 50.0% |" in report

