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
    assert "Experiment type: fixed-workload bandwidth sweep" in report
    assert "Successful cases: 4 / 4" in report
    assert "metrics_warmup_steps=1" in report
    assert "Best stage by bandwidth:" in report
    assert "Stage ranking by bandwidth:" in report
    assert "| unlimited | 1 | 120.0 | 6.0 |" in report
    assert "| 5 Gbps | 0 | 80.0 | 8.0 |" in report
    assert "Throughput relative to the stage baseline:" in report
    assert "| stage 0 | 100.0% | 80.0% |" in report
    assert "| stage 1 | 100.0% | 50.0% |" in report


def test_generate_report_markdown_includes_fit_to_memory_tuning_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "name": "fit_memory_demo",
                "args": {
                    "bandwidth_mode": "socket",
                    "nproc_per_node": 4,
                    "steps": 8,
                    "metrics_warmup_steps": 2,
                },
                "results": [
                    {
                        "config": {
                            "stage": 0,
                            "model_size": "medium",
                            "bandwidth_gbps": 0.0,
                            "batch_size": 1,
                            "grad_accum_steps": 1,
                            "seq_len": 128,
                        },
                        "log_path": "logs/s0.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 500.0,
                        "mean_comm_ms": 40.0,
                        "return_code": 0,
                    },
                    {
                        "config": {
                            "stage": 0,
                            "model_size": "medium",
                            "bandwidth_gbps": 1.0,
                            "batch_size": 1,
                            "grad_accum_steps": 1,
                            "seq_len": 128,
                        },
                        "log_path": "logs/s0_bw1.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 250.0,
                        "mean_comm_ms": 80.0,
                        "return_code": 0,
                    },
                    {
                        "config": {
                            "stage": 3,
                            "model_size": "medium",
                            "bandwidth_gbps": 0.0,
                            "batch_size": 4,
                            "grad_accum_steps": 1,
                            "seq_len": 128,
                        },
                        "log_path": "logs/s3.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 900.0,
                        "mean_comm_ms": 60.0,
                        "return_code": 0,
                    },
                    {
                        "config": {
                            "stage": 3,
                            "model_size": "medium",
                            "bandwidth_gbps": 1.0,
                            "batch_size": 4,
                            "grad_accum_steps": 1,
                            "seq_len": 128,
                        },
                        "log_path": "logs/s3_bw1.log",
                        "profile_path": "",
                        "mean_tokens_per_s": 200.0,
                        "mean_comm_ms": 140.0,
                        "return_code": 0,
                    },
                ],
            }
        )
    )
    (run_dir / "tuning_summary.json").write_text(
        json.dumps(
            {
                "memory_budget_mb": 22000.0,
                "memory_metric": "peak_cuda_max_reserved_mb",
                "per_stage": {
                    "0": {
                        "selected_batch_size": 1,
                        "selected_peak_memory_mb": 21000.0,
                        "selected_global_tokens_per_step": 512,
                    },
                    "3": {
                        "selected_batch_size": 4,
                        "selected_peak_memory_mb": 20500.0,
                        "selected_global_tokens_per_step": 2048,
                    },
                },
            }
        )
    )

    report = bandwidth_report.generate_report_markdown(run_dir)

    assert "Experiment type: fit-to-memory bandwidth sweep" in report
    assert "fit-to-memory: each stage is first tuned" in report
    assert "selected cuda max reserved MB" in report
    assert "| 0 | 1 | 512 | 21000.0 | 95.5% |" in report
    assert "| 3 | 4 | 2048 | 20500.0 | 93.2% |" in report
