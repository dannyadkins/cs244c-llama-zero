from __future__ import annotations

import json
from pathlib import Path

from analysis import visualize


def test_parse_summary_week3_schema(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "config": {"stage": 3, "model_size": "tiny", "bandwidth_gbps": 5.0},
                "log_path": "experiments/results/run/logs/case.log",
                "mean_tokens_per_s": 123.0,
                "mean_comm_ms": 45.0,
                "mean_fb_ms": 67.0,
                "mean_opt_ms": 8.0,
                "peak_host_rss_mb": 321.0,
                "peak_cuda_allocated_mb": 111.0,
                "peak_cuda_reserved_mb": 222.0,
                "peak_cuda_max_allocated_mb": 333.0,
                "peak_cuda_max_reserved_mb": 444.0,
                "measured_state_memory_mb": {
                    "params_mb": 90.0,
                    "grads_mb": 45.0,
                    "optimizer_mb": 70.0,
                    "total_mb": 205.0,
                },
                "final_loss": 4.2,
                "return_code": 0,
            }
        ]
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(payload))

    cases = visualize.parse_summary(summary_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.stage == 3
    assert case.model_size == "tiny"
    assert case.bandwidth_gbps == 5.0
    assert case.mean_tokens_per_s == 123.0
    assert case.peak_host_rss_mb == 321.0
    assert case.peak_cuda_max_reserved_mb == 444.0
    assert case.measured_state_memory_mb == {
        "params_mb": 90.0,
        "grads_mb": 45.0,
        "optimizer_mb": 70.0,
        "total_mb": 205.0,
    }
    assert case.return_code == 0


def test_parse_summary_week2_compat_schema(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "stage": 2,
                "log_path": "experiments/results/run/stage2.log",
                "final_loss": 6.5,
                "return_code": 0,
            }
        ]
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(payload))

    cases = visualize.parse_summary(summary_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.stage == 2
    assert case.model_size == "unknown"
    assert case.bandwidth_gbps == 0.0


def test_parse_loss_log_handles_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.log"
    steps, losses = visualize.parse_loss_log(missing)
    assert steps == []
    assert losses == []


def test_case_peak_memory_prefers_host_when_cuda_is_zero() -> None:
    case = visualize.CaseView(
        stage=3,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        mean_tokens_per_s=None,
        mean_comm_ms=None,
        mean_fb_ms=None,
        mean_opt_ms=None,
        peak_host_rss_mb=512.0,
        peak_cuda_allocated_mb=0.0,
        peak_cuda_reserved_mb=0.0,
        peak_cuda_max_allocated_mb=0.0,
        peak_cuda_max_reserved_mb=0.0,
        final_loss=None,
        return_code=0,
        measured_state_memory_mb=None,
        theoretical_memory_mb=None,
    )

    assert visualize._case_peak_memory_mb(case) == 512.0


def test_representative_cases_prefer_unlimited_baseline() -> None:
    base = dict(
        model_size="tiny",
        log_path=Path("/tmp/missing.log"),
        mean_tokens_per_s=None,
        mean_comm_ms=None,
        mean_fb_ms=None,
        mean_opt_ms=None,
        peak_host_rss_mb=None,
        peak_cuda_allocated_mb=None,
        peak_cuda_reserved_mb=None,
        peak_cuda_max_allocated_mb=None,
        peak_cuda_max_reserved_mb=None,
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 1.0, "grads_mb": 1.0, "optimizer_mb": 1.0, "total_mb": 3.0},
        theoretical_memory_mb={"params_mb": 1.0, "grads_mb": 1.0, "optimizer_mb": 1.0, "total_mb": 3.0},
    )
    cases = [
        visualize.CaseView(stage=0, bandwidth_gbps=5.0, **base),
        visualize.CaseView(stage=0, bandwidth_gbps=0.0, **base),
        visualize.CaseView(stage=1, bandwidth_gbps=2.0, **base),
        visualize.CaseView(stage=1, bandwidth_gbps=5.0, **base),
    ]

    selected = visualize._representative_cases_by_stage(cases)

    assert [(case.stage, case.bandwidth_gbps) for case in selected] == [(0, 0.0), (1, 5.0)]
