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
                "mean_tflops_per_s": 0.456,
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
    assert case.profile_path is None
    assert case.mean_tokens_per_s == 123.0
    assert case.mean_tflops_per_s == 0.456
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
    assert case.profile_path is None


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
        profile_path=None,
        mean_tokens_per_s=None,
        mean_tflops_per_s=None,
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


def test_case_peak_memory_prefers_true_cuda_peak() -> None:
    case = visualize.CaseView(
        stage=0,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=None,
        mean_tokens_per_s=None,
        mean_tflops_per_s=None,
        mean_comm_ms=None,
        mean_fb_ms=None,
        mean_opt_ms=None,
        peak_host_rss_mb=256.0,
        peak_cuda_allocated_mb=111.0,
        peak_cuda_reserved_mb=222.0,
        peak_cuda_max_allocated_mb=333.0,
        peak_cuda_max_reserved_mb=444.0,
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 50.0, "grads_mb": 25.0, "optimizer_mb": 75.0, "total_mb": 150.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_peak_memory_mb(case) == 333.0


def test_case_peak_breakdown_uses_measured_state_total() -> None:
    case = visualize.CaseView(
        stage=0,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=None,
        mean_tokens_per_s=None,
        mean_tflops_per_s=None,
        mean_comm_ms=None,
        mean_fb_ms=None,
        mean_opt_ms=None,
        peak_host_rss_mb=None,
        peak_cuda_allocated_mb=111.0,
        peak_cuda_reserved_mb=222.0,
        peak_cuda_max_allocated_mb=333.0,
        peak_cuda_max_reserved_mb=444.0,
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 50.0, "grads_mb": 25.0, "optimizer_mb": 75.0, "total_mb": 150.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_peak_breakdown_mb(case) == (150.0, 183.0)


def test_case_peak_breakdown_prefers_logical_state_over_live_peak_state(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {"label": "measured_step_1_peak", "cuda_allocated_mb": 210.0, "cuda_max_allocated_mb": 260.0},
                ],
                "measured_step_state_timeline": [
                    {"label": "measured_step_1_peak", "params_mb": 60.0, "grads_mb": 80.0, "optimizer_mb": 40.0, "total_mb": 180.0},
                ],
            }
        )
    )

    case = visualize.CaseView(
        stage=2,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
        mean_tokens_per_s=None,
        mean_tflops_per_s=None,
        mean_comm_ms=None,
        mean_fb_ms=None,
        mean_opt_ms=None,
        peak_host_rss_mb=None,
        peak_cuda_allocated_mb=210.0,
        peak_cuda_reserved_mb=None,
        peak_cuda_max_allocated_mb=260.0,
        peak_cuda_max_reserved_mb=None,
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 60.0, "grads_mb": 20.0, "optimizer_mb": 40.0, "total_mb": 120.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_peak_breakdown_mb(case) == (120.0, 140.0)


def test_representative_cases_prefer_unlimited_baseline() -> None:
    base = dict(
        model_size="tiny",
        log_path=Path("/tmp/missing.log"),
        profile_path=None,
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


def test_case_average_live_breakdown_uses_profile_snapshots(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {"label": "start", "cuda_allocated_mb": 80.0},
                    {"label": "step_1_start", "cuda_allocated_mb": 200.0},
                    {"label": "step_1_post_backward", "cuda_allocated_mb": 260.0},
                    {"label": "step_1_end", "cuda_allocated_mb": 220.0},
                ]
            }
        )
    )

    case = visualize.CaseView(
        stage=2,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
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
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 50.0, "grads_mb": 20.0, "optimizer_mb": 30.0, "total_mb": 100.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_average_live_breakdown_mb(case) == (100.0, 126.66666666666666)


def test_case_average_live_breakdown_prefers_measured_step_state_timeline(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {"label": "measured_step_1_start", "cuda_allocated_mb": 10.0},
                    {"label": "measured_step_1_after_forward", "cuda_allocated_mb": 20.0},
                    {"label": "measured_step_1_end", "cuda_allocated_mb": 12.0},
                ],
                "measured_step_state_timeline": [
                    {"label": "measured_step_1_start", "total_mb": 8.0},
                    {"label": "measured_step_1_after_forward", "total_mb": 8.0},
                    {"label": "measured_step_1_end", "total_mb": 8.0},
                ],
            }
        )
    )

    case = visualize.CaseView(
        stage=2,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
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
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 5.0, "grads_mb": 5.0, "optimizer_mb": 5.0, "total_mb": 15.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_average_live_breakdown_mb(case) == (8.0, 6.0)


def test_case_average_live_breakdown_uses_timestamp_weighting(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {"label": "measured_step_1_start", "timestamp_s": 0.0, "cuda_allocated_mb": 10.0},
                    {"label": "measured_step_1_mid", "timestamp_s": 9.0, "cuda_allocated_mb": 30.0},
                    {"label": "measured_step_1_end", "timestamp_s": 10.0, "cuda_allocated_mb": 10.0},
                ],
                "measured_step_state_timeline": [
                    {"label": "measured_step_1_start", "total_mb": 8.0},
                    {"label": "measured_step_1_mid", "total_mb": 8.0},
                    {"label": "measured_step_1_end", "total_mb": 8.0},
                ],
            }
        )
    )

    case = visualize.CaseView(
        stage=2,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
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
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 5.0, "grads_mb": 5.0, "optimizer_mb": 5.0, "total_mb": 15.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_average_live_breakdown_mb(case) == (8.0, 4.0)


def test_case_peak_state_breakdown_prefers_timeline_max(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "measured_step_state_timeline": [
                    {"label": "a", "params_mb": 10.0, "grads_mb": 0.0, "optimizer_mb": 5.0, "total_mb": 15.0},
                    {"label": "b", "params_mb": 10.0, "grads_mb": 4.0, "optimizer_mb": 5.0, "total_mb": 19.0},
                ]
            }
        )
    )

    case = visualize.CaseView(
        stage=2,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
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
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 1.0, "grads_mb": 1.0, "optimizer_mb": 1.0, "total_mb": 3.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_peak_state_breakdown(case) == {
        "params_mb": 10.0,
        "grads_mb": 4.0,
        "optimizer_mb": 5.0,
        "total_mb": 19.0,
    }


def test_case_state_breakdown_at_peak_total_prefers_peak_snapshot_label(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {"label": "a", "cuda_allocated_mb": 50.0, "cuda_max_allocated_mb": 80.0},
                    {"label": "b", "cuda_allocated_mb": 60.0, "cuda_max_allocated_mb": 90.0},
                ],
                "measured_step_state_timeline": [
                    {"label": "a", "params_mb": 10.0, "grads_mb": 6.0, "optimizer_mb": 5.0, "total_mb": 21.0},
                    {"label": "b", "params_mb": 10.0, "grads_mb": 4.0, "optimizer_mb": 5.0, "total_mb": 19.0},
                ],
            }
        )
    )

    case = visualize.CaseView(
        stage=2,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
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
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 1.0, "grads_mb": 1.0, "optimizer_mb": 1.0, "total_mb": 3.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_state_breakdown_at_peak_total(case) == {
        "params_mb": 10.0,
        "grads_mb": 4.0,
        "optimizer_mb": 5.0,
        "total_mb": 19.0,
    }


def test_case_post_backward_state_breakdown_prefers_post_backward_label(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "measured_step_state_timeline": [
                    {"label": "measured_step_1_start", "params_mb": 10.0, "grads_mb": 0.0, "optimizer_mb": 5.0, "total_mb": 15.0},
                    {"label": "measured_step_1_post_backward", "params_mb": 10.0, "grads_mb": 2.0, "optimizer_mb": 5.0, "total_mb": 17.0},
                    {"label": "measured_step_1_post_optimizer", "params_mb": 10.0, "grads_mb": 0.0, "optimizer_mb": 5.0, "total_mb": 15.0},
                ]
            }
        )
    )

    case = visualize.CaseView(
        stage=2,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
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
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 1.0, "grads_mb": 1.0, "optimizer_mb": 1.0, "total_mb": 3.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_post_backward_state_breakdown(case) == {
        "params_mb": 10.0,
        "grads_mb": 2.0,
        "optimizer_mb": 5.0,
        "total_mb": 17.0,
    }


def test_case_peak_breakdown_backfills_legacy_stage3_local_shards(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {
                        "label": "measured_step_1_start",
                        "cuda_allocated_mb": 90.0,
                        "cuda_max_allocated_mb": 90.0,
                    },
                    {
                        "label": "measured_step_1_peak",
                        "cuda_allocated_mb": 130.0,
                        "cuda_max_allocated_mb": 150.0,
                    },
                ],
                "measured_step_state_timeline": [
                    {"label": "measured_step_1_start", "total_mb": 60.0},
                    {"label": "measured_step_1_peak", "total_mb": 60.0},
                ],
            }
        )
    )

    case = visualize.CaseView(
        stage=3,
        model_size="tiny",
        bandwidth_gbps=0.0,
        log_path=Path("/tmp/missing.log"),
        profile_path=profile_path,
        mean_tokens_per_s=None,
        mean_tflops_per_s=None,
        mean_comm_ms=None,
        mean_fb_ms=None,
        mean_opt_ms=None,
        peak_host_rss_mb=None,
        peak_cuda_allocated_mb=150.0,
        peak_cuda_reserved_mb=None,
        peak_cuda_max_allocated_mb=150.0,
        peak_cuda_max_reserved_mb=None,
        final_loss=None,
        return_code=0,
        measured_state_memory_mb={"params_mb": 20.0, "grads_mb": 20.0, "optimizer_mb": 40.0, "total_mb": 80.0},
        theoretical_memory_mb=None,
    )

    assert visualize._case_peak_breakdown_mb(case) == (80.0, 70.0)
