from analysis.stage3_scaling import render_report, scaling_points


def test_scaling_points_cast_and_sort() -> None:
    summary = {
        "scaling_points": [
            {
                "gpu_count": 8,
                "selected_batch_size": 6,
                "selected_peak_memory_mb": 20800.0,
                "global_tokens_per_step": 12288,
                "mean_tokens_per_s": 20000.0,
                "mean_tflops_per_s": 80.0,
                "per_gpu_tflops_per_s": 10.0,
                "perfect_linear_tflops_per_s": 64.0,
                "speedup_vs_base": 4.0,
                "scaling_efficiency_vs_base": 1.0,
                "superlinear_gain_vs_perfect_linear": 1.25,
                "peak_cuda_max_allocated_mb": 21000.0,
            },
            {
                "gpu_count": 2,
                "selected_batch_size": 2,
                "global_tokens_per_step": 1024,
                "mean_tokens_per_s": 5000.0,
                "mean_tflops_per_s": 16.0,
                "per_gpu_tflops_per_s": 8.0,
                "perfect_linear_tflops_per_s": 16.0,
                "speedup_vs_base": 1.0,
                "scaling_efficiency_vs_base": 1.0,
                "superlinear_gain_vs_perfect_linear": 1.0,
                "peak_cuda_max_allocated_mb": 22000.0,
            },
        ]
    }

    points = scaling_points(summary)

    assert [point["gpu_count"] for point in points] == [2, 8]
    assert points[0]["selected_batch_size"] == 2
    assert points[1]["per_gpu_tflops_per_s"] == 10.0
    assert points[1]["peak_memory_mb"] == 20800.0


def test_render_report_mentions_skipped_counts_and_reference() -> None:
    summary = {
        "args": {
            "model_size": "medium",
            "seq_len": 256,
            "dtype": "bfloat16",
            "tflops_mode": "profile",
            "steps": 10,
        },
        "per_gpu_count": {
            "1": {"fit_status": "no_fit"},
            "2": {"fit_status": "fit"},
            "4": {"fit_status": "fit"},
        },
    }
    points = [
        {
            "gpu_count": 2,
            "selected_batch_size": 1,
            "global_tokens_per_step": 512,
            "mean_tokens_per_s": 4000.0,
            "mean_tflops_per_s": 14.0,
            "per_gpu_tflops_per_s": 7.0,
            "perfect_linear_tflops_per_s": 14.0,
            "speedup_vs_base": 1.0,
            "scaling_efficiency_vs_base": 1.0,
            "superlinear_gain_vs_perfect_linear": 1.0,
            "peak_memory_mb": 23000.0,
            "peak_cuda_max_allocated_mb": 23000.0,
        },
        {
            "gpu_count": 4,
            "selected_batch_size": 2,
            "global_tokens_per_step": 2048,
            "mean_tokens_per_s": 10000.0,
            "mean_tflops_per_s": 36.0,
            "per_gpu_tflops_per_s": 9.0,
            "perfect_linear_tflops_per_s": 28.0,
            "speedup_vs_base": 2.57,
            "scaling_efficiency_vs_base": 1.286,
            "superlinear_gain_vs_perfect_linear": 1.286,
            "peak_memory_mb": 18000.0,
            "peak_cuda_max_allocated_mb": 18000.0,
        },
    ]

    report = render_report(summary, points, baseline_gpu_count=2)

    assert "1" in report
    assert "ZeRO paper" in report
    assert "Strongest superlinear gain" in report
