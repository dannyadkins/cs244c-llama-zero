from __future__ import annotations

from argparse import Namespace

from experiments import run_stage3_scaling_sweep
from experiments.run_fit_memory_bandwidth import TuningTrial


def _trial(batch_size: int, peak_memory_mb: float) -> TuningTrial:
    return TuningTrial(
        batch_size=batch_size,
        fits=True,
        peak_memory_mb=peak_memory_mb,
        return_code=0,
        reason="ok",
        case_id=f"case_{batch_size}_{int(peak_memory_mb)}",
        log_path=f"log_{batch_size}.txt",
        profile_path=f"profile_{batch_size}.json",
        mean_tokens_per_s=None,
    )


def _args(*, batch_size_multiple: int = 1, max_batch_size: int = 128) -> Namespace:
    return Namespace(
        min_batch_size=1,
        max_batch_size=max_batch_size,
        batch_size_multiple=batch_size_multiple,
        initial_batch_size=0,
        tuning_strategy="predictive",
    )


def test_predict_initial_batch_size_falls_back_to_shard_ratio() -> None:
    predicted, payload = run_stage3_scaling_sweep._predict_initial_batch_size(
        _args(batch_size_multiple=4),
        gpu_count=4,
        previous_selected_batch_size=20,
        previous_gpu_count=2,
        selected_trials_by_gpu_count={2: _trial(20, 22_000.0)},
        trial_history_by_gpu_count={2: [_trial(20, 22_000.0)]},
    )

    assert predicted == 40
    assert payload["kind"] == "shard_ratio"


def test_predict_initial_batch_size_uses_memory_model_when_available() -> None:
    args = _args(batch_size_multiple=2, max_batch_size=128)
    selected_trials_by_gpu_count = {
        1: _trial(50, 10_000.0),
        2: _trial(70, 10_000.0),
    }
    trial_history_by_gpu_count = {
        1: [_trial(20, 7_000.0), _trial(40, 9_000.0), selected_trials_by_gpu_count[1]],
        2: [_trial(40, 7_000.0), _trial(60, 9_000.0), selected_trials_by_gpu_count[2]],
    }

    predicted, payload = run_stage3_scaling_sweep._predict_initial_batch_size(
        args,
        gpu_count=4,
        previous_selected_batch_size=70,
        previous_gpu_count=2,
        selected_trials_by_gpu_count=selected_trials_by_gpu_count,
        trial_history_by_gpu_count=trial_history_by_gpu_count,
    )

    assert predicted == 80
    assert payload["kind"] == "memory_model"
    assert payload["budget_mb"] == 10_000.0


def test_predict_initial_batch_size_prefers_shard_ratio_until_two_gpu_counts_exist() -> None:
    args = _args(batch_size_multiple=1, max_batch_size=128)
    selected_trials_by_gpu_count = {1: _trial(19, 22_000.0)}
    trial_history_by_gpu_count = {
        1: [_trial(16, 20_800.0), _trial(18, 21_600.0), selected_trials_by_gpu_count[1]],
    }

    predicted, payload = run_stage3_scaling_sweep._predict_initial_batch_size(
        args,
        gpu_count=2,
        previous_selected_batch_size=19,
        previous_gpu_count=1,
        selected_trials_by_gpu_count=selected_trials_by_gpu_count,
        trial_history_by_gpu_count=trial_history_by_gpu_count,
    )

    assert predicted == 38
    assert payload["kind"] == "shard_ratio"
