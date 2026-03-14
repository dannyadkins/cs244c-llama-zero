from __future__ import annotations

from experiments import run_fit_memory_bandwidth


def _trial(batch_size: int, fits: bool) -> run_fit_memory_bandwidth.TuningTrial:
    return run_fit_memory_bandwidth.TuningTrial(
        batch_size=batch_size,
        fits=fits,
        peak_memory_mb=None if not fits else float(batch_size * 100),
        return_code=0 if fits else 1,
        reason="ok" if fits else "over_budget",
        case_id=f"case_{batch_size}",
        log_path=f"log_{batch_size}.txt",
        profile_path=f"profile_{batch_size}.json",
        mean_tokens_per_s=None,
    )


def test_select_max_batch_size_finds_largest_fitting_batch() -> None:
    seen: list[int] = []

    def evaluator(batch_size: int) -> run_fit_memory_bandwidth.TuningTrial:
        seen.append(batch_size)
        return _trial(batch_size=batch_size, fits=batch_size <= 10)

    best, trials = run_fit_memory_bandwidth._select_max_batch_size(
        min_batch_size=1,
        max_batch_size=16,
        batch_size_multiple=1,
        growth_factor=2.0,
        initial_batch_size=0,
        evaluator=evaluator,
    )

    assert best.batch_size == 10
    assert [trial.batch_size for trial in trials] == [8, 10, 11, 12, 16]
    assert seen == [8, 16, 12, 10, 11]


def test_select_max_batch_size_respects_batch_size_multiple() -> None:
    def evaluator(batch_size: int) -> run_fit_memory_bandwidth.TuningTrial:
        return _trial(batch_size=batch_size, fits=batch_size <= 18)

    best, trials = run_fit_memory_bandwidth._select_max_batch_size(
        min_batch_size=4,
        max_batch_size=32,
        batch_size_multiple=4,
        growth_factor=2.0,
        initial_batch_size=0,
        evaluator=evaluator,
    )

    assert best.batch_size == 16
    assert [trial.batch_size for trial in trials] == [16, 20, 24, 32]


def test_select_max_batch_size_can_recover_from_high_initial_guess() -> None:
    seen: list[int] = []

    def evaluator(batch_size: int) -> run_fit_memory_bandwidth.TuningTrial:
        seen.append(batch_size)
        return _trial(batch_size=batch_size, fits=batch_size <= 84)

    best, trials = run_fit_memory_bandwidth._select_max_batch_size(
        min_batch_size=4,
        max_batch_size=512,
        batch_size_multiple=4,
        growth_factor=2.0,
        initial_batch_size=256,
        evaluator=evaluator,
    )

    assert best.batch_size == 84
    assert seen == [256, 128, 64, 96, 80, 88, 84]
    assert [trial.batch_size for trial in trials] == [64, 80, 84, 88, 96, 128, 256]
