from experiments.run_pairwise_crossover_search import PairwisePoint, choose_next_finite_bandwidth


def _point(bandwidth_gbps: float, ratio_stage_b_over_stage_a: float) -> PairwisePoint:
    return PairwisePoint(
        bandwidth_gbps=bandwidth_gbps,
        stage_a_tokens_per_s=100.0,
        stage_b_tokens_per_s=100.0 * ratio_stage_b_over_stage_a,
        stage_a_comm_ms=0.0,
        stage_b_comm_ms=0.0,
        stage_a_fb_ms=0.0,
        stage_b_fb_ms=0.0,
        ratio_stage_b_over_stage_a=ratio_stage_b_over_stage_a,
        winner_stage=3 if ratio_stage_b_over_stage_a >= 1.0 else 2,
    )


def test_choose_next_bandwidth_prioritizes_sign_flip_interval() -> None:
    sampled = {
        0.01: _point(0.01, 1.5),
        0.1: _point(0.1, 0.8),
        0.5: _point(0.5, 0.6),
    }
    next_bandwidth = choose_next_finite_bandwidth(sampled, tolerance_ratio=1.25)
    assert next_bandwidth is not None
    assert next_bandwidth == (0.01 * 0.1) ** 0.5


def test_choose_next_bandwidth_targets_closest_to_parity_when_no_flip() -> None:
    sampled = {
        0.01: _point(0.01, 2.0),
        0.1: _point(0.1, 1.1),
        0.5: _point(0.5, 1.8),
    }
    next_bandwidth = choose_next_finite_bandwidth(sampled, tolerance_ratio=1.25)
    assert next_bandwidth is not None
    assert next_bandwidth == (0.1 * 0.5) ** 0.5


def test_choose_next_bandwidth_returns_none_when_interval_already_tight() -> None:
    sampled = {
        0.4: _point(0.4, 1.1),
        0.45: _point(0.45, 0.95),
    }
    assert choose_next_finite_bandwidth(sampled, tolerance_ratio=1.25) is None
