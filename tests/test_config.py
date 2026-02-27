from model import build_medium_config, build_small_config, build_tiny_config, estimate_num_parameters


def test_presets_scale_monotonically() -> None:
    tiny = build_tiny_config()
    small = build_small_config()
    medium = build_medium_config()

    tiny_n = estimate_num_parameters(tiny)
    small_n = estimate_num_parameters(small)
    medium_n = estimate_num_parameters(medium)

    assert tiny_n < small_n < medium_n

    assert 40_000_000 <= tiny_n <= 60_000_000
    assert 250_000_000 <= small_n <= 400_000_000
    assert 1_100_000_000 <= medium_n <= 1_500_000_000
