from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments import harness


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        config="",
        name="unit",
        results_dir="experiments/results",
        skip_existing=False,
        dry_run=False,
        stages=[0, 1],
        model_sizes=["tiny", "small"],
        bandwidth_gbps=[0.0, 5.0],
        nproc_per_node=2,
        steps=10,
        seq_len=128,
        batch_size=4,
        grad_accum_steps=1,
        collective_impl="ring",
        data_mode="synthetic",
        seed=1337,
        dtype="float32",
        max_grad_norm=0.0,
        profile_memory_interval=0,
        bandwidth_mode="simulated",
        sim_latency_ms=0.0,
        tc_interface="eth0",
        theory_vocab_size=0,
        extra_args="",
    )


def test_build_cases_cartesian_product() -> None:
    args = _base_args()
    cases = harness._build_cases(args)

    assert len(cases) == 8
    ids = {harness._case_id(case) for case in cases}
    assert len(ids) == 8
    assert any(case_id.startswith("s0_") for case_id in ids)
    assert any(case_id.startswith("s1_") for case_id in ids)


def test_parse_step_metrics_extracts_means() -> None:
    log_text = "\n".join(
        [
            "[step 00001] loss=8.1000 avg100=8.1000 tokens/s=1,200 grad_norm=0.900 fb_ms=10.0 comm_ms=5.0 opt_ms=2.0",
            "[step 00002] loss=8.0000 avg100=8.0500 tokens/s=800 grad_norm=1.100 fb_ms=20.0 comm_ms=15.0 opt_ms=4.0",
        ]
    )
    metrics = harness._parse_step_metrics(log_text)

    assert metrics["final_loss"] == 8.0
    assert metrics["logged_steps"] == 2
    assert metrics["mean_tokens_per_s"] == 1000.0
    assert metrics["mean_fb_ms"] == 15.0
    assert metrics["mean_comm_ms"] == 10.0
    assert metrics["mean_opt_ms"] == 3.0


def test_merge_config_file_overrides_defaults_and_matrix(tmp_path: Path) -> None:
    cfg = {
        "defaults": {
            "steps": 99,
            "seq_len": 256,
            "batch_size": 2,
            "dtype": "bfloat16",
        },
        "matrix": {
            "stages": [0, 1, 2, 3],
            "model_sizes": ["medium"],
            "bandwidth_gbps": [0.0, 1.0],
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    args = _base_args()
    args.config = str(cfg_path)
    merged = harness._merge_config_file(args)

    assert merged.steps == 99
    assert merged.seq_len == 256
    assert merged.batch_size == 2
    assert merged.dtype == "bfloat16"
    assert merged.stages == [0, 1, 2, 3]
    assert merged.model_sizes == ["medium"]
    assert merged.bandwidth_gbps == [0.0, 1.0]


def test_theoretical_memory_monotonic_for_stages() -> None:
    args = _base_args()
    args.stages = [0, 1, 2, 3]
    args.model_sizes = ["tiny"]
    args.bandwidth_gbps = [0.0]
    cases = sorted(harness._build_cases(args), key=lambda c: c.stage)

    totals = []
    for case in cases:
        _num_params, breakdown = harness._theoretical_memory(case)
        totals.append(breakdown["total_mb"])

    assert totals[0] > totals[1] > totals[2] > totals[3]
