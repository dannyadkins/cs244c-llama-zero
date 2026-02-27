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
