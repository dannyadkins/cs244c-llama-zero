from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
        nnodes=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port_base=29500,
        case_timeout_s=1800.0,
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
        metrics_warmup_steps=0,
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
            "[step 00001] loss=8.1000 avg100=8.1000 tokens/s=1,200 grad_norm=0.900 fb_ms=10.0 comm_ms=5.0 opt_ms=2.0 tflops=0.500",
            "[step 00002] loss=8.0000 avg100=8.0500 tokens/s=800 grad_norm=1.100 fb_ms=20.0 comm_ms=15.0 opt_ms=4.0 tflops=0.250",
        ]
    )
    metrics = harness._parse_step_metrics(log_text)

    assert metrics["final_loss"] == 8.0
    assert metrics["logged_steps"] == 2
    assert metrics["mean_tokens_per_s"] == 1000.0
    assert metrics["mean_tflops_per_s"] == 0.375
    assert metrics["mean_fb_ms"] == 15.0
    assert metrics["mean_comm_ms"] == 10.0
    assert metrics["mean_opt_ms"] == 3.0


def test_parse_step_metrics_can_skip_warmup_steps() -> None:
    log_text = "\n".join(
        [
            "[step 00001] loss=8.3000 avg100=8.3000 tokens/s=300 grad_norm=0.900 fb_ms=30.0 comm_ms=10.0 opt_ms=3.0 tflops=0.100",
            "[step 00002] loss=8.1000 avg100=8.2000 tokens/s=900 grad_norm=1.100 fb_ms=12.0 comm_ms=4.0 opt_ms=2.0 tflops=0.400",
            "[step 00003] loss=8.0000 avg100=8.1333 tokens/s=1,200 grad_norm=1.000 fb_ms=9.0 comm_ms=3.0 opt_ms=1.0 tflops=0.500",
        ]
    )

    metrics = harness._parse_step_metrics(log_text, metrics_warmup_steps=1)

    assert metrics["final_loss"] == 8.0
    assert metrics["logged_steps"] == 3
    assert metrics["mean_tokens_per_s"] == 1050.0
    assert metrics["mean_tflops_per_s"] == 0.45
    assert metrics["mean_fb_ms"] == 10.5
    assert metrics["mean_comm_ms"] == 3.5
    assert metrics["mean_opt_ms"] == 1.5


def test_parse_profile_memory_extracts_peaks(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {
                        "label": "start",
                        "host_maxrss_mb": 100.0,
                        "cuda_allocated_mb": 10.0,
                        "cuda_reserved_mb": 20.0,
                        "cuda_max_allocated_mb": 30.0,
                        "cuda_max_reserved_mb": 40.0,
                    },
                    {
                        "label": "end",
                        "host_maxrss_mb": 150.0,
                        "cuda_allocated_mb": 12.0,
                        "cuda_reserved_mb": 18.0,
                        "cuda_max_allocated_mb": 35.0,
                        "cuda_max_reserved_mb": 45.0,
                    },
                ]
            }
        )
    )

    metrics = harness._parse_profile_memory(profile_path)

    assert metrics == {
        "peak_host_rss_mb": 150.0,
        "peak_cuda_allocated_mb": 12.0,
        "peak_cuda_reserved_mb": 20.0,
        "peak_cuda_max_allocated_mb": 35.0,
        "peak_cuda_max_reserved_mb": 45.0,
        "measured_state_memory_mb": None,
    }


def test_parse_profile_memory_prefers_measured_step_peaks(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "memory": [
                    {
                        "label": "step_1_end",
                        "host_maxrss_mb": 150.0,
                        "cuda_allocated_mb": 12.0,
                        "cuda_reserved_mb": 20.0,
                        "cuda_max_allocated_mb": 35.0,
                        "cuda_max_reserved_mb": 45.0,
                    }
                ],
                "measured_step_memory": {
                    "step": 4,
                    "peak_allocated_mb": 77.0,
                    "peak_reserved_mb": 88.0,
                },
            }
        )
    )

    metrics = harness._parse_profile_memory(profile_path)

    assert metrics == {
        "peak_host_rss_mb": 150.0,
        "peak_cuda_allocated_mb": 77.0,
        "peak_cuda_reserved_mb": 88.0,
        "peak_cuda_max_allocated_mb": 77.0,
        "peak_cuda_max_reserved_mb": 88.0,
        "measured_state_memory_mb": None,
    }


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


def test_parse_args_uses_config_as_defaults_but_preserves_explicit_cli(tmp_path: Path) -> None:
    cfg = {
        "defaults": {
            "name": "from_config",
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

    parsed = harness.parse_args(
        [
            "--config",
            str(cfg_path),
            "--name",
            "from_cli",
            "--steps",
            "5",
            "--bandwidth-gbps",
            "5.0",
        ]
    )

    assert parsed.name == "from_cli"
    assert parsed.steps == 5
    assert parsed.seq_len == 256
    assert parsed.batch_size == 2
    assert parsed.dtype == "bfloat16"
    assert parsed.stages == [0, 1, 2, 3]
    assert parsed.model_sizes == ["medium"]
    assert parsed.bandwidth_gbps == [5.0]


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


def test_build_train_zero_cmd_uses_active_python_and_explicit_master() -> None:
    args = _base_args()
    case = harness._build_cases(args)[0]
    launch = harness._launch_config_from_args(args)

    cmd = harness._build_train_zero_cmd(
        case=case,
        profile_path=Path("/tmp/profile.json"),
        launch=launch,
        case_index=3,
    )

    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--master_addr" in cmd
    assert cmd[cmd.index("--master_addr") + 1] == "127.0.0.1"
    assert "--master_port" in cmd
    assert cmd[cmd.index("--master_port") + 1] == "29503"
    assert str(harness.PROJECT_ROOT / "train_zero.py") in cmd


def test_build_launch_env_sets_simulated_bandwidth(monkeypatch) -> None:
    args = _base_args()
    case = harness._build_cases(args)[-1]
    monkeypatch.setenv("ZERO_SIM_BW_GBPS", "999")
    monkeypatch.setenv("ZERO_SIM_LATENCY_MS", "999")

    env = harness._build_launch_env(case)

    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["ZERO_SIM_BW_GBPS"] == "5.0"
    assert "ZERO_SIM_LATENCY_MS" not in env

    case.bandwidth_mode = "none"
    env = harness._build_launch_env(case)
    assert "ZERO_SIM_BW_GBPS" not in env
    assert "ZERO_SIM_LATENCY_MS" not in env


def test_run_case_timeout_records_failure(tmp_path: Path, monkeypatch) -> None:
    args = _base_args()
    case = harness._build_cases(args)[0]
    launch = harness._launch_config_from_args(args)

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python", "-m", "torch.distributed.run"],
            timeout=launch.case_timeout_s,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    result = harness._run_case(
        case=case,
        run_dir=tmp_path,
        skip_existing=False,
        dry_run=False,
        launch=launch,
        case_index=0,
    )

    assert result.return_code == 124
    assert result.notes == "timeout_after_s=1800.0"
    assert "timeout" in Path(result.log_path).read_text()
    assert Path(result.profile_path).name.endswith(".json")
    assert Path(tmp_path / "cases" / f"{result.case_id}.json").exists()


def test_run_case_records_profile_memory_metrics(tmp_path: Path, monkeypatch) -> None:
    args = _base_args()
    args.profile_memory_interval = 1
    case = harness._build_cases(args)[0]
    launch = harness._launch_config_from_args(args)

    def fake_run(cmd, capture_output, text, env, cwd, timeout):
        profile_path = Path(cmd[cmd.index("--profile-json") + 1])
        profile_path.write_text(
            json.dumps(
                {
                    "memory": [
                        {
                            "label": "step0",
                            "host_maxrss_mb": 200.0,
                            "cuda_allocated_mb": 1.0,
                            "cuda_reserved_mb": 2.0,
                            "cuda_max_allocated_mb": 3.0,
                            "cuda_max_reserved_mb": 4.0,
                        },
                        {
                            "label": "step1",
                            "host_maxrss_mb": 250.0,
                            "cuda_allocated_mb": 1.5,
                            "cuda_reserved_mb": 2.5,
                            "cuda_max_allocated_mb": 3.5,
                            "cuda_max_reserved_mb": 4.5,
                        },
                    ]
                }
            )
        )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="[step 00001] loss=8.1000 avg100=8.1000 tokens/s=1,200 grad_norm=0.900 fb_ms=10.0 comm_ms=5.0 opt_ms=2.0 tflops=0.750\n",
            stderr="",
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    result = harness._run_case(
        case=case,
        run_dir=tmp_path,
        skip_existing=False,
        dry_run=False,
        launch=launch,
        case_index=0,
    )

    assert result.return_code == 0
    assert result.peak_host_rss_mb == 250.0
    assert result.mean_tflops_per_s == 0.75
    assert result.peak_cuda_allocated_mb == 1.5
    assert result.peak_cuda_reserved_mb == 2.5
    assert result.peak_cuda_max_allocated_mb == 3.5
    assert result.peak_cuda_max_reserved_mb == 4.5
