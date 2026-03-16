from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "remote_4gpu_small_pairwise_crossover_socket.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ship the current repo to a clean remote workspace and run a pairwise stage crossover search"
    )
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--user", type=str, default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--ssh-option", action="append", default=[], help="Repeated raw ssh -o options")

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--name-override", type=str, default="")
    parser.add_argument("--extra-runner-args", type=str, default="")

    parser.add_argument("--remote-base-dir", type=str, default="/workspace/codex-runs")
    parser.add_argument("--remote-python", type=str, default="python3")
    parser.add_argument("--keep-remote-workdir", action="store_true")

    parser.add_argument("--local-results-dir", type=str, default="experiments/results/remote")
    parser.add_argument("--overwrite-local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "run"


def _remote_target(args: argparse.Namespace) -> str:
    return f"{args.user}@{args.host}"


def _ssh_base(args: argparse.Namespace) -> List[str]:
    base = ["ssh", "-p", str(args.port)]
    for option in args.ssh_option:
        base.extend(["-o", option])
    base.append(_remote_target(args))
    return base


def _scp_base(args: argparse.Namespace) -> List[str]:
    base = ["scp", "-P", str(args.port)]
    for option in args.ssh_option:
        base.extend(["-o", option])
    return base


def _run_local(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def _run_ssh(args: argparse.Namespace, remote_cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _ssh_base(args) + [remote_cmd],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def _stream_ssh(args: argparse.Namespace, remote_cmd: str) -> None:
    process = subprocess.Popen(
        _ssh_base(args) + [remote_cmd],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="")
    finally:
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"remote command failed with exit code {return_code}: {remote_cmd}")


def _project_relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path must live under the repo root: {path}") from exc


def _load_config_defaults(config_path: Path) -> dict[str, object]:
    payload = json.loads(config_path.read_text())
    defaults = payload.get("defaults", {})
    return defaults if isinstance(defaults, dict) else {}


def _create_bundle() -> Path:
    proc = _run_local(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    rel_paths = [item for item in proc.stdout.split("\0") if item]
    if not rel_paths:
        raise RuntimeError("bundle would be empty")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        bundle_path = Path(tmp_file.name)

    with tarfile.open(bundle_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for rel_path in rel_paths:
            src = PROJECT_ROOT / rel_path
            if not src.exists():
                continue
            archive.add(src, arcname=rel_path, recursive=True)
    return bundle_path


def _create_remote_workspace(args: argparse.Namespace, run_name: str) -> str:
    prefix = _sanitize_token(run_name)
    remote_cmd = (
        f"mkdir -p {shlex.quote(args.remote_base_dir)} && "
        f"mktemp -d {shlex.quote(args.remote_base_dir.rstrip('/'))}/{shlex.quote(prefix)}.XXXXXX"
    )
    proc = _run_ssh(args, remote_cmd)
    return proc.stdout.strip()


def _upload_bundle(args: argparse.Namespace, bundle_path: Path, remote_bundle_path: str) -> None:
    subprocess.run(
        _scp_base(args) + [str(bundle_path), f"{_remote_target(args)}:{remote_bundle_path}"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _pull_run_dir(args: argparse.Namespace, remote_run_dir: str, local_parent: Path) -> Path:
    local_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tmp_tar_path = Path(tmp_file.name)
    remote_archive_path = f"{remote_run_dir.rstrip('/')}.tar.gz"

    try:
        _run_ssh(
            args,
            f"tar -C {shlex.quote(remote_run_dir)} -czf {shlex.quote(remote_archive_path)} .",
        )
        subprocess.run(
            _scp_base(args) + [f"{_remote_target(args)}:{remote_archive_path}", str(tmp_tar_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
        shutil.unpack_archive(str(tmp_tar_path), str(local_parent))
    finally:
        subprocess.run(
            _ssh_base(args) + [f"rm -f {shlex.quote(remote_archive_path)}"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        tmp_tar_path.unlink(missing_ok=True)

    return local_parent


def _write_metadata(local_run_dir: Path, payload: dict[str, object]) -> None:
    (local_run_dir / "remote_run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")

    config_defaults = _load_config_defaults(config_path)
    default_name = str(config_defaults.get("name", config_path.stem))
    default_results_dir = str(config_defaults.get("results_dir", "experiments/results"))
    run_name = _sanitize_token(args.name_override or default_name)
    host_dir = _sanitize_token(args.host)
    local_run_dir = (PROJECT_ROOT / args.local_results_dir / host_dir / run_name).resolve()
    if local_run_dir.exists():
        if not args.overwrite_local:
            raise FileExistsError(f"local run dir already exists: {local_run_dir}")
        shutil.rmtree(local_run_dir)

    config_rel = _project_relative(config_path)
    bundle_path = _create_bundle()
    remote_workspace = ""
    remote_run_dir = ""
    start_s = time.time()

    try:
        remote_workspace = _create_remote_workspace(args, run_name)
        remote_bundle_path = f"{remote_workspace}/repo_bundle.tar.gz"
        _upload_bundle(args, bundle_path, remote_bundle_path)
        _stream_ssh(
            args,
            (
                f"mkdir -p {shlex.quote(remote_workspace)} && "
                f"tar -xzf {shlex.quote(remote_bundle_path)} -C {shlex.quote(remote_workspace)} && "
                f"rm -f {shlex.quote(remote_bundle_path)}"
            ),
        )
        _stream_ssh(
            args,
            (
                f"cd {shlex.quote(remote_workspace)} && "
                f"{shlex.quote(args.remote_python)} -c "
                f"\"import torch, matplotlib; print('torch', torch.__version__, 'cuda', torch.cuda.device_count())\""
            ),
        )

        remote_cmd = [
            args.remote_python,
            "experiments/run_pairwise_crossover_search.py",
            "--config",
            str(config_rel),
            "--name",
            run_name,
        ]
        if args.extra_runner_args.strip():
            remote_cmd.extend(shlex.split(args.extra_runner_args))
        if args.dry_run:
            remote_cmd.append("--dry-run")

        remote_run_dir = f"{remote_workspace}/{default_results_dir.rstrip('/')}/{run_name}"
        _stream_ssh(
            args,
            f"cd {shlex.quote(remote_workspace)} && " + " ".join(shlex.quote(part) for part in remote_cmd),
        )

        _pull_run_dir(args, remote_run_dir, local_run_dir)
        _write_metadata(
            local_run_dir,
            {
                "host": args.host,
                "user": args.user,
                "port": args.port,
                "config": str(config_rel),
                "run_name": run_name,
                "remote_workspace": remote_workspace,
                "remote_run_dir": remote_run_dir,
                "local_run_dir": str(local_run_dir),
                "elapsed_s": time.time() - start_s,
            },
        )
        print(f"[remote-pairwise] local results: {local_run_dir}")
    finally:
        bundle_path.unlink(missing_ok=True)
        if remote_workspace and not args.keep_remote_workdir:
            subprocess.run(
                _ssh_base(args) + [f"rm -rf {shlex.quote(remote_workspace)}"],
                cwd=PROJECT_ROOT,
                check=False,
            )


if __name__ == "__main__":
    main()
