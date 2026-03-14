from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "infra" / "build_socket_shaper.py"
SHARED_OBJECT = PROJECT_ROOT / "infra" / "socket_shaper.so"


pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="socket shaper integration test is Linux-only")


def _transfer_script() -> str:
    return textwrap.dedent(
        """
        import json
        import socket
        import threading
        import time

        TOTAL = 8 * 1024 * 1024
        payload = b"x" * (1 << 20)

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        ready = threading.Event()

        def server():
            ready.set()
            conn, _ = listener.accept()
            received = 0
            while received < TOTAL:
                chunk = conn.recv(1 << 20)
                if not chunk:
                    break
                received += len(chunk)
            conn.close()
            listener.close()

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        ready.wait()

        client = socket.socket()
        client.connect(("127.0.0.1", port))
        sent = 0
        t0 = time.perf_counter()
        while sent < TOTAL:
            chunk = payload[: min(len(payload), TOTAL - sent)]
            client.sendall(chunk)
            sent += len(chunk)
        client.shutdown(socket.SHUT_WR)
        thread.join()
        dt = time.perf_counter() - t0
        print(json.dumps({"seconds": dt, "mbps": (TOTAL * 8) / (dt * 1e6)}))
        """
    )


def _client_script() -> str:
    return textwrap.dedent(
        """
        import json
        import socket
        import sys
        import time

        total = int(sys.argv[1])
        host = sys.argv[2]
        port = int(sys.argv[3])
        payload = b"x" * (1 << 20)

        client = socket.socket()
        client.connect((host, port))
        sent = 0
        t0 = time.perf_counter()
        while sent < total:
            chunk = payload[: min(len(payload), total - sent)]
            client.sendall(chunk)
            sent += len(chunk)
        client.shutdown(socket.SHUT_WR)
        client.close()
        print(json.dumps({"seconds": time.perf_counter() - t0}))
        """
    )


def _run_parallel_clients(shared_name_a: str | None, shared_name_b: str | None) -> float:
    total = 4 * 1024 * 1024
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    host, port = listener.getsockname()

    def _reader(conn: socket.socket) -> None:
        received = 0
        while received < total:
            chunk = conn.recv(1 << 20)
            if not chunk:
                break
            received += len(chunk)
        conn.close()

    import threading
    readers = []

    def _accept_loop() -> None:
        for _ in range(2):
            conn, _addr = listener.accept()
            thread = threading.Thread(target=_reader, args=(conn,), daemon=True)
            thread.start()
            readers.append(thread)

    accept_thread = threading.Thread(target=_accept_loop, daemon=True)
    accept_thread.start()

    def _client_env(shared_name: str | None) -> dict[str, str]:
        env = os.environ.copy()
        env["LD_PRELOAD"] = str(SHARED_OBJECT)
        env["ZERO_SOCKET_SHAPER_BW_GBPS"] = "0.1"
        env["ZERO_SOCKET_SHAPER_LATENCY_MS"] = "0"
        env["ZERO_SOCKET_SHAPER_BURST_BYTES"] = "32768"
        if shared_name:
            env["ZERO_SOCKET_SHAPER_SHARED_NAME"] = shared_name
        else:
            env.pop("ZERO_SOCKET_SHAPER_SHARED_NAME", None)
        return env

    t0 = time.perf_counter()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _client_script(), str(total), host, str(port)],
            cwd=PROJECT_ROOT,
            env=_client_env(shared_name_a),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
        subprocess.Popen(
            [sys.executable, "-c", _client_script(), str(total), host, str(port)],
            cwd=PROJECT_ROOT,
            env=_client_env(shared_name_b),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
    ]
    for proc in procs:
        proc.wait(timeout=30)
        assert proc.returncode == 0, proc.stderr.read() if proc.stderr else ""

    accept_thread.join(timeout=30)
    for reader in readers:
        reader.join(timeout=30)
    listener.close()
    return time.perf_counter() - t0


def test_socket_shaper_builds_and_limits_local_tcp_throughput(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=PROJECT_ROOT, check=True)
    assert SHARED_OBJECT.exists()

    unshaped = subprocess.run(
        [sys.executable, "-c", _transfer_script()],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    unshaped_metrics = json.loads(unshaped.stdout)

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(SHARED_OBJECT)
    env["ZERO_SOCKET_SHAPER_BW_GBPS"] = "0.1"
    env["ZERO_SOCKET_SHAPER_LATENCY_MS"] = "0"
    env["ZERO_SOCKET_SHAPER_BURST_BYTES"] = "65536"
    shaped = subprocess.run(
        [sys.executable, "-c", _transfer_script()],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    shaped_metrics = json.loads(shaped.stdout)

    assert shaped_metrics["seconds"] > unshaped_metrics["seconds"]
    assert shaped_metrics["mbps"] < unshaped_metrics["mbps"]
    assert shaped_metrics["mbps"] < 200.0


def test_socket_shaper_shared_name_enforces_global_bucket_across_processes() -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=PROJECT_ROOT, check=True)
    assert SHARED_OBJECT.exists()

    independent_elapsed = _run_parallel_clients("zero_socket_a", "zero_socket_b")
    shared_elapsed = _run_parallel_clients("zero_socket_shared", "zero_socket_shared")

    assert shared_elapsed > independent_elapsed * 1.4
