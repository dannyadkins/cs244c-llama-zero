from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "infra" / "socket_shaper.c"
OUTPUT_PATH = PROJECT_ROOT / "infra" / "socket_shaper.so"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Linux LD_PRELOAD socket shaper shared library")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    return parser.parse_args()


def build_socket_shaper(output_path: Path) -> Path:
    if sys.platform != "linux":
        raise RuntimeError("socket shaper is only supported on Linux")

    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise FileNotFoundError("missing C compiler: expected 'cc' or 'gcc'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            cc,
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(output_path),
            str(SOURCE_PATH),
            "-ldl",
            "-pthread",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return output_path


def main() -> None:
    args = parse_args()
    output_path = build_socket_shaper(Path(args.output))
    print(output_path)


if __name__ == "__main__":
    main()
