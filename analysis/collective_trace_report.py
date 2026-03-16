from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize collective trace data from train_zero profile JSON")
    parser.add_argument("profiles", nargs="+", help="One or more profile JSON files")
    return parser.parse_args()


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        key = (str(record.get("op", "")), str(record.get("label", "")))
        entry = by_key.get(key)
        if entry is None:
            entry = {
                "op": key[0],
                "label": key[1],
                "calls": 0,
                "elapsed_ms": 0.0,
                "input_bytes": 0.0,
                "output_bytes": 0.0,
            }
            by_key[key] = entry
        entry["calls"] += 1
        entry["elapsed_ms"] += float(record.get("elapsed_ms", 0.0))
        entry["input_bytes"] += float(record.get("input_bytes", 0.0))
        entry["output_bytes"] += float(record.get("output_bytes", 0.0))
    return sorted(by_key.values(), key=lambda item: (-item["elapsed_ms"], item["label"], item["op"]))


def bytes_to_mb(value: float) -> float:
    return value / (1024.0 * 1024.0)


def render_profile(path: Path) -> str:
    payload = json.loads(path.read_text())
    args = payload.get("args", {})
    records: list[dict[str, object]] = []
    steps = payload.get("steps", [])
    for step in steps:
        records.extend(step.get("collective_trace", []))
    summary = summarize_records(records)

    lines: list[str] = []
    lines.append(f"# {path}")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- stage: `{payload.get('stage')}`")
    lines.append(f"- batch size / GPU: `{payload.get('microbatch_size_per_gpu')}`")
    lines.append(f"- seq len: `{payload.get('args', {}).get('seq_len')}`")
    lines.append(f"- steps with trace: `{len(steps)}`")
    lines.append("")
    lines.append("## Aggregate Trace")
    lines.append("")
    lines.append("| op | label | calls | elapsed ms | input MB | output MB |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in summary:
        lines.append(
            f"| {item['op']} | {item['label']} | {item['calls']} | {item['elapsed_ms']:.1f} | "
            f"{bytes_to_mb(item['input_bytes']):.2f} | {bytes_to_mb(item['output_bytes']):.2f} |"
        )

    by_op: dict[str, float] = defaultdict(float)
    for item in summary:
        by_op[str(item["op"])] += float(item["elapsed_ms"])
    lines.append("")
    lines.append("## By Operation")
    lines.append("")
    for op, elapsed_ms in sorted(by_op.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{op}`: `{elapsed_ms:.1f} ms`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for raw_path in args.profiles:
        print(render_profile(Path(raw_path)))


if __name__ == "__main__":
    main()
