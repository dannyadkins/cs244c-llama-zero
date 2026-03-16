from __future__ import annotations

import torch

from collectives import LocalCollectives, TracingCollectives
from train_zero import aggregate_collective_trace


def test_tracing_collectives_records_consistent_tensor_sizes() -> None:
    traced = TracingCollectives(LocalCollectives())
    shard = torch.arange(4, dtype=torch.bfloat16)
    with traced.label_scope("stage2.step.allgather.updated_param_shard"):
        gathered = traced.allgather(shard)

    records = traced.consume_trace()
    assert gathered.numel() == 4
    assert len(records) == 1
    record = records[0]
    assert record["op"] == "allgather"
    assert record["label"] == "stage2.step.allgather.updated_param_shard"
    assert record["input_bytes"] == shard.numel() * shard.element_size()
    assert record["output_bytes"] == gathered.numel() * gathered.element_size()
    assert traced.consume_trace() == []


def test_aggregate_collective_trace_merges_by_op_and_label() -> None:
    summary = aggregate_collective_trace(
        [
            {"op": "allgather", "label": "stage3.forward.allgather.layers.0", "elapsed_ms": 2.0, "input_bytes": 4, "output_bytes": 16},
            {"op": "allgather", "label": "stage3.forward.allgather.layers.0", "elapsed_ms": 3.0, "input_bytes": 4, "output_bytes": 16},
            {"op": "reduce_scatter", "label": "stage3.backward.reduce_scatter.layers.0", "elapsed_ms": 1.5, "input_bytes": 16, "output_bytes": 4},
        ]
    )

    assert summary == [
        {
            "op": "allgather",
            "label": "stage3.forward.allgather.layers.0",
            "calls": 2,
            "elapsed_ms": 5.0,
            "input_bytes": 8.0,
            "output_bytes": 32.0,
        },
        {
            "op": "reduce_scatter",
            "label": "stage3.backward.reduce_scatter.layers.0",
            "calls": 1,
            "elapsed_ms": 1.5,
            "input_bytes": 16.0,
            "output_bytes": 4.0,
        },
    ]
