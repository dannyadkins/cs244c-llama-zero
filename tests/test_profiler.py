import time

from profiler import MemoryTracker, TimerRegistry, overlap_efficiency


def test_overlap_efficiency_bounds() -> None:
    # no overlap
    assert overlap_efficiency(step_ms=15.0, compute_ms=10.0, communication_ms=5.0) == 0.0
    # full overlap
    assert overlap_efficiency(step_ms=10.0, compute_ms=10.0, communication_ms=5.0) == 1.0


def test_timer_registry_records_samples() -> None:
    registry = TimerRegistry(device=None)
    timer = registry.timer("sleep")

    timer.start()
    time.sleep(0.01)
    timer.stop()

    summary = registry.summarize()["sleep"]
    assert summary.count == 1
    assert summary.total_ms > 0.0


def test_memory_tracker_records_snapshots() -> None:
    tracker = MemoryTracker(device=None)
    tracker.record("before")
    tracker.record("after")

    snapshots = tracker.as_dicts()
    assert len(snapshots) == 2
    assert snapshots[0]["label"] == "before"
    assert snapshots[1]["label"] == "after"
