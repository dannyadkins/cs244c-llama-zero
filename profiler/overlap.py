from __future__ import annotations


def overlap_efficiency(step_ms: float, compute_ms: float, communication_ms: float) -> float:
    """Fraction of potential comm time hidden by overlap.

    0.0 means no overlap (step ~= compute + comm).
    1.0 means full overlap (step ~= max(compute, comm)).
    """

    if step_ms <= 0 or compute_ms < 0 or communication_ms < 0:
        raise ValueError("step_ms must be > 0 and component times must be >= 0")

    no_overlap = compute_ms + communication_ms
    full_overlap = max(compute_ms, communication_ms)

    if no_overlap <= full_overlap:
        return 1.0

    hidden = no_overlap - step_ms
    possible_hidden = no_overlap - full_overlap
    return max(0.0, min(1.0, hidden / possible_hidden))
