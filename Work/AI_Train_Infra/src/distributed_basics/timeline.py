"""Interval-union accounting for DDP compute/communication overlap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class Interval:
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("interval requires 0 <= start_ns < end_ns")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(intervals)
    merged: list[Interval] = []
    for current in ordered:
        if not merged or current.start_ns > merged[-1].end_ns:
            merged.append(current)
        else:
            previous = merged[-1]
            merged[-1] = Interval(previous.start_ns, max(previous.end_ns, current.end_ns))
    return merged


def union_duration_ns(intervals: Iterable[Interval]) -> int:
    return sum(interval.duration_ns for interval in merge_intervals(intervals))


def intersection_duration_ns(left: Iterable[Interval], right: Iterable[Interval]) -> int:
    a = merge_intervals(left)
    b = merge_intervals(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i].start_ns, b[j].start_ns)
        end = min(a[i].end_ns, b[j].end_ns)
        if end > start:
            total += end - start
        if a[i].end_ns <= b[j].end_ns:
            i += 1
        else:
            j += 1
    return total


@dataclass(frozen=True)
class OverlapSummary:
    compute_total_ns: int
    communication_total_ns: int
    compute_communication_overlap_ns: int
    exposed_communication_ns: int
    communication_overlap_fraction: float | None
    gpu_window_ns: int | None
    gpu_idle_or_other_ns: int | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def summarize_overlap(
    compute: Iterable[Interval], communication: Iterable[Interval]
) -> OverlapSummary:
    compute_merged = merge_intervals(compute)
    communication_merged = merge_intervals(communication)
    compute_ns = union_duration_ns(compute_merged)
    communication_ns = union_duration_ns(communication_merged)
    overlap_ns = intersection_duration_ns(compute_merged, communication_merged)
    exposed_ns = communication_ns - overlap_ns
    all_intervals = compute_merged + communication_merged
    if all_intervals:
        window_ns = max(item.end_ns for item in all_intervals) - min(
            item.start_ns for item in all_intervals
        )
        busy_union_ns = compute_ns + communication_ns - overlap_ns
        idle_or_other_ns = window_ns - busy_union_ns
    else:
        window_ns = None
        idle_or_other_ns = None
    fraction = overlap_ns / communication_ns if communication_ns else None
    return OverlapSummary(
        compute_total_ns=compute_ns,
        communication_total_ns=communication_ns,
        compute_communication_overlap_ns=overlap_ns,
        exposed_communication_ns=exposed_ns,
        communication_overlap_fraction=fraction,
        gpu_window_ns=window_ns,
        gpu_idle_or_other_ns=idle_or_other_ns,
    )
