"""Explicit timing helpers for reproducible GPU benchmarks.

The single most important fact encoded here: a CUDA kernel launch is
asynchronous.  A host ``time.perf_counter`` interval without a terminal
synchronization measures *enqueue* latency, not *completion* latency.  Two
distinct helpers make the choice explicit instead of burying it in call sites:

* :func:`sync_wall_latency` - wall-clock time between two host-side
  ``torch.cuda.synchronize`` calls.  This is what an end-to-end caller actually
  experiences for one request, including launch overhead and any host/device
  pipeline serialization, but it deliberately breaks cross-request overlap.

* :func:`cuda_event_latency` - device time between two CUDA events recorded on
  a stream.  This isolates GPU execution time and excludes host launch cost,
  but it does not reflect wall-clock latency as seen by a client.

Both return a :class:`TimingSummary` that keeps the raw per-sample values, so
percentiles are computed from the same population the mean comes from.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Optional

import torch

Microseconds = float


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile (same convention as NumPy default)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class TimingSummary:
    """Latency statistics in the caller-chosen unit (default microseconds)."""

    unit: str
    samples: int
    mean: float
    stddev: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    raw: List[float]

    def as_dict(self) -> dict:
        return {
            "unit": self.unit,
            "samples": self.samples,
            "mean": self.mean,
            "stddev": self.stddev,
            "median": self.median,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def summarize(values: List[Microseconds], unit: str = "us") -> TimingSummary:
    if not values:
        raise ValueError("cannot summarize an empty sample list")
    return TimingSummary(
        unit=unit,
        samples=len(values),
        mean=statistics.mean(values),
        stddev=statistics.pstdev(values),
        median=statistics.median(values),
        p50=percentile(values, 0.50),
        p90=percentile(values, 0.90),
        p95=percentile(values, 0.95),
        p99=percentile(values, 0.99),
        minimum=min(values),
        maximum=max(values),
        raw=values,
    )


def sync_wall_latency(
    fn: Callable[[], None],
    *,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 100,
) -> TimingSummary:
    """Measure wall-clock latency of ``fn`` with a sync barrier on each side.

    Returns microseconds per call.  ``fn`` runs with no arguments and must
    leave its work enqueued on the default (or an explicit) stream; the helper
    only inserts the start/end synchronization barriers.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    for _ in range(warmup):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    samples: List[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = perf_counter()
        samples.append((t1 - t0) * 1e6)
    return summarize(samples, unit="us")


def cuda_event_latency(
    fn: Callable[[], None],
    *,
    device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
    warmup: int = 10,
    iterations: int = 100,
) -> TimingSummary:
    """Measure device-side execution time of ``fn`` using CUDA events.

    Events are recorded on ``stream`` (default stream if ``None``) immediately
    before and after ``fn``.  This excludes host launch overhead and measures
    only the GPU work enqueued by ``fn`` on that stream.  Requires a CUDA
    device.
    """
    if device.type != "cuda":
        raise RuntimeError("cuda_event_latency requires a CUDA device")
    s = stream or torch.cuda.current_stream(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: List[float] = []
    for _ in range(iterations):
        start.record(s)
        fn()
        end.record(s)
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return summarize(samples, unit="us")
