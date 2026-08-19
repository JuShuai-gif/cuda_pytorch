"""Observability: the three pillars (metrics, logs, traces).

The goal is to answer "where did this request's 200ms go?" across a Cloud ->
Edge -> Robot -> Model chain.  Three primitives, with the IDs that tie them
together:

  Trace    a request's path across tiers, as a tree of timed spans, keyed by a
           request_id (which links to task_id and robot_id)
  Metrics  aggregate numeric series (QPS, latency p50/p99, error rate)
  Logs     structured events carrying the same request_id for correlation

A span is a timed region (start/duration) with metadata.  The tracer records
spans per trace_id; the request_id is the *correlation id* that every tier
propagates, so a single request can be reconstructed end to end.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Span:
    name: str
    start_ms: float
    duration_ms: float
    metadata: dict = field(default_factory=dict)


class Tracer:
    def __init__(self):
        self.traces: Dict[str, List[Span]] = {}

    @contextmanager
    def span(self, trace_id: str, name: str, **metadata):
        t0 = time.perf_counter() * 1e3
        yield
        dur = time.perf_counter() * 1e3 - t0
        self.traces.setdefault(trace_id, []).append(
            Span(name, t0, dur, metadata))

    def trace(self, trace_id: str) -> List[Span]:
        return self.traces.get(trace_id, [])

    def total_ms(self, trace_id: str) -> float:
        return sum(s.duration_ms for s in self.trace(trace_id))


class Metrics:
    def __init__(self):
        self._values: Dict[str, List[float]] = {}

    def record(self, name: str, value: float):
        self._values.setdefault(name, []).append(value)

    def summary(self, name: str) -> dict:
        vals = sorted(self._values.get(name, []))
        if not vals:
            return {}
        n = len(vals)

        def pct(q):
            return vals[min(n - 1, int(n * q))]

        return {
            "count": n,
            "mean": sum(vals) / n,
            "p50": pct(0.50),
            "p99": pct(0.99),
            "max": vals[-1],
        }


class StructuredLogger:
    def __init__(self):
        self.entries: List[dict] = []

    def log(self, request_id: str, level: str, msg: str, **fields):
        self.entries.append({
            "ts_ms": time.perf_counter() * 1e3,
            "request_id": request_id,
            "level": level,
            "msg": msg,
            **fields,
        })
