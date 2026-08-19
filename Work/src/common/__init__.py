"""Shared, honest measurement infrastructure for the inference lab.

This package is intentionally dependency-light (PyTorch only) and exposes the
timing / environment / report helpers reused by the ``inference``, ``kernel``
and ``profiling`` modules.  Timing policy is explicit everywhere: a CUDA launch
is asynchronous, so every helper states whether it measures enqueue time or
device completion time.
"""

from .env import collect_environment, device_properties, resolve_device, resolve_dtype
from .measure import (
    TimingSummary,
    cuda_event_latency,
    percentile,
    sync_wall_latency,
    summarize,
)
from .report import write_report

__all__ = [
    "TimingSummary",
    "collect_environment",
    "cuda_event_latency",
    "device_properties",
    "percentile",
    "resolve_device",
    "resolve_dtype",
    "summarize",
    "sync_wall_latency",
    "write_report",
]
