#!/usr/bin/env python3
"""Shared benchmarking utilities for all chapters."""

import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def cuda_timer(name: str):
    """CUDA event timer context manager."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    print(f"[{name}] {elapsed:.3f} ms")


def get_gpu_info() -> dict:
    """Get GPU properties for the current device."""
    if not torch.cuda.is_available():
        return {"device": "CPU"}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "max_shared_memory_per_block": props.max_shared_memory_per_block,
        "warp_size": 32,
    }


def save_results(results: list, path: str | Path):
    """Save benchmark results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def estimate_bandwidth(bytes_moved: float, time_ms: float) -> float:
    """Compute bandwidth in GB/s."""
    return (bytes_moved / 1e9) / (time_ms / 1000.0)


def estimate_tflops(flops: float, time_ms: float) -> float:
    """Compute TFLOPS."""
    return (flops / 1e12) / (time_ms / 1000.0)


DEFAULT_WARMUP = 10
DEFAULT_ITERS = 100
