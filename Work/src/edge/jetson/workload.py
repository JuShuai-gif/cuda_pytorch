"""Sustained GPU workload for thermal/power observation.

The whole point of the edge lab: a workload that runs continuously (not a
10-second burst) so we can watch temperature rise, clocks settle, and power
stabilize - the "runs fine for 10s" vs "stable for 24h" distinction.  The
workload is a tight loop of fp16 GEMMs that keeps the GPU compute-heavy.
"""

from __future__ import annotations

import time

import torch


def run_sustained_gpu(seconds: float, size: int = 2048) -> int:
    """Run a GEMM loop for `seconds`; returns the number of iterations done."""
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    n = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a @ b
        n += 1
    torch.cuda.synchronize()
    return n


def run_sustained_cpu(seconds: float, size: int = 2048) -> int:
    """A CPU-bound loop (matrix multiply) for CPU power/thermal contrast."""
    import numpy as np

    a = np.random.randn(size, size).astype("float32")
    b = np.random.randn(size, size).astype("float32")
    n = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        np.dot(a, b)
        n += 1
    return n
