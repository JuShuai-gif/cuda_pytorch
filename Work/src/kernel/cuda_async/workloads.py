"""CUDA async execution and stream experiments.

These workloads make the host/device execution model observable: whether a
copy blocks the CPU, whether host memory is page-locked (pinned) or pageable,
and whether independent kernels overlap across streams.

Note on the Jetson/Thor unified-memory platform: host and device share the
same physical DRAM, so "H2D" copies do not cross a PCIe bus.  The pinned vs
pageable difference is therefore about *DMA efficiency and page migration*
rather than the huge PCIe penalty seen on discrete GPUs.  The benchmark
reports the measured numbers so the platform difference shows up instead of
being assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List

import torch


@dataclass(frozen=True)
class H2DResult:
    label: str
    pinned: bool
    non_blocking: bool
    bytes: int
    wall_ms: float
    event_ms: float
    gbps_wall: float


def _copy_once(src: torch.Tensor, dst: torch.Tensor, *, non_blocking: bool) -> None:
    dst.copy_(src, non_blocking=non_blocking)


def benchmark_h2d(
    n_bytes: int,
    *,
    device: torch.device,
    pinned: bool,
    non_blocking: bool,
    warmup: int = 10,
    iterations: int = 50,
) -> H2DResult:
    """Measure host-to-device copy bandwidth for a given memory configuration."""
    n_floats = max(1, n_bytes // 4)
    if pinned:
        src = torch.empty(n_floats, dtype=torch.float32, pin_memory=True)
    else:
        src = torch.empty(n_floats, dtype=torch.float32)
    dst = torch.empty(n_floats, dtype=torch.float32, device=device)
    src.normal_()

    for _ in range(warmup):
        _copy_once(src, dst, non_blocking=non_blocking)
    torch.cuda.synchronize(device)

    wall_samples: List[float] = []
    event_samples: List[float] = []
    for _ in range(iterations):
        t0 = perf_counter()
        _copy_once(src, dst, non_blocking=non_blocking)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_samples.append((perf_counter() - t0) * 1e3)

    # Event-based device time for the copy itself.
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            start.record()
            _copy_once(src, dst, non_blocking=non_blocking)
            end.record()
            torch.cuda.synchronize(device)
            event_samples.append(start.elapsed_time(end))

    wall_ms = sum(wall_samples) / len(wall_samples)
    event_ms = sum(event_samples) / len(event_samples) if event_samples else 0.0
    gbps = (n_bytes / (event_ms * 1e-3)) / 1e9 if event_ms > 0 else 0.0
    label = ("pinned" if pinned else "pageable") + ("_nb" if non_blocking else "_sync")
    return H2DResult(
        label=label,
        pinned=pinned,
        non_blocking=non_blocking,
        bytes=n_bytes,
        wall_ms=wall_ms,
        event_ms=event_ms,
        gbps_wall=gbps,
    )


def benchmark_streams(
    *,
    device: torch.device,
    n_streams: int,
    mat_size: int,
    work_per_stream: int,
    warmup: int = 3,
    iterations: int = 20,
) -> dict:
    """Compare total time for independent GEMMs across 1 stream vs N streams.

    Each stream computes ``work_per_stream`` square matmuls of ``mat_size``.
    The single-stream case serializes everything on the default stream; the
    multi-stream case spreads chunks across ``n_streams`` so independent work
    can overlap.
    """
    if device.type != "cuda":
        raise RuntimeError("stream overlap benchmark requires CUDA")

    def run_single() -> None:
        a = torch.randn(mat_size, mat_size, device=device)
        b = torch.randn(mat_size, mat_size, device=device)
        for _ in range(n_streams * work_per_stream):
            torch.mm(a, b)

    def run_multi() -> None:
        streams = [torch.cuda.Stream() for _ in range(n_streams)]
        chunks = [
            (
                torch.randn(mat_size, mat_size, device=device),
                torch.randn(mat_size, mat_size, device=device),
            )
            for _ in range(n_streams)
        ]
        for s, (a, b) in zip(streams, chunks):
            with torch.cuda.stream(s):
                for _ in range(work_per_stream):
                    torch.mm(a, b)
        for s in streams:
            s.synchronize()

    for _ in range(warmup):
        run_single()
        run_multi()
    torch.cuda.synchronize(device)

    def timed(fn) -> float:
        t0 = perf_counter()
        fn()
        torch.cuda.synchronize(device)
        return (perf_counter() - t0) * 1e3

    single = [timed(run_single) for _ in range(iterations)]
    multi = [timed(run_multi) for _ in range(iterations)]
    return {
        "n_streams": n_streams,
        "mat_size": mat_size,
        "work_per_stream": work_per_stream,
        "single_stream_ms": sum(single) / len(single),
        "multi_stream_ms": sum(multi) / len(multi),
    }
