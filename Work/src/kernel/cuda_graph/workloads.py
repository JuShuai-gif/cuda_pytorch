"""CUDA Graph vs normal launch experiments.

A launch-bound workload is a long chain of tiny elementwise kernels.  Each
kernel does almost no device work, so the wall-clock cost is dominated by CPU
launch overhead, not GPU execution.  This is exactly the batch=1, tiny-op
regime that hurts real-time robotics inference.

CUDA Graphs capture the whole chain once and replay it with a single launch,
amortizing the CPU cost.  The benchmark measures both device time (CUDA
events) and wall time (host synchronizations) so the launch-overhead win shows
up in ``wall`` even when ``event`` barely changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch

from common.measure import cuda_event_latency, sync_wall_latency


@dataclass(frozen=True)
class GraphResult:
    mode: str
    wall_summary: dict
    event_summary: dict


def make_chain_input(n: int, *, device: torch.device) -> torch.Tensor:
    return torch.randn(n, device=device, dtype=torch.float32)


def run_chain_normal(x: torch.Tensor, *, n_ops: int, scalar: float) -> None:
    """Run a chain of tiny elementwise ops as individual launches."""
    for _ in range(n_ops):
        x.mul_(scalar)
        x.add_(1.0)


def benchmark_normal(
    x: torch.Tensor, *, n_ops: int, scalar: float, warmup: int, iterations: int
) -> GraphResult:
    def fn() -> None:
        run_chain_normal(x, n_ops=n_ops, scalar=scalar)

    device = x.device
    wall = sync_wall_latency(fn, device=device, warmup=warmup, iterations=iterations)
    event = cuda_event_latency(fn, device=device, warmup=warmup, iterations=iterations)
    return GraphResult(mode="normal", wall_summary=wall.as_dict(), event_summary=event.as_dict())


def build_graph(x: torch.Tensor, *, n_ops: int, scalar: float) -> torch.cuda.CUDAGraph:
    """Capture the elementwise chain into a CUDA Graph on a side stream."""
    graph = torch.cuda.CUDAGraph()
    # Warm up the capture path on a side stream, per the documented pattern.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run_chain_normal(x, n_ops=n_ops, scalar=scalar)
    torch.cuda.current_stream().wait_stream(side)

    with torch.cuda.graph(graph):
        run_chain_normal(x, n_ops=n_ops, scalar=scalar)
    return graph


def benchmark_graph(
    x: torch.Tensor, *, n_ops: int, scalar: float, warmup: int, iterations: int
) -> GraphResult:
    graph = build_graph(x, n_ops=n_ops, scalar=scalar)
    device = x.device

    def fn() -> None:
        graph.replay()

    wall = sync_wall_latency(fn, device=device, warmup=warmup, iterations=iterations)
    event = cuda_event_latency(fn, device=device, warmup=warmup, iterations=iterations)
    return GraphResult(mode="graph", wall_summary=wall.as_dict(), event_summary=event.as_dict())
