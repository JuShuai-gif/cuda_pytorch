"""Batch=1 policy inference: naive launch vs CUDA Graph.

Robot online control is almost always batch=1.  A batch=1 small policy is
launch-bound: the CPU cost of issuing each op dominates the tiny GPU work.  The
standard fix is to capture the whole forward pass into a CUDA Graph and replay
it with one launch.  This module compares the wall and event latency of the
naive path vs the graph path.
"""

from __future__ import annotations

import torch

from common.measure import cuda_event_latency, sync_wall_latency
from robotics.policy_inference.pipeline import VLAPolicy


def build_graph(model: VLAPolicy, static_input: torch.Tensor) -> torch.cuda.CUDAGraph:
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        model.infer(static_input)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(g):
        model.infer(static_input)
    return g


def compare_batch1(device: torch.device, warmup=20, iterations=100) -> dict:
    model = VLAPolicy().to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)

    def naive_fn():
        model.infer(x)

    wall = sync_wall_latency(naive_fn, device=device, warmup=warmup, iterations=iterations)
    event = cuda_event_latency(naive_fn, device=device, warmup=warmup, iterations=iterations)

    graph = build_graph(model, x)

    def graph_fn():
        graph.replay()

    gwall = sync_wall_latency(graph_fn, device=device, warmup=warmup, iterations=iterations)
    gevent = cuda_event_latency(graph_fn, device=device, warmup=warmup, iterations=iterations)

    return {
        "naive_wall_us": wall.mean,
        "naive_event_us": event.mean,
        "graph_wall_us": gwall.mean,
        "graph_event_us": gevent.mean,
        "wall_speedup_x": wall.mean / gwall.mean,
    }
