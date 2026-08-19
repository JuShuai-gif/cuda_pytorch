"""Nsight Systems target for the CUDA Graph experiment.

Runs the normal chain and the graph replay with NVTX ranges so the timeline
shows the launch gap between tiny kernels (normal) versus the single replay
launch (graph).  NVTX names use underscores, not ``/``.
"""

from __future__ import annotations

import argparse

import torch

from common.env import resolve_device
from kernel.cuda_graph.workloads import (
    build_graph,
    make_chain_input,
    run_chain_normal,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-ops", type=int, default=64)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--steps", type=int, default=5)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("nsys target requires CUDA")

    x = make_chain_input(args.n, device=device)
    scalar = 1.0001
    graph = build_graph(x, n_ops=args.n_ops, scalar=scalar)

    for step in range(args.steps):
        torch.cuda.nvtx.range_push(f"normal_step_{step}")
        run_chain_normal(x, n_ops=args.n_ops, scalar=scalar)
        torch.cuda.nvtx.range_pop()

    for step in range(args.steps):
        torch.cuda.nvtx.range_push(f"graph_step_{step}")
        graph.replay()
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
