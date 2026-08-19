"""Benchmark a launch-bound kernel chain with and without CUDA Graph.

The key comparison is ``wall`` latency between the two modes: the graph replay
issues one launch instead of ``n_ops``, so the host-side cost collapses.  The
``event`` device time may stay similar because the GPU work is identical; that
is expected and is itself the lesson (launch-bound means CPU-bound, not
GPU-bound).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from common.env import collect_environment, resolve_device
from common.report import write_report
from kernel.cuda_graph.workloads import (
    benchmark_graph,
    benchmark_normal,
    make_chain_input,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-ops", type=int, default=64, help="number of tiny ops in the chain")
    p.add_argument("--n", type=int, default=1024, help="element count of the working tensor")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", required=True, help="JSON report path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA Graph benchmark requires CUDA")

    x = make_chain_input(args.n, device=device)
    scalar = 1.0001

    normal = benchmark_normal(
        x, n_ops=args.n_ops, scalar=scalar,
        warmup=args.warmup, iterations=args.iterations,
    )
    graph = benchmark_graph(
        x, n_ops=args.n_ops, scalar=scalar,
        warmup=args.warmup, iterations=args.iterations,
    )

    report: dict[str, Any] = {
        "kind": "cuda_graph",
        "environment": collect_environment(device),
        "config": {
            "n_ops": args.n_ops,
            "n": args.n,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "normal": {"wall": normal.wall_summary, "event": normal.event_summary},
        "graph": {"wall": graph.wall_summary, "event": graph.event_summary},
    }
    path = write_report(args.output, report)
    print(json.dumps(
        {
            "normal_wall_mean_us": normal.wall_summary["mean"],
            "graph_wall_mean_us": graph.wall_summary["mean"],
            "normal_event_mean_us": normal.event_summary["mean"],
            "graph_event_mean_us": graph.event_summary["mean"],
        },
        indent=2,
    ))
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
