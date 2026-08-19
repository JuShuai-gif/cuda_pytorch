"""Single-request inference latency benchmark.

Measures the latency of one forward pass of a small residual-MLP model.  The
whole point of this module is to demonstrate the two *different* notions of
latency and why they disagree:

* ``event`` - CUDA-event device time.  Excludes host launch overhead, so on a
  small model it can report a number much smaller than what a client sees.
* ``wall`` - wall-clock time bracketed by host synchronizations.  This is what
  an end-to-end caller observes for one isolated request and includes Python,
  dispatcher, allocator and launch overhead.

For a batch=1, launch-bound model the ``wall`` number is dominated by host
overhead while ``event`` is dominated by GPU execution.  The gap *is* the
lesson: reducing device time is pointless until the host overhead is attacked
(fusion, CUDA Graph, fewer launches).

Reported statistics always include mean/p50/p90/p95/p99 plus raw samples, never
a single "runtime = X ms" figure.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from common.env import collect_environment, resolve_device, resolve_dtype
from common.measure import cuda_event_latency, sync_wall_latency
from common.report import write_report
from inference.workloads import (
    InferenceConfig,
    flops_per_forward,
    make_input,
    make_model,
    parameter_count,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--output", required=True, help="JSON report path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    config = InferenceConfig(args.hidden, args.layers, args.batch, args.seq_len)

    model = make_model(config, device=device, dtype=dtype)
    x = make_input(config, device=device, dtype=dtype)

    def forward_once() -> None:
        with torch.no_grad():
            model(x)

    # Warm up outside the timed region, then measure both latency notions.
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    wall = sync_wall_latency(
        forward_once, device=device, warmup=0, iterations=args.iterations
    )
    event = (
        cuda_event_latency(
            forward_once, device=device, warmup=0, iterations=args.iterations
        )
        if device.type == "cuda"
        else None
    )

    report: dict[str, Any] = {
        "kind": "latency",
        "environment": collect_environment(device),
        "config": {
            "hidden": config.hidden,
            "layers": config.layers,
            "batch": config.batch,
            "seq_len": config.seq_len,
            "dtype": str(dtype),
            "parameters": parameter_count(config),
            "flops_per_forward": flops_per_forward(config),
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "wall_latency": wall.as_dict(),
        "event_latency": event.as_dict() if event else None,
    }
    path = write_report(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
