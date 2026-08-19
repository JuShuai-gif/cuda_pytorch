"""Benchmark CUDA async execution: pinned memory, non-blocking H2D, streams.

Runs three experiments and writes a single JSON report:

1. pageable vs pinned H2D copy bandwidth
2. blocking vs non-blocking H2D (does the copy release the CPU early?)
3. single stream vs N streams for independent GEMM work

The pinned/stream experiments require a CUDA device; the pageable/pinned H2D
experiment degrades to a CPU-to-CPU copy on CPU and is only meaningful on
CUDA.  CPU fallback verifies the code path, not the CUDA behavior.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from common.env import collect_environment, resolve_device
from common.report import write_report
from kernel.cuda_async.workloads import benchmark_h2d, benchmark_streams


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--bytes", type=int, default=64 * 1024 * 1024, help="H2D payload bytes")
    p.add_argument("--mat-size", type=int, default=512)
    p.add_argument("--work-per-stream", type=int, default=8)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--output", required=True, help="JSON report path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)

    h2d = [
        benchmark_h2d(
            args.bytes, device=device, pinned=False, non_blocking=False,
            warmup=args.warmup, iterations=args.iterations,
        ),
        benchmark_h2d(
            args.bytes, device=device, pinned=True, non_blocking=False,
            warmup=args.warmup, iterations=args.iterations,
        ),
        benchmark_h2d(
            args.bytes, device=device, pinned=True, non_blocking=True,
            warmup=args.warmup, iterations=args.iterations,
        ),
    ]

    streams = (
        benchmark_streams(
            device=device,
            n_streams=args.n_streams,
            mat_size=args.mat_size,
            work_per_stream=args.work_per_stream,
            warmup=3,
            iterations=args.iterations,
        )
        if device.type == "cuda"
        else None
    )

    report: dict[str, Any] = {
        "kind": "cuda_async",
        "environment": collect_environment(device),
        "config": {
            "bytes": args.bytes,
            "mat_size": args.mat_size,
            "work_per_stream": args.work_per_stream,
            "n_streams": args.n_streams,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "h2d": [r.__dict__ for r in h2d],
        "streams": streams,
    }
    path = write_report(args.output, report)
    print(json.dumps(report["h2d"], indent=2))
    if streams:
        print(json.dumps(streams, indent=2))
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
