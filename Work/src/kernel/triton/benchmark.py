"""Unified Triton-vs-PyTorch benchmark.

For each operator, run a correctness check first (triton output vs torch
reference), then benchmark both implementations with CUDA-event (device) time
and synchronized wall time, and report the mean + speedup.  The report refuses
to overwrite an existing file.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m kernel.triton.benchmark --device cuda --dtype float32 \
      --warmup 20 --iterations 100 --output /tmp/triton.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

import kernel.triton  # noqa: F401  (sets TRITON_PTXAS_BLACKWELL_PATH)
from common.env import collect_environment, resolve_device, resolve_dtype
from common.measure import cuda_event_latency, sync_wall_latency
from common.report import write_report
from kernel.triton.operators import (
    attention,
    gemm,
    layernorm,
    quantize,
    reduction,
    rmsnorm,
    softmax,
    vector_add,
)

OPERATORS = [vector_add, reduction, softmax, layernorm, rmsnorm, gemm, attention, quantize]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def bench_fn(fn, *, device, warmup, iterations):
    wall = sync_wall_latency(fn, device=device, warmup=warmup, iterations=iterations)
    event = (
        cuda_event_latency(fn, device=device, warmup=warmup, iterations=iterations)
        if device.type == "cuda"
        else None
    )
    return wall, event


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type != "cuda":
        raise RuntimeError("Triton benchmark requires CUDA")

    results: list[dict[str, Any]] = []
    for mod in OPERATORS:
        op = mod.build()
        ok, diff = op.check(device, dtype)

        def triton_fn():
            with torch.no_grad():
                op.triton(*op.inputs(device, dtype), **op.kwargs)

        def ref_fn():
            with torch.no_grad():
                op.reference(*op.inputs(device, dtype), **op.kwargs)

        t_wall, t_event = bench_fn(triton_fn, device=device, warmup=args.warmup,
                                   iterations=args.iterations)
        r_wall, r_event = bench_fn(ref_fn, device=device, warmup=args.warmup,
                                   iterations=args.iterations)

        results.append({
            "name": op.name,
            "note": op.note,
            "dtype": str(dtype),
            "correct": ok,
            "max_abs_diff": diff,
            "triton_wall_us": t_wall.mean,
            "reference_wall_us": r_wall.mean,
            "triton_event_us": t_event.mean,
            "reference_event_us": r_event.mean,
            "wall_speedup_x": r_wall.mean / t_wall.mean,
            "event_speedup_x": r_event.mean / t_event.mean,
        })
        print(f"{op.name:14s} correct={ok}  wall: triton {t_wall.mean:8.2f}us vs "
              f"torch {r_wall.mean:8.2f}us  event: {t_event.mean:8.2f} vs {r_event.mean:8.2f}us")

    report = {
        "kind": "triton_operators",
        "environment": collect_environment(device),
        "config": {"dtype": str(dtype), "warmup": args.warmup, "iterations": args.iterations},
        "results": results,
    }
    path = write_report(args.output, report)
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
