"""Inference throughput benchmark.

Throughput is the inverse problem of latency.  Here the loop is *not*
synchronized per request: we enqueue a burst of forward passes and synchronize
only once at the end, so successive kernels can overlap and the queue stays
full.  This is the measurement that tells you the sustainable samples/s (or
tokens/s) of a model, distinct from the isolated per-request latency.

Optionally sweeps batch sizes to make the latency/throughput tradeoff
concrete: bigger batches raise throughput (better tensor-core utilization) but
also raise the per-request latency, because each request waits longer in the
batch.

Tokens are defined here as ``batch * seq_len`` positions per forward pass.
For a real LLM the meaningful units are TTFT/TPOT/tokens-per-second; those
arrive with KV cache and decode in a later stage.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import Any

import torch

from common.env import collect_environment, resolve_device, resolve_dtype
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
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--batch", type=int, default=1, help="single batch, or see --batch-sweep")
    p.add_argument("--batch-sweep", type=int, nargs="*", default=None,
                   help="list of batch sizes to sweep")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--output", required=True, help="JSON report path")
    return p.parse_args(argv)


def measure_throughput(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    iterations: int,
) -> dict[str, Any]:
    """Enqueue a burst and sync once, returning samples/s and tokens/s."""
    batch = x.shape[0]
    seq_len = x.shape[1]
    with torch.no_grad():
        for _ in range(10):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - t0

    samples_per_sec = (iterations * batch) / elapsed
    tokens_per_sec = samples_per_sec * seq_len
    return {
        "batch": batch,
        "iterations": iterations,
        "wall_seconds": elapsed,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "avg_batch_latency_ms": (elapsed / iterations) * 1e3,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    batches = args.batch_sweep if args.batch_sweep else [args.batch]
    results: list[dict[str, Any]] = []
    for b in batches:
        config = InferenceConfig(args.hidden, args.layers, b, args.seq_len)
        model = make_model(config, device=device, dtype=dtype)
        x = make_input(config, device=device, dtype=dtype)
        results.append(measure_throughput(model, x, device=device, iterations=args.iterations))

    report: dict[str, Any] = {
        "kind": "throughput",
        "environment": collect_environment(device),
        "config": {
            "hidden": args.hidden,
            "layers": args.layers,
            "seq_len": args.seq_len,
            "dtype": str(dtype),
            "parameters": parameter_count(
                InferenceConfig(args.hidden, args.layers, args.batch, args.seq_len)
            ),
            "flops_per_forward": flops_per_forward(
                InferenceConfig(args.hidden, args.layers, args.batch, args.seq_len)
            ),
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "throughput": results,
    }
    path = write_report(args.output, report)
    print(json.dumps(report["throughput"], indent=2))
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
