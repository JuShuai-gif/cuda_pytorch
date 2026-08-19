"""Benchmark naive vs optimized robot runtime.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m projects.robot_runtime.benchmark --device cuda --output /tmp/robot_runtime.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from common.env import collect_environment, resolve_device
from common.report import write_report
from inference.vlm.pipeline import make_image_bytes
from projects.robot_runtime.runtime import NaiveRuntime, OptimizedRuntime


def run_latency(runtime, frames, device, iterations=100, sync_fn=None) -> dict:
    latencies = []
    for i in range(iterations):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        runtime.infer(frame)
        if sync_fn:
            sync_fn()
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1e3)
    latencies.sort()
    n = len(latencies)

    def pct(q):
        return latencies[min(n - 1, int(n * q))]

    return {
        "mean_ms": sum(latencies) / n,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "jitter_ms": pct(0.99) - pct(0.50),
        "max_ms": latencies[-1],
    }


def run_throughput(runtime, frames, device, n=200, sync_fn=None) -> dict:
    t0 = time.perf_counter()
    for i in range(n):
        runtime.infer(frames[i % len(frames)])
    if sync_fn:
        sync_fn()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return {"frames_per_sec": n / elapsed, "avg_ms": elapsed / n * 1e3}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("requires CUDA")

    frames = [make_image_bytes(seed=i) for i in range(4)]

    naive = NaiveRuntime(device)
    opt = OptimizedRuntime(device)

    # Warm up.
    for _ in range(20):
        naive.infer(frames[0])
        opt.infer(frames[0])
    torch.cuda.synchronize(device)

    naive_r = run_latency(naive, frames, device)
    opt_r = run_latency(opt, frames, device, sync_fn=opt.sync)
    naive_tp = run_throughput(naive, frames, device)
    opt_tp = run_throughput(opt, frames, device, sync_fn=opt.sync)

    report = {
        "kind": "robot_runtime",
        "environment": collect_environment(device),
        "latency": {"naive": naive_r, "optimized": opt_r},
        "throughput": {"naive": naive_tp, "optimized": opt_tp},
    }
    write_report(args.output, report)

    print(f"{'runtime':12s} {'mean_ms':>8s} {'p50_ms':>8s} {'p99_ms':>8s} "
          f"{'jitter':>8s}")
    for name, r in [("naive", naive_r), ("optimized", opt_r)]:
        print(f"{name:12s} {r['mean_ms']:8.2f} {r['p50_ms']:8.2f} {r['p99_ms']:8.2f} "
              f"{r['jitter_ms']:8.2f}")
    print(f"latency speedup (p50): {naive_r['p50_ms'] / opt_r['p50_ms']:.2f}x")
    print(f"{'runtime':12s} {'fps':>8s} {'avg_ms':>8s}")
    for name, r in [("naive", naive_tp), ("optimized", opt_tp)]:
        print(f"{name:12s} {r['frames_per_sec']:8.1f} {r['avg_ms']:8.2f}")
    print(f"throughput speedup: {opt_tp['frames_per_sec'] / naive_tp['frames_per_sec']:.2f}x")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
