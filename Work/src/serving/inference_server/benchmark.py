"""Benchmark batching strategies: throughput vs latency.

Sends a burst of concurrent requests (threads) against the server under each
batching strategy and reports achieved throughput and per-request latency, so
the tradeoff - batching raises throughput but raises latency, and queue depth
adds waiting - is visible in real numbers.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m serving.inference_server.benchmark --device cuda --output /tmp/server.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import torch

from common.env import collect_environment, resolve_device
from common.report import write_report
from serving.inference_server.server import InferenceServer, make_model


def run_requests(server: InferenceServer, x: torch.Tensor, n: int, concurrency: int) -> dict:
    latencies = []
    lock = threading.Lock()
    idx = iter(range(n))

    def worker():
        while True:
            try:
                i = next(idx)
            except StopIteration:
                return
            t0 = time.perf_counter()
            try:
                server.infer(x)
            except (RuntimeError, TimeoutError):
                continue
            with lock:
                latencies.append((time.perf_counter() - t0) * 1e3)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    latencies.sort()
    n_done = len(latencies)
    return {
        "requests": n_done,
        "throughput_req_per_s": n_done / elapsed,
        "mean_ms": sum(latencies) / n_done if n_done else 0,
        "p50_ms": latencies[n_done // 2] if n_done else 0,
        "p99_ms": latencies[min(n_done - 1, int(n_done * 0.99))] if n_done else 0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("inference server benchmark requires CUDA")

    x = torch.randn(512, device=device)  # (hidden,) input

    results = {}
    for strategy in ["no_batch", "static", "dynamic"]:
        server = InferenceServer(make_model(), device, strategy=strategy,
                                 max_batch=8, max_wait=0.005)
        results[strategy] = run_requests(server, x, args.n, args.concurrency)
        server.stop()

    report = {"kind": "inference_server", "environment": collect_environment(device),
              "config": {"n": args.n, "concurrency": args.concurrency},
              "results": results}
    write_report(args.output, report)

    print(f"{'strategy':12s} {'throughput':>12s} {'mean_ms':>9s} {'p50_ms':>8s} {'p99_ms':>8s}")
    for s, r in results.items():
        print(f"{s:12s} {r['throughput_req_per_s']:11.0f}/s {r['mean_ms']:9.2f} "
              f"{r['p50_ms']:8.2f} {r['p99_ms']:8.2f}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
