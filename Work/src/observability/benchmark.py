"""Demonstrate distributed tracing + metrics + logs across the robot stack.

Simulates requests flowing Cloud -> Edge -> Robot -> Model, recording spans,
metrics and logs all correlated by request_id, then (1) reconstructs one
request's trace to see where its latency went, and (2) aggregates metrics.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m observability.benchmark --output /tmp/observability.json
"""

from __future__ import annotations

import argparse
import json
import random
import time

from common.report import write_report
from observability.observability import Metrics, StructuredLogger, Tracer


def serve_request(tracer, metrics, logger, request_id: str, task_id: str,
                  robot_id: str, *, slow_model: bool = False):
    # Cloud scheduling span.
    with tracer.span(request_id, "cloud.schedule", task_id=task_id):
        time.sleep(0.001)

    # Edge forwarding span.
    with tracer.span(request_id, "edge.forward", robot_id=robot_id):
        time.sleep(0.001)

    # Robot model inference span (the likely bottleneck).
    model_ms = 0.020 if slow_model else 0.003
    with tracer.span(request_id, "robot.model_infer", model_version="v2"):
        time.sleep(model_ms)

    total = tracer.total_ms(request_id)
    metrics.record("latency_ms", total)
    metrics.record("model_latency_ms", model_ms * 1e3)
    if slow_model:
        metrics.record("error_rate", 1.0)
        logger.log(request_id, "WARN", "slow model inference", robot_id=robot_id)
    else:
        metrics.record("error_rate", 0.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    tracer = Tracer()
    metrics = Metrics()
    logger = StructuredLogger()

    n = 200
    for i in range(n):
        slow = (i % 50 == 0)  # every 50th request is slow
        serve_request(tracer, metrics, logger,
                      request_id=f"req_{i}", task_id=f"task_{i % 20}",
                      robot_id=f"robot_{i % 10}", slow_model=slow)

    # Reconstruct the trace of a slow request.
    slow_id = "req_50"
    slow_trace = [{"name": s.name, "duration_ms": round(s.duration_ms, 3),
                   "metadata": s.metadata} for s in tracer.trace(slow_id)]

    report = {
        "kind": "observability",
        "latency_summary": metrics.summary("latency_ms"),
        "error_rate": metrics.summary("error_rate")["mean"],
        "slow_request_trace": slow_trace,
        "log_entries": len(logger.entries),
    }
    write_report(args.output, report)

    print("== metrics ==")
    s = metrics.summary("latency_ms")
    print(f"  latency: mean={s['mean']:.2f}ms p50={s['p50']:.2f}ms p99={s['p99']:.2f}ms")
    print(f"  error_rate: {metrics.summary('error_rate')['mean']:.1%}")
    print(f"== slow request {slow_id} trace ==")
    for sp in slow_trace:
        print(f"  {sp['name']:20s} {sp['duration_ms']:6.2f}ms  {sp['metadata']}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
