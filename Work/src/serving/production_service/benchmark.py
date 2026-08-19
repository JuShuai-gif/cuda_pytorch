"""Overload and reliability demonstration.

Simulates a GPU serving at a fixed rate while requests arrive in a burst far
above capacity, and shows what each protection mechanism does to the latency
distribution and to the drop rate.  Then demonstrates a circuit breaker
tripping on a failing downstream.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m serving.production_service.benchmark --output /tmp/prod.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque

from common.report import write_report
from serving.production_service import CircuitBreaker, LoadShedder, TokenBucket


def simulate_burst(service_ms: float, arrival_times: list[float], *,
                   queue_capacity: int | None = None,
                   rate_limiter: TokenBucket | None = None) -> dict:
    """Process arrivals through a single worker; return latency + drop stats."""
    queue = deque()
    latencies = []
    dropped = 0
    admitted = 0
    t = 0.0
    i = 0

    while i < len(arrival_times) or queue:
        # Admit arrivals that occur at or before time t.
        while i < len(arrival_times) and arrival_times[i] <= t:
            if rate_limiter is not None and not rate_limiter.allow():
                dropped += 1
                i += 1
                continue
            if queue_capacity is not None and len(queue) >= queue_capacity:
                dropped += 1
                i += 1
                continue
            queue.append(arrival_times[i])
            admitted += 1
            i += 1

        if queue:
            arrival = queue.popleft()
            start = max(t, arrival)
            finish = start + service_ms
            latencies.append((finish - arrival) / 1000.0)  # seconds
            t = finish
        else:
            # No work: jump to the next arrival.
            if i < len(arrival_times):
                t = arrival_times[i]
            else:
                break

    latencies.sort()
    n = len(latencies)
    return {
        "admitted": admitted,
        "dropped": dropped,
        "drop_rate": dropped / (admitted + dropped) if (admitted + dropped) else 0,
        "p50_s": latencies[n // 2] if n else 0,
        "p99_s": latencies[min(n - 1, int(n * 0.99))] if n else 0,
        "max_s": latencies[-1] if n else 0,
    }


def burst_arrivals(rate_per_s: float, duration_s: float) -> list[float]:
    """A uniform burst of arrivals at `rate_per_s` for `duration_s` seconds."""
    n = int(rate_per_s * duration_s)
    return [i / rate_per_s for i in range(n)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    service_ms = 1.0          # GPU serves 1 req/ms = 1000 req/s
    arrivals = burst_arrivals(5000, 2.0)  # 10000 requests in 2s (5x overload)

    unprotected = simulate_burst(service_ms, arrivals)
    shed = simulate_burst(service_ms, arrivals, queue_capacity=100)
    limited = simulate_burst(service_ms, arrivals, queue_capacity=100,
                             rate_limiter=TokenBucket(rate=1000, capacity=100))

    report = {
        "kind": "production_service",
        "config": {"service_ms": service_ms, "arrival_rate": 5000,
                   "duration_s": 2.0, "total_requests": len(arrivals)},
        "unprotected": unprotected,
        "load_shedding": shed,
        "rate_limit_plus_shedding": limited,
    }
    write_report(args.output, report)

    print("== overload: 10000 req in 2s vs 1000 req/s GPU ==")
    for name, r in [("unprotected", unprotected), ("load_shedding", shed),
                    ("rate_limit+shedding", limited)]:
        print(f"  {name:20s} admitted={r['admitted']:6d} dropped={r['dropped']:6d} "
              f"p50={r['p50_s']:.3f}s p99={r['p99_s']:.3f}s max={r['max_s']:.3f}s")

    # Circuit breaker demo: a downstream that fails N times in a row.
    cb = CircuitBreaker(fail_threshold=3, reset_timeout=5.0)
    states = []
    import time
    for k in range(10):
        if not cb.allow():
            states.append("reject(open)")
            continue
        if k < 5:  # first 5 calls fail
            cb.record_failure()
            states.append(f"fail->{cb.state}")
        else:
            cb.record_success()
            states.append(f"ok->{cb.state}")
    print("== circuit breaker (downstream fails 5x) ==")
    print("  " + "  ".join(states))
    report["circuit_breaker_trace"] = states
    write_report(args.output, report, overwrite=True)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
