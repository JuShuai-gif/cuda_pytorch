"""Benchmark the three autoscaling metrics under a load spike.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m serving.autoscaling.benchmark --output /tmp/autoscale.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from serving.autoscaling.autoscaler import simulate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    results = {}
    for metric in ["cpu", "queue", "latency"]:
        results[metric] = simulate(metric)

    report = {"kind": "autoscaling", "results": results}
    write_report(args.output, report)

    print("load spike: 100 -> 500 req/s at t=30 (GPU worker serves 100 req/s)")
    print(f"{'metric':10s} {'final_workers':>13s} {'mean_workers':>12s} "
          f"{'dropped':>8s} {'mean_lat_s':>11s}")
    for metric, r in results.items():
        print(f"{metric:10s} {r['final_workers']:13d} {r['mean_workers']:12.2f} "
              f"{r['total_dropped']:8d} {r['mean_latency_s']:11.3f}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
