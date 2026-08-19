"""Demonstrate the accuracy-vs-business-metric misjudgment in A/B tests.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m serving.ab_test.benchmark --output /tmp/ab.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from serving.ab_test.ab_test import ModelVariant, summarize


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    deadline = 50.0  # 50ms control loop (20 Hz)

    # A: slower but more accurate; B: faster but slightly less accurate.
    model_a = ModelVariant("A_slow_accurate", accuracy=0.95, latency_ms=80.0, failure_rate=0.01)
    model_b = ModelVariant("B_fast_less_accurate", accuracy=0.90, latency_ms=15.0, failure_rate=0.01)

    a = summarize(model_a, 10000, deadline)
    b = summarize(model_b, 10000, deadline)

    report = {"kind": "ab_test", "deadline_ms": deadline, "models": [a, b]}
    write_report(args.output, report)

    print(f"robot task (deadline {deadline}ms) - which model do you ship?")
    print(f"{'model':22s} {'accuracy':>9s} {'latency':>8s} {'success_rate':>12s}")
    for m in [a, b]:
        print(f"{m['name']:22s} {m['accuracy']:9.1%} {m['latency_ms']:7.0f}ms "
              f"{m['robot_success_rate']:12.1%}")
    print()
    print(f"accuracy says: ship {a['name'] if a['accuracy'] > b['accuracy'] else b['name']}")
    print(f"robot success says: ship {a['name'] if a['robot_success_rate'] > b['robot_success_rate'] else b['name']}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
