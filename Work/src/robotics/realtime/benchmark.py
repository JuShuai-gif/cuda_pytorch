"""Compare control latency distributions: constant vs jittery.

The headline: a jittery 15ms-mean latency (p99 = 200ms) makes a control loop
unusable even though the mean is low, because occasional long delays drive
large tracking errors.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m robotics.realtime.benchmark --output /tmp/realtime.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from robotics.realtime.control import simulate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    target = 1.0
    dt = 0.01          # 10ms per step
    n = 2000
    settle = 400       # first 400 steps at low latency to settle

    # After settling, apply each latency pattern.
    constant_low = simulate(target, [1] * n, dt=dt, settle_start=settle)   # 10ms
    constant_high = simulate(target, [1] * settle + [20] * (n - settle),
                             dt=dt, settle_start=settle)                   # 200ms
    # Jitter: mostly 10ms, but a 50ms stall (5 steps of 200ms) every 40 steps.
    post = [1] * (n - settle)
    for i in range(40, len(post), 40):
        for kk in range(min(5, len(post) - i)):
            post[i + kk] = 20
    jittery = simulate(target, [1] * settle + post, dt=dt, settle_start=settle)

    report = {
        "kind": "realtime",
        "config": {"target": target, "dt_s": dt},
        "constant_low_10ms": constant_low,
        "constant_high_200ms": constant_high,
        "jittery_mean15ms_p99_200ms": jittery,
    }
    write_report(args.output, report)

    print("control target = 1.0, dt = 10ms (errors after settling)")
    print(f"{'latency':26s} {'rms_error':>11s} {'max_error':>11s} {'final':>9s} {'settled':>8s}")
    for name, r in [("constant 10ms", constant_low),
                    ("constant 200ms", constant_high),
                    ("jittery (mean ~15ms)", jittery)]:
        print(f"{name:26s} {r['rms_error']:11.4f} {r['max_error']:11.4f} "
              f"{r['final_error']:9.4f} {str(r['settled']):>8s}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
