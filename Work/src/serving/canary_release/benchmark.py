"""Benchmark canary release: rollback on regression vs blind 100% deploy.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m serving.canary_release.benchmark --output /tmp/canary.json
"""

from __future__ import annotations

import argparse
import json
import random

from common.report import write_report
from serving.canary_release.canary import CanaryController, ModelVersion


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    stable = ModelVersion("V1", error_rate=0.01, latency_ms=10.0)

    # Scenario 1: V2 has an accuracy regression (8x error rate).
    acc_reg = ModelVersion("V2-acc-regression", error_rate=0.08, latency_ms=10.0)
    ctrl = CanaryController(stable, acc_reg)
    acc_trace = ctrl.run(random.Random(0))

    # Scenario 2: V2 has a latency regression (5x slower).
    lat_reg = ModelVersion("V2-lat-regression", error_rate=0.01, latency_ms=50.0)
    ctrl2 = CanaryController(stable, lat_reg)
    lat_trace = ctrl2.run(random.Random(1))

    # Scenario 3: healthy V2 (no regression) -> reaches 100%.
    healthy = ModelVersion("V2-healthy", error_rate=0.01, latency_ms=9.0)
    ctrl3 = CanaryController(stable, healthy)
    healthy_trace = ctrl3.run(random.Random(2))

    report = {
        "kind": "canary_release",
        "accuracy_regression": acc_trace,
        "latency_regression": lat_trace,
        "healthy": healthy_trace,
    }
    write_report(args.output, report)

    def show(title, trace):
        print(f"== {title} ==")
        for t in trace:
            print(f"  share={t['stage_share']:4.0%}  err {t['stable_err']:.1%}->{t['candidate_err']:.1%}  "
                  f"lat {t['stable_lat_ms']:.0f}->{t['candidate_lat_ms']:.0f}ms  -> {t['action']}")

    show("V2 accuracy regression", acc_trace)
    show("V2 latency regression", lat_trace)
    show("V2 healthy", healthy_trace)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
