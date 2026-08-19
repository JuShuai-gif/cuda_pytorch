"""Run the data-loop flywheel: failure rate drops as failures are mined and
fed back into training.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m robotics.data_loop.benchmark --output /tmp/data_loop.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from robotics.data_loop.data_loop import FAILURE_TYPES, DataLoop, Model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int, default=4)
    args = p.parse_args(argv)

    # Initial model is weak on all failure types.
    loop = DataLoop(Model({ft: 0.5 for ft in FAILURE_TYPES}))

    rounds = []
    for r in range(args.rounds):
        loop.run_robot_fleet(300)
        failure_counts = loop.mine_failures()
        rounds.append({
            "round": r + 1,
            "failure_rate": loop.failure_rate(),
            "failure_counts": failure_counts,
            "model_success_rates": dict(loop.model.success_rates),
        })
        loop.train()

    report = {"kind": "data_loop", "rounds": rounds}
    write_report(args.output, report)

    print(f"{'round':6s} {'failure_rate':>13s} {'low_light':>10s} "
          f"{'occlusion':>10s} {'novel_obj':>10s}")
    for r in rounds:
        c = r["failure_counts"]
        print(f"{r['round']:6d} {r['failure_rate']:13.1%} {c['low_light']:10d} "
              f"{c['occlusion']:10d} {c['novel_object']:10d}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
