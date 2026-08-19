"""Benchmark delivery semantics + idempotency for robot commands.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m robotics.distributed.benchmark --output /tmp/distributed.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from robotics.distributed.delivery import Command, deliver_at_least_once, deliver_at_most_once
from robotics.distributed.idempotency import RobotExecutor


def run_scenario(n_commands: int, loss_rate: float, idempotent: bool, strategy: str,
                 seed: int = 0) -> dict:
    commands = [Command(i, f"move_forward_{i}") for i in range(n_commands)]
    if strategy == "at_most_once":
        delivered = deliver_at_most_once(commands, loss_rate, seed)
    else:
        delivered = deliver_at_least_once(commands, loss_rate, seed)

    robot = RobotExecutor(idempotent=idempotent)
    for c in delivered:
        robot.apply(c)

    return {
        "strategy": strategy,
        "idempotent": idempotent,
        "commands_sent": n_commands,
        "deliveries_received": len(delivered),
        "robot_position": robot.position,
        "expected_position": n_commands,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    loss = 0.2
    n = 100
    results = [
        run_scenario(n, loss, idempotent=False, strategy="at_most_once"),
        run_scenario(n, loss, idempotent=False, strategy="at_least_once"),
        run_scenario(n, loss, idempotent=True, strategy="at_least_once"),
    ]
    report = {"kind": "distributed_fundamentals", "loss_rate": loss,
              "n_commands": n, "results": results}
    write_report(args.output, report)

    print(f"100 commands 'move 1m', link loss {loss:.0%} - final robot position:")
    for r in results:
        label = f"{r['strategy']}" + ("+idempotent" if r["idempotent"] else "")
        print(f"  {label:28s} deliveries={r['deliveries_received']:3d}  "
              f"position={r['robot_position']:5.0f}m (expected {r['expected_position']}m)")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
