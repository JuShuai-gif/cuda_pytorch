"""Benchmark OTA: healthy upgrade and each fault scenario.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m cloud_edge.ota.benchmark --output /tmp/ota.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from cloud_edge.ota.ota import ModelArtifact, ModelRegistry, RobotOTA


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    v2 = ModelArtifact.make("v2", b"model_v2_weights")
    registry = ModelRegistry([v2])

    scenarios = []

    def run(name, **faults):
        robot = RobotOTA("v1", b"model_v1_weights", disk_capacity=1024)
        result = robot.update(registry, "v2", **faults)
        scenarios.append({
            "scenario": name,
            "result": result,
            "final_version": robot.current_version,
            "log": list(robot.log),
        })
        return robot

    run("healthy")
    run("download_interrupt", download_fails=True)
    run("corrupted_artifact", corrupt=True)
    run("disk_full", disk_too_small=True)
    run("load_failure", load_fails=True)

    write_report(args.output, {"kind": "ota", "scenarios": scenarios})

    print(f"{'scenario':20s} {'result':22s} {'final_version':15s}")
    for s in scenarios:
        print(f"{s['scenario']:20s} {s['result']:22s} {s['final_version']:15s}")
    print()
    print("== load_failure log (shows rollback) ==")
    for s in scenarios:
        if s["scenario"] == "load_failure":
            for line in s["log"]:
                print(f"  {line}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
