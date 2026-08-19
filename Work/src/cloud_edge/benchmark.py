"""Run the cloud-edge cooperation simulation.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m cloud_edge.benchmark --output /tmp/cloud_edge.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from cloud_edge.simulate import simulate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    result = simulate()
    write_report(args.output, {"kind": "cloud_edge", **result})

    print("== cloud-edge cooperation ==")
    for e in result["events"]:
        print(f"  {e}")
    print("== robot versions ==")
    for rid, v in result["robot_versions"].items():
        print(f"  {rid}: {v}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
