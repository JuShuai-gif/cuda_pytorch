"""Fault injection + watchdog recovery demonstration.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m robotics.reliability.benchmark --output /tmp/reliability.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from robotics.reliability.faults import FAULTS
from robotics.reliability.watchdog import InferenceProcess, Watchdog


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    # Fault 1: crash on the 3rd call.
    wd = Watchdog(lambda: InferenceProcess(crash_on_call=3))
    outcomes = [wd.guarded_infer() for _ in range(6)]
    crash_report = {
        "outcomes": outcomes,
        "restarts": wd.restarts,
        "fallbacks": wd.fallbacks,
    }

    # Fault 2: hang (inference never returns) -> fallback.
    wd2 = Watchdog(lambda: InferenceProcess(hang_seconds=10.0), timeout_s=0.1)
    hang_report = {
        "outcome": wd2.guarded_infer(),
        "fallbacks": wd2.fallbacks,
    }

    # The incident playbook (subset).
    fault_names = ["gpu_oom", "model_load_failure", "thermal_throttling"]
    playbook = {n: {k: getattr(FAULTS[n], k) for k in
                    ["symptom", "first_evidence", "root_cause", "recovery", "fix"]}
                for n in fault_names}

    report = {
        "kind": "reliability",
        "crash_recovery": crash_report,
        "hang_recovery": hang_report,
        "playbook": playbook,
    }
    write_report(args.output, report)

    print("== crash fault (crash on 3rd call) ==")
    print(f"  outcomes: {crash_report['outcomes']}")
    print(f"  restarts={crash_report['restarts']} fallbacks={crash_report['fallbacks']}")
    print("== hang fault (inference hangs) ==")
    print(f"  outcome: {hang_report['outcome']}  fallbacks={hang_report['fallbacks']}")
    print("== playbook sample (gpu_oom) ==")
    oom = playbook["gpu_oom"]
    print(f"  symptom={oom['symptom']}")
    print(f"  root_cause={oom['root_cause']}")
    print(f"  fix={oom['fix']}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
