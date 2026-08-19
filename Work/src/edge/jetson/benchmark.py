"""Sustained-load thermal/power benchmark.

Runs a continuous GPU workload while sampling tegrastats, then reports how
temperature and power evolve: the idle baseline, the peak under load, and the
steady state.  This is what distinguishes "runs fine for 10 seconds" from
"stable for 24 hours".

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m edge.jetson.benchmark --seconds 30 --output /tmp/edge.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from common.report import write_report
from edge.jetson.monitor import TegrastatsSampler
from edge.jetson.platform import platform_profile
from edge.jetson.workload import run_sustained_gpu


def summarize(samples: list[dict], key: str):
    vals = [s[key] for s in samples if key in s]
    if not vals:
        return None
    return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals),
            "last": vals[-1]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=int, default=30)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    profile = platform_profile()

    # Idle baseline (2 seconds of no load).
    sampler = TegrastatsSampler()
    sampler.start()
    time.sleep(2.0)
    idle = sampler.snapshot()
    sampler.stop()

    # Sustained load with sampling.
    sampler = TegrastatsSampler()
    sampler.start()
    n_iters = run_sustained_gpu(args.seconds)
    time.sleep(1.0)  # let the last samples land
    loaded = sampler.snapshot()
    sampler.stop()

    def report(samples):
        return {
            "gpu_temp": summarize(samples, "gpu_temp_c"),
            "cpu_temp": summarize(samples, "cpu_temp_c"),
            "gpu_power_mw": summarize(samples, "gpu_power_mw"),
            "total_power_mw": summarize(samples, "total_power_mw"),
            "cpu_freq_mhz": summarize(samples, "cpu_freq_mhz"),
        }

    out = {
        "kind": "edge_jetson",
        "platform": profile,
        "config": {"seconds": args.seconds, "gpu_iterations": n_iters},
        "idle": report(idle),
        "sustained_load": report(loaded),
    }
    write_report(args.output, out)

    print("== platform ==")
    print(f"  arch={profile['arch']} gpu={profile['gpu']} sm={profile['gpu_sm_count']} "
          f"unified_memory={profile['unified_memory']}")
    print(f"  power_mode={profile['power_mode']}")
    print("== idle vs sustained load ==")
    for label, r in [("idle", out["idle"]), ("load", out["sustained_load"])]:
        gt = r["gpu_temp"] or {}
        pt = r["total_power_mw"] or {}
        print(f"  {label:6s} gpu_temp={gt.get('mean', 0):.1f}C  "
              f"total_power={pt.get('mean', 0):.0f}mW")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
