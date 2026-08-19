"""Benchmark robot policy inference: control loop + batch=1.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m robotics.policy_inference.benchmark --device cuda --output /tmp/robot.json
"""

from __future__ import annotations

import argparse
import json

import torch

from common.env import collect_environment, resolve_device
from common.report import write_report
from robotics.policy_inference.batch1 import compare_batch1
from robotics.policy_inference.pipeline import VLAPolicy
from robotics.policy_inference.realtime import run_control_loop


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    p.add_argument("--cycles", type=int, default=200)
    p.add_argument("--deadline", type=float, default=15.0)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("robot benchmark requires CUDA")

    model = VLAPolicy().to(device).eval()

    # Control loop without and with injected CPU jitter.
    clean = run_control_loop(model, device, cycles=args.cycles, deadline_ms=args.deadline,
                             inject_cpu_jitter=False)
    jitter = run_control_loop(model, device, cycles=args.cycles, deadline_ms=args.deadline,
                              inject_cpu_jitter=True)
    batch1 = compare_batch1(device)

    report = {
        "kind": "robot_policy_inference",
        "environment": collect_environment(device),
        "control_loop_clean": clean,
        "control_loop_with_jitter": jitter,
        "batch1": batch1,
    }
    write_report(args.output, report)

    print("== control loop (clean) ==")
    print(f"  mean={clean['mean_ms']:.2f}ms p50={clean['p50_ms']:.2f}ms "
          f"p99={clean['p99_ms']:.2f}ms  miss={clean['deadline_miss_rate']:.1%}")
    print("== control loop (with CPU jitter) ==")
    print(f"  mean={jitter['mean_ms']:.2f}ms p50={jitter['p50_ms']:.2f}ms "
          f"p99={jitter['p99_ms']:.2f}ms  jitter={jitter['jitter_ms']:.2f}ms  "
          f"miss={jitter['deadline_miss_rate']:.1%}")
    print("== batch=1 naive vs CUDA Graph ==")
    print(f"  naive wall={batch1['naive_wall_us']:.1f}us event={batch1['naive_event_us']:.1f}us")
    print(f"  graph wall={batch1['graph_wall_us']:.1f}us event={batch1['graph_event_us']:.1f}us  "
          f"speedup={batch1['wall_speedup_x']:.2f}x")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
