"""Distillation benchmark: accuracy / params / latency tradeoff.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m compression.distillation.benchmark --device cuda --output /tmp/distill.json
"""

from __future__ import annotations

import argparse
import json

import torch

from common.env import collect_environment, resolve_device
from common.measure import cuda_event_latency
from common.report import write_report
from compression.distillation.distill import (
    MLP,
    run_distillation,
    temperature_sweep,
)


def measure_latency(model, device, batch=1):
    model.eval()
    x = torch.randn(batch, 784, device=device)
    with torch.no_grad():
        t = cuda_event_latency(lambda: model(x), device=device, warmup=20, iterations=200)
    return t.mean


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("distillation benchmark requires CUDA")

    r = run_distillation(device)
    sweep = temperature_sweep(device)

    teacher = MLP(784, 512, 3, 10).to(device).eval()
    student = MLP(784, 256, 2, 10).to(device).eval()
    teacher_lat = measure_latency(teacher, device)
    student_lat = measure_latency(student, device)

    report = {
        "kind": "distillation",
        "environment": collect_environment(device),
        "results": r,
        "temperature_sweep": sweep,
        "latency": {
            "teacher_us": teacher_lat,
            "student_us": student_lat,
            "latency_ratio": teacher_lat / student_lat,
        },
    }
    write_report(args.output, report)

    print(f"teacher acc: {r['teacher_acc']:.3f}  params {r['teacher_params']}")
    print(f"student direct acc: {r['student_direct_acc']:.3f}")
    print(f"student distilled acc: {r['student_distilled_acc']:.3f}  params {r['student_params']}")
    print(f"latency: teacher {teacher_lat:.1f}us vs student {student_lat:.1f}us "
          f"({teacher_lat/student_lat:.1f}x)")
    print("== temperature sweep ==")
    for s in sweep:
        print(f"  T={s['T']:5.1f}  acc={s['accuracy']:.3f}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
