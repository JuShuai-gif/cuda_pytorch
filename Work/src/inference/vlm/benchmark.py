"""Per-stage latency breakdown of the VLM pipeline.

Measures each stage separately so the bottleneck is found by data, not guess:
decode (JPEG -> PIL), preprocess (resize/normalize on CPU), H2D, vision
encoder, connector, and the language model.  Runs each stage enough times to
report stable per-call latency, and prints the share of the end-to-end time.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m inference.vlm.benchmark --device cuda --output /tmp/vlm.json
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import torch

from common.env import collect_environment, resolve_device
from common.measure import cuda_event_latency
from common.report import write_report
from inference.vlm.pipeline import (
    VLM,
    decode_image,
    make_image_bytes,
    preprocess,
)


def cpu_stage_time(fn, iterations=100):
    fn()  # warm
    t0 = perf_counter()
    for _ in range(iterations):
        fn()
    return (perf_counter() - t0) / iterations * 1e3  # ms


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    device = resolve_device(args.device)

    img_bytes = make_image_bytes()
    img = decode_image(img_bytes)
    x_cpu = preprocess(img)  # (3, 224, 224)

    decode_ms = cpu_stage_time(lambda: decode_image(img_bytes), iterations=200)
    preproc_ms = cpu_stage_time(lambda: preprocess(img), iterations=200)

    model = VLM().to(device).eval()

    # H2D: CPU -> device.
    x_dev = x_cpu.to(device)

    def h2d_fn():
        x_cpu.to(device)

    h2d_ms = cuda_event_latency(h2d_fn, device=device, warmup=50, iterations=200).mean / 1e3

    x = x_dev.unsqueeze(0)

    def vision_fn():
        model.vision_encode(x)

    vision_ms = cuda_event_latency(vision_fn, device=device, warmup=10, iterations=50).mean / 1e3

    # Connector is fused into vision_encode; measure llm separately on the
    # vision token sequence.
    with torch.no_grad():
        vision_tokens = model.vision_encode(x)

    def llm_fn():
        model.llm_forward(vision_tokens)

    llm_ms = cuda_event_latency(llm_fn, device=device, warmup=10, iterations=50).mean / 1e3

    stages = [
        {"stage": "decode", "device": "cpu", "ms": decode_ms},
        {"stage": "preprocess", "device": "cpu", "ms": preproc_ms},
        {"stage": "h2d", "device": "h2d", "ms": h2d_ms},
        {"stage": "vision_encoder", "device": "gpu", "ms": vision_ms},
        {"stage": "llm", "device": "gpu", "ms": llm_ms},
    ]
    total = sum(s["ms"] for s in stages)
    for s in stages:
        s["share"] = s["ms"] / total

    report = {"kind": "vlm_inference", "environment": collect_environment(device),
              "total_ms": total, "stages": stages}
    write_report(args.output, report)

    print(f"{'stage':16s} {'device':7s} {'ms':>9s} {'share':>7s}")
    for s in stages:
        print(f"{s['stage']:16s} {s['device']:7s} {s['ms']:9.3f} {s['share']:7.1%}")
    print(f"total: {total:.3f} ms")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
