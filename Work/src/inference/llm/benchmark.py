"""Prefill vs decode benchmark.

Measures a single transformer layer's prefill (full sequence) and decode (one
token + KV cache) latency across sequence lengths, and reports the achieved
FLOPs and arithmetic intensity so the compute-bound vs memory-bound split is
visible in real numbers, not just theory.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m inference.llm.benchmark --device cuda --output /tmp/llm.json
"""

from __future__ import annotations

import argparse
import json

import torch

from common.env import collect_environment, resolve_device
from common.measure import cuda_event_latency
from common.report import write_report
from inference.llm.model import TransformerLayer
from inference.llm.roofline import decode_metrics, prefill_metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("LLM benchmark requires CUDA")

    d, B = args.d, args.batch
    layer = TransformerLayer(d).to(device).eval()

    results = []
    for S in [128, 512, 2048, 8192]:
        x = torch.randn(B, S, d, device=device)
        # Prefill: full sequence.
        def prefill_fn():
            layer.prefill(x)
        p_us = cuda_event_latency(prefill_fn, device=device, warmup=10, iterations=50).mean

        # Decode: one token + KV cache (S accumulated tokens).
        with torch.no_grad():
            _, k, v = layer.prefill(x)
        xt = torch.randn(B, 1, d, device=device)
        def decode_fn():
            layer.decode(xt, k, v)
        q_us = cuda_event_latency(decode_fn, device=device, warmup=10, iterations=50).mean

        pm = prefill_metrics(1, d, S, B, 0)  # single layer
        qm = decode_metrics(1, d, S, B, 0)

        results.append({
            "seq_len": S,
            "prefill_us": p_us,
            "decode_us": q_us,
            "prefill_tflops": pm.flops / (p_us * 1e-6) / 1e12,
            "decode_tflops": qm.flops / (q_us * 1e-6) / 1e12,
            "prefill_ai": pm.arithmetic_intensity,
            "decode_ai": qm.arithmetic_intensity,
            "kv_cache_mb": qm.kv_cache_bytes / 1e6,
        })

    report = {"kind": "llm_inference", "environment": collect_environment(device),
              "config": {"d": d, "batch": B}, "results": results}
    write_report(args.output, report)

    print(f"{'seq':>6s} {'prefill_us':>11s} {'decode_us':>10s} {'p_tflops':>9s} "
          f"{'d_tflops':>9s} {'p_ai':>7s} {'d_ai':>7s} {'kv_mb':>7s}")
    for r in results:
        print(f"{r['seq_len']:6d} {r['prefill_us']:11.1f} {r['decode_us']:10.1f} "
              f"{r['prefill_tflops']:9.2f} {r['decode_tflops']:9.2f} "
              f"{r['prefill_ai']:7.1f} {r['decode_ai']:7.1f} {r['kv_cache_mb']:7.1f}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
