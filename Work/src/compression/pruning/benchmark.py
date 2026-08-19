"""Pruning benchmark: FLOPs reduction vs real hardware speedup.

The central question of pruning: does removing parameters actually make
inference faster on the GPU?  Three regimes, with three different answers:

1. unstructured - zeros scattered randomly; a dense matmul still computes
   them, so ~no speedup despite large FLOPs reduction.
2. structured (row/channel) - whole rows removed, the matrix genuinely
   shrinks, so speedup tracks FLOPs reduction.
3. 2:4 sparse - hardware-native structured sparsity (Ampere+ tensor core),
   real ~2x speedup with a small accuracy cost.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m compression.pruning.benchmark --device cuda --output /tmp/prune.json
"""

from __future__ import annotations

import argparse
import json

import torch

from common.env import collect_environment, resolve_device
from common.measure import cuda_event_latency
from common.report import write_report
from compression.pruning.prune import (
    magnitude_prune_unstructured,
    sparsity,
    structured_row_prune,
    to_2to4,
)


def matmul_latency(a, b, device, warmup=20, iterations=100):
    def fn():
        a @ b
    return cuda_event_latency(fn, device=device, warmup=warmup, iterations=iterations).mean


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("pruning benchmark requires CUDA")

    torch.manual_seed(0)
    M = N = K = 1024
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w = torch.randn(K, N, device=device, dtype=torch.float16)

    base_us = matmul_latency(x, w, device)

    results = []

    # 1. Unstructured magnitude pruning at several sparsities.
    for sp in [0.5, 0.9, 0.99]:
        w_pruned, mask = magnitude_prune_unstructured(w, sp)
        sp_actual = sparsity(w_pruned)
        # Dense matmul still computes the zeros -> little/no speedup.
        us = matmul_latency(x, w_pruned, device)
        results.append({
            "method": "unstructured",
            "sparsity": sp_actual,
            "flops_fraction": 1.0 - sp_actual,
            "latency_us": us,
            "speedup_x": base_us / us,
        })

    # 2. Structured row pruning (input-channel removal).
    for sp in [0.5, 0.75]:
        w_pruned, kept = structured_row_prune(w, sp)
        x_pruned = x[:, kept]
        us = matmul_latency(x_pruned, w_pruned, device)
        results.append({
            "method": "structured_row",
            "sparsity": sp,
            "flops_fraction": 1.0 - sp,
            "latency_us": us,
            "speedup_x": base_us / us,
        })

    # 3. 2:4 structured sparsity.
    try:
        w_24 = to_2to4(w)
        us = matmul_latency(x, w_24, device)
        # Accuracy cost of the sparsification itself.
        err = ((x @ w_24).float() - (x @ w).float()).abs().max().item()
        results.append({
            "method": "2to4",
            "sparsity": 0.5,
            "flops_fraction": 0.5,
            "latency_us": us,
            "speedup_x": base_us / us,
            "max_abs_err": err,
        })
    except Exception as e:  # noqa: BLE001
        results.append({"method": "2to4", "error": str(e)[:120]})

    report = {
        "kind": "pruning",
        "environment": collect_environment(device),
        "config": {"M": M, "N": N, "K": K, "dtype": "float16",
                   "baseline_us": base_us},
        "results": results,
    }
    write_report(args.output, report)

    print(f"baseline dense matmul: {base_us:.2f}us")
    print(f"{'method':16s} {'sparsity':>8s} {'flops%':>8s} {'latency':>9s} {'speedup':>8s}")
    for r in results:
        if "error" in r:
            print(f"  2to4 error: {r['error']}")
            continue
        extra = f"  err={r['max_abs_err']:.4f}" if "max_abs_err" in r else ""
        print(f"{r['method']:16s} {r['sparsity']:8.2f} {r['flops_fraction']:8.2f} "
              f"{r['latency_us']:8.2f}us {r['speedup_x']:7.2f}x{extra}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
