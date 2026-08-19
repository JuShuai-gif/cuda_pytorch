"""Unified quantization benchmark: granularity, formats, PTQ.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m compression.quantization.benchmark --device cuda --output /tmp/quant.json
"""

from __future__ import annotations

import argparse
import json

import torch

import kernel.triton  # noqa: F401  (TRITON_PTXAS_BLACKWELL_PATH)
from common.env import collect_environment, resolve_device
from common.report import write_report
from compression.quantization.awq import awq_experiment, make_outlier_weight as make_weight_outlier
from compression.quantization.dtypes import format_metadata, gemm_precision_speed
from compression.quantization.ptq import run_accuracy
from compression.quantization.quantize import granularity_error, make_outlier_weight
from compression.quantization.smoothquant import (
    make_outlier_activation,
    smoothquant_experiment,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("quantization benchmark requires CUDA")

    w = make_outlier_weight()(device, torch.float32)
    granularity = [r.__dict__ for r in granularity_error(w)]

    formats = format_metadata()
    gemm = gemm_precision_speed(device)
    ptq = run_accuracy(device)

    # SmoothQuant + AWQ on an outlier-activation linear layer.
    M, K, N = 4096, 1024, 1024
    x = make_outlier_activation(M, K)(device, torch.float32)
    w_lin = torch.randn(K, N, device=device, dtype=torch.float32) * 0.05
    smoothquant = smoothquant_experiment(x, w_lin)
    x_awq = torch.randn(M, K, device=device, dtype=torch.float32)
    w_awq = make_weight_outlier(K, N)(device, torch.float32)
    awq = awq_experiment(x_awq, w_awq)

    report = {
        "kind": "quantization",
        "environment": collect_environment(device),
        "granularity_error": granularity,
        "format_metadata": formats,
        "gemm_precision_speed": gemm,
        "ptq_weight_only_int8": ptq,
        "smoothquant": smoothquant,
        "awq": awq,
    }
    write_report(args.output, report)

    print("== granularity error (outlier weight) ==")
    for r in granularity:
        print(f"  {r['granularity']:16s} max_abs_err={r['max_abs_err']:.4f}  mse={r['mse']:.6f}")

    print("== gemm format precision/speed ==")
    for r in gemm:
        print(f"  {r['format']:12s} max_abs_err={r['max_abs_err']:.4f}  "
              f"{r['event_us']:8.2f}us  speedup={r['speedup_vs_fp32_ieee']:.2f}x")

    print("== weight-only int8 PTQ ==")
    print(f"  max_abs_diff={ptq['max_abs_diff']:.4f}  mse={ptq['mse']:.6f}  "
          f"size {ptq['fp16_weight_bytes']} -> {ptq['int8_weight_bytes']} bytes "
          f"({ptq['size_ratio']:.2f}x)")
    print("== SmoothQuant ==")
    print(f"  direct_err={smoothquant['direct_max_abs_err']:.4f}  "
          f"smooth_err={smoothquant['smooth_max_abs_err']:.4f}  "
          f"({smoothquant['error_reduction_x']:.1f}x)")
    print("== AWQ ==")
    print(f"  naive_weighted={awq['naive_weighted_error']:.4f}  "
          f"awq_weighted={awq['awq_weighted_error']:.4f}  "
          f"({awq['error_reduction_x']:.2f}x, m={awq['best_multiplier']:.2f})")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
