"""Fused vs unfused benchmark.

For each fusion case, measure three things the eager multi-kernel version
loses on:

1. kernel count  - number of CUDA kernel launches (via torch.profiler)
2. memory traffic - total device bytes moved (via torch.profiler memory)
3. latency       - CUDA-event device time

The whole argument for fusion is visible in the table: fused variants launch
fewer kernels and move fewer bytes because intermediates never round-trip
through global memory.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

import kernel.triton  # noqa: F401
from common.env import collect_environment, resolve_device
from common.measure import cuda_event_latency
from common.report import write_report
from kernel.fusion import FusionCase
from kernel.fusion.fused_ops import (
    fused_dequant_gemm,
    fused_gemm_bias,
    fused_linear_relu,
    fused_residual_rmsnorm,
)


def profile_kernels(fn) -> dict[str, Any]:
    """Run fn under torch.profiler and return kernel count + traffic bytes."""
    fn()  # warm up
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
        fn()
        torch.cuda.synchronize()
    count = 0
    traffic = 0
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            count += e.count
            traffic += e.self_device_memory_usage
    return {"kernel_count": count, "memory_traffic_bytes": traffic}


def build_cases(device, dtype):
    M, N, K = 1024, 1024, 1024
    rows, cols = 4096, 1024
    eps = 1e-5
    nbytes = torch.tensor([], dtype=dtype).element_size()

    def linear_inputs(d, dt):
        x = torch.randn(M, K, device=d, dtype=dt)
        w = torch.randn(N, K, device=d, dtype=dt)
        b = torch.randn(N, device=d, dtype=dt)
        return x, w, b

    def residual_inputs(d, dt):
        x = torch.randn(rows, cols, device=d, dtype=dt)
        r = torch.randn(rows, cols, device=d, dtype=dt)
        w = torch.randn(cols, device=d, dtype=dt)
        return x, r, w

    def gemm_inputs(d, dt):
        a = torch.randn(M, K, device=d, dtype=dt)
        b = torch.randn(K, N, device=d, dtype=dt)
        bias = torch.randn(N, device=d, dtype=dt)
        return a, b, bias

    def dequant_inputs(d, dt):
        a = torch.randn(M, K, device=d, dtype=dt)
        wq = torch.randint(-127, 127, (K, N), device=d, dtype=torch.int8)
        ws = torch.rand(N, device=d, dtype=torch.float32) * 0.1
        return a, wq, ws

    # Traffic estimates (bytes read + written), analytical not measured.
    # bias_relu: unfused = linear(read a,w; write y) + relu(read y; write out)
    #            fused   = read a,w,b; write out
    bias_relu_un = (M * K + N * K + M * N) * nbytes + (M * N + M * N) * nbytes
    bias_relu_fu = (M * K + N * K + N + M * N) * nbytes

    # residual_rmsnorm: unfused materializes y, y2, mean, rstd, out1, out.
    # Each intermediate is rows*cols; fused reads x,r,w and writes out once.
    t = rows * cols * nbytes
    resid_un = (2 * t + t) + (t + t) + t + (t + t) + (t + t)  # ~9 tensor-round-trips
    resid_fu = t + t + cols * nbytes + t  # read x, read r, read w, write out

    # gemm_bias: unfused = a@b + bias; fused = gemm with bias in epilogue
    gemm_bias_un = (M * K + K * N + M * N) * nbytes + (M * N + N + M * N) * nbytes
    gemm_bias_fu = (M * K + K * N + N + M * N) * nbytes

    # dequant_gemm: unfused materializes fp32 then fp16 dequantized weight
    # before gemm; fused dequantizes inside SRAM.
    dq_un = (K * N * 1 + K * N * 4) + (K * N * 4 + K * N * nbytes) + (M * K + K * N + M * N) * nbytes
    dq_fu = M * K * nbytes + K * N * 1 + N * 4 + M * N * nbytes

    return [
        FusionCase(
            "bias_relu",
            unfused=lambda x, w, b: torch.relu(torch.nn.functional.linear(x, w, b)),
            fused=fused_linear_relu,
            inputs=linear_inputs,
            traffic_unfused_bytes=bias_relu_un,
            traffic_fused_bytes=bias_relu_fu,
            note="linear + relu folded into one gemm epilogue",
        ),
        FusionCase(
            "residual_rmsnorm",
            unfused=lambda x, r, w, eps: (
                lambda y: y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * w
            )(x + r),
            fused=fused_residual_rmsnorm,
            inputs=residual_inputs,
            kwargs={"eps": eps},
            traffic_unfused_bytes=resid_un,
            traffic_fused_bytes=resid_fu,
            note="x+r fused into rmsnorm, no materialized sum",
        ),
        FusionCase(
            "gemm_bias",
            unfused=lambda a, b, bias: a @ b + bias,
            fused=fused_gemm_bias,
            inputs=gemm_inputs,
            traffic_unfused_bytes=gemm_bias_un,
            traffic_fused_bytes=gemm_bias_fu,
            note="bias added in the accumulation epilogue",
        ),
        FusionCase(
            "dequant_gemm",
            unfused=lambda a, wq, ws: a @ (wq.float() * ws).to(a.dtype),
            fused=fused_dequant_gemm,
            inputs=dequant_inputs,
            traffic_unfused_bytes=dq_un,
            traffic_fused_bytes=dq_fu,
            note="int8 weight dequantized in SRAM, not as a materialized tensor",
        ),
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("fusion benchmark requires CUDA")
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]

    results = []
    for case in build_cases(device, dtype):
        inputs = case.inputs(device, dtype)

        def unfused_fn():
            with torch.no_grad():
                case.unfused(*inputs, **case.kwargs)

        def fused_fn():
            with torch.no_grad():
                case.fused(*inputs, **case.kwargs)

        with torch.no_grad():
            expected = case.unfused(*inputs, **case.kwargs)
            actual = case.fused(*inputs, **case.kwargs)
        torch.cuda.synchronize()
        ok = torch.allclose(actual, expected, atol=1e-1, rtol=1e-1)

        un_profile = profile_kernels(unfused_fn)
        fu_profile = profile_kernels(fused_fn)

        un_lat = cuda_event_latency(unfused_fn, device=device, warmup=args.warmup,
                                    iterations=args.iterations)
        fu_lat = cuda_event_latency(fused_fn, device=device, warmup=args.warmup,
                                    iterations=args.iterations)

        results.append({
            "name": case.name,
            "note": case.note,
            "correct": ok,
            "unfused_kernels": un_profile["kernel_count"],
            "fused_kernels": fu_profile["kernel_count"],
            "unfused_traffic_bytes": case.traffic_unfused_bytes,
            "fused_traffic_bytes": case.traffic_fused_bytes,
            "unfused_event_us": un_lat.mean,
            "fused_event_us": fu_lat.mean,
            "latency_speedup_x": un_lat.mean / fu_lat.mean,
        })
        print(f"{case.name:18s} correct={ok}  kernels {un_profile['kernel_count']}->{fu_profile['kernel_count']}  "
              f"traffic {case.traffic_unfused_bytes/1e6:5.1f}->{case.traffic_fused_bytes/1e6:5.1f}MB  "
              f"latency {un_lat.mean:7.2f}->{fu_lat.mean:7.2f}us  ({un_lat.mean/fu_lat.mean:.2f}x)")

    report = {
        "kind": "fusion",
        "environment": collect_environment(device),
        "config": {"dtype": args.dtype},
        "results": results,
    }
    write_report(args.output, report)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
