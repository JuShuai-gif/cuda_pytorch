"""Parameter sweep for Triton kernels (BLOCK size and num_warps).

The point of a sweep is to show *why* a configuration is fast or slow, not just
*that* one is faster.  This script sweeps the gemm tile sizes and num_warps and
reports device time for each combination, so the occupancy / register / DRAM
tradeoffs can be read off the table.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m kernel.triton.sweep --device cuda --dtype float16 --output /tmp/sweep.json
"""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl

import kernel.triton  # noqa: F401
from common.env import collect_environment, resolve_device, resolve_dtype
from common.measure import cuda_event_latency
from common.report import write_report


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def run(a, b, c, *, bm, bn, bk, nw):
    M, K = a.shape
    N = b.shape[1]
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type != "cuda":
        raise RuntimeError("sweep requires CUDA")

    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    c = torch.empty(M, N, device=device, dtype=dtype)

    configs = [
        (64, 64, 32, 4),
        (128, 64, 32, 4),
        (64, 64, 64, 4),
        (128, 128, 32, 8),
        (64, 128, 64, 4),
        (128, 128, 64, 8),
    ]
    results = []
    for bm, bn, bk, nw in configs:
        def fn():
            run(a, b, c, bm=bm, bn=bn, bk=bk, nw=nw)
        s = cuda_event_latency(fn, device=device, warmup=args.warmup,
                               iterations=args.iterations)
        results.append({
            "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw,
            "event_us_mean": s.mean, "event_us_p99": s.p99,
        })
        print(f"BM={bm:4d} BN={bn:4d} BK={bk:4d} nw={nw}  ->  {s.mean:8.2f}us  p99 {s.p99:8.2f}us")

    report = {
        "kind": "triton_sweep",
        "environment": collect_environment(device),
        "config": {"M": M, "N": N, "K": K, "dtype": str(dtype)},
        "results": results,
    }
    write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
