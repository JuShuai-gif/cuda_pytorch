#!/usr/bin/env python3
"""
Benchmark: Autotuned kernels vs fixed-config vs PyTorch.

Measures:
  - Autotuned matmul vs fixed-config matmul vs torch.matmul
  - Performance variance across configs (sensitivity analysis)
  - Worst/best config ratio for each problem shape
  - Shows why autotune matters in production

Run: python 11_autotune/benchmark_autotune.py
"""

from __future__ import annotations

import time
from typing import Any

import torch
import triton

from triton_autotune_demo import autotuned_matmul, autotuned_matmul_kernel

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def _cuda_available() -> bool:
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return False
    return True


def _format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _benchmark(fn, warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def bench_autotuned_vs_fixed_vs_torch() -> None:
    """Compare autotuned matmul against fixed config and torch.matmul."""
    if not _cuda_available():
        return

    print("=" * 80)
    print("  AUTOTUNE: Matmul Performance Comparison")
    print("=" * 80)

    shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 512),
        (2048, 2048, 1024),
    ]

    fixed_config = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}
    fixed_num_warps = 4
    fixed_num_stages = 2

    results: list[list[str]] = []

    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # --- torch.matmul (cuBLAS) ---
        t_torch = _benchmark(lambda: torch.matmul(a, b))

        # --- Autotuned ---
        # First call: trigggers autotune (expensive, skip from timing)
        _ = autotuned_matmul(a, b)
        t_auto = _benchmark(lambda: autotuned_matmul(a, b))

        # --- Fixed config ---
        c_fixed = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = (
            triton.cdiv(M, fixed_config["BLOCK_M"]) * triton.cdiv(N, fixed_config["BLOCK_N"]),
            1,
            1,
        )

        # Need a non-autotuned version for fixed config
        @triton.jit
        def _fixed_matmul_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr,
        ):
            import triton.language as tl

            pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            rk = tl.arange(0, BLOCK_K)

            a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_start in range(0, K, BLOCK_K):
                k_offs = k_start + rk
                a_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
                b_mask = (k_offs[:, None] < K) & (rn[None, :] < N)
                a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
                acc += tl.dot(a_tile, b_tile)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

            c_mask = (rm[:, None] < M) & (rn[None, :] < N)
            c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            tl.store(c_ptrs, acc, mask=c_mask)

        _fixed_matmul_kernel[grid](
            a,
            b,
            c_fixed,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c_fixed.stride(0),
            c_fixed.stride(1),
            BLOCK_M=fixed_config["BLOCK_M"],
            BLOCK_N=fixed_config["BLOCK_N"],
            BLOCK_K=fixed_config["BLOCK_K"],
            GROUP_M=fixed_config["GROUP_M"],
            num_warps=fixed_num_warps,
            num_stages=fixed_num_stages,
        )

        t_fixed = _benchmark(
            lambda: _fixed_matmul_kernel[grid](
                a,
                b,
                c_fixed,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c_fixed.stride(0),
                c_fixed.stride(1),
                BLOCK_M=fixed_config["BLOCK_M"],
                BLOCK_N=fixed_config["BLOCK_N"],
                BLOCK_K=fixed_config["BLOCK_K"],
                GROUP_M=fixed_config["GROUP_M"],
                num_warps=fixed_num_warps,
                num_stages=fixed_num_stages,
            )
        )

        speedup_torch = t_torch / t_auto if t_auto > 0 else 0
        speedup_fixed = t_fixed / t_auto if t_auto > 0 else 0

        cfg = autotuned_matmul_kernel.best_config
        best = cfg.kwargs if cfg else {}

        results.append(
            [
                f"{M}x{N}x{K}",
                _format_time(t_torch),
                _format_time(t_fixed),
                _format_time(t_auto),
                f"BLOCK_M={best.get('BLOCK_M', '?')}, BLOCK_N={best.get('BLOCK_N', '?')}",
                f"{speedup_torch:.2f}x",
                f"{speedup_fixed:.2f}x",
            ]
        )

    header = [
        "Shape",
        "torch.matmul",
        "Fixed Config",
        "Autotuned",
        "Best Config",
        "vs torch",
        "vs fixed",
    ]
    if tabulate:
        print(tabulate(results, headers=header, tablefmt="grid", stralign="right"))
    else:
        for row in results:
            print(
                f"  {row[0]:<16} Torch:{row[1]:>10} Fixed:{row[2]:>10} Auto:{row[3]:>10} {row[4]} "
                f"vsT:{row[5]:>6} vsF:{row[6]:>6}"
            )


def bench_config_sensitivity() -> None:
    """Show how performance varies across configs for a fixed shape.

    This demonstrates why autotune matters: the wrong config can be
    significantly slower than the best one.
    """
    if not _cuda_available():
        return

    print("\n" + "=" * 80)
    print("  AUTOTUNE: Config Sensitivity Analysis")
    print("=" * 80)

    M, N, K = 1024, 1024, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # Sample a subset of configs
    config_variants = [
        {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_warps": 8,
            "num_stages": 4,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        },
    ]

    # Create non-autotuned kernel
    @triton.jit
    def _bench_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        import triton.language as tl

        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + rk
            a_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
            b_mask = (k_offs[:, None] < K) & (rn[None, :] < N)
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a_tile, b_tile)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=c_mask)

    times: list[tuple[dict, float]] = []
    for cfg in config_variants:
        c_out = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = (
            triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),
            1,
            1,
        )

        # Warmup
        for _ in range(5):
            _bench_kernel[grid](
                a,
                b,
                c_out,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c_out.stride(0),
                c_out.stride(1),
                BLOCK_M=cfg["BLOCK_M"],
                BLOCK_N=cfg["BLOCK_N"],
                BLOCK_K=cfg["BLOCK_K"],
                GROUP_M=cfg["GROUP_M"],
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(30):
            _bench_kernel[grid](
                a,
                b,
                c_out,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c_out.stride(0),
                c_out.stride(1),
                BLOCK_M=cfg["BLOCK_M"],
                BLOCK_N=cfg["BLOCK_N"],
                BLOCK_K=cfg["BLOCK_K"],
                GROUP_M=cfg["GROUP_M"],
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 30

        times.append((cfg, elapsed))

    times.sort(key=lambda x: x[1])

    best_t = times[0][1]
    worst_t = times[-1][1]
    ratio = worst_t / best_t if best_t > 0 else 0

    print(f"\n  Shape: {M}x{N}x{K}")
    print(f"  Configs tested: {len(times)}")
    print(f"  Best time:  {_format_time(best_t)}")
    print(f"  Worst time: {_format_time(worst_t)}")
    print(f"  Worst/Best ratio: {ratio:.2f}x")
    print(
        f"\n  {'Rank':<5} {'BLOCK_M':<10} {'BLOCK_N':<10} {'BLOCK_K':<10} "
        f"{'warps':<7} {'stages':<7} {'Time':<12} {'vs Best':<8}"
    )
    print(f"  {'-' * 70}")

    for i, (cfg, t) in enumerate(times):
        ratio_str = f"{t / best_t:.2f}x" if i > 0 else "baseline"
        print(
            f"  {i + 1:<5} {cfg['BLOCK_M']:<10} {cfg['BLOCK_N']:<10} "
            f"{cfg['BLOCK_K']:<10} {cfg['num_warps']:<7} {cfg['num_stages']:<7} "
            f"{_format_time(t):<12} {ratio_str:<8}"
        )


def bench_autotune_overhead() -> None:
    """Measure the cost of the autotune search itself."""
    if not _cuda_available():
        return

    print("\n" + "=" * 80)
    print("  AUTOTUNE: Overhead Measurement")
    print("=" * 80)

    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # Time the full first call (includes autotune search)
    t0 = time.perf_counter()
    _ = autotuned_matmul(a, b)
    torch.cuda.synchronize()
    first_call = time.perf_counter() - t0

    # Time subsequent calls (cached)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = autotuned_matmul(a, b)
    torch.cuda.synchronize()
    cached = (time.perf_counter() - t0) / 50

    print(f"  Shape: {M}x{N}x{K}")
    print(f"  Configs in search space: {len(autotuned_matmul_kernel.configs)}")
    print(f"  First call (autotune search): {_format_time(first_call)}")
    print(f"  Cached call (avg):              {_format_time(cached)}")
    print(f"  Overhead ratio: {first_call / cached:.0f}x" if cached > 0 else "")


if __name__ == "__main__":
    bench_autotuned_vs_fixed_vs_torch()
    bench_config_sensitivity()
    bench_autotune_overhead()
