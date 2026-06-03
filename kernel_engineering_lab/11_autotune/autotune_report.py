#!/usr/bin/env python3
"""
Autotune report generation.

Runs all autotuned kernels across a range of shapes, collects best
configurations and performance data, and exports a comprehensive report
as JSON and Markdown table.

Run: python 11_autotune/autotune_report.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from layernorm_autotune import (
    autotuned_layernorm,
    autotuned_layernorm_kernel,
    autotuned_rmsnorm,
    autotuned_rmsnorm_kernel,
)
from softmax_autotune import autotuned_softmax, autotuned_softmax_kernel
from triton_autotune_demo import autotuned_matmul, autotuned_matmul_kernel


@dataclass
class KernelReport:
    kernel_name: str
    shape: tuple[int, ...]
    best_config: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    pytorch_time_ms: float = 0.0
    speedup: float = 1.0
    max_error: float = 0.0


def _benchmark(fn, warmup: int = 5, repeat: int = 20) -> float:
    """Time a function in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / repeat * 1000.0  # ms


def generate_autotune_report(benchmark_dir: str = "../benchmarks") -> list[KernelReport]:
    """Run all autotuned kernels and collect performance data.

    Args:
        benchmark_dir: Directory for saving output files.

    Returns:
        List of KernelReport objects.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot generate report.")
        return []

    os.makedirs(benchmark_dir, exist_ok=True)

    reports: list[KernelReport] = []

    # ------------------------------------------------------------------
    # Matmul
    # ------------------------------------------------------------------
    matmul_shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 512),
        (2048, 2048, 1024),
    ]

    for M, N, K in matmul_shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Run autotuned
        out = autotuned_matmul(a, b)  # includes autotune on first call

        ref = torch.matmul(a, b)
        err = (out - ref).abs().max().item()

        t_auto = _benchmark(lambda: autotuned_matmul(a, b))
        t_ref = _benchmark(lambda: torch.matmul(a, b))

        cfg = autotuned_matmul_kernel.best_config
        best = cfg.kwargs if cfg else {}

        reports.append(
            KernelReport(
                kernel_name="autotuned_matmul",
                shape=(M, N, K),
                best_config={
                    "BLOCK_M": best.get("BLOCK_M"),
                    "BLOCK_N": best.get("BLOCK_N"),
                    "BLOCK_K": best.get("BLOCK_K"),
                    "num_warps": best.get("num_warps"),
                    "num_stages": best.get("num_stages"),
                },
                execution_time_ms=t_auto,
                pytorch_time_ms=t_ref,
                speedup=t_ref / t_auto if t_auto > 0 else 0,
                max_error=err,
            )
        )

    # ------------------------------------------------------------------
    # LayerNorm
    # ------------------------------------------------------------------
    ln_shapes = [
        (4, 512),
        (4, 1024),
        (16, 2048),
        (64, 4096),
        (4, 768),
        (16, 768),
        (64, 768),
    ]

    for B, N in ln_shapes:
        x = torch.randn(B, N, device="cuda", dtype=torch.float32)
        w = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)

        out = autotuned_layernorm(x, w, b)

        ref = torch.nn.functional.layer_norm(
            x.float(), [N], weight=w.float(), bias=b.float(), eps=1e-5
        )
        err = (out.float() - ref).abs().max().item()

        t_auto = _benchmark(lambda: autotuned_layernorm(x, w, b))
        t_ref = _benchmark(
            lambda: torch.nn.functional.layer_norm(x, [N], weight=w, bias=b, eps=1e-5)
        )

        cfg = autotuned_layernorm_kernel.best_config
        best = cfg.kwargs if cfg else {}

        reports.append(
            KernelReport(
                kernel_name="autotuned_layernorm",
                shape=(B, N),
                best_config={
                    "BLOCK_SIZE": best.get("BLOCK_SIZE"),
                    "num_warps": best.get("num_warps"),
                },
                execution_time_ms=t_auto,
                pytorch_time_ms=t_ref,
                speedup=t_ref / t_auto if t_auto > 0 else 0,
                max_error=err,
            )
        )

    # ------------------------------------------------------------------
    # RMSNorm
    # ------------------------------------------------------------------
    for B, N in [(4, 512), (4, 1024), (16, 2048), (64, 4096)]:
        x = torch.randn(B, N, device="cuda", dtype=torch.float32)
        w = torch.randn(N, device="cuda", dtype=torch.float32)

        out = autotuned_rmsnorm(x, w)

        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        ref = (x.float() * rms * w.float()).to(x.dtype)
        err = (out.float() - ref.float()).abs().max().item()

        t_auto = _benchmark(lambda: autotuned_rmsnorm(x, w))
        t_ref = _benchmark(
            lambda: (
                x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()
            )
        )

        cfg = autotuned_rmsnorm_kernel.best_config
        best = cfg.kwargs if cfg else {}

        reports.append(
            KernelReport(
                kernel_name="autotuned_rmsnorm",
                shape=(B, N),
                best_config={
                    "BLOCK_SIZE": best.get("BLOCK_SIZE"),
                    "num_warps": best.get("num_warps"),
                    "num_stages": best.get("num_stages"),
                },
                execution_time_ms=t_auto,
                pytorch_time_ms=t_ref,
                speedup=t_ref / t_auto if t_auto > 0 else 0,
                max_error=err,
            )
        )

    # ------------------------------------------------------------------
    # Softmax
    # ------------------------------------------------------------------
    for B, N in [(4, 256), (16, 512), (64, 1024), (16, 4096)]:
        x = torch.randn(B, N, device="cuda", dtype=torch.float32)

        out = autotuned_softmax(x)
        ref = torch.softmax(x.float(), dim=-1)
        err = (out.float() - ref).abs().max().item()

        t_auto = _benchmark(lambda: autotuned_softmax(x))
        t_ref = _benchmark(lambda: torch.softmax(x, dim=-1))

        cfg = autotuned_softmax_kernel.best_config
        best = cfg.kwargs if cfg else {}

        reports.append(
            KernelReport(
                kernel_name="autotuned_softmax",
                shape=(B, N),
                best_config={
                    "BLOCK_SIZE": best.get("BLOCK_SIZE"),
                    "num_warps": best.get("num_warps"),
                    "num_stages": best.get("num_stages"),
                },
                execution_time_ms=t_auto,
                pytorch_time_ms=t_ref,
                speedup=t_ref / t_auto if t_auto > 0 else 0,
                max_error=err,
            )
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    # JSON
    json_path = os.path.join(benchmark_dir, "autotune_report.json")
    json_data = [asdict(r) for r in reports]
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"JSON report saved to {json_path}")

    # Markdown table
    md_path = os.path.join(benchmark_dir, "autotune_report.md")
    _write_markdown_table(reports, md_path)
    print(f"Markdown report saved to {md_path}")

    return reports


def _write_markdown_table(reports: list[KernelReport], path: str) -> None:
    """Write a Markdown table summarizing all reports."""
    lines: list[str] = []
    lines.append("# Autotune Benchmark Report")
    lines.append("")
    lines.append("Generated automatically by `autotune_report.py`.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Kernel | Shape | Config | Triton (ms) | PyTorch (ms) | Speedup | Error |")
    lines.append("|--------|-------|--------|------------|-------------|---------|-------|")

    for r in reports:
        cfg_str = ", ".join(f"{k}={v}" for k, v in r.best_config.items() if v is not None)
        lines.append(
            f"| {r.kernel_name} | {r.shape} | {cfg_str} | "
            f"{r.execution_time_ms:.4f} | {r.pytorch_time_ms:.4f} | "
            f"{r.speedup:.2f}x | {r.max_error:.2e} |"
        )

    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")

    # Compute aggregate stats
    speedups = [r.speedup for r in reports if r.speedup > 0]
    errors = [r.max_error for r in reports]

    if speedups:
        lines.append(f"- **Mean speedup**: {sum(speedups) / len(speedups):.2f}x")
        lines.append(f"- **Max speedup**: {max(speedups):.2f}x")
        lines.append(f"- **Min speedup**: {min(speedups):.2f}x")
    if errors:
        lines.append(f"- **Max error**: {max(errors):.2e}")
        lines.append(f"- **Mean error**: {sum(errors) / len(errors):.2e}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def print_summary_table(reports: list[KernelReport]) -> None:
    """Print a condensed summary to stdout."""
    if not reports:
        print("No reports to display.")
        return

    print("\n" + "=" * 100)
    print("  AUTOTUNE REPORT SUMMARY")
    print("=" * 100)
    print(f"  {'Kernel':<25} {'Shape':<20} {'Time (ms)':>10} {'PyTorch (ms)':>12} {'Speedup':>8}")
    print(f"  {'-' * 25} {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 8}")

    for r in reports:
        print(
            f"  {r.kernel_name:<25} {str(r.shape):<20} "
            f"{r.execution_time_ms:>10.4f} {r.pytorch_time_ms:>12.4f} "
            f"{r.speedup:>7.2f}x"
        )

    print(f"\n  Total configs evaluated: {len(reports)}")
    print(f"  Report saved to: benchmarks/autotune_report.json")


if __name__ == "__main__":
    reports = generate_autotune_report()
    print_summary_table(reports)
