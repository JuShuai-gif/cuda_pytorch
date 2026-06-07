"""
注意力实现对比基准测试。

比较三种注意力实现：
1. Naive（ScaledDotProductAttention）：完整 O(n^2) 矩阵，无优化。
2. FlashAttentionSimple：分块计算配合 online softmax，O(n^2) 计算量但 O(n) 内存。
3. PyTorch SDPA：torch.nn.functional.scaled_dot_product_attention（在可用时使用 FlashAttention
   或 Memory-Efficient Attention 后端）。

指标：速度、内存使用、数值精度。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# 允许直接运行此文件或作为包的一部分运行
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from transformer.attention import (
    ScaledDotProductAttention,
    CausalAttention,
    FlashAttentionSimple,
)


def benchmark_attention_implementations(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_lengths: list[int] | None = None,
    head_dim: int = 64,
    device: str = "cuda",
    num_warmup: int = 5,
    num_runs: int = 20,
) -> dict[str, Any]:
    """
    对不同注意力实现进行基准测试和对比。

    Args:
        batch_size: 批大小。
        num_heads: 注意力头数。
        seq_lengths: 要测试的序列长度列表。
        head_dim: 每个头的维度。
        device: 计算设备。
        num_warmup: 预热迭代次数。
        num_runs: 基准测试迭代次数。

    Returns:
        按序列长度和实现分组的基准测试结果字典。
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024]

    results: dict[str, Any] = {}

    for seq_len in seq_lengths:
        print(f"\n  Sequence length: {seq_len}")
        seq_results: dict[str, dict[str, float]] = {}

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # 1. Naive 缩放点积注意力
        naive_attn = CausalAttention()
        naive_attn.to(device)

        # 预热
        for _ in range(num_warmup):
            _ = naive_attn(q, k, v)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        for _ in range(num_runs):
            out_naive = naive_attn(q, k, v)

        if device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000  # 毫秒 -> 秒
            peak_mem_naive = torch.cuda.max_memory_allocated() / 1024**2
        else:
            elapsed = time.perf_counter() - start_time
            peak_mem_naive = 0.0

        avg_time_naive: float = elapsed / num_runs
        seq_results["naive"] = {
            "avg_time_ms": avg_time_naive * 1000,
            "peak_memory_mb": peak_mem_naive,
            "tokens_per_second": seq_len * batch_size / max(avg_time_naive, 1e-6),
        }
        print(
            f"    Naive:        {avg_time_naive * 1000:.2f}ms, {peak_mem_naive:.1f}MB"
        )

        # 2. FlashAttentionSimple（分块）
        flash_attn = FlashAttentionSimple(block_size=min(128, seq_len))
        flash_attn.to(device)

        for _ in range(num_warmup):
            _ = flash_attn(q, k, v, causal=True)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        for _ in range(num_runs):
            out_flash = flash_attn(q, k, v, causal=True)

        if device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000
            peak_mem_flash = torch.cuda.max_memory_allocated() / 1024**2
        else:
            elapsed = time.perf_counter() - start_time
            peak_mem_flash = 0.0

        avg_time_flash: float = elapsed / num_runs
        seq_results["flash_simple"] = {
            "avg_time_ms": avg_time_flash * 1000,
            "peak_memory_mb": peak_mem_flash,
            "tokens_per_second": seq_len * batch_size / max(avg_time_flash, 1e-6),
        }
        print(
            f"    FlashSimple:  {avg_time_flash * 1000:.2f}ms, {peak_mem_flash:.1f}MB"
        )

        # 3. PyTorch SDPA（使用 FlashAttention 后端）
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True, attn_mask=None)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        for _ in range(num_runs):
            out_sdpa = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, attn_mask=None
            )

        if device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000
            peak_mem_sdpa = torch.cuda.max_memory_allocated() / 1024**2
        else:
            elapsed = time.perf_counter() - start_time
            peak_mem_sdpa = 0.0

        avg_time_sdpa: float = elapsed / num_runs
        seq_results["torch_sdpa"] = {
            "avg_time_ms": avg_time_sdpa * 1000,
            "peak_memory_mb": peak_mem_sdpa,
            "tokens_per_second": seq_len * batch_size / max(avg_time_sdpa, 1e-6),
        }
        print(f"    PyTorch SDPA: {avg_time_sdpa * 1000:.2f}ms, {peak_mem_sdpa:.1f}MB")

        # 数值精度对比
        diff_flash_vs_naive: float = (
            (out_flash.float() - out_naive.float()).abs().max().item()
        )
        diff_sdpa_vs_naive: float = (
            (out_sdpa.float() - out_naive.float()).abs().max().item()
        )
        print(
            f"    Accuracy: Flash vs Naive max_diff={diff_flash_vs_naive:.2e}, "
            f"SDPA vs Naive max_diff={diff_sdpa_vs_naive:.2e}"
        )

        seq_results["accuracy"] = {
            "flash_vs_naive_max_diff": diff_flash_vs_naive,
            "sdpa_vs_naive_max_diff": diff_sdpa_vs_naive,
        }

        results[f"seq_{seq_len}"] = seq_results

    return results


def print_comparison_table(results: dict[str, Any]) -> None:
    """打印所有结果的格式化对比表。"""
    print("\n" + "=" * 80)
    print("Attention Implementation Comparison")
    print("=" * 80)
    header: str = (
        f"{'Seq Len':<10} {'Impl':<15} {'Time(ms)':<12} {'Mem(MB)':<10} {'Tok/s':<12}"
    )
    print(header)
    print("-" * 80)

    for key, data in results.items():
        seq_len: str = key.replace("seq_", "")
        for impl in ["naive", "flash_simple", "torch_sdpa"]:
            if impl in data:
                m = data[impl]
                print(
                    f"{seq_len:<10} {impl:<15} {m['avg_time_ms']:<12.2f} "
                    f"{m['peak_memory_mb']:<10.1f} {m['tokens_per_second']:<12.0f}"
                )
        if "accuracy" in data:
            acc = data["accuracy"]
            print(
                f"{'':>10} {'accuracy':<15} flash_diff={acc['flash_vs_naive_max_diff']:.2e}, "
                f"sdpa_diff={acc['sdpa_vs_naive_max_diff']:.2e}"
            )
        print()


# 快速测试
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Attention Implementation Benchmark")
    print("=" * 60)

    # 在 CPU 上使用较小序列长度进行测试
    results = benchmark_attention_implementations(
        batch_size=1,
        num_heads=4,
        seq_lengths=[32, 64, 128],
        head_dim=32,
        device="cpu",
        num_warmup=2,
        num_runs=5,
    )

    print_comparison_table(results)
    print("Attention benchmark tests passed!")
