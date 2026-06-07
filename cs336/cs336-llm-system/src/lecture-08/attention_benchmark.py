"""
对不同注意力实现的内存占用和速度进行 benchmark。

对比项目：
  - 朴素 scaled dot-product attention
  - PyTorch 的 torch.nn.functional.scaled_dot_product_attention（使用/未使用 flash）
  - 我们的自定义 MHA、GQA、MQA
  - Sliding window attention
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark_utils

from .attention import (
    CausalAttention,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
    create_causal_mask,
    scaled_dot_product_attention,
)


# =========================================================================
# 内存测量
# =========================================================================


def measure_peak_memory(
    fn: Any,
    *args: Any,
    **kwargs: Any,
) -> int:
    """测量函数调用的 GPU 峰值内存使用量。"""
    if not torch.cuda.is_available():
        return -1

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad():
        result = fn(*args, **kwargs)

    if isinstance(result, torch.Tensor):
        # 保留结果，防止内存被释放
        _ = result

    peak_mem = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    return peak_mem


# =========================================================================
# 速度 benchmark
# =========================================================================


def benchmark_speed(
    attn_fn: Any,
    inputs: tuple,
    num_iters: int = 10,
    warmup: int = 5,
    label: str = "",
) -> dict[str, float]:
    """对注意力函数进行速度 benchmark。"""
    # 预热
    for _ in range(warmup):
        _ = attn_fn(*inputs)

    # 同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = attn_fn(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    return {
        "label": label,
        "avg_ms": avg_time * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


# =========================================================================
# 主 benchmark
# =========================================================================


def run_benchmarks() -> None:
    """运行全面的注意力变体 benchmark。"""
    print("=" * 70)
    print("Attention Benchmark Suite")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    use_amp = torch.cuda.is_available()

    configs = [
        # (batch, seq_len, hidden, num_heads, num_kv_heads, label)
        (1, 128, 512, 8, 8, "Small"),
        (1, 512, 768, 12, 12, "Medium"),
        (1, 1024, 1024, 16, 16, "Large"),
        (1, 2048, 1024, 16, 16, "X-Large"),
    ]

    # --- 原始 SDPA 对比 ---
    print("\n--- Raw Scaled Dot-Product Attention ---")
    print(
        f"{'Config':<12} {'Naive (ms)':<14} {'PyTorch SDPA (ms)':<18} {'Speedup':<10} {'Naive Mem (MB)':<15} {'SDPA Mem (MB)':<15}"
    )
    print("-" * 85)

    for batch, seq_len, hidden, num_heads, _, label in configs:
        head_dim = hidden // num_heads
        q = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, num_heads, seq_len, head_dim, device=device)

        # 计时：朴素实现
        naive_result = benchmark_speed(
            lambda: scaled_dot_product_attention(q, k, v),
            (),
            label="Naive SDPA",
        )

        # 计时：PyTorch SDPA
        pt_result = benchmark_speed(
            lambda: F.scaled_dot_product_attention(q, k, v),
            (),
            label="PyTorch SDPA",
        )

        speedup = (
            naive_result["avg_ms"] / pt_result["avg_ms"]
            if pt_result["avg_ms"] > 0
            else 0
        )

        # 内存
        naive_mem = measure_peak_memory(lambda: scaled_dot_product_attention(q, k, v))
        pt_mem = measure_peak_memory(lambda: F.scaled_dot_product_attention(q, k, v))

        print(
            f"{label:<12} {naive_result['avg_ms']:>10.3f} ms  {pt_result['avg_ms']:>10.3f} ms  "
            f"{speedup:>6.1f}x  {naive_mem / 1e6:>10.2f} MB  {pt_mem / 1e6:>10.2f} MB"
        )

    # --- 注意力变体对比 ---
    print("\n--- Attention Variant Comparison (seq_len=256, hidden=512, 8 heads) ---")
    print(f"{'Variant':<25} {'Time (ms)':<14} {'Params':<12} {'KV Cache (MB)':<15}")
    print("-" * 70)

    batch, seq_len, hidden = 1, 256, 512
    x = torch.randn(batch, seq_len, hidden, device=device)

    variants_args: list[dict] = [
        {
            "name": "MHA",
            "cls": MultiHeadAttention,
            "kwargs": {"hidden_size": hidden, "num_heads": 8},
        },
        {
            "name": "GQA (KV=4)",
            "cls": GroupedQueryAttention,
            "kwargs": {"hidden_size": hidden, "num_heads": 8, "num_kv_heads": 4},
        },
        {
            "name": "GQA (KV=2)",
            "cls": GroupedQueryAttention,
            "kwargs": {"hidden_size": hidden, "num_heads": 8, "num_kv_heads": 2},
        },
        {
            "name": "MQA (KV=1)",
            "cls": MultiQueryAttention,
            "kwargs": {"hidden_size": hidden, "num_heads": 8},
        },
        {
            "name": "Causal MHA",
            "cls": CausalAttention,
            "kwargs": {"hidden_size": hidden, "num_heads": 8},
        },
        {
            "name": "Sliding Win (w=64)",
            "cls": SlidingWindowAttention,
            "kwargs": {"hidden_size": hidden, "num_heads": 8, "window_size": 64},
        },
    ]

    for var in variants_args:
        attn = var["cls"](**var["kwargs"]).to(device)
        attn.eval()

        params = sum(p.numel() for p in attn.parameters())
        head_dim = hidden // 8
        num_kv_heads = getattr(attn, "num_kv_heads", 8)
        kv_cache_mb = 2 * batch * seq_len * num_kv_heads * head_dim * 2 / 1e6  # fp16

        result = benchmark_speed(lambda: attn(x), (), label=var["name"])

        print(
            f"{var['name']:<25} {result['avg_ms']:>10.3f}   {params:>8,}   {kv_cache_mb:>10.2f}"
        )

    # --- 序列长度扩展分析 ---
    print("\n--- Sequence Length Scaling (Causal MHA) ---")
    print(f"{'Seq Len':<10} {'Time (ms)':<14}")
    print("-" * 30)

    for seq_l in [64, 128, 256, 512, 1024, 2048]:
        x_s = torch.randn(1, seq_l, 512, device=device)
        attn = CausalAttention(hidden_size=512, num_heads=8).to(device)
        attn.eval()
        result = benchmark_speed(
            lambda: attn(x_s), (), num_iters=5, label=f"seq={seq_l}"
        )
        print(f"{seq_l:<10} {result['avg_ms']:>10.3f}")

    # --- tiled 注意力的 block size 影响 ---
    print("\n--- Tiled Attention: Block Size Effect ---")
    print(f"{'Block Size':<12} {'Time (ms)':<14}")
    print("-" * 30)

    from .flash_attention_simple import flash_attention_tiled

    seq_l = 512
    q = torch.randn(1, 4, seq_l, 64, device=device)
    k = torch.randn(1, 4, seq_l, 64, device=device)
    v = torch.randn(1, 4, seq_l, 64, device=device)

    for bs in [16, 32, 64, 128, 256]:
        result = benchmark_speed(
            lambda: flash_attention_tiled(q, k, v, block_size=bs),
            (),
            num_iters=5,
            label=f"bs={bs}",
        )
        print(f"{bs:<12} {result['avg_ms']:>10.3f}")


def main() -> None:
    run_benchmarks()


if __name__ == "__main__":
    main()
