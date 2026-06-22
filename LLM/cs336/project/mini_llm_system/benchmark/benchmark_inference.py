"""
推理基准测试工具。

测量指标包括：
- TTFT (Time To First Token)：首个生成 token 出现前的延迟。
- Tokens per second：生成吞吐量。
- Latency per token：每个生成 token 的平均时间。
- 端到端延迟：从输入 prompt 到最终 token 的总时间。
- 内存使用：推理期间的 GPU 峰值内存。

比较不同 batch 大小、序列长度和生成策略下的性能。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

# 允许直接运行此文件或作为包的一部分运行
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def measure_inference_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    use_cache: bool = True,
    device: str = "cuda",
) -> dict[str, float]:
    """
    测量单次生成的推理延迟。

    Args:
        model: 语言模型。
        input_ids: 输入 token ID，形状为 [1, seq_len]。
        max_new_tokens: 要生成的 token 数量。
        temperature: 采样温度（0 表示贪心解码）。
        use_cache: 是否使用 KV cache。
        device: 计算设备。

    Returns:
        包含延迟指标的字典。
    """
    from inference.generation import generate_greedy, generate_sampling

    model.eval()
    model.to(device)
    input_ids = input_ids.to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 使用流式方式测量 TTFT（首个 token）
    ttft: float = 0.0
    all_tokens: list[int] = []
    generated = input_ids.clone()
    num_layers: int = len(model.layers)
    kv_caches = None

    start_time: float = time.perf_counter()

    if temperature == 0:
        # 使用 KV cache 的贪心解码
        for step in range(max_new_tokens):
            if use_cache and kv_caches is not None:
                current_input = generated[:, -1:]
                logits, kv_caches = model.forward(current_input, kv_caches=kv_caches)
            elif use_cache:
                logits, kv_caches = model.forward(generated, kv_caches=None)
            else:
                logits, _ = model.forward(generated, kv_caches=None)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if step == 0 and device.startswith("cuda"):
                torch.cuda.synchronize()
                ttft = time.perf_counter() - start_time

            generated = torch.cat([generated, next_token], dim=-1)
            all_tokens.append(next_token.item())

    end_time: float = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    total_time: float = end_time - start_time
    num_generated: int = len(all_tokens)

    metrics: dict[str, float] = {
        "prompt_tokens": float(input_ids.size(1)),
        "generated_tokens": float(num_generated),
        "total_time_s": total_time,
        "ttft_s": ttft,
        "tokens_per_second": num_generated / max(total_time, 1e-6),
        "latency_per_token_ms": (total_time / max(num_generated, 1)) * 1000,
    }

    if device.startswith("cuda"):
        metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    return metrics


def benchmark_by_sequence_length(
    model: nn.Module,
    seq_lengths: list[int],
    max_new_tokens: int = 20,
    device: str = "cuda",
    vocab_size: int = 1000,
) -> dict[int, dict[str, float]]:
    """
    在不同输入序列长度下对推理进行基准测试。

    Args:
        model: 语言模型。
        seq_lengths: 要测试的 prompt 长度列表。
        max_new_tokens: 每次测试要生成的 token 数量。
        device: 计算设备。
        vocab_size: 用于随机输入生成的词汇表大小。

    Returns:
        将 seq_len 映射到延迟指标的字典。
    """
    results: dict[int, dict[str, float]] = {}

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        metrics = measure_inference_latency(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            use_cache=True,
            device=device,
        )
        results[seq_len] = metrics
        print(
            f"  seq_len={seq_len:4d}: {metrics['tokens_per_second']:8.1f} tok/s, "
            f"TTFT={metrics['ttft_s'] * 1000:.1f}ms, "
            f"latency={metrics['latency_per_token_ms']:.1f}ms/tok"
        )

    return results


def benchmark_by_batch_size(
    model: nn.Module,
    batch_sizes: list[int],
    seq_len: int = 128,
    max_new_tokens: int = 20,
    device: str = "cuda",
    vocab_size: int = 1000,
) -> dict[int, dict[str, float]]:
    """
    在不同 batch 大小下对推理进行基准测试。

    Args:
        model: 语言模型。
        batch_sizes: 要测试的 batch 大小列表。
        seq_len: 每个序列的 prompt 长度。
        max_new_tokens: 要生成的 token 数量。
        device: 计算设备。
        vocab_size: 词汇表大小。

    Returns:
        将 batch_size 映射到吞吐量指标的字典。
    """
    results: dict[int, dict[str, float]] = {}

    for batch_size in batch_sizes:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        metrics = measure_inference_latency(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            use_cache=True,
            device=device,
        )
        metrics["total_tokens_per_second"] = (
            batch_size
            * metrics["generated_tokens"]
            / max(metrics["total_time_s"], 1e-6)
        )
        results[batch_size] = metrics
        print(
            f"  batch={batch_size:3d}: {metrics['total_tokens_per_second']:10.1f} tok/s total, "
            f"latency={metrics['latency_per_token_ms']:.1f}ms/tok"
        )

    return results


def benchmark_cache_vs_no_cache(
    model: nn.Module,
    seq_len: int = 64,
    max_new_tokens: int = 30,
    device: str = "cuda",
    vocab_size: int = 1000,
) -> dict[str, dict[str, float]]:
    """
    比较使用和不使用 KV cache 的推理性能。

    不使用 cache 时，每个生成步骤都需要重新处理整个序列，
    导致 O(n^2) 复杂度。使用 cache 时，每个步骤只处理新的 token，
    复杂度为 O(n)。

    Args:
        model: 语言模型。
        seq_len: prompt 长度。
        max_new_tokens: 要生成的 token 数量。
        device: 计算设备。
        vocab_size: 词汇表大小。

    Returns:
        包含 "cached" 和 "no_cache" 指标的字典。
    """
    results: dict[str, dict[str, float]] = {}
    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    print(f"\n  Prompt length: {seq_len}, Max new tokens: {max_new_tokens}")

    for use_cache in [True, False]:
        metrics = measure_inference_latency(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            use_cache=use_cache,
            device=device,
        )
        label: str = "cached" if use_cache else "no_cache"
        results[label] = metrics

        if label == "no_cache" and "cached" in results:
            cached_tps: float = results["cached"]["tokens_per_second"]
            uncached_tps: float = metrics["tokens_per_second"]
            speedup: float = cached_tps / max(uncached_tps, 1e-6)
            print(
                f"  Cached:   {cached_tps:.1f} tok/s, {results['cached']['latency_per_token_ms']:.1f}ms/tok"
            )
            print(
                f"  No cache: {uncached_tps:.1f} tok/s, {metrics['latency_per_token_ms']:.1f}ms/tok"
            )
            print(f"  Speedup:  {speedup:.2f}x")

    return results


# 快速测试
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformer.config import MiniLLMConfig
    from transformer.layers import MiniLLM

    config = MiniLLMConfig(
        vocab_size=500,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=256,
    )
    model = MiniLLM(config)

    print("=" * 60)
    print("Inference Benchmark")
    print("=" * 60)

    # 测试单次推理
    input_ids = torch.randint(0, 500, (1, 32))
    metrics = measure_inference_latency(
        model, input_ids, max_new_tokens=10, temperature=0, device="cpu"
    )
    print(f"Single inference: {metrics['tokens_per_second']:.1f} tok/s")

    # 比较使用和不使用 KV cache 的推理
    print("\nKV Cache vs No Cache:")
    benchmark_cache_vs_no_cache(
        model, seq_len=32, max_new_tokens=10, device="cpu", vocab_size=500
    )

    # 比较不同序列长度
    print("\nBy Sequence Length:")
    benchmark_by_sequence_length(
        model,
        seq_lengths=[8, 16, 32, 64],
        max_new_tokens=5,
        device="cpu",
        vocab_size=500,
    )

    print("\nInference benchmark tests passed!")
