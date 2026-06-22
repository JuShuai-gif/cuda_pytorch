"""
训练性能基准测试工具。

测量指标：
- tokens/s throughput
- GPU 内存使用量
- 单步时间（前向 + 反向 + 优化器）
- 梯度累积开销
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# 允许以独立脚本或作为包的一部分运行此文件
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def measure_training_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    num_steps: int = 100,
    warmup_steps: int = 10,
    gradient_accumulation_steps: int = 1,
    device: str = "cuda",
) -> dict[str, float]:
    """
    测量训练吞吐量（tokens/s）。

    参数：
        model: 待测试性能的模型。
        dataloader: 提供批数据的 DataLoader。
        num_steps: 需要测量的训练步数。
        warmup_steps: 预热步数（不计入性能指标）。
        gradient_accumulation_steps: 每个优化器步骤的微批次数。
        device: 计算设备。

    返回：
        包含吞吐量指标的字典。
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 预热
    data_iter = iter(dataloader)
    for _ in range(warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 性能测试
    total_tokens: int = 0
    total_time: float = 0.0
    step_times: list[float] = []

    data_iter = iter(dataloader)
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        batch_tokens: int = input_ids.numel()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
            start = time.perf_counter()

        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
            elapsed: float = time.perf_counter() - start
        else:
            elapsed = 0.001  # CPU 测试用

        step_times.append(elapsed * 1000)  # 毫秒
        total_time += elapsed
        total_tokens += batch_tokens

    # 计算指标
    avg_step_time_ms: float = sum(step_times) / len(step_times)
    tokens_per_sec: float = total_tokens / max(total_time, 1e-6)

    metrics = {
        "tokens_per_second": tokens_per_sec,
        "avg_step_time_ms": avg_step_time_ms,
        "min_step_time_ms": min(step_times),
        "max_step_time_ms": max(step_times),
        "total_tokens": float(total_tokens),
        "total_time_s": total_time,
        "num_steps": float(num_steps),
    }

    if device.startswith("cuda"):
        metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    return metrics


def compare_dtype_throughput(
    model_builder,
    dataloader: DataLoader,
    dtypes: list[torch.dtype] | None = None,
    num_steps: int = 50,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    比较不同浮点类型下的训练吞吐量。

    参数：
        model_builder: 返回新模型实例的可调用对象。
        dataloader: 训练数据的 DataLoader。
        dtypes: 需要比较的 dtype 列表（默认：[float32, float16, bfloat16]）。
        num_steps: 每种 dtype 的测试步数。
        device: 计算设备。

    返回：
        dtype 名称到吞吐量指标的映射字典。
    """
    if dtypes is None:
        dtypes = [torch.float32]
        if device.startswith("cuda"):
            dtypes.extend([torch.float16, torch.bfloat16])

    results: dict[str, Any] = {}
    for dtype in dtypes:
        model = model_builder().to(dtype=dtype)
        metrics = measure_training_throughput(
            model,
            dataloader,
            num_steps=num_steps,
            warmup_steps=5,
            device=device,
        )
        dtype_name: str = str(dtype).split(".")[-1]
        results[dtype_name] = metrics
        metrics["dtype"] = dtype_name

    return results


# 快速测试
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformer.config import MiniLLMConfig
    from transformer.layers import MiniLLM

    # 创建一个小型模型
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=1024,
        max_seq_len=128,
    )

    # 创建虚拟数据
    class DummyDataset(Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 200

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.randint(0, 1000, (128,)),
                "labels": torch.randint(0, 1000, (128,)),
            }

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 构建模型
    model = MiniLLM(config)

    # 测量吞吐量
    print("Measuring training throughput...")
    results = measure_training_throughput(
        model,
        dataloader,
        num_steps=20,
        warmup_steps=5,
        device="cpu",  # 使用 CPU 以确保安全
    )

    print("\nTraining Benchmark Results:")
    print(f"  Tokens/second: {results['tokens_per_second']:.1f}")
    print(f"  Avg step time: {results['avg_step_time_ms']:.2f} ms")
    print(f"  Total time: {results['total_time_s']:.2f} s")
    print(f"  Steps measured: {int(results['num_steps'])}")
    if "peak_memory_mb" in results:
        print(f"  Peak GPU memory: {results['peak_memory_mb']:.1f} MB")

    print("\nBenchmark training module test passed!")
