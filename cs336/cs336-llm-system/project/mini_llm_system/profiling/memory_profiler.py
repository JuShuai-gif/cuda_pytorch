"""
GPU 内存分析与性能剖析工具。

提供以下功能：
- 跟踪 GPU 内存分配随时间的变化。
- 识别内存瓶颈（模型参数、梯度、优化器状态、激活值）。
- 估算不同模型配置的内存需求。
- 比较训练各阶段的内存使用情况。
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn


class MemoryTracker:
    """
    跟踪 GPU 内存分配与峰值使用量。

    用法示例:
        tracker = MemoryTracker()
        tracker.start()

        # ... 运行操作 ...
        tracker.snapshot("after_forward")

        # ... 更多操作 ...
        tracker.snapshot("after_backward")

        tracker.report()
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device: str = device
        self.snapshots: list[dict[str, Any]] = []
        self.start_time: float = 0.0

    def start(self) -> None:
        """开始跟踪。记录基准内存状态。"""
        self.start_time = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.snapshot("baseline")

    def snapshot(self, label: str) -> dict[str, Any]:
        """
        记录当前内存状态。

        参数:
            label: 此快照的描述性标签。

        返回:
            包含内存指标的字典。
        """
        elapsed: float = time.perf_counter() - self.start_time

        metrics: dict[str, Any] = {
            "label": label,
            "elapsed_s": elapsed,
        }

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            metrics["allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            metrics["reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            metrics["allocated_mb"] = 0.0
            metrics["reserved_mb"] = 0.0
            metrics["peak_allocated_mb"] = 0.0

        self.snapshots.append(metrics)
        return metrics

    def report(self) -> None:
        """打印格式化的内存使用报告。"""
        if not self.snapshots:
            print("No snapshots recorded.")
            return

        print("\n" + "=" * 70)
        print("GPU Memory Usage Report")
        print("=" * 70)
        header: str = f"{'Snapshot':<20} {'Elapsed(s)':<12} {'Alloc(MB)':<12} {'Reserved(MB)':<14} {'Peak(MB)':<10}"
        print(header)
        print("-" * 70)

        for snap in self.snapshots:
            print(
                f"{snap['label']:<20} {snap['elapsed_s']:<12.4f} "
                f"{snap['allocated_mb']:<12.2f} {snap['reserved_mb']:<14.2f} "
                f"{snap['peak_allocated_mb']:<10.2f}"
            )

        # 显示从第一个快照到最后一个快照的变化量
        first = self.snapshots[0]
        last = self.snapshots[-1]
        delta: float = last["allocated_mb"] - first["allocated_mb"]
        print("-" * 70)
        print(f"Net memory change: {delta:+.2f} MB")


def estimate_model_memory(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 2048,
    dtype_bytes: int = 4,  # float32 对应 4, float16 对应 2
    optimizer_states: int = 2,  # Adam 有 2 个状态量 (m, v)
) -> dict[str, float]:
    """
    估算模型训练所需的内存。

    将内存分解为以下几部分：
    - 模型参数
    - 梯度
    - 优化器状态（Adam: 动量 + 方差 = 参数量的 2 倍）
    - 激活值（基于隐藏层大小和序列长度的粗略估算）

    参数:
        model: PyTorch 模型。
        batch_size: 训练时的 batch size。
        seq_len: 序列长度。
        dtype_bytes: 每个参数的字节数（float32 为 4，float16 为 2）。
        optimizer_states: 每个参数对应的优化器状态张量数。

    返回:
        以 MB 为单位的内存估算字典。
    """
    num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_mb: float = (num_params * dtype_bytes) / 1024**2
    grads_mb: float = params_mb  # 梯度与参数大小相同
    optimizer_mb: float = params_mb * optimizer_states

    # 粗略的激活值内存估算
    # 对于 Transformer，激活值与 batch * seq * hidden * layers 成正比
    hidden_size: int = 0
    for p in model.parameters():
        if p.dim() >= 2:
            hidden_size = max(hidden_size, p.shape[-1])

    if hidden_size == 0:
        hidden_size = 768  # 默认回退值

    num_layers: int = sum(
        1 for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)
    )
    if num_layers == 0:
        num_layers = 4  # 保守估计

    # 激活值内存: ~ (batch * seq * hidden * num_layers * 4 bytes) 对于 float32
    activation_bytes: int = (
        batch_size * seq_len * hidden_size * num_layers * dtype_bytes
    )
    # 注意力机制中间结果的粗略乘数
    activation_bytes *= 4
    activation_mb: float = activation_bytes / 1024**2

    total_mb: float = params_mb + grads_mb + optimizer_mb + activation_mb

    return {
        "num_params": float(num_params),
        "dtype": "float32" if dtype_bytes == 4 else "float16",
        "params_mb": params_mb,
        "gradients_mb": grads_mb,
        "optimizer_states_mb": optimizer_mb,
        "activations_mb": activation_mb,
        "total_estimated_mb": total_mb,
    }


def track_training_memory(
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 1000,
    num_steps: int = 3,
) -> None:
    """
    跟踪各训练步骤的内存使用情况。

    参数:
        model: 要训练的模型。
        batch_size: batch size。
        seq_len: 序列长度。
        vocab_size: 词表大小。
        num_steps: 训练步数。
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    tracker = MemoryTracker(device)
    tracker.start()

    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        tracker.snapshot(f"step_{step}_zero_grad")

        logits, _ = model(input_ids)
        tracker.snapshot(f"step_{step}_forward")

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        loss.backward()
        tracker.snapshot(f"step_{step}_backward")

        optimizer.step()
        tracker.snapshot(f"step_{step}_optimizer")

    tracker.report()

    # 打印估算的内存分解
    est = estimate_model_memory(model, batch_size, seq_len)
    print(f"\nEstimated Memory Breakdown:")
    print(f"  Parameters:       {est['params_mb']:.1f} MB")
    print(f"  Gradients:        {est['gradients_mb']:.1f} MB")
    print(f"  Optimizer states: {est['optimizer_states_mb']:.1f} MB")
    print(f"  Activations:      {est['activations_mb']:.1f} MB")
    print(f"  Total estimated:  {est['total_estimated_mb']:.1f} MB")


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
    num_params: int = model.get_num_params()

    print(f"Model: {num_params:,} parameters")
    print()

    # 估算内存
    est = estimate_model_memory(model, batch_size=2, seq_len=128)
    print("Memory Estimate:")
    for k, v in est.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")
    print()

    # 跟踪训练内存（CPU 安全）- 使用匹配的 vocab_size
    track_training_memory(model, batch_size=1, seq_len=32, num_steps=2, vocab_size=500)

    print("\nMemory profiler test passed!")
