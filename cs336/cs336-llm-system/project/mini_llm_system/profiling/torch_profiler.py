"""
MiniLLM 的 torch profiler 使用示例。

演示如何使用 torch.profiler 分析模型性能、识别瓶颈并可视化计算时间线。

用法：
    python torch_profiler.py
    # 然后使用以下命令查看：tensorboard --logdir=./profiler_logs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)


def profile_model_forward(
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 1000,
    log_dir: str = "./profiler_logs",
) -> None:
    """
    对模型的单次前向传播进行 profiling。

    Args:
        model: 要进行 profiling 的模型。
        batch_size: profiling 使用的 batch size。
        seq_len: 序列长度。
        vocab_size: 词汇表大小。
        log_dir: 保存 profile trace 文件的目录。
    """
    model.train()
    model.cuda()

    input_ids: torch.Tensor = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    labels: torch.Tensor = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Profiling model forward pass...")
    print(f"  Batch: {batch_size}, Seq len: {seq_len}")
    print(f"  Logs saved to: {log_dir_path.absolute()}")

    # 预热
    for _ in range(3):
        logits, _ = model(input_ids)

    torch.cuda.synchronize()

    # 使用 schedule 进行 profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(log_dir_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(8):
            with torch.no_grad():
                logits, _ = model(input_ids)
            prof.step()

    print(f"Profiling complete!")
    print(f"To view: tensorboard --logdir={log_dir_path.absolute()}")
    print()

    # 打印摘要
    print("Operator time summary (top 10 by CUDA time):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def profile_training_step(
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 1000,
    log_dir: str = "./profiler_logs_training",
) -> None:
    """
    对完整训练步骤（前向 + 反向 + 优化器）进行 profiling。

    Args:
        model: 要进行 profiling 的模型。
        batch_size: batch size。
        seq_len: 序列长度。
        vocab_size: 词汇表大小。
        log_dir: 存放 trace 文件的目录。
    """
    model.train()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    input_ids: torch.Tensor = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device
    )
    labels: torch.Tensor = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 预热
    for _ in range(3):
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    print(f"\nProfiling training step (forward + backward + optimizer)...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(log_dir_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(5):
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            prof.step()

    print(f"Training profiling complete!")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def profile_attention_only(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
) -> None:
    """
    仅对 attention 运算进行 profiling，以了解其开销。
    """
    from transformer.attention import CausalAttention

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    causal = CausalAttention().to(device)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"\nProfiling attention operation (seq_len={seq_len})...")

    # 预热
    for _ in range(3):
        _ = causal(q, k, v)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(5):
            _ = causal(q, k, v)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# 快速测试
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformer.config import MiniLLMConfig
    from transformer.layers import MiniLLM

    # 创建一个小模型
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

    print("Torch Profiler Examples")
    print("=" * 60)
    print()

    # 仅对 attention 进行 profiling（较轻量）
    profile_attention_only(batch_size=1, num_heads=4, seq_len=64, head_dim=32)

    print("\nTorch profiler module loaded successfully!")
    print("Run with CUDA for detailed GPU timeline traces.")
