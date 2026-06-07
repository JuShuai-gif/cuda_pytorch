"""
第六讲 — GPU 编程：torch.profiler 使用示例。

演示如何使用 ``torch.profiler`` 分析 kernel 执行、
内存使用和 trace 导出。实际的性能分析受保护；参见
下方的 ``_ENABLE_PROFILE``。
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 被分析模型
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """用于性能分析演示的小型类 transformer 模型堆栈。"""

    def __init__(self, dim: int = 128, hidden: int = 512, vocab: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.ln1(h)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out
        h = self.ln2(h)
        mlp_out = self.mlp(h)
        h = h + mlp_out
        return self.head(h)


# ---------------------------------------------------------------------------
# Profiler 上下文管理器
# ---------------------------------------------------------------------------


@contextmanager
def profiler_context(
    enable: bool = True,
    activities: Optional[List[Any]] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    log_dir: Optional[str] = None,
):
    """上下文管理器，可选择性地用 torch.profiler 包裹代码。

    使用方式::

        with profiler_context(enable=True):
            output = model(input)
            loss.backward()
    """
    if not enable or not torch.cuda.is_available():
        yield
        return

    if activities is None:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

    if log_dir is None:
        log_dir = tempfile.mkdtemp(prefix="torch_profile_")

    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        ) as prof:
            yield prof
    finally:
        # 打印摘要（始终执行，即使部分失败）
        if "prof" in locals() and prof is not None:
            try:
                print(
                    prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
                )
            except Exception:
                pass
            print(f"Profile trace saved to: {log_dir}")


# ---------------------------------------------------------------------------
# 演示辅助函数（不执行重型计算）
# ---------------------------------------------------------------------------


def demo_profiler_api() -> None:
    """展示如何使用 profiler，而不实际执行性能分析。"""
    print("torch.profiler API usage example:")
    print()
    print("  # Basic profiling:")
    print("  with torch.profiler.profile(")
    print("      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],")
    print("      record_shapes=True,")
    print("      profile_memory=True,")
    print("      with_stack=True,")
    print("  ) as prof:")
    print("      output = model(input_data)")
    print("      loss.backward()")
    print()
    print("  # Print summary:")
    print("  print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))")
    print()
    print("  # Export Chrome trace:")
    print("  prof.export_chrome_trace('trace.json')")
    print()
    print("  # TensorBoard trace:")
    print("  torch.profiler.tensorboard_trace_handler('./log')")
    print()
    print("  # Schedule-based profiling (skip first N steps):")
    print("  schedule = torch.profiler.schedule(")
    print("      wait=2, warmup=2, active=3, repeat=1)")
    print("  with torch.profiler.profile(")
    print("      schedule=schedule,")
    print("      on_trace_ready=tensorboard_trace_handler('./log'),")
    print("  ) as prof:")
    print("      for step, batch in enumerate(dataloader):")
    print("          train_step(batch)")
    print("          prof.step()")


def demo_model_info(model: nn.Module) -> None:
    """打印模型统计信息。"""
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel summary:")
    print(f"  Total parameters:  {params:,}")
    print(f"  Trainable:         {trainable:,}")
    print(f"  Modules:           {len(list(model.modules()))}")

    # 逐模块细分
    print(f"\n  Module breakdown:")
    for name, module in model.named_modules():
        if name == "":
            continue
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            print(f"    {name:30s}: {n:>10,} params")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    device = torch.device("cpu")

    # 创建模型
    model = TinyModel(dim=128, hidden=512, vocab=1024).to(device)
    demo_model_info(model)

    # 展示 profiler API
    print()
    demo_profiler_api()

    # 执行一次小型前向传播（不进行性能分析）以验证形状
    x = torch.randint(0, 1024, (2, 16))
    with torch.no_grad():
        out = model(x)
    print(f"\nForward pass output shape: {out.shape}  ✓")

    print("\nAll checks passed.")
