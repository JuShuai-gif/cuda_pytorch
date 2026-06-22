"""
简单的流水线并行模拟。
将模型拆分到不同的"设备"上，演示 micro-batch 如何在模型各个阶段之间流水线化执行。

演示的概念：
  - 模型按阶段划分
  - Micro-batch 流水线调度（GPipe / 1F1B 调度）
  - Bubble 时间开销
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 用于划分的简单模型
# ---------------------------------------------------------------------------


class SimpleTransformerBlock(nn.Module):
    """用于流水线划分的单个 Transformer 块。"""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x + residual
        return x


class PipelineStage:
    """
    流水线中的一个阶段。表示分配给某个虚拟设备的一层或多层。
    """

    def __init__(self, name: str, layers: nn.ModuleList, device_id: int):
        self.name = name
        self.layers = layers
        self.device_id = device_id
        self.activations: list[torch.Tensor | None] = []  # 存储前向激活以供反向使用
        self.grads: list[Any] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行本阶段的前向传播。"""
        for layer in self.layers:
            x = layer(x)
        self.activations.append(x.detach())
        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """执行反向传播（模拟为原样返回梯度）。"""
        return grad


@dataclass
class PipelineSchedule:
    """表示流水线执行调度。"""

    num_stages: int
    num_microbatches: int

    def gpipe_schedule(self) -> list[list[tuple[str, int]]]:
        """
        GPipe 调度：先注入所有 microbatch 的前向传播，
        再处理所有反向传播。
        """
        schedule: list[list[tuple[str, int]]] = []
        return schedule

    def one_f_one_b_schedule(self) -> list[list[tuple[str, int]]]:
        """1F1B 调度：交替执行前向和反向以降低内存使用。"""
        return []


# =========================================================================
# 模拟
# =========================================================================


def demo_pipeline_stages() -> None:
    """演示将模型拆分为流水线阶段。"""
    print("=" * 60)
    print("流水线并行 - 模型划分演示")
    print("=" * 60)

    num_layers = 8
    num_stages = 4
    layers_per_stage = num_layers // num_stages

    print(f"\n总层数：{num_layers}")
    print(f"流水线阶段数：{num_stages}")
    print(f"每阶段层数：{layers_per_stage}")
    print()

    # 创建模型并划分
    all_layers = nn.ModuleList([SimpleTransformerBlock(128) for _ in range(num_layers)])
    stages: list[PipelineStage] = []

    for stage_idx in range(num_stages):
        start = stage_idx * layers_per_stage
        end = start + layers_per_stage
        stage_layers = all_layers[start:end]
        stage = PipelineStage(
            name=f"Stage-{stage_idx}",
            layers=nn.ModuleList(stage_layers),
            device_id=stage_idx,
        )
        stages.append(stage)

    for stage in stages:
        num_params = sum(p.numel() for p in stage.layers.parameters())
        print(
            f"  {stage.name}（设备 {stage.device_id}）：{len(stage.layers)} 层，{num_params:,} 参数"
        )

    # 每设备内存
    total_params = sum(p.numel() for p in all_layers.parameters())
    per_device = total_params / num_stages
    print(f"\n  模型总参数量：{total_params:,}")
    print(f"  每设备参数量：{per_device:,.0f}")
    print(
        f"  内存节省：相比完整副本节省 {((total_params - per_device) / total_params) * 100:.0f}%"
    )


def demo_gpipe_bubble() -> None:
    """演示 GPipe 调度中的 bubble 开销。"""
    print("\n" + "=" * 60)
    print("GPipe Bubble 开销分析")
    print("=" * 60)

    num_microbatches = 8
    num_stages = 4

    # GPipe 有三个阶段：
    # 1. 预热：各阶段逐个启动（流水线填充）
    # 2. 稳态：所有阶段均在工作
    # 3. 冷却：各阶段逐个结束（流水线排空）

    print(f"\nMicrobatch 数：{num_microbatches}，阶段数：{num_stages}")

    # 每个 microbatch 在每个阶段耗时 1 个时间单位（简化）
    # 总时间 = (num_microbatches + num_stages - 1) * 每阶段时间
    total_slots = num_microbatches + num_stages - 1
    useful_slots = num_microbatches * num_stages  # 完美利用率下
    actual_slots = total_slots * num_stages  # 每个阶段活跃 total_slots 个时间单位

    bubble_slots = total_slots * num_stages - num_microbatches * num_stages
    bubble_ratio = bubble_slots / (total_slots * num_stages)

    print(f"\n  总时间槽：{total_slots}")
    print(f"  完美利用率槽：{useful_slots}")
    print(f"  实际计算量：{num_microbatches * num_stages}")
    print(f"  Bubble 槽：{bubble_slots}")
    print(f"  Bubble 比例：{bubble_ratio:.1%}")
    print(f"\n  公式：bubble = (S - 1) / (M + S - 1)")
    print(f"         其中 S = 阶段数，M = microbatch 数")
    print(f"  当 M={num_microbatches}，S={num_stages} 时：{bubble_ratio:.1%}")

    # 可视化调度
    print(f"\n  GPipe 调度可视化（F=前向，B=反向）：")
    print(f"  {'时间':>5}", end="")
    for t in range(total_slots + 1):
        print(f"{t:>5}", end="")
    print()

    for stage in range(num_stages):
        print(f"  S{stage:<4}", end="")
        for t in range(total_slots + 1):
            # 确定该阶段在时刻 t 执行什么操作
            microbatch_idx = t - stage
            if 0 <= microbatch_idx < num_microbatches:
                print(f" F{stage}{microbatch_idx:<2}", end="")
            elif num_microbatches <= microbatch_idx < 2 * num_microbatches:
                b_idx = microbatch_idx - num_microbatches
                # 所有前向完成后再执行反向
                backward_t = b_idx + stage + num_microbatches
                if backward_t <= t <= backward_t:
                    b_actual = num_microbatches - 1 - b_idx
                    print(f" B{stage}{b_actual:<2}", end="")
                else:
                    print(f"{'':>5}", end="")
            else:
                print(f"{'':>5}", end="")
        print()

    print(f"\n  关键洞察：增大 M 可降低 bubble 比例。")
    print(f"  M=8：  bubble={(num_stages - 1) / (8 + num_stages - 1):.1%}")
    print(f"  M=32： bubble={(num_stages - 1) / (32 + num_stages - 1):.1%}")
    print(f"  M=128：bubble={(num_stages - 1) / (128 + num_stages - 1):.1%}")


def demo_1f1b_schedule() -> None:
    """演示 1F1B（一次前向、一次反向）调度。"""
    print("\n" + "=" * 60)
    print("1F1B（One-Forward-One-Backward）调度")
    print("=" * 60)

    num_microbatches = 4
    num_stages = 3

    print(f"\nMicrobatch 数：{num_microbatches}，阶段数：{num_stages}")
    print("\n1F1B 内存优势：")
    print("  GPipe：将所有 microbatch 的激活一直存到反向 → O(M) 内存")
    print("  1F1B：尽可能早地开始反向 → O(1) 峰值激活")
    print("\n1F1B 调度：")
    print("  预热：注入 M 次前向传播（与 GPipe 相同）")
    print("  稳态：交替执行 1F 和 1B")
    print("  冷却：完成剩余的反向传播")

    # 显示时间线
    total_time = 2 * num_microbatches + num_stages - 1
    print(f"\n  时间线（{total_time} 步）：")
    for t in range(total_time):
        activities = []
        for s in range(num_stages):
            f_idx = t - s
            b_idx = t - (s + num_microbatches)
            if 0 <= f_idx < num_microbatches:
                activities.append(f"S{s}:F{f_idx}")
            elif 0 <= b_idx < num_microbatches:
                activities.append(f"S{s}:B{b_idx}")
        if activities:
            print(f"  步骤 {t}：{', '.join(activities)}")
        else:
            print(f"  步骤 {t}：（空闲）")


def demo_pipeline_simulation() -> None:
    """模拟流水线化模型的前向传播。"""
    print("\n" + "=" * 60)
    print("流水线前向传播模拟")
    print("=" * 60)

    hidden_size = 128
    num_stages = 3
    layers_per_stage = 2

    stages = []
    for s in range(num_stages):
        layers = nn.ModuleList(
            [SimpleTransformerBlock(hidden_size) for _ in range(layers_per_stage)]
        )
        stages.append(PipelineStage(f"Stage-{s}", layers, s))

    batch_size, seq_len = 2, 16
    num_microbatches = 4
    microbatch_size = batch_size // num_microbatches

    print(
        f"\nBatch size：{batch_size}，Microbatch 数：{num_microbatches}，Microbatch size：{microbatch_size}"
    )
    print(f"阶段数：{num_stages}")

    # 模拟 GPipe 流水线
    data = torch.randn(batch_size, seq_len, hidden_size)
    microbatches = data.chunk(num_microbatches, dim=0)

    print(f"\n--- GPipe 前向传播 ---")
    start_time = time.time()

    for mb_idx, mb in enumerate(microbatches):
        # 通过所有阶段
        for stage_idx, stage in enumerate(stages):
            # 模拟"设备"间的通信延迟
            if stage_idx > 0:
                # 在实际流水线中，这里是 send/recv 操作
                pass
            output = stage.forward(mb)
            mb = output
            print(
                f"  MB{mb_idx} Stage{stage_idx}：输入 shape → 输出 shape {output.shape}"
            )

        if mb_idx < num_microbatches - 1:
            mb = microbatches[mb_idx + 1]

    elapsed = time.time() - start_time
    print(f"\n  模拟总耗时：{elapsed:.4f}s")


def main() -> None:
    demo_pipeline_stages()
    demo_gpipe_bubble()
    demo_1f1b_schedule()
    demo_pipeline_simulation()


if __name__ == "__main__":
    main()
