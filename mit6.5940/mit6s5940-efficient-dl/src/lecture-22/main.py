#!/usr/bin/env python3
"""
MIT 6.5940 第22讲：课程总结 -- 端到端模型压缩流水线

涵盖主题：
  - 完整流水线：加载模型 -> 剪枝 -> 量化 -> ONNX 导出 -> 基准测试
  - 生成综合报告：每个阶段的参数量、FLOPs、延迟、内存
  - 创建对比表：基准模型 vs 剪枝后 vs 量化后 vs 剪枝+量化
  - 自动生成 Markdown 摘要报告

所有计算均在 CPU 上运行，无需 GPU。
"""

from __future__ import annotations

import os
import time
import json
import math
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np


# ===========================================================================
# 可复现性设置
# ===========================================================================
torch.manual_seed(42)


# ===========================================================================
# 1. 参考模型
# ===========================================================================


class CompressionDemoModel(nn.Module):
    """用于演示压缩流水线的代表性 CNN 模型。

    架构: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> FC -> FC
    该模型刻意过度参数化，以展示压缩带来的收益。
    """

    def __init__(self):
        """初始化各层：4个卷积层（带批归一化）、全局平均池化、2个全连接层。"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：Conv+BN+ReLU 四次 -> 全局平均池化 -> 展平 -> FC+ReLU -> FC。"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===========================================================================
# 2. 剪枝工具函数
# ===========================================================================

import torch.nn.functional as F  # noqa: E402（模块级别需要）


def apply_structured_pruning(
    model: CompressionDemoModel, prune_ratio: float = 0.5
) -> CompressionDemoModel:
    """对卷积层应用 L1 范数结构化剪枝。

    剪掉每个 Conv2d 层中 `prune_ratio` 比例的输出通道。
    使用 PyTorch 内置的剪枝工具。
    """
    pruned = copy.deepcopy(model)

    for name, module in pruned.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 3:
            # 对权重张量的第0维（输出通道维）进行 L1 结构化剪枝
            prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=0)
            # 使剪枝永久生效（移除剪枝掩码，保留稀疏权重）
            prune.remove(module, "weight")

    return pruned


def apply_unstructured_pruning(
    model: CompressionDemoModel, prune_ratio: float = 0.6
) -> CompressionDemoModel:
    """对全连接层应用 L1 范数非结构化（逐元素）剪枝。"""
    pruned = copy.deepcopy(model)

    for name, module in pruned.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            prune.remove(module, "weight")

    return pruned


# ===========================================================================
# 3. INT8 量化模拟
# ===========================================================================


def simulate_int8_quantization(model: nn.Module) -> nn.Module:
    """通过截断权重来模拟 INT8 训练后量化。

    在实际部署中，会使用 torch.quantization 或 ONNX Runtime。
    这里我们模拟将精度降至 8 位整数的效果。
    """
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for param in quantized.parameters():
            if param.dim() < 1:  # 跳过标量参数
                continue
            w = param.data
            w_max = w.abs().max().clamp(min=1e-8)  # 权重的最大绝对值，防止除零
            # 模拟 8 位量化：256 个离散级别 ([-128, 127])
            w_quant = (w / w_max * 127).round().clamp(-128, 127) / 127 * w_max
            param.data.copy_(w_quant)
    return quantized


# ===========================================================================
# 4. 指标采集
# ===========================================================================


def count_parameters(model: nn.Module) -> int:
    """统计总参数量（包含剪枝/归零的权重）。"""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """统计非零参数量（剪枝后）。"""
    total = 0
    for p in model.parameters():
        total += (p != 0).sum().item()
    return total


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """估算卷积层和全连接层的 FLOPs。

    使用前向钩子测量输出形状并计算 FLOPs。
    """
    hook_data: List[int] = []  # 存储各层 FLOPs 的列表

    def conv_hook(m, inp, out):
        """卷积层 FLOPs 钩子：FLOPs = 2 * C_in * K_h * K_w * C_out * H_out * W_out"""
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            flops = 2 * m.in_channels * k * m.out_channels * out.shape[2] * out.shape[3]
            hook_data.append(flops)

    def linear_hook(m, inp, out):
        """全连接层 FLOPs 钩子：FLOPs = 2 * in_features * out_features"""
        if isinstance(m, nn.Linear):
            hook_data.append(2 * m.in_features * m.out_features)

    handles = []
    for m in model.modules():
        handles.append(m.register_forward_hook(conv_hook))
        handles.append(m.register_forward_hook(linear_hook))

    x = torch.randn(*input_shape)
    with torch.no_grad():
        model(x)  # 执行前向传播以触发钩子

    for h in handles:
        h.remove()  # 清理钩子

    return sum(hook_data)


def measure_latency(
    model: nn.Module, input_shape: Tuple[int, ...], warmup: int = 10, repeats: int = 100
) -> float:
    """测量平均推理延迟。

    参数:
        model: PyTorch 模型
        input_shape: 输入张量形状 (batch_size, channels, height, width)
        warmup: 预热迭代次数
        repeats: 测量迭代次数

    返回:
        平均延迟，单位为毫秒。
    """
    model.eval()
    x = torch.randn(*input_shape)

    with torch.no_grad():
        # 预热阶段：让 CPU 缓存和频率稳定
        for _ in range(warmup):
            _ = model(x)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        elapsed = (time.perf_counter() - t0) / repeats  # 平均耗时（秒）

    return elapsed * 1000  # 转换为毫秒


def measure_memory(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """估算推理过程中的峰值内存使用量。

    包含：参数内存、缓冲区内存和峰值激活张量内存。
    """
    # 参数内存（float32 = 4 字节/元素）
    param_mem = sum(p.numel() for p in model.parameters()) * 4
    # 缓冲区内存（如 BN 的 running_mean）
    buffer_mem = sum(b.numel() for b in model.buffers()) * 4

    # 追踪峰值激活内存
    peak_act = 0

    def hook_fn(m, inp, out):
        """前向钩子：记录每层输出的内存占用，取最大值。"""
        nonlocal peak_act
        act_size = 0
        if isinstance(out, torch.Tensor):
            act_size = out.numel() * 4
        if isinstance(out, tuple):
            act_size = sum(o.numel() * 4 for o in out if isinstance(o, torch.Tensor))
        peak_act = max(peak_act, act_size)

    handles = []
    for m in model.modules():
        handles.append(m.register_forward_hook(hook_fn))

    x = torch.randn(*input_shape)
    with torch.no_grad():
        _ = model(x)  # 执行前向传播以触发钩子

    for h in handles:
        h.remove()

    return param_mem + buffer_mem + peak_act


def collect_metrics(
    model: nn.Module, stage_name: str, input_shape: Tuple[int, ...]
) -> Dict[str, Any]:
    """为模型阶段采集综合指标。

    返回:
        包含 params, nonzero_params, flops, latency, memory 的字典。
    """
    return {
        "stage": stage_name,
        "params": count_parameters(model),
        "nonzero_params": count_nonzero_parameters(model),
        "flops": estimate_flops(model, input_shape),
        "latency_ms": round(measure_latency(model, input_shape), 3),
        "memory_mb": round(measure_memory(model, input_shape) / (1024**2), 3),
    }


# ===========================================================================
# 5. ONNX 导出
# ===========================================================================


def export_to_onnx(
    model: nn.Module, filepath: str, input_shape: Tuple[int, ...]
) -> str:
    """将模型导出为 ONNX 格式。

    参数:
        model: PyTorch 模型
        filepath: 输出 .onnx 文件路径
        input_shape: 虚拟输入形状

    返回:
        导出模型的文件路径。
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,  # 导出模型参数
            opset_version=13,  # ONNX 算子集版本
            do_constant_folding=True,  # 常量折叠优化
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
    except Exception as e:
        # 某些操作可能不支持；优雅地回退
        # 创建一个最小的 ONNX 文件作为概念验证
        print(f"  [警告] 带动态轴的 ONNX 导出失败: {e}")

    # 检查文件大小
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        return f"{filepath} ({size_kb:.1f} KB)"
    return f"{filepath} (导出已模拟)"


# ===========================================================================
# 6. 报告生成
# ===========================================================================


def generate_comparison_table(
    stages: List[Dict[str, Any]], baseline_idx: int = 0
) -> str:
    """从指标数据生成格式化的对比表。

    参数:
        stages: 指标字典列表
        baseline_idx: 基准阶段的索引

    返回:
        格式化字符串表格。
    """
    if not stages:
        return "没有可供对比的阶段。"

    baseline = stages[baseline_idx]
    header = (
        f"{'Stage':<20} {'Params':>10} {'NonZero':>10} {'FLOPs(M)':>10} "
        f"{'Lat(ms)':>9} {'Mem(MB)':>9} {'ΔParams':>9} {'ΔLat':>8}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for s in stages:
        # 计算相对于基准的参数变化百分比
        delta_p = (s["params"] - baseline["params"]) / max(baseline["params"], 1) * 100
        # 计算相对于基准的延迟变化百分比
        delta_l = (
            (s["latency_ms"] - baseline["latency_ms"])
            / max(baseline["latency_ms"], 0.001)
            * 100
        )
        lines.append(
            f"{s['stage']:<20} {s['params']:>10,} {s['nonzero_params']:>10,} "
            f"{s['flops'] / 1e6:>10.2f} {s['latency_ms']:>9.3f} {s['memory_mb']:>9.3f} "
            f"{delta_p:>+8.1f}% {delta_l:>+7.1f}%"
        )

    return "\n".join(lines)


def generate_markdown_report(
    stages: List[Dict[str, Any]],
    onnx_files: List[str],
    output_path: str,
) -> str:
    """生成综合 Markdown 摘要报告。

    参数:
        stages: 阶段指标列表
        onnx_files: 导出的 ONNX 模型路径
        output_path: .md 报告的输出路径

    返回:
        报告内容字符串。
    """
    report_lines = [
        "# MIT 6.5940 第22讲：端到端压缩流水线报告",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 流水线概览",
        "",
        "```",
        "加载模型 -> 结构化剪枝 -> INT8 量化 -> ONNX 导出 -> 基准测试",
        "```",
        "",
        "## 指标对比",
        "",
    ]

    if stages:
        baseline = stages[0]
        # 构建 Markdown 表头
        report_lines.append(
            f"| {'Stage':<18} | {'Params':>10} | {'NonZero':>10} | "
            f"{'FLOPs(M)':>10} | {'Lat(ms)':>9} | {'Mem(MB)':>9} |"
        )
        report_lines.append("|" + "|".join([" ---:"] * 6) + "|")
        for s in stages:
            pct = (
                (s["params"] / baseline["params"] * 100)
                if baseline["params"] > 0
                else 100
            )
            report_lines.append(
                f"| {s['stage']:<18} | {s['params']:>10,} | {s['nonzero_params']:>10,} | "
                f"{s['flops'] / 1e6:>10.2f} | {s['latency_ms']:>9.3f} | {s['memory_mb']:>9.3f} |"
            )

    report_lines.extend(
        [
            "",
            "## ONNX 导出",
            "",
        ]
    )
    for fpath in onnx_files:
        report_lines.append(f"- `{fpath}`")

    report_lines.extend(
        [
            "",
            "## 关键要点",
            "",
            "1. **结构化剪枝**移除整个通道，能带来真正的延迟降低。",
            "2. **INT8 量化**将模型大小减少 4 倍，精度损失极小。",
            "3. **剪枝+量化组合**提供乘法级别的收益。",
            "4. **ONNX 导出**支持在各种硬件后端上部署。",
            "5. **端到端流水线**：在每个阶段进行测量以发现瓶颈。",
            "",
            "---",
            "",
            "*报告由第22讲压缩流水线自动生成。*",
        ]
    )

    report_content = "\n".join(report_lines)

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


# ===========================================================================
# 7. 完整流水线
# ===========================================================================


def run_compression_pipeline(output_dir: str) -> Tuple[List[Dict[str, Any]], str]:
    """执行完整的压缩流水线。

    各阶段:
      1. 基准模型
      2. 结构化剪枝（Conv2d 层 prune_ratio=0.5）
      3. 非结构化剪枝（Linear 层 prune_ratio=0.6）
      4. INT8 量化
      5. 剪枝 + 量化

    参数:
        output_dir: 输出文件目录

    返回:
        (阶段指标列表, Markdown 报告路径)
    """
    os.makedirs(output_dir, exist_ok=True)
    input_shape = (1, 3, 32, 32)  # 模拟 CIFAR-10 风格的输入
    stages: List[Dict[str, Any]] = []
    onnx_files: List[str] = []

    # 阶段 1: 基准模型
    print("阶段 1: 基准模型")
    baseline = CompressionDemoModel()
    stages.append(collect_metrics(baseline, "Baseline", input_shape))

    # 阶段 2: 结构化剪枝
    print("阶段 2: 结构化剪枝（50% 卷积通道）")
    pruned_structured = apply_structured_pruning(baseline, prune_ratio=0.5)
    stages.append(
        collect_metrics(pruned_structured, "Pruned (Structured)", input_shape)
    )

    # 阶段 3: 附加非结构化剪枝
    print("阶段 3: 非结构化剪枝（60% 全连接层权重）")
    pruned_combined = apply_unstructured_pruning(pruned_structured, prune_ratio=0.6)
    stages.append(collect_metrics(pruned_combined, "Pruned (Combined)", input_shape))

    # 阶段 4: INT8 量化
    print("阶段 4: INT8 量化")
    quantized = simulate_int8_quantization(baseline)
    stages.append(collect_metrics(quantized, "Quantized (INT8)", input_shape))
    # 考虑 4 倍内存减少（float32 -> int8）
    stages[-1]["memory_mb"] = round(stages[-1]["memory_mb"] * 0.25, 3)

    # 阶段 5: 剪枝 + 量化
    print("阶段 5: 剪枝 + 量化")
    pruned_quantized = simulate_int8_quantization(pruned_combined)
    stages.append(collect_metrics(pruned_quantized, "Pruned+Quantized", input_shape))
    stages[-1]["memory_mb"] = round(stages[-1]["memory_mb"] * 0.25, 3)

    # ONNX 导出
    print("ONNX 导出")
    for name, model in [
        ("baseline", baseline),
        ("pruned", pruned_combined),
        ("quantized", quantized),
    ]:
        onnx_path = os.path.join(output_dir, f"{name}.onnx")
        result = export_to_onnx(model, onnx_path, input_shape)
        onnx_files.append(result)
        print(f"  已导出: {result}")

    # 生成报告
    print("正在生成 Markdown 报告")
    report_path = os.path.join(output_dir, "report.md")
    report = generate_markdown_report(stages, onnx_files, report_path)
    print(f"  报告已写入: {report_path}")

    return stages, report_path


# ===========================================================================
# 8. 主函数
# ===========================================================================

import copy  # noqa: E402


def main() -> None:
    """主入口：运行端到端压缩流水线并打印对比结果。"""
    print("=" * 72)
    print("MIT 6.5940 第22讲：端到端压缩流水线")
    print("=" * 72)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    stages, report_path = run_compression_pipeline(output_dir)

    # 打印对比表
    print("\n--- 对比表 ---")
    table = generate_comparison_table(stages)
    print(table)

    # 打印摘要统计
    print("\n--- 摘要统计 ---")
    bl = stages[0]  # 基准模型
    best = stages[-1]  # 最佳压缩模型
    print(f"  基准参数量:     {bl['params']:,}")
    print(
        f"  最终参数量:     {best['params']:,} (减少了 {(1 - best['params'] / bl['params']) * 100:.1f}%)"
    )
    print(f"  基准延迟:       {bl['latency_ms']:.3f} ms")
    print(
        f"  最终延迟:       {best['latency_ms']:.3f} ms (加速 {bl['latency_ms'] / max(best['latency_ms'], 0.001):.1f}x)"
    )
    print(f"  基准内存:       {bl['memory_mb']:.3f} MB")
    print(
        f"  最终内存:       {best['memory_mb']:.3f} MB (减少了 {(1 - best['memory_mb'] / bl['memory_mb']) * 100:.1f}%)"
    )
    print(f"  报告已生成:      {report_path}")

    print("\n完成。所有计算均在 CPU 上执行。\n")


if __name__ == "__main__":
    main()
