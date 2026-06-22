"""
MCUNet / TinyML 内存预算仿真 (第10讲)
==========================================
本模块模拟在微控制器单元(MCU)上部署微型神经网络时的内存约束。
它实现了一个 TinyCNN 构建器，在实例化模型*之前*分析性地检查
SRAM 和 Flash 预算，并生成格式化的内存预算报告。

演示的关键概念:
  - SRAM 预算  (激活内存 / 运行时内存, 通常 ~256 KB)
  - Flash 预算 (参数存储, 通常 ~1 MB 片上Flash)
  - 解析性 MAC / 参数量 / 激活内存计算
  - 预算感知的架构构建, 拒绝无效配置

所有计算仅在 CPU 上进行; 仅依赖标准库 (torch, numpy)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

BYTES_PER_FLOAT32 = 4  # float32 元素大小（字节）

# 内存单位换算
KB = 1024
MB = 1024 * KB

# 默认 MCU 内存预算（可通过构造函数参数覆盖）
DEFAULT_SRAM_BUDGET = 256 * KB  # 256 KB  – 典型 Cortex-M4/M7 的 SRAM
DEFAULT_FLASH_BUDGET = 1 * MB  # 1 MB    – 典型片上 Flash


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """网络中单层的解析统计信息。"""

    name: str  # 层名称
    layer_type: str  # 层类型 (conv / relu / fc 等)
    spatial_in: str  # 输入空间形状, 例如 "1x28x28"
    spatial_out: str  # 输出空间形状, 例如 "8x26x26"
    params: int  # 可训练参数数量
    param_bytes: int  # 参数所占 Flash 空间（字节）
    macs: int  # 乘加运算次数
    activation_elements: int  # 输出张量元素个数
    activation_bytes: int  # 输出激活所占 SRAM 空间（字节）


@dataclass
class MCUMemoryReport:
    """TinyCNN 候选模型的聚合内存报告。"""

    model_name: str  # 模型名称
    layers: List[LayerStats] = field(default_factory=list)  # 各层统计信息
    total_params: int = 0  # 总参数数量
    total_param_bytes: int = 0  # 总参数 Flash 占用（字节）
    total_macs: int = 0  # 总 MAC 运算量
    peak_activation_bytes: int = 0  # 峰值激活内存（字节）
    peak_activation_kb: float = 0.0  # 峰值激活内存（KB）
    sram_budget_bytes: int = DEFAULT_SRAM_BUDGET  # SRAM 预算（字节）
    flash_budget_bytes: int = DEFAULT_FLASH_BUDGET  # Flash 预算（字节）
    sram_ok: bool = True  # SRAM 预算是否满足
    flash_ok: bool = True  # Flash 预算是否满足
    passed: bool = True  # 整体是否通过
    rejection_reason: str = ""  # 拒绝原因（如果未通过）


# ---------------------------------------------------------------------------
# 解析性辅助函数
# ---------------------------------------------------------------------------


def _conv_output_size(
    h_in: int,
    w_in: int,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
    dilation: int | Tuple[int, int] = 1,
) -> Tuple[int, int]:
    """计算 Conv2d / Pool2d 层的空间输出尺寸。

    使用标准卷积输出公式:
        H_out = floor((H_in + 2*P - D*(K-1) - 1) / S + 1)
    """
    # 将标量参数统一转换为元组
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # 分别计算高度和宽度方向的输出尺寸
    h_out = math.floor(
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w_out = math.floor(
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )
    return h_out, w_out


def _compute_conv2d_macs(
    in_c: int,
    out_c: int,
    k_h: int,
    k_w: int,
    out_h: int,
    out_w: int,
    groups: int = 1,
) -> int:
    """计算 Conv2d 层的 MAC（乘加运算）次数（不含偏置加法开销）。

    MACs = out_c * (in_c / groups) * k_h * k_w * out_h * out_w
    每次 MAC 对应一次乘法 + 一次加法，但这里仅计乘加对的数量。
    """
    return out_c * (in_c // groups) * k_h * k_w * out_h * out_w


def _compute_conv2d_params(
    in_c: int,
    out_c: int,
    k_h: int,
    k_w: int,
    bias: bool = True,
    groups: int = 1,
) -> int:
    """计算 Conv2d 层的可训练参数数量。

    权重参数量 = out_c * (in_c / groups) * k_h * k_w
    如果启用偏置, 额外加上 out_c 个参数。
    """
    params = out_c * (in_c // groups) * k_h * k_w
    if bias:
        params += out_c
    return params


def _compute_linear_macs(in_features: int, out_features: int) -> int:
    """计算全连接层的 MAC 次数。

    每个输出元素需要 in_features 次乘加操作。
    """
    return in_features * out_features


def _compute_linear_params(
    in_features: int, out_features: int, bias: bool = True
) -> int:
    """计算全连接层的可训练参数数量。

    权重矩阵大小 = in_features * out_features
    如果启用偏置, 额外加上 out_features 个参数。
    """
    params = in_features * out_features
    if bias:
        params += out_features
    return params


# ---------------------------------------------------------------------------
# TinyCNN 构建器
# ---------------------------------------------------------------------------


class TinyCNN(nn.Module):
    """用于 MCU 部署仿真的内存预算感知型微型 CNN。

    请勿直接实例化; 使用 ``build_tiny_cnn()`` 函数,
    它会验证预算并同时返回模型和 MCU 内存报告。
    """

    def __init__(self, layers: nn.ModuleList, report: MCUMemoryReport):
        super().__init__()
        self.features = layers  # 顺序排列的各层模块
        self.memory_report = report  # 关联的内存预算报告

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：依次通过所有特征层。"""
        for layer in self.features:
            x = layer(x)
        return x


def build_tiny_cnn(
    model_name: str,
    layer_configs: List[dict],
    input_shape: Tuple[int, int, int],  # (C, H, W) — 单样本的形状, 不含 batch 维度
    sram_budget: int = DEFAULT_SRAM_BUDGET,
    flash_budget: int = DEFAULT_FLASH_BUDGET,
    byte_width: int = BYTES_PER_FLOAT32,
) -> Tuple[Optional[TinyCNN], MCUMemoryReport]:
    """验证内存预算, 并在预算通过时构建 TinyCNN 模型。

    Parameters
    ----------
    model_name : str
        报告中显示的人类可读名称。
    layer_configs : list of dict
        按顺序排列的层描述字典列表。每个字典必须包含 ``"type"`` 键。
        支持的类型及其额外键值:

          - ``"conv"`` : ``out_channels``, ``kernel_size``, ``stride`` (默认 1),
            ``padding`` (默认 0), ``bias`` (默认 True), ``groups`` (默认 1)。
            ``in_channels`` 从当前空间跟踪器中推断。
          - ``"maxpool"`` / ``"avgpool"`` : ``kernel_size``, ``stride``,
            ``padding`` (默认 0)。
          - ``"relu"`` : 无需额外键值 (原地激活, 不改变空间尺寸)。
          - ``"flatten"`` : 无需额外键值。
          - ``"fc"`` : ``out_features``, ``bias`` (默认 True)。
            ``in_features`` 从展平后的维度推断。

    input_shape : (C, H, W)
        单个样本的形状 (排除 batch 维度)。
    sram_budget : int
        SRAM 预算, 以字节为单位 (默认 256 KB)。
    flash_budget : int
        Flash 预算, 以字节为单位 (默认 1 MB)。
    byte_width : int
        每个元素的字节数 (默认 4, 对应 float32)。

    Returns
    -------
    model : TinyCNN or None
        构建好的模型, 或 ``None`` (如果超出预算)。
    report : MCUMemoryReport
        详细的解析性内存报告。
    """
    # 初始化内存报告对象
    report = MCUMemoryReport(
        model_name=model_name,
        sram_budget_bytes=sram_budget,
        flash_budget_bytes=flash_budget,
    )

    # 当前空间跟踪器: 通道数 C, 高度 H, 宽度 W
    c, h, w = input_shape
    # 跟踪 FC 层之后的展平特征维度
    flattened_dim: Optional[int] = None
    flattened: bool = False
    total_params = 0  # 累计参数总数
    total_macs = 0  # 累计 MAC 总量
    peak_act_bytes = 0  # 峰值激活内存

    layers: List[LayerStats] = []
    modules: List[nn.Module] = []

    # 输入激活的内存占用 (不含 batch 维度)
    input_act_bytes = c * h * w * byte_width
    peak_act_bytes = max(peak_act_bytes, input_act_bytes)

    # 逐层解析配置
    for idx, cfg in enumerate(layer_configs):
        lt = cfg["type"]
        layer_name = f"{lt}_{idx}"

        # ===================================================================
        # 卷积层处理
        # ===================================================================
        if lt == "conv":
            out_c = cfg["out_channels"]  # 输出通道数
            k = cfg.get("kernel_size", 3)  # 卷积核尺寸, 默认 3x3
            s = cfg.get("stride", 1)  # 步长, 默认 1
            p = cfg.get("padding", 0)  # 填充, 默认 0
            bias = cfg.get("bias", True)  # 是否使用偏置
            groups = cfg.get("groups", 1)  # 分组数, 默认 1

            # 将标量核尺寸展开为 (k_h, k_w)
            if isinstance(k, int):
                k_h, k_w = k, k
            else:
                k_h, k_w = k

            # 计算输出空间尺寸
            h_out, w_out = _conv_output_size(h, w, k, s, p)

            # 解析性统计: 参数量、MAC、激活内存
            params = _compute_conv2d_params(c, out_c, k_h, k_w, bias, groups)
            macs = _compute_conv2d_macs(c, out_c, k_h, k_w, h_out, w_out, groups)
            act_elems = out_c * h_out * w_out
            act_bytes = act_elems * byte_width

            # 构建实际的 nn.Conv2d 模块
            modules.append(
                nn.Conv2d(
                    c,
                    out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=bias,
                    groups=groups,
                )
            )

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = f"{out_c}x{h_out}x{w_out}"

            # 更新空间跟踪器: 输出变为下一层的输入
            c, h, w = out_c, h_out, w_out

        # ===================================================================
        # 池化层处理 (maxpool / avgpool)
        # ===================================================================
        elif lt in ("maxpool", "avgpool"):
            k = cfg.get("kernel_size", 2)  # 池化核尺寸
            s = cfg.get("stride", k)  # 步长, 默认等于核尺寸
            p = cfg.get("padding", 0)  # 填充

            h_out, w_out = _conv_output_size(h, w, k, s, p)
            params = 0  # 池化层无可训练参数
            macs = 0  # 池化 MAC 极小, 忽略不计
            act_elems = c * h_out * w_out
            act_bytes = act_elems * byte_width

            # 根据类型选择相应的池化模块
            pool_cls = nn.MaxPool2d if lt == "maxpool" else nn.AvgPool2d
            modules.append(pool_cls(kernel_size=k, stride=s, padding=p))

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = f"{c}x{h_out}x{w_out}"

            # 更新空间跟踪器 (通道数不变, 仅 H/W 变化)
            h, w = h_out, w_out

        # ===================================================================
        # ReLU 激活层处理
        # ===================================================================
        elif lt == "relu":
            # 原地操作 (inplace=True) 以节省内存
            modules.append(nn.ReLU(inplace=True))
            params = 0
            macs = 0
            act_elems = c * h * w
            act_bytes = act_elems * byte_width

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = spatial_in_str  # ReLU 不改变形状

        # ===================================================================
        # Flatten 展平层处理
        # ===================================================================
        elif lt == "flatten":
            flattened_dim = c * h * w  # 计算展平后的特征维度
            flattened = True
            modules.append(nn.Flatten())
            params = 0
            macs = 0
            act_elems = flattened_dim
            act_bytes = act_elems * byte_width

            spatial_in_str = f"{c}x{h}x{w}"
            spatial_out_str = str(flattened_dim)

        # ===================================================================
        # 全连接层处理
        # ===================================================================
        elif lt == "fc":
            # 如果前面没有显式展平, 则自动展平
            if not flattened:
                flattened_dim = c * h * w
                flattened = True

            out_f = cfg["out_features"]
            bias = cfg.get("bias", True)

            params = _compute_linear_params(flattened_dim, out_f, bias)
            macs = _compute_linear_macs(flattened_dim, out_f)
            act_elems = out_f
            act_bytes = act_elems * byte_width

            modules.append(nn.Linear(flattened_dim, out_f, bias=bias))

            spatial_in_str = str(flattened_dim)
            spatial_out_str = str(out_f)

            # FC 层之后不再处于空间模式; 更新 tracker 为虚拟值
            flattened_dim = out_f
            c, h, w = out_f, 1, 1  # 虚拟空间跟踪器

        else:
            raise ValueError(f"不支持的层类型: {lt}")

        # ----- 累积全局统计 -----
        param_bytes = params * byte_width
        total_params += params
        total_macs += macs
        peak_act_bytes = max(peak_act_bytes, act_bytes)

        # ----- 逐层预算检查 (快速失败) -----
        layer_fail_reason = ""
        # 检查该层的输出激活是否超出 SRAM 预算
        if act_bytes > sram_budget:
            layer_fail_reason = (
                f"Layer '{layer_name}' output activation ({act_bytes:,} bytes) "
                f"exceeds SRAM budget ({sram_budget:,} bytes)"
            )
        # 检查该层的参数是否超出 Flash 预算
        if param_bytes > flash_budget:
            layer_fail_reason += (
                f"{' | ' if layer_fail_reason else ''}"
                f"Layer '{layer_name}' parameters ({param_bytes:,} bytes) "
                f"exceed Flash budget ({flash_budget:,} bytes)"
            )

        # 如果该层失败, 记录原因并停止构建
        if layer_fail_reason:
            report.rejection_reason = layer_fail_reason
            report.passed = False
            report.sram_ok = False if "SRAM" in layer_fail_reason else report.sram_ok
            report.flash_ok = False if "Flash" in layer_fail_reason else report.flash_ok
            # 仍记录该层的统计信息, 以便报告具有诊断价值
            layers.append(
                LayerStats(
                    name=layer_name,
                    layer_type=lt,
                    spatial_in=spatial_in_str,
                    spatial_out=spatial_out_str,
                    params=params,
                    param_bytes=param_bytes,
                    macs=macs,
                    activation_elements=act_elems,
                    activation_bytes=act_bytes,
                )
            )
            break  # 终止构建: 该模型不可行

        # 通过检查, 记录该层统计信息
        layers.append(
            LayerStats(
                name=layer_name,
                layer_type=lt,
                spatial_in=spatial_in_str,
                spatial_out=spatial_out_str,
                params=params,
                param_bytes=param_bytes,
                macs=macs,
                activation_elements=act_elems,
                activation_bytes=act_bytes,
            )
        )

    # ----- 全局聚合检查 -----
    total_param_bytes = total_params * byte_width

    # 检查总参数大小是否超出 Flash 预算
    if report.passed and total_param_bytes > flash_budget:
        report.passed = False
        report.flash_ok = False
        report.rejection_reason = (
            f"Total parameter storage ({total_param_bytes:,} bytes) "
            f"exceeds Flash budget ({flash_budget:,} bytes)"
        )

    # 检查峰值激活内存是否超出 SRAM 预算
    if report.passed and peak_act_bytes > sram_budget:
        report.passed = False
        report.sram_ok = False
        report.rejection_reason = (
            f"Peak activation memory ({peak_act_bytes:,} bytes) "
            f"exceeds SRAM budget ({sram_budget:,} bytes)"
        )

    # ----- 填充报告 -----
    report.layers = layers
    report.total_params = total_params
    report.total_param_bytes = total_param_bytes
    report.total_macs = total_macs
    report.peak_activation_bytes = peak_act_bytes
    report.peak_activation_kb = peak_act_bytes / KB

    # 未通过预算检查, 返回 None 作为模型
    if not report.passed:
        return None, report

    # 通过预算检查, 构建并返回实际模型
    model = TinyCNN(nn.ModuleList(modules), report)
    return model, report


# ---------------------------------------------------------------------------
# 报告格式化输出
# ---------------------------------------------------------------------------


def format_report_table(report: MCUMemoryReport) -> str:
    """将 MCU 内存预算报告格式化为多行字符串。

    输出包含:
      - 预算摘要 (SRAM / Flash 上限及实际使用量)
      - 通过 / 失败状态
      - 逐层详细信息表 (参数量、MAC、激活内存)
      - 内存利用率百分比
    """
    sep = "-" * 100
    lines: List[str] = [
        sep,
        f"  MCU Memory Budget Report  |  Model: {report.model_name}",
        sep,
        f"  SRAM Budget : {report.sram_budget_bytes:>10,} bytes  "
        f"({report.sram_budget_bytes / KB:8.1f} KB)  |  "
        f"Peak Activation : {report.peak_activation_bytes:>8,} bytes  "
        f"({report.peak_activation_kb:8.1f} KB)",
        f"  Flash Budget: {report.flash_budget_bytes:>10,} bytes  "
        f"({report.flash_budget_bytes / KB:8.1f} KB)  |  "
        f"Total Parameters: {report.total_param_bytes:>8,} bytes  "
        f"({report.total_param_bytes / KB:8.1f} KB)",
        sep,
        f"  {'RESULT':>8s}: {'PASS' if report.passed else 'FAIL'}",
    ]
    if not report.passed:
        lines.append(f"  Reason: {report.rejection_reason}")
    lines.append(sep)

    # 层详细信息表头
    lines.append(
        f"  {'Layer':<16s} {'Type':<8s} {'Spatial In':>12s} {'Spatial Out':>12s} "
        f"{'Params':>8s} {'Param(B)':>10s} {'MACs':>12s} {'Act(B)':>10s}"
    )
    lines.append("  " + "-" * 94)

    # 逐层输出统计
    for ls in report.layers:
        lines.append(
            f"  {ls.name:<16s} {ls.layer_type:<8s} "
            f"{ls.spatial_in:>12s} {ls.spatial_out:>12s} "
            f"{ls.params:>8,d} {ls.param_bytes:>10,d} "
            f"{ls.macs:>12,d} {ls.activation_bytes:>10,d}"
        )

    lines.append("  " + "-" * 94)
    # 总计行
    lines.append(
        f"  {'TOTAL':<16s} {'':8s} {'':>12s} {'':>12s} "
        f"{report.total_params:>8,d} {report.total_param_bytes:>10,d} "
        f"{report.total_macs:>12,d} {report.peak_activation_bytes:>10,d} (peak)"
    )
    lines.append(sep)

    # 内存利用率摘要
    sram_pct = (report.peak_activation_bytes / report.sram_budget_bytes) * 100
    flash_pct = (report.total_param_bytes / report.flash_budget_bytes) * 100
    lines.append(f"  SRAM  utilization: {sram_pct:5.1f}%")
    lines.append(f"  Flash utilization: {flash_pct:5.1f}%")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 快速完整性检查（自检 / 冒烟测试）
# ---------------------------------------------------------------------------


def _run_sanity_check() -> None:
    """验证一个已知的小型架构能产生正确的解析统计值。

    测试配置: Conv(1->2, k=3) + ReLU + Flatten + FC(2*6*6=72, 5)
    输入: (1, 8, 8)
    使用宽松的预算以确保通过。
    """
    configs: List[dict] = [
        {"type": "conv", "out_channels": 2, "kernel_size": 3},
        {"type": "relu"},
        {"type": "flatten"},
        {"type": "fc", "out_features": 5},
    ]
    _, report = build_tiny_cnn(
        "SanityCheck",
        configs,
        input_shape=(1, 8, 8),
        sram_budget=1 * MB,  # 宽容预算, 确保通过
        flash_budget=10 * MB,
    )
    model = TinyCNN.__new__(TinyCNN)  # 绕过 __init__ 手动计算期望值
    # 手动计算期望值
    with torch.no_grad():
        conv = nn.Conv2d(1, 2, 3)
        fc = nn.Linear(2 * 6 * 6, 5)
    # 期望总参数量 = conv 参数 + fc 参数
    expected_params = sum(
        p.numel() for p in list(conv.parameters()) + list(fc.parameters())
    )
    # 期望卷积 MACs = 2 * 1 * 3 * 3 * 6 * 6 = 648
    expected_conv_macs = 2 * 1 * 3 * 3 * 6 * 6  # 648
    # 期望 FC MACs = 72 * 5 = 360
    expected_fc_macs = 72 * 5  # 360
    expected_macs = expected_conv_macs + expected_fc_macs
    # 期望峰值激活内存 = max(输入, conv输出, relu输出, flatten输出, fc输出) * 4 字节
    expected_peak_act = max(
        1 * 8 * 8 * 4,  # 输入: 256 字节
        2 * 6 * 6 * 4,  # conv 输出: 288 字节
        2 * 6 * 6 * 4,  # relu (相同): 288 字节
        72 * 4,  # flatten: 288 字节
        5 * 4,  # fc: 20 字节
    )

    # 断言验证
    assert report.total_params == expected_params, (
        f"Params: {report.total_params} != {expected_params}"
    )
    assert report.total_macs == expected_macs, (
        f"MACs: {report.total_macs} != {expected_macs}"
    )
    assert report.peak_activation_bytes == expected_peak_act, (
        f"Peak act: {report.peak_activation_bytes} != {expected_peak_act}"
    )
    assert report.passed, "Sanity check model should pass budgets"
    print("[sanity] 所有断言通过。")


# ---------------------------------------------------------------------------
# 主演示程序
# ---------------------------------------------------------------------------


def main() -> None:
    """展示 TinyCNN 预算仿真器: 分别测试有效和无效的架构。

    演示内容:
      1. 自检 (冒烟测试)
      2. 有效架构: TinyNet (适配 256 KB SRAM / 1 MB Flash)
      3. 无效架构: WideNet (超出 SRAM 预算)
      4. 无效架构: FatFC (超大 FC 层超出 Flash 预算)
    """
    print("=" * 100)
    print("  MCUNet / TinyML 内存预算仿真器  |  第10讲")
    print("=" * 100)
    print()

    # ------------------------------------------------------------------
    # 1. 自检 (冒烟测试)
    # ------------------------------------------------------------------
    print("--- 自检 ---")
    _run_sanity_check()
    print()

    # ------------------------------------------------------------------
    # 2. 有效架构: TinyNet (适配 256 KB SRAM / 1 MB Flash)
    # ------------------------------------------------------------------
    tiny_net_configs: List[dict] = [
        # 输入: 1x28x28 (MNIST 风格灰度图)
        {
            "type": "conv",
            "out_channels": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        {"type": "relu"},
        {"type": "maxpool", "kernel_size": 2, "stride": 2},
        {
            "type": "conv",
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        {"type": "relu"},
        {"type": "maxpool", "kernel_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "fc", "out_features": 10},
    ]

    model_valid, report_valid = build_tiny_cnn(
        "TinyNet (有效)",
        tiny_net_configs,
        input_shape=(1, 28, 28),
    )

    print(format_report_table(report_valid))
    print()

    if model_valid is not None:
        # 快速 CPU 前向传播验证
        with torch.no_grad():
            dummy = torch.randn(1, 1, 28, 28)
            out = model_valid(dummy)
        print(f"  前向传播通过。输出形状: {tuple(out.shape)}")
        print(
            f"  磁盘模型大小 (state_dict): "
            f"{sum(p.numel() for p in model_valid.parameters()) * BYTES_PER_FLOAT32:,} 字节"
        )
    print()
    print()

    # ------------------------------------------------------------------
    # 3. 无效架构: WideNet (超出 SRAM 预算)
    #     128 通道 x 24x24 空间 = 73,728 元素 x 4 字节 = 294,912 字节 > 256 KB
    # ------------------------------------------------------------------
    wide_net_configs: List[dict] = [
        # 输入: 1x28x28
        {
            "type": "conv",
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        {"type": "relu"},
        {
            "type": "conv",
            "out_channels": 128,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
        },
        # ^^^ 128x24x24 = 73,728 元素 = 294,912 字节 > 256 KB → 超出 SRAM 预算
        {"type": "relu"},
        {"type": "flatten"},
        {"type": "fc", "out_features": 10},
    ]

    model_wide, report_wide = build_tiny_cnn(
        "WideNet (无效 - 超出SRAM预算)",
        wide_net_configs,
        input_shape=(1, 28, 28),
    )

    print(format_report_table(report_wide))
    print()

    assert model_wide is None, "WideNet 应该被拒绝"
    print("  (正确拒绝了 WideNet 构建 – SRAM 预算超出)")
    print()

    # ------------------------------------------------------------------
    # 4. 无效架构: FatFC (超大 FC 层超出 Flash 预算)
    #     4x26x26 = 2704 → FC(2704, 1024)
    #     2704*1024 ≈ 2.77M 参数 ≈ 11 MB → 超出 Flash 预算
    # ------------------------------------------------------------------
    fat_fc_configs: List[dict] = [
        {"type": "conv", "out_channels": 4, "kernel_size": 3},
        {"type": "relu"},
        {"type": "flatten"},
        # 4x26x26 = 2704 → FC(2704, 1024)
        # 2704*1024 ≈ 2.77 M 参数 ≈ 11 MB → 超出 Flash 预算
        {"type": "fc", "out_features": 1024},
    ]

    model_fat, report_fat = build_tiny_cnn(
        "FatFC (无效 - 超出Flash预算)",
        fat_fc_configs,
        input_shape=(1, 28, 28),
    )

    print(format_report_table(report_fat))
    print()

    assert model_fat is None, "FatFC 应该被拒绝"
    print("  (正确拒绝了 FatFC 构建 – Flash 预算超出)")
    print()
    print("=" * 100)
    print("  演示完成。")
    print("=" * 100)


if __name__ == "__main__":
    main()
