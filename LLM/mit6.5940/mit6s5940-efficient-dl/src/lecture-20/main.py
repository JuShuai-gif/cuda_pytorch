#!/usr/bin/env python3
"""
MIT 6.5940 第20讲：梯度压缩与混合并行

涵盖主题：
  - 模拟深度梯度压缩（DGC）：Top-k稀疏化 + 动量修正
  - 1-Bit SGD模拟：将梯度量化为1比特并配合误差反馈
  - 对比：通信量减少 vs 精度影响
  - 混合并行内存计算：在给定模型上的DP + PP + TP

所有计算均在CPU上运行，无需GPU。
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 随机种子
# ===========================================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ===========================================================================
# 1. 深度梯度压缩（DGC）：Top-k稀疏化
# ===========================================================================


class DeepGradientCompression:
    """实现带动量修正和误差反馈的深度梯度压缩（DGC）。

    参考：Lin等人，"Deep Gradient Compression: Reducing the
    Communication Bandwidth for Distributed Training"，ICLR 2018。

    算法：
      1. 按幅值选择最大的k%梯度（稀疏化）
      2. 将小梯度累积到误差残差中（误差反馈）
      3. 应用动量修正以补偿陈旧性
    """

    def __init__(self, sparsity: float = 0.99, momentum: float = 0.9):
        """
        参数：
            sparsity: 要置零的梯度比例（0.99 = 保留最大的1%）
            momentum: 用于修正的动量因子
        """
        self.sparsity = sparsity
        self.momentum = momentum
        self.residual: torch.Tensor | None = None  # 误差残差缓存
        self.momentum_buffer: torch.Tensor | None = None  # 动量缓存

    def compress(self, grad: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """使用Top-k稀疏化压缩梯度张量。

        参数：
            grad: 待压缩的梯度张量

        返回：
            (压缩后的梯度, 非零元素数量)
        """
        # 添加残差（误差反馈机制）
        if self.residual is not None and self.residual.shape == grad.shape:
            grad = grad + self.residual

        g = grad.flatten()
        total_elements = g.numel()
        # 计算需要保留的元素数量k（基于稀疏度）
        k = max(1, int(total_elements * (1.0 - self.sparsity)))

        # Top-k：仅保留绝对值最大的k个元素
        _, indices = torch.topk(g.abs(), k)
        mask = torch.zeros_like(g)
        mask[indices] = 1.0
        compressed = g * mask

        # 存储残差（未被发送的值，将在下一轮累加）
        self.residual = (g - compressed).reshape(grad.shape)

        # 动量修正：平滑压缩后的梯度方向
        if self.momentum_buffer is None or self.momentum_buffer.shape != grad.shape:
            self.momentum_buffer = torch.zeros_like(grad)
        compressed_reshaped = compressed.reshape(grad.shape)
        self.momentum_buffer = (
            self.momentum * self.momentum_buffer + compressed_reshaped
        )

        return self.momentum_buffer.clone(), k


def simulate_dgc_impact(
    grad_magnitudes: List[float], sparsities: List[float]
) -> Dict[float, Dict[str, float]]:
    """模拟DGC对梯度重构质量的影响。

    参数：
        grad_magnitudes: 按降序排列的梯度幅值列表
        sparsities: 要测试的稀疏度比例列表

    返回：
        将稀疏度映射到 {保留能量, 压缩比} 的字典。
    """
    results = {}
    g = torch.tensor(grad_magnitudes, dtype=torch.float32)
    total_energy = (g**2).sum().item()  # 总的梯度能量

    for s in sparsities:
        k = max(1, int(len(g) * (1.0 - s)))
        # 计算Top-k元素所保留的能量比例
        topk_energy = (g[:k] ** 2).sum().item()
        energy_retained = topk_energy / total_energy if total_energy > 0 else 1.0
        compression = 1.0 / (1.0 - s) if s < 1.0 else float("inf")
        results[s] = {
            "energy_retained": energy_retained,
            "compression_ratio": compression,
            "values_sent": k,
        }
    return results


# ===========================================================================
# 2. 1-Bit SGD（带误差反馈）
# ===========================================================================


class OneBitSGD:
    """1-Bit SGD：将梯度量化为1比特并配合误差反馈。

    参考：Seide等人，"1-Bit Stochastic Gradient Descent and its
    Application to Data-Parallel Distributed Training of Speech DNNs"，
    Interspeech 2014。

    算法：
      1. 将残差误差加到当前梯度上
      2. 计算sign(grad) * mean(|grad|) —— 1比特用于符号，1个标量用于缩放
      3. 用量化误差更新残差
    """

    def __init__(self):
        self.residual: torch.Tensor | None = None  # 误差残差缓存

    def compress(self, grad: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """将梯度压缩为1比特表示。

        参数：
            grad: 梯度张量

        返回：
            (量化后的梯度, 实际通信的比特数)
        """
        # 添加误差反馈
        if self.residual is not None and self.residual.shape == grad.shape:
            grad = grad + self.residual

        # 1比特量化：仅保留符号，乘以平均绝对值作为缩放因子
        scale = grad.abs().mean()
        quantized = torch.sign(grad) * scale

        # 存储残差误差
        self.residual = grad - quantized

        # 通信量：每个元素1比特 + 1个float32标量
        bits = grad.numel() * 1 + 32
        return quantized, bits


def compare_compression_methods(
    grad: torch.Tensor,
    sparsity: float = 0.99,
) -> Dict[str, Dict[str, float]]:
    """在同一个梯度张量上比较DGC和1-Bit SGD。

    参数：
        grad: 样本梯度张量
        sparsity: DGC稀疏度比例

    返回：
        每种方法的对比指标。
    """
    orig_size = grad.numel() * 32  # 原始bit数（FP32）

    # DGC压缩
    dgc = DeepGradientCompression(sparsity=sparsity)
    dgc_comp, dgc_nnz = dgc.compress(grad)
    # DGC通信量 = 值(32bit) + 索引(log2(N) bit)
    dgc_bits = dgc_nnz * 32 + dgc_nnz * math.ceil(math.log2(grad.numel()))  # 值 + 索引
    dgc_cosine = F.cosine_similarity(grad.flatten(), dgc_comp.flatten(), dim=0).item()

    # 1-Bit SGD压缩
    onebit = OneBitSGD()
    onebit_comp, onebit_bits = onebit.compress(grad)
    onebit_cosine = F.cosine_similarity(
        grad.flatten(), onebit_comp.flatten(), dim=0
    ).item()

    return {
        "original": {"bits": float(orig_size), "compression": 1.0},
        "dgc": {
            "bits": float(dgc_bits),
            "compression": orig_size / max(dgc_bits, 1),
            "cosine_sim": dgc_cosine,
        },
        "1bit_sgd": {
            "bits": float(onebit_bits),
            "compression": orig_size / max(onebit_bits, 1),
            "cosine_sim": onebit_cosine,
        },
    }


# ===========================================================================
# 3. 混合并行内存计算器
# ===========================================================================


def hybrid_parallelism_memory(
    model_params: int,
    hidden_dim: int,
    num_layers: int,
    dp_size: int = 1,
    pp_size: int = 1,
    tp_size: int = 1,
    batch_size: int = 64,
    seq_len: int = 512,
) -> Dict[str, float]:
    """计算混合DP + PP + TP模式下的内存使用。

    内存组成：
      - 模型参数（P）：按PP（每阶段）和TP划分
      - 梯度（G）：与参数相同的划分方式
      - 优化器状态（O）：Adam的m+v，与参数相同的方式划分
      - 激活值（A）：与每个PP阶段的batch/seq长度成正比

    参数：
        model_params: 总参数数量
        hidden_dim: 模型隐藏维度
        num_layers: Transformer层数
        dp_size: 数据并行副本数
        pp_size: 流水线并行阶段数
        tp_size: 张量并行大小
        batch_size: 全局批次大小
        seq_len: 序列长度

    返回：
        每个设备的内存拆解（GB）。
    """
    total_devices = dp_size * pp_size * tp_size
    device_factor = pp_size * tp_size

    # 参数：按PP和TP划分，按DP复制
    P = model_params * 4 / device_factor

    # 梯度：与参数相同的划分方式
    G = P

    # 优化器状态（Adam）：m + v，每个可训练参数占2倍FP32
    O = P * 2

    # 激活值（粗略估计）：每个Transformer层约 34 * b * s * h
    # 每个PP微批次存储用于反向传播
    micro_batch = max(1, batch_size // dp_size)
    layers_per_stage = max(1, num_layers // pp_size)
    A = 34 * micro_batch * seq_len * hidden_dim * layers_per_stage * 4

    total_mem = P + G + O + A

    return {
        "config": f"DP={dp_size} PP={pp_size} TP={tp_size} (total_devices={total_devices})",
        "params_gb": P / 1e9,
        "grads_gb": G / 1e9,
        "optimizer_gb": O / 1e9,
        "activations_gb": A / 1e9,
        "total_mem_gb": total_mem / 1e9,
    }


# ===========================================================================
# 4. 用于DGC模拟的梯度分布生成
# ===========================================================================


def generate_power_law_gradients(n: int, alpha: float = 1.5) -> torch.Tensor:
    """生成服从幂律分布的梯度幅值。

    真实神经网络的梯度通常服从幂律分布，
    这使得Top-k稀疏化非常有效。
    """
    ranks = np.arange(1, n + 1)
    magnitudes = ranks ** (-alpha)
    magnitudes = magnitudes / magnitudes.sum()
    # 添加噪声使分布更真实
    noise = np.random.normal(0, 0.01, n)
    magnitudes = np.abs(magnitudes + noise)
    # 按降序排列（最大的在前）
    magnitudes.sort()
    return torch.tensor(magnitudes[::-1].copy(), dtype=torch.float32)


# ===========================================================================
# 5. 主演示
# ===========================================================================


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 第20讲：梯度压缩与混合并行")
    print("=" * 72)

    # ---------- DGC：Top-k稀疏化 ----------
    print("\n--- 1. 深度梯度压缩（DGC）---")
    n_elements = 1_000_000
    grad = generate_power_law_gradients(n_elements, alpha=1.5)

    print(f"  梯度张量大小: {n_elements:,} 个元素")
    print(f"  分布: 幂律分布 (alpha=1.5)")

    sparsities = [0.90, 0.95, 0.99, 0.999]
    dgc_results = simulate_dgc_impact(grad.tolist(), sparsities)
    print(f"  {'稀疏度':>10} {'保留能量':>16} {'压缩比':>14} {'发送值数':>13}")
    print(f"  {'-' * 55}")
    for s in sparsities:
        r = dgc_results[s]
        print(
            f"  {s:>10.1%} {r['energy_retained']:>16.1%} {r['compression_ratio']:>14.1f}x "
            f"{r['values_sent']:>13,}"
        )

    # ---------- 带动量修正的DGC ----------
    print("\n--- 2. 带动量修正的DGC ---")
    dgc = DeepGradientCompression(sparsity=0.99, momentum=0.9)
    sample_grad = torch.randn(100, 100)
    compressed, nnz = dgc.compress(sample_grad)
    orig_norm = sample_grad.norm().item()
    comp_norm = compressed.norm().item()
    print(f"  原始梯度范数: {orig_norm:.4f}")
    print(f"  压缩后梯度范数: {comp_norm:.4f}")
    print(f"  非零比例: {nnz / sample_grad.numel():.2%}")

    # 模拟多步训练中的误差反馈效果
    print("  多步误差反馈模拟:")
    dgc2 = DeepGradientCompression(sparsity=0.99, momentum=0.9)
    grad_sequence = [torch.randn(100) * 0.1 + torch.ones(100) for _ in range(5)]
    for step, g in enumerate(grad_sequence):
        compressed, nnz = dgc2.compress(g)
        error_norm = dgc2.residual.norm().item() if dgc2.residual is not None else 0
        print(f"    步骤 {step}: 非零数={nnz}/{g.numel()}, 残差范数={error_norm:.4f}")

    # ---------- 1-Bit SGD ----------
    print("\n--- 3. 1-Bit SGD模拟 ---")
    sample_grad_1bit = torch.randn(10000)
    onebit = OneBitSGD()

    for step in range(3):
        grad_step = sample_grad_1bit + torch.randn(10000) * 0.01
        quantized, bits = onebit.compress(grad_step)
        # 模拟：仅符号方向重要
        sign_match = (
            (torch.sign(grad_step) == torch.sign(quantized)).float().mean().item()
        )
        print(
            f"  步骤 {step}: 比特数={bits} ({bits / (grad_step.numel() * 32):.1%} of FP32), "
            f"符号匹配率={sign_match:.1%}"
        )

    # ---------- 压缩方法对比 ----------
    print("\n--- 4. 压缩方法对比 ---")
    test_grad = generate_power_law_gradients(10000, alpha=1.2)
    comparison = compare_compression_methods(test_grad, sparsity=0.99)
    print(f"  {'方法':<14} {'比特数':>10} {'压缩比':>14} {'余弦相似度':>12}")
    print(f"  {'-' * 52}")
    for method, metrics in comparison.items():
        cs = (
            f"{metrics.get('cosine_sim', 1.0):.4f}"
            if "cosine_sim" in metrics
            else "1.0000"
        )
        print(
            f"  {method:<14} {metrics['bits']:>10.0f} {metrics['compression']:>14.1f}x {cs:>12}"
        )

    # ---------- 混合并行内存 ----------
    print("\n--- 5. 混合并行（DP+PP+TP）内存 ---")
    # 模拟一个类似GPT-3的模型（175B参数，96层，hidden=12288）
    model_params_large = 175_000_000_000
    hidden_dim_large = 12288
    num_layers_large = 96

    configs = [
        (64, 1, 1),  # 纯DP：64路数据并行
        (8, 8, 1),  # DP+PP：中等配置
        (4, 4, 4),  # DP+PP+TP：均衡配置
        (1, 16, 4),  # PP为主 + TP
        (1, 8, 8),  # TP为主
    ]

    print(f"  模型: ~175B参数, 96层, hidden={hidden_dim_large}")
    print(
        f"  {'配置':<30} {'参数':>8} {'梯度':>8} {'优化器':>8} "
        f"{'激活值':>8} {'总计':>8}"
    )
    print(f"  {'-' * 72}")

    for dp, pp, tp in configs:
        mem = hybrid_parallelism_memory(
            model_params_large,
            hidden_dim_large,
            num_layers_large,
            dp,
            pp,
            tp,
            batch_size=2048,
            seq_len=2048,
        )
        print(
            f"  {mem['config']:<30} {mem['params_gb']:>8.2f} {mem['grads_gb']:>8.2f} "
            f"{mem['optimizer_gb']:>8.2f} {mem['activations_gb']:>8.2f} "
            f"{mem['total_mem_gb']:>8.2f}"
        )

    # ---------- 通信量分析 ----------
    print("\n--- 6. 通信量减少分析 ---")
    grad_size_gb = 175e9 * 4 / 1e9  # ~700 GB
    print(f"  基线梯度大小: {grad_size_gb:.1f} GB (FP32)")

    for s, label in [(0.99, "DGC (保留1%)"), (0.999, "DGC (保留0.1%)")]:
        comp_vol = grad_size_gb * (1 - s)
        print(f"  {label:<22}: {comp_vol:.1f} GB  ({1 / (1 - s):.0f}x 压缩)")

    onebit_vol = grad_size_gb / 32  # 每个值1比特 vs 32比特
    print(f"  1-Bit SGD {'':<14}: {onebit_vol:.1f} GB  (32x 压缩)")

    # ---------- 总结 ----------
    print("\n--- 7. 总结 ---")
    print("  核心要点：")
    print("    - DGC：100-1000倍梯度压缩，精度损失<1%")
    print("    - 1-Bit SGD：32倍压缩，配合误差反馈效果良好")
    print("    - 误差反馈至关重要：残差机制防止信息丢失")
    print("    - 动量修正补偿DGC中的梯度陈旧性")
    print("    - 混合并行（DP+PP+TP）对于100B+模型必不可少")
    print("    - 通信是分布式训练中的瓶颈")

    print("\n完成。所有计算均在CPU上运行。\n")


if __name__ == "__main__":
    main()
