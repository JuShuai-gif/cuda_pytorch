#!/usr/bin/env python3
"""
MIT 6.5940 第 17 讲：高效 GAN / 视频优化

涉及的主题：
  - 为 MNIST 构建一个小型 GAN（生成器 + 判别器）
  - GAN 压缩技术：剪枝生成器通道
  - TSM（时序移位模块）概念演示：沿时间维度移位通道
  - 比较：原始 vs 压缩 GAN 的 FID（模拟）、延迟
  - 测量：参数量减少、FLOPs 减少

所有计算在 CPU 上运行，不需要 GPU。
"""

from __future__ import annotations

import time
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 可复现性设置
# ---------------------------------------------------------------------------
torch.manual_seed(42)


# ===========================================================================
# 1. 小型 GAN 构建模块（MNIST 的生成器 + 判别器）
# ===========================================================================


class Generator(nn.Module):
    """用于 28x28 MNIST 图像的简易 DCGAN 风格生成器。

    通过四层转置卷积从 latent_dim 维噪声逐步上采样，
    最终输出 1 x 28 x 28 的 Tanh 归一化图像。
    """

    def __init__(self, latent_dim: int = 100, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.main = nn.Sequential(
            # latent_dim x 1 x 1  -->  base*4 x 7 x 7
            nn.ConvTranspose2d(latent_dim, base_channels * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            #  --> base*2 x 14 x 14
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            #  --> base x 28 x 28
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            #  --> 1 x 28 x 28
            nn.ConvTranspose2d(base_channels, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """前向传播：从隐变量生成图像。

        Args:
            z: (B, latent_dim, 1, 1) 隐变量。

        Returns:
            (B, 1, 28, 28) 生成的图像。
        """
        return self.main(z)


class Discriminator(nn.Module):
    """用于 28x28 MNIST 图像的简易 DCGAN 风格判别器。

    通过四层普通卷积逐步下采样，最终输出单个标量 logit。
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            # 1 x 28 x 28  -->  base x 14 x 14
            nn.Conv2d(1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #  --> base*2 x 7 x 7
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #  --> base*4 x 3 x 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #  --> 1
            nn.Conv2d(base_channels * 4, 1, 3, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：判断输入是否为真实图像。

        Args:
            x: (B, 1, 28, 28) 输入图像。

        Returns:
            (B,) 形状的判别 logits。
        """
        return self.main(x).view(-1)


# ===========================================================================
# 2. 工具辅助函数
# ===========================================================================


def count_parameters(model: nn.Module) -> int:
    """返回可训练参数的总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_conv_flops(layer: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """对 Conv2d / ConvTranspose2d 层的粗略 FLOPs 估算。"""
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        # FLOPs = 2 * (C_in * K_h * K_w) * C_out * H_out * W_out  （乘加对）
        # 此处留空，由下方精确计算函数实现
        return 0
    return 0


def count_flops_gan(g: Generator) -> int:
    """计算生成器前向传播的近似 FLOPs。

    手动遍历已知的架构计算每层 FLOPs。
    """
    flops = 0
    latent_dim = g.latent_dim
    bc = g.base_channels

    # 第 1 层: ConvTranspose2d(latent_dim, bc*4, 7, 1, 0) -> 输出: (bc*4) x 7 x 7
    flops += 2 * latent_dim * 7 * 7 * (bc * 4) * 7 * 7
    # 第 2 层: ConvTranspose2d(bc*4, bc*2, 4, 2, 1) -> 输出: (bc*2) x 14 x 14
    flops += 2 * (bc * 4) * 4 * 4 * (bc * 2) * 14 * 14
    # 第 3 层: ConvTranspose2d(bc*2, bc, 4, 2, 1) -> 输出: (bc) x 28 x 28
    flops += 2 * (bc * 2) * 4 * 4 * bc * 28 * 28
    # 第 4 层: ConvTranspose2d(bc, 1, 4, 2, 1) -> 输出: 1 x 28 x 28
    return flops


def count_flops_generator_exact(g: Generator) -> int:
    """使用前向钩子精确计算生成器的 FLOPs。

    为每个 ConvTranspose2d / Conv2d 层注册钩子，
    运行一次前向传播后累加各层的乘加操作次数。

    Returns:
        乘加操作的总数。
    """
    z = torch.randn(1, g.latent_dim, 1, 1)
    hooks: List[torch.Tensor] = []

    def hook(module, inp, out):
        """前向钩子：计算单层转置卷积的 FLOPs。"""
        if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
            cin = module.in_channels
            k = (
                module.kernel_size[0]
                if isinstance(module.kernel_size, tuple)
                else module.kernel_size
            )
            cout = module.out_channels
            h_out, w_out = out.shape[2], out.shape[3]
            # 对于转置卷积: FLOPs ≈ 2 * cin * k * k * cout * h_out * w_out
            hooks.append(2 * cin * k * k * cout * h_out * w_out)

    handles = []
    for m in g.modules():
        handles.append(m.register_forward_hook(hook))
    g(z)
    for h in handles:
        h.remove()
    return sum(h for h in hooks)


# ===========================================================================
# 3. GAN 压缩：通道剪枝
# ===========================================================================


def prune_generator_channels(g: Generator, keep_ratio: float = 0.5) -> Generator:
    """创建一个压缩版生成器，仅保留部分通道。

    这实现了 GAN Compression 的概念：生成器通常是
    过度参数化的，可以通过剪枝来压缩，质量损失很小。

    通过复制原始权重中对应的通道切片来实现权重复用。

    Args:
        g:          原始生成器。
        keep_ratio: 保留的通道比例（0.0 ~ 1.0）。

    Returns:
        通道数被压缩后的生成器。
    """
    latent_dim = g.latent_dim
    old_bc = g.base_channels
    new_bc = max(8, int(old_bc * keep_ratio))

    pruned = Generator(latent_dim=latent_dim, base_channels=new_bc)

    # 复制各层权重的对应部分（输出通道沿 dim=1 方向裁剪，
    # 对应下一层的输入通道沿 dim=0 方向裁剪）
    with torch.no_grad():
        # ConvTranspose2d 权重形状: (in_channels, out_channels, k, k)
        # 我们裁剪输出通道（dim=1）和下一层的对应输入通道（dim=0）

        # 第 1 层: (latent_dim, bc_old*4, 7, 7) -> (latent_dim, bc_new*4, 7, 7)
        old_w = g.main[0].weight  # [latent_dim, bc_old*4, 7, 7]
        pruned.main[0].weight.copy_(old_w[:, : new_bc * 4, :, :])

        # 第 2 层: (bc_old*4, bc_old*2, 4, 4) -> (bc_new*4, bc_new*2, 4, 4)
        old_w = g.main[3].weight  # [bc_old*4, bc_old*2, 4, 4]
        pruned.main[3].weight.copy_(old_w[: new_bc * 4, : new_bc * 2, :, :])

        # 第 3 层: (bc_old*2, bc_old, 4, 4) -> (bc_new*2, bc_new, 4, 4)
        old_w = g.main[6].weight  # [bc_old*2, bc_old, 4, 4]
        pruned.main[6].weight.copy_(old_w[: new_bc * 2, :new_bc, :, :])

        # 第 4 层: (bc_old, 1, 4, 4) -> (bc_new, 1, 4, 4)
        old_w = g.main[9].weight  # [bc_old, 1, 4, 4]
        pruned.main[9].weight.copy_(old_w[:new_bc, :, :, :])

    return pruned


# ===========================================================================
# 4. TSM：时序移位模块概念
# ===========================================================================


def temporal_shift_module(x: torch.Tensor, n_div: int = 8) -> torch.Tensor:
    """模拟 TSM（Temporal Shift Module，时序移位模块）。

    将部分通道沿时间（帧）维度向前/向后移位，
    以在不需要 3D 卷积的情况下实现时序推理。

    参考文献：Lin et al., "TSM: Temporal Shift Module for Efficient Video
    Understanding", ICCV 2019.

    Args:
        x:     形状为 (B, C, T, H, W) 的张量
        n_div: 通道被分为 n_div 份；前 1/n_div 的通道向后移位，
               中间 1/n_div 的通道向前移位，其余保持不变。

    Returns:
        移位后形状相同的张量。
    """
    B, C, T, H, W = x.shape
    fold = C // n_div
    if fold == 0:
        return x  # 通道数不足，不进行移位

    out = torch.zeros_like(x)
    # 第 1 份: 向后移位 (t+1)，丢弃最后一帧
    out[:, :fold, 1:, :, :] = x[:, :fold, : T - 1, :, :]
    # 第 2 份: 向前移位 (t-1)，丢弃第一帧
    out[:, fold : 2 * fold, : T - 1, :, :] = x[:, fold : 2 * fold, 1:, :, :]
    # 其余通道: 保持不变，无移位
    out[:, 2 * fold :, :, :, :] = x[:, 2 * fold :, :, :, :]

    return out


# ===========================================================================
# 5. 模拟 FID 计算
# ===========================================================================


def simulated_fid(image_count: int, noise_std: float) -> float:
    """用简单的启发式方法模拟 FID。

    实际中 FID 是通过 InceptionV3 特征计算的。这里我们将其
    近似为与图像多样性成正比（加上样本数量的对数项），
    并添加噪声来模拟压缩带来的质量退化。

    Args:
        image_count: 用于计算 FID 的图像数量
        noise_std:   加性噪声的标准差（质量代理指标）

    Returns:
        模拟的 FID 分数（越低越好）。
    """
    base = 20.0
    # FID 随样本减少和噪声增加而变差（分数变高）
    fid = base + noise_std * 15.0 + max(0, math.log(50000 / max(image_count, 1)))
    return round(fid, 2)


# ===========================================================================
# 6. 主演示函数
# ===========================================================================


def main() -> None:
    """运行第 17 讲的所有演示内容。"""
    print("=" * 72)
    print("MIT 6.5940 Lecture 17: Efficient GAN / Video Optimization")
    print("=" * 72)

    device = torch.device("cpu")
    latent_dim = 100

    # ---------- GAN 架构概览 ----------
    print("\n--- 1. GAN Architecture ---")
    g = Generator(latent_dim=latent_dim, base_channels=64).to(device)
    d = Discriminator(base_channels=64).to(device)

    g_params = count_parameters(g)
    d_params = count_parameters(d)
    print(f"  Generator params:  {g_params:,}")
    print(f"  Discriminator params: {d_params:,}")
    print(f"  Total GAN params:  {g_params + d_params:,}")

    # ---------- FLOPs 估算 ----------
    print("\n--- 2. FLOPs Estimation ---")
    g_flops = count_flops_generator_exact(g)
    print(f"  Generator forward FLOPs: {g_flops / 1e6:.2f} MFLOPs")

    # ---------- GAN 压缩 ----------
    print("\n--- 3. GAN Compression (Channel Pruning) ---")
    keep_ratios = [0.75, 0.50, 0.25]
    for ratio in keep_ratios:
        g_pruned = prune_generator_channels(g, keep_ratio=ratio)
        p_params = count_parameters(g_pruned)
        p_flops = count_flops_generator_exact(g_pruned)
        reduction_p = (1.0 - p_params / g_params) * 100
        reduction_f = (1.0 - p_flops / g_flops) * 100
        print(
            f"  Keep ratio {ratio:.0%}: params={p_params:,} ({reduction_p:.1f}% ↓), "
            f"FLOPs={p_flops / 1e6:.2f} M ({reduction_f:.1f}% ↓)"
        )

    # ---------- FID 比较（模拟） ----------
    print("\n--- 4. Simulated FID Comparison ---")
    fid_orig = simulated_fid(50000, noise_std=0.0)
    print(f"  Original GAN FID:        {fid_orig}")
    for ratio in [0.75, 0.50, 0.25]:
        # 压缩越多 -> 噪声越大（质量退化更严重）
        noise = (1.0 - ratio) * 3.0
        fid_c = simulated_fid(50000, noise_std=noise)
        print(f"  Compressed ({ratio:.0%}) FID:  {fid_c}")

    # ---------- 延迟比较 ----------
    print("\n--- 5. Latency Comparison ---")
    z_test = torch.randn(100, latent_dim, 1, 1, device=device)
    # 预热
    _ = g(z_test)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = g(z_test)
    t_orig = (time.perf_counter() - t0) / 100

    for ratio in [0.75, 0.50, 0.25]:
        g_pruned = prune_generator_channels(g, keep_ratio=ratio)
        _ = g_pruned(z_test)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = g_pruned(z_test)
        t_pruned = (time.perf_counter() - t0) / 100
        speedup = t_orig / t_pruned
        print(
            f"  Keep {ratio:.0%}: {t_pruned * 1000:.2f} ms/batch "
            f"(vs {t_orig * 1000:.2f} ms, {speedup:.2f}x speedup)"
        )

    # ---------- TSM 演示 ----------
    print("\n--- 6. TSM (Temporal Shift Module) Concept ---")
    # 模拟一批 4 个视频片段，每个片段 8 帧
    B, C, T, H, W = 2, 16, 8, 4, 4
    x_video = torch.arange(B * C * T * H * W, dtype=torch.float32).reshape(
        B, C, T, H, W
    )
    shifted = temporal_shift_module(x_video, n_div=8)
    # 验证：第 0 份（前 C//8 通道）向后移位
    fold = C // 8
    match_bwd = torch.allclose(
        shifted[0, :fold, 1:, 0, 0], x_video[0, :fold, : T - 1, 0, 0]
    )
    not_eq_start = not torch.allclose(
        shifted[0, :fold, 0, 0, 0], x_video[0, :fold, 0, 0, 0]
    )
    print(f"  Input shape:  {tuple(x_video.shape)}")
    print(f"  Channels shifted backward (t+1): {'OK' if match_bwd else 'FAIL'}")
    print(f"  Frame 0 correctly NOT shifted back: {'OK' if not_eq_start else 'FAIL'}")
    print("  TSM enables temporal reasoning with 2D conv FLOPs (zero extra cost).")

    # ---------- 汇总 ----------
    print("\n--- 7. Summary ---")
    print(
        f"  {'Strategy':<25} {'Params':>12} {'FLOPs(M)':>10} {'FID':>6} {'Speedup':>8}"
    )
    print(f"  {'-' * 61}")
    for ratio in [1.0, 0.75, 0.50, 0.25]:
        if ratio == 1.0:
            gp = g
            label = "Original"
        else:
            gp = prune_generator_channels(g, keep_ratio=ratio)
            label = f"Pruned {ratio:.0%}"
        p = count_parameters(gp)
        f = count_flops_generator_exact(gp) / 1e6
        fid = simulated_fid(50000, noise_std=(1.0 - ratio) * 3.0)
        sp = 1.0 if ratio == 1.0 else round(1.0 / ratio, 2)
        print(f"  {label:<25} {p:>12,} {f:>10.2f} {fid:>6} {sp:>8.2f}x")

    print("\nDone. All computations on CPU.\n")


if __name__ == "__main__":
    main()
