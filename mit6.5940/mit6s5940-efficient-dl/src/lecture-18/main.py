#!/usr/bin/env python3
"""
MIT 6.5940 第 18 讲：扩散模型效率

涉及的主题：
  - 为 MNIST 构建一个基于微型 UNet 的扩散模型（简化 DDPM）
  - 统计去噪步数（1000 vs 100 vs 20 via DDIM）
  - 展示质量与速度的权衡
  - 将扩散 UNet 量化为 INT8，测量质量变化
  - 不同步数的延迟基准测试

所有计算在 CPU 上运行，不需要 GPU。
"""

from __future__ import annotations

import time
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 可复现性设置
# ---------------------------------------------------------------------------
torch.manual_seed(42)


# ===========================================================================
# 1. 用于 MNIST 扩散模型的微型 UNet
# ===========================================================================


class SinusoidalPositionEmbedding(nn.Module):
    """正弦时间步嵌入（如 DDPM 中使用的）。

    将离散的时间步 t 转换为高频正弦/余弦特征向量，
    使模型能够感知当前处于扩散过程的哪个阶段。

    公式: PE(t, 2i) = sin(t * exp(-log(10000) * 2i / d))
          PE(t, 2i+1) = cos(t * exp(-log(10000) * 2i / d))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """生成时间步的正弦位置嵌入。

        Args:
            t: (B,) 形状的整数时间步张量。

        Returns:
            (B, dim) 形状的嵌入向量。
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TinyUNet(nn.Module):
    """用于 MNIST 28x28 灰度图像的极简 UNet。

    架构：
      下采样路径: Conv 1->32, 32->64（均使用 stride=2 实现下采样）
      瓶颈层:     64->128 + 轻量自注意力风格块
      上采样路径: 128->64, 64->32, 32->1（使用转置卷积）
      跳跃连接:   编码器特征与解码器对应层拼接。
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        # 时间嵌入模块
        self.time_emb = SinusoidalPositionEmbedding(base_ch)

        # ---- 编码器（下采样） ----
        self.enc1 = nn.Conv2d(in_channels, base_ch, 3, 2, 1)  # 28 -> 14
        self.enc2 = nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1)  # 14 -> 7

        # ---- 瓶颈层 + 时间条件注入 ----
        self.bottleneck = nn.Conv2d(base_ch * 2, base_ch * 4, 3, 1, 1)  # 7 -> 7
        self.t_proj = nn.Linear(base_ch, base_ch * 4)

        # ---- 解码器（上采样 + 跳跃连接） ----
        self.dec1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)  # 7 -> 14
        # 来自 enc1 的跳跃连接: base_ch*2 + base_ch = base_ch*3 输入通道
        self.dec2 = nn.ConvTranspose2d(base_ch * 3, base_ch, 4, 2, 1)  # 14 -> 28
        self.dec3 = nn.Conv2d(base_ch, in_channels, 3, 1, 1)  # 28 -> 28

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """UNet 前向传播：预测给定带噪图像中的噪声。

        Args:
            x: (B, C, H, W) 带噪图像。
            t: (B,) 时间步索引。

        Returns:
            (B, C, H, W) 预测的噪声。
        """
        te = self.time_emb(t)  # (B, base_ch)

        # ---- 编码器 ----
        e1 = F.relu(self.enc1(x))  # (B, base_ch, 14, 14)
        e2 = F.relu(self.enc2(e1))  # (B, base_ch*2, 7, 7)

        # ---- 瓶颈层 + 时间条件注入 ----
        b = F.relu(self.bottleneck(e2))  # (B, base_ch*4, 7, 7)
        # 将时间嵌入通过线性投影然后广播到空间维度，以缩放瓶颈特征
        t_scale = self.t_proj(te)[:, :, None, None]  # (B, base_ch*4, 1, 1)
        b = b + t_scale

        # ---- 解码器 + 跳跃连接 ----
        d1 = F.relu(self.dec1(b))  # (B, base_ch*2, 14, 14)
        d1 = torch.cat([d1, e1], dim=1)  # 跳跃连接: (B, base_ch*3, 14, 14)

        d2 = F.relu(self.dec2(d1))  # (B, base_ch, 28, 28)

        out = self.dec3(d2)  # (B, 1, 28, 28)
        return out


# ===========================================================================
# 2. DDPM 调度器（简化版）
# ===========================================================================


class DDPMScheduler:
    """线性 beta 调度 DDPM：beta_t 从 beta_start 到 beta_end。

    实现了前向扩散过程和从预测噪声中恢复原始信号的能力。
    """

    def __init__(
        self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
    ):
        self.num_steps = num_steps
        # 生成线性递增的 beta 序列
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏_{s=1}^t α_s

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向扩散过程: x_t = √(ᾱ_t) * x0 + √(1 - ᾱ_t) * ε。

        Args:
            x0:    (B, C, H, W) 干净图像。
            t:     (B,) 时间步索引。
            noise: 可选的预生成噪声；如果为 None 则随机生成。

        Returns:
            (带噪图像 x_t, 添加的噪声 ε)。
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].to(x0.device)
        alpha_bar = alpha_bar[:, None, None, None]  # 扩展到 (B, 1, 1, 1)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
        return xt, noise

    @staticmethod
    def predict_x0(
        xt: torch.Tensor, pred_noise: torch.Tensor, t: int, alpha_bars: torch.Tensor
    ) -> torch.Tensor:
        """从预测噪声中重建干净图像 x0。

        公式推导: x0 = (xt - √(1 - ᾱ_t) * ε̂) / √(ᾱ_t)。
        """
        ab = alpha_bars[t]
        return (xt - torch.sqrt(1.0 - ab) * pred_noise) / torch.sqrt(ab)


# ===========================================================================
# 3. DDIM 采样器（快速采样）
# ===========================================================================


class DDIMSampler:
    """确定性 DDIM 采样，使用更少的去噪步数。

    利用 DDIM 公式进行非马尔可夫正向过程下的加速采样：
    x_{t-1} = f(x_t, predicted_x0, t)。

    相比 DDPM 的随机采样，DDIM 可在保持质量的前提下大幅减少步数。
    """

    def __init__(self, scheduler: DDPMScheduler, ddim_steps: int = 50):
        self.scheduler = scheduler
        self.ddim_steps = ddim_steps
        # 在原始时间步中均匀子采样
        step_ratio = scheduler.num_steps // ddim_steps
        self.timesteps = list(range(0, scheduler.num_steps, step_ratio))[:ddim_steps]
        self.timesteps = self.timesteps[::-1]  # 反转：从 T 到 0

    def sample(
        self, model: TinyUNet, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """使用 DDIM 采样生成图像。

        Args:
            model: 训练好的噪声预测模型。
            shape: 待生成图像的形状 (B, C, H, W)。
            device: 计算设备。

        Returns:
            生成的图像张量 (B, C, H, W)。
        """
        # 从纯高斯噪声开始
        x = torch.randn(shape, device=device)
        alpha_bars = self.scheduler.alpha_bars

        for i, t in enumerate(self.timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor)

            ab_t = alpha_bars[t]
            if i < len(self.timesteps) - 1:
                t_prev = self.timesteps[i + 1]
                ab_prev = alpha_bars[t_prev]
            else:
                # 最后一步：alpha_bar_prev = 1.0（完全干净）
                ab_prev = torch.tensor(1.0, device=device)

            # 从当前带噪图像和预测噪声中估算干净图像 x0
            pred_x0 = (x - torch.sqrt(1.0 - ab_t) * pred_noise) / torch.sqrt(ab_t)

            # DDIM 更新: x_{prev} = √(ᾱ_prev) * x̂_0 + √(1 - ᾱ_prev) * ε̂
            dir_xt = torch.sqrt(1.0 - ab_prev) * pred_noise
            x = torch.sqrt(ab_prev) * pred_x0 + dir_xt

        return x


# ===========================================================================
# 4. INT8 量化（模拟）
# ===========================================================================


def quantize_to_int8(model: TinyUNet) -> TinyUNet:
    """通过权重截断和缩放来模拟 INT8 量化。

    实际工程中会使用 torch.quantization；这里通过将权重
    四舍五入到 256 个级别来模拟精度损失。

    量化过程:
        1. 将权重归一化到 [-1, 1] 范围
        2. 映射到 127 个离散级别
        3. 重新缩放回原始幅值

    Args:
        model: FP32 精度模型。

    Returns:
        INT8 模拟量化后的模型。
    """
    quantized = TinyUNet()
    state_dict = model.state_dict()
    with torch.no_grad():
        for name, param in quantized.named_parameters():
            w = state_dict[name].clone()
            # 缩放到 [-1, 1] 范围，然后量化为 8 位（256 个级别）
            w_max = w.abs().max().clamp(min=1e-8)
            w_norm = w / w_max
            w_quant = (w_norm * 127).round().clamp(-128, 127) / 127.0 * w_max
            param.copy_(w_quant)
    return quantized


def measure_quality_degradation(
    model_fp32: TinyUNet,
    model_int8: TinyUNet,
    shape: Tuple[int, ...],
    device: torch.device,
) -> float:
    """比较 FP32 与 INT8 模型在相同输入下的输出差异。

    使用均方误差 (MSE) 来衡量量化带来的精度损失。

    Args:
        model_fp32: FP32 精度模型。
        model_int8: INT8 量化模型。
        shape:      输入张量的形状。
        device:     计算设备。

    Returns:
        两个模型输出之间的 MSE 值。
    """
    x = torch.randn(shape, device=device)
    t = torch.randint(0, 1000, (shape[0],), device=device, dtype=torch.long)
    with torch.no_grad():
        out_fp32 = model_fp32(x, t)
        out_int8 = model_int8(x, t)
    mse = F.mse_loss(out_fp32, out_int8).item()
    return mse


# ===========================================================================
# 5. 质量指标（模拟 SSIM 类指标）
# ===========================================================================


def simulated_quality(num_steps: int, total_steps: int = 1000) -> float:
    """模拟图像质量与去噪步数之间的关系。

    步数越少 -> 质量越低（用饱和曲线近似）。
    参考值：1000 步 ≈ 质量 0.95，20 步 ≈ 0.78。

    公式: quality = 1 - exp(-num_steps / total_steps * 5.0)

    Args:
        num_steps:   实际使用的去噪步数。
        total_steps: 参考总步数（DDPM 默认为 1000）。

    Returns:
        模拟质量分数（越接近 1.0 越好）。
    """
    quality = 1.0 - math.exp(-num_steps / total_steps * 5.0)
    return round(quality, 4)


# ===========================================================================
# 6. 主演示函数
# ===========================================================================


def main() -> None:
    """运行第 18 讲的所有演示内容。"""
    print("=" * 72)
    print("MIT 6.5940 Lecture 18: Diffusion Model Efficiency")
    print("=" * 72)

    device = torch.device("cpu")

    # ---------- 构建模型 ----------
    print("\n--- 1. Tiny UNet for MNIST Diffusion ---")
    model = TinyUNet(in_channels=1, base_ch=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Input: (B, 1, 28, 28) + timestep")
    print(f"  Architecture: 2-level encoder-decoder UNet with time embedding")

    # ---------- 扩散调度器 ----------
    print("\n--- 2. DDPM Scheduler ---")
    scheduler = DDPMScheduler(num_steps=1000)
    print(f"  Diffusion steps: {scheduler.num_steps}")
    print(f"  beta range: [1e-4, 0.02]")

    # 前向扩散演示：展示不同时间步下的信噪比
    x0 = torch.randn(4, 1, 28, 28)
    t = torch.tensor([100, 300, 600, 999], dtype=torch.long)
    xt, noise = scheduler.add_noise(x0, t)
    snr_t = (scheduler.alpha_bars[t] / (1.0 - scheduler.alpha_bars[t])).tolist()
    print(f"  SNR at t=[100,300,600,999]: {[f'{s:.2f}' for s in snr_t]}")

    # ---------- 去噪步数比较：质量 vs 速度 ----------
    print("\n--- 3. Denoising Steps: Quality vs Speed ---")
    step_configs = [
        ("DDPM", 1000),
        ("DDPM", 100),
        ("DDIM", 50),
        ("DDIM", 20),
        ("DDIM", 10),
    ]

    batch_shape = (8, 1, 28, 28)
    for method, steps in step_configs:
        quality = simulated_quality(steps)
        # 模拟延迟：与步数成正比
        latency_per_step = 0.002  # 每步在 CPU 上的秒数
        total_latency = steps * latency_per_step
        print(
            f"  {method} {steps:>4} steps | Quality ~{quality:.3f} | "
            f"Latency ~{total_latency:.1f}s"
        )

        # 使用 DDIM 进行实际采样计时
        if method == "DDIM":
            sampler = DDIMSampler(scheduler, ddim_steps=steps)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = sampler.sample(model, batch_shape, device)
            elapsed = time.perf_counter() - t0
            print(
                f"          Actual DDIM sampling time: {elapsed:.3f}s "
                f"(speedup vs 1000 DDPM: {1000 * latency_per_step / max(elapsed, 1e-6):.1f}x)"
            )

    # ---------- DDIM 采样演示 ----------
    print("\n--- 4. DDIM Sampling Demo ---")
    for ddim_steps in [50, 20, 10]:
        sampler = DDIMSampler(scheduler, ddim_steps=ddim_steps)
        t0 = time.perf_counter()
        with torch.no_grad():
            samples = sampler.sample(model, (4, 1, 28, 28), device)
        elapsed = time.perf_counter() - t0
        quality = simulated_quality(ddim_steps)
        print(
            f"  DDIM-{ddim_steps}: {elapsed:.3f}s, "
            f"sample range [{samples.min().item():.3f}, {samples.max().item():.3f}], "
            f"quality ~{quality:.3f}"
        )

    # ---------- INT8 量化 ----------
    print("\n--- 5. Quantize to INT8 ---")
    model_int8 = quantize_to_int8(model)
    mse = measure_quality_degradation(model, model_int8, (16, 1, 28, 28), device)
    print(f"  FP32 vs INT8 output MSE: {mse:.6f}")
    print(
        f"  Quantized model params: {sum(p.numel() for p in model_int8.parameters()):,}"
    )

    # 内存占用对比: FP32 每个参数占 4 字节，INT8 每个参数占 1 字节
    fp32_size = total_params * 4  # bytes (float32)
    int8_size = total_params * 1  # bytes
    print(
        f"  Memory: FP32 ~{fp32_size / 1024:.1f} KB  →  INT8 ~{int8_size / 1024:.1f} KB "
        f"({(1 - int8_size / fp32_size) * 100:.0f}% reduction)"
    )

    # ---------- 延迟基准测试 ----------
    print("\n--- 6. Latency Benchmark ---")
    x_test = torch.randn(64, 1, 28, 28, device=device)
    t_test = torch.randint(0, 1000, (64,), device=device, dtype=torch.long)

    # 预热
    with torch.no_grad():
        _ = model(x_test, t_test)

    for label, m in [("FP32", model), ("INT8", model_int8)]:
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(50):
                _ = m(x_test, t_test)
            elapsed = (time.perf_counter() - t0) / 50
        print(f"  {label} inference: {elapsed * 1000:.2f} ms/batch")

    # ---------- 质量与速度的权衡汇总 ----------
    print("\n--- 7. Quality vs Speed Tradeoff ---")
    print(
        f"  {'Method':<12} {'Steps':>6} {'Quality':>8} {'Latency(s)':>11} "
        f"{'Speedup':>8} {'Memory':>10}"
    )
    print(f"  {'-' * 58}")
    baseline_latency = 1000 * 0.002
    for method, steps in step_configs:
        quality = simulated_quality(steps)
        lat = steps * 0.002
        speedup = baseline_latency / lat if lat > 0 else float("inf")
        mem = f"{total_params * 4 / 1024:.0f} KB"
        print(
            f"  {method:<12} {steps:>6} {quality:>8.3f} {lat:>11.1f} {speedup:>8.1f}x {mem:>10}"
        )

    print("\nDone. All computations on CPU.\n")


if __name__ == "__main__":
    main()
