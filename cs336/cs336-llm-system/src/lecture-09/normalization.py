"""
从零实现的归一化层（不调用 nn.LayerNorm 等）。

实现：
  - LayerNorm：标准层归一化
  - RMSNorm：均方根归一化（不去中心，更简单）
  - Pre-Norm Transformer 块：LN → Attn → Residual → LN → FFN → Residual
  - Post-Norm Transformer 块：Attn → LN → Residual → FFN → LN → Residual
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# LayerNorm（从零实现）
# =========================================================================


class LayerNorm(nn.Module):
    """
    从零实现的 Layer Normalization。

    公式：
      y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

    在最后一个维度（通常是 hidden_size）上做归一化。
    gamma 和 beta 是可学习的仿射参数。
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 可学习的仿射参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在最后 D 个维度上计算均值和方差
        dims = tuple(range(-len(self.normalized_shape), 0))

        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 仿射变换
        return x_norm * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


# =========================================================================
# RMSNorm（从零实现）
# =========================================================================


class RMSNorm(nn.Module):
    """
    从零实现的 RMS（均方根）归一化。

    公式：
      y = x / sqrt(mean(x^2) + eps) * gamma

    与 LayerNorm 的关键区别：
      - 不去中心（不减去均值）→ 更快，但稳定性稍差
      - 只有可学习的缩放参数（gamma），无 bias

    用于：LLaMA、Mistral、Gemma 和许多现代 LLM。
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 仅保留可学习的缩放参数（无 bias）
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = torch.sqrt(torch.mean(x.pow(2), dim=dims, keepdim=True) + self.eps)

        # 归一化
        return x / rms * self.weight

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


# =========================================================================
# Pre-Norm Transformer 块
# =========================================================================


class PreNormTransformerBlock(nn.Module):
    """
    使用 Pre-Normalization 的 Transformer 块（现代标准）。

    架构：
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-norm 在每个子层之前应用归一化。
    这可以稳定训练，用于 GPT-2/3、LLaMA 等。
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int | None = None, eps: float = 1e-5
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        # 归一化层
        self.attn_norm = LayerNorm(hidden_size, eps=eps)
        self.ffn_norm = LayerNorm(hidden_size, eps=eps)

        # Attention（简化：用线性层作为占位）
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm：先归一化，再应用子层，再加残差
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =========================================================================
# Post-Norm Transformer 块
# =========================================================================


class PostNormTransformerBlock(nn.Module):
    """
    使用 Post-Normalization 的 Transformer 块（原始 Transformer，现已过时）。

    架构：
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))

    Post-norm 在残差加法之后应用归一化。
    这曾用于原始 "Attention Is All You Need" 论文，
    但对深层网络不太稳定。
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int | None = None, eps: float = 1e-5
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.attn_norm = LayerNorm(hidden_size, eps=eps)
        self.ffn_norm = LayerNorm(hidden_size, eps=eps)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm：先应用子层，再加残差，再做归一化
        x = self.attn_norm(x + self.attention(x))
        x = self.ffn_norm(x + self.ffn(x))
        return x


# =========================================================================
# 对比分析
# =========================================================================


def analyze_pre_post_norm() -> None:
    """分析并对比 pre-norm 与 post-norm 的行为。"""
    print("=" * 70)
    print("Pre-Norm vs Post-Norm Analysis")
    print("=" * 70)
    print("""
    Pre-Norm（现代）：
      x_{l+1} = x_l + F(Norm(x_l))

      - 梯度沿残差路径无阻碍地流动
      - 恒等映射更易学习（Norm(x) 近似恒等）
      - 对深层网络（>100 层）训练更稳定
      - 用于：GPT-2/3、LLaMA、BERT（早期为 post-norm）、Mistral

    Post-Norm（原始）：
      x_{l+1} = Norm(x_l + F(x_l))

      - 梯度必须穿过归一化层
      - 归一化会削弱梯度的量级
      - 对深层网络不太稳定；需要谨慎的初始化
      - 用于：原始 Transformer（Vaswani et al. 2017）

    梯度流动对比：
    ┌──────────┬──────────────────────┬────────────────────┐
    │          │ Pre-Norm             │ Post-Norm          │
    ├──────────┼──────────────────────┼────────────────────┤
    │ 梯度     │ ∂L/∂x_l ≈ ∂L/∂x_L    │ ∂L/∂x_l = ∏ ...   │
    │ 量级     │ （近似恒等）           │ （随深度衰减）       │
    ├──────────┼──────────────────────┼────────────────────┤
    │ LR Warmup │ 不太关键             │ 通常需要            │
    ├──────────┼──────────────────────┼────────────────────┤
    │ 深度     │ 可良好扩展            │ 约 20 层后退化       │
    └──────────┴──────────────────────┴────────────────────┘

    数值演示：对比深层网络中的梯度范数。
    """)


def demo_numerical_stability() -> None:
    """演示 LayerNorm 与 RMSNorm 的数值行为。"""
    print("\n" + "=" * 70)
    print("Numerical Stability: LayerNorm vs RMSNorm")
    print("=" * 70)

    torch.manual_seed(42)

    hidden_size = 512
    batch_size = 4
    seq_len = 32

    ln = LayerNorm(hidden_size)
    rms = RMSNorm(hidden_size)

    # 用不同输入量级测试
    for scale_name, scale in [
        ("Normal", 1.0),
        ("Large", 100.0),
        ("Small", 0.01),
        ("Very Large", 1e6),
    ]:
        x = torch.randn(batch_size, seq_len, hidden_size) * scale

        with torch.no_grad():
            ln_out = ln(x)
            rms_out = rms(x)

        ln_mean = ln_out.mean().item()
        ln_std = ln_out.std().item()
        rms_mean = rms_out.mean().item()
        rms_std = rms_out.std().item()

        print(f"\n  Input scale: {scale_name} ({scale})")
        print(f"    LayerNorm output - mean: {ln_mean:.4f}, std: {ln_std:.4f}")
        print(f"    RMSNorm output   - mean: {rms_mean:.4f}, std: {rms_std:.4f}")
        print(f"    Note: RMSNorm does NOT center the output, so mean != 0")


def check_correctness() -> None:
    """验证我们的实现与 PyTorch 内置实现的匹配程度。"""
    print("\n" + "=" * 70)
    print("Correctness Check: Custom vs PyTorch")
    print("=" * 70)

    torch.manual_seed(123)
    hidden_size = 256
    x = torch.randn(2, 8, hidden_size)

    # ----- LayerNorm -----
    ln_custom = LayerNorm(hidden_size)
    ln_torch = nn.LayerNorm(hidden_size)

    # 复制权重
    with torch.no_grad():
        ln_torch.weight.copy_(ln_custom.weight)
        ln_torch.bias.copy_(ln_custom.bias)

    with torch.no_grad():
        custom_out = ln_custom(x)
        torch_out = ln_torch(x)

    diff = (custom_out - torch_out).abs().max().item()
    print(f"\n  LayerNorm max diff: {diff:.8f}")
    print(f"  LayerNorm allclose: {torch.allclose(custom_out, torch_out, rtol=1e-5)}")

    # ----- RMSNorm -----
    rms_custom = RMSNorm(hidden_size)
    # PyTorch 2.1+ 有 RMSNorm
    if hasattr(nn, "RMSNorm"):
        rms_torch = nn.RMSNorm(hidden_size)
        with torch.no_grad():
            rms_torch.weight.copy_(rms_custom.weight)
        with torch.no_grad():
            custom_out_rms = rms_custom(x)
            torch_out_rms = rms_torch(x)
        diff_rms = (custom_out_rms - torch_out_rms).abs().max().item()
        print(f"\n  RMSNorm max diff: {diff_rms:.8f}")
        print(
            f"  RMSNorm allclose: {torch.allclose(custom_out_rms, torch_out_rms, rtol=1e-5)}"
        )
    else:
        print(f"\n  RMSNorm: 无可对比的 PyTorch 内置实现（需要 PyTorch >= 2.1）")
        # 验证输出统计量
        with torch.no_grad():
            out = rms_custom(x)
        rms_val = torch.sqrt(torch.mean(out.pow(2), dim=-1)).mean().item()
        print(f"  RMSNorm output RMS: {rms_val:.4f} (should be near weight norm)")


def demo_pre_post_norm_gradient() -> None:
    """展示梯度如何流经 pre-norm 与 post-norm 块。"""
    print("\n" + "=" * 70)
    print("Gradient Flow: Pre-Norm vs Post-Norm")
    print("=" * 70)

    hidden_size = 128
    num_layers = 12

    # 构建深层网络
    pre_blocks = nn.Sequential(
        *[PreNormTransformerBlock(hidden_size) for _ in range(num_layers)]
    )
    post_blocks = nn.Sequential(
        *[PostNormTransformerBlock(hidden_size) for _ in range(num_layers)]
    )

    x = torch.randn(2, 16, hidden_size, requires_grad=True)

    # 前向传播
    pre_out = pre_blocks(x)
    post_out = post_blocks(x.clone())

    # 反向传播
    pre_out.sum().backward()
    pre_grad_norm = x.grad.norm().item()

    x.grad = None
    post_out.sum().backward()
    post_grad_norm = x.grad.norm().item()

    print(f"\n  使用 {num_layers} 层时：")
    print(f"  Pre-norm  输入处的梯度范数：{pre_grad_norm:.6f}")
    print(f"  Post-norm 输入处的梯度范数：{post_grad_norm:.6f}")
    print(f"  比值（pre/post）：{pre_grad_norm / post_grad_norm:.2f}x")
    print(f"\n  Pre-norm 在深层网络中保留了更强的梯度，")
    print(f"  这解释了为什么它对多层网络的可扩展性更好。")


def main() -> None:
    analyze_pre_post_norm()
    demo_numerical_stability()
    check_correctness()
    demo_pre_post_norm_gradient()


if __name__ == "__main__":
    main()
