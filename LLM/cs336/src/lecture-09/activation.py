"""
从零实现的激活函数。

包含：
  - ReLU、GeLU、Swish (SiLU)、SwiGLU
  - SwiGLU 前馈网络
  - 梯度行为分析
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 激活函数（从零实现）
# =========================================================================


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    ReLU（修正线性单元）。

    f(x) = max(0, x)
    f'(x) = 0（当 x < 0），否则为 1
    """
    return torch.maximum(x, torch.zeros_like(x))


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GeLU（高斯误差线性单元，精确版本）。

    f(x) = x * Phi(x)，其中 Phi 是标准高斯 CDF
         = x * 0.5 * (1 + erf(x / sqrt(2)))

    ReLU 的光滑近似。用于 BERT、GPT-2、ViT。
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """
    使用 tanh 的快速近似 GeLU。

    f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    用于 GPT-3 及许多实现中以提升速度。
    """
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Swish（当 beta=1 时为 SiLU）激活函数。

    f(x) = x * sigmoid(beta * x)
    f'(x) = f(x) + sigmoid(beta * x) * (1 - f(x))

    自门控激活函数。用于 EfficientNet、LLaMA。
    """
    return x * torch.sigmoid(beta * x)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU = beta=1 时的 Swish。用作 LLaMA 中的激活函数。"""
    return swish(x, beta=1.0)


# =========================================================================
# SwiGLU
# =========================================================================


def swiglu(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU 激活（Swi-门控线性单元）。

    SwiGLU(x) = (x @ W_gate · SiLU(x @ W_up)) @ W_down

    其中：
      - x @ W_gate：即 "gate"（门控）投影
      - x @ W_up：即 "up"（上投影），经过 SiLU 处理
      - gate 与经激活的 up 的 Hadamard 积
      - 最终的下投影

    SwiGLU 只是对 FFN 的中间隐藏状态逐元素应用的激活函数。
    它取代了现代 LLM（LLaMA、PaLM 等）中标准的 ReLU/GeLU。
    """
    gate = x @ W_gate
    up = silu(x @ W_up)
    return gate * up


class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络。

    标准 FFN：  x → FC1 → ReLU → FC2 → output
    SwiGLU FFN：x → [Gate Proj, Up Proj] → SwiGLU → Down Proj → output

    SwiGLU 使用 3 个权重矩阵而非 2 个，但中间维度通常会相应减小以补偿。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU 有三个投影
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # SwiGLU：gate 与 SiLU(up) 的逐元素乘积
        hidden = gate * silu(up)
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)

    def num_parameters(self) -> tuple[int, int]:
        """返回 (swiglu_params, equivalent_standard_params) 以进行比较。"""
        h, i = self.gate_proj.in_features, self.gate_proj.out_features
        swiglu_params = 3 * h * i
        # 具有相当容量的标准 FFN 使用 4h * 2h/3 作为中间维度
        equivalent_i = int(2 * i / 3 * 2)  # 粗略等效
        standard_params = 2 * h * equivalent_i
        return swiglu_params, standard_params


class StandardFFN(nn.Module):
    """用于对比的标准 FFN。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "relu":
            x = relu(x)
        elif self.activation == "gelu":
            x = gelu(x)
        elif self.activation == "silu":
            x = silu(x)
        else:
            x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


# =========================================================================
# 梯度分析
# =========================================================================


def analyze_gradients() -> None:
    """分析不同激活函数的梯度行为。"""
    print("=" * 70)
    print("Activation Function Gradient Analysis")
    print("=" * 70)

    # 用于分析的数据点
    x_vals = torch.linspace(-3, 3, 1000, requires_grad=True)

    activations = {
        "ReLU": lambda x: relu(x),
        "GeLU (exact)": lambda x: gelu(x),
        "GeLU (approx)": lambda x: gelu_approx(x),
        "Swish/SiLU": lambda x: silu(x),
    }

    print("\n  Properties at key points:")
    fp_label = "f'(0)"
    print(
        f"  {'Activation':<20} {'f(0)':>8} {'f(inf)':>8} {'f(-inf)':>8} {fp_label:>8} {'Non-zero grad':>14}"
    )
    print("  " + "-" * 72)

    for name, fn in activations.items():
        y = fn(x_vals)

        # 梯度
        grad = torch.autograd.grad(y.sum(), x_vals, create_graph=True)[0]

        f0 = fn(torch.tensor(0.0)).item()
        finf = fn(torch.tensor(100.0)).item()
        fninf = fn(torch.tensor(-100.0)).item()

        # f'(0)
        x0 = torch.tensor(0.0, requires_grad=True)
        y0 = fn(x0)
        grad0 = torch.autograd.grad(y0, x0)[0].item()

        # 非零梯度区域
        nonzero = (grad.abs() > 0.01).float().mean().item() * 100

        print(
            f"  {name:<20} {f0:>8.3f} {finf:>8.1f} {fninf:>8.3f} {grad0:>8.3f} {nonzero:>9.1f}%"
        )

    print("\n  Key observations:")
    print("  - ReLU:   x < 0 时梯度为 0 → 'dying ReLU' 问题")
    print("  - GeLU:   光滑、处处非零梯度，概率门控")
    print("  - Swish:  对于 x < 0 非单调（小幅负向凸起），自门控")
    print("  - SwiGLU: 将门控思想扩展为双线性形式 → 更丰富的表达能力")


def analyze_swiglu_capacity() -> None:
    """对比 SwiGLU 和标准 FFN 的参数数量与表达能力。"""
    print("\n" + "=" * 70)
    print("SwiGLU vs Standard FFN: Parameter Efficiency")
    print("=" * 70)

    configs = [
        (512, 2048),
        (768, 3072),
        (1024, 4096),
        (4096, 14336),
    ]

    print(
        f"\n  {'Hidden':<10} {'SwGLU Int':<12} {'SwGLU Params':<14} {'Std Params':<14} {'Ratio':<10}"
    )
    print("  " + "-" * 60)

    for hidden, intermediate in configs:
        swiglu_params = 3 * hidden * intermediate
        # 具有相当计算量的标准 FFN：使用更小的中间维度
        # SwiGLU 有 3 个 dim=intermediate 的矩阵；标准有 2 个
        # 要匹配参数量，标准需要 intermediate * 1.5
        std_intermediate = int(intermediate * 1.5)
        std_params = 2 * hidden * std_intermediate

        ratio = swiglu_params / std_params
        print(
            f"  {hidden:<10} {intermediate:<12} {swiglu_params:>10,}   {std_params:>10,}   {ratio:>6.2f}"
        )

    print(
        f"\n  SwiGLU uses 3 weight matrices instead of 2, but delivers better quality"
    )
    print(f"  per parameter due to the multiplicative gating mechanism.")


def demo_ffn_forward() -> None:
    """对比不同类型 FFN 的前向传播。"""
    print("\n" + "=" * 70)
    print("FFN Forward Comparison")
    print("=" * 70)

    hidden = 512
    intermediate = 2048
    x = torch.randn(2, 16, hidden)

    ffn_relu = StandardFFN(hidden, intermediate, "relu")
    ffn_gelu = StandardFFN(hidden, intermediate, "gelu")
    ffn_silu = StandardFFN(hidden, intermediate, "silu")
    ffn_swiglu = SwiGLUFFN(
        hidden, intermediate // 3 * 2
    )  # 标准：SwiGLU 使用约 2/3 * intermediate

    with torch.no_grad():
        out_relu = ffn_relu(x)
        out_gelu = ffn_gelu(x)
        out_silu = ffn_silu(x)
        out_swiglu = ffn_swiglu(x)

    print(f"\n  Input shape:  {x.shape}")
    print(
        f"  ReLU FFN:     output {out_relu.shape}, params {sum(p.numel() for p in ffn_relu.parameters()):,}"
    )
    print(
        f"  GeLU FFN:     output {out_gelu.shape}, params {sum(p.numel() for p in ffn_gelu.parameters()):,}"
    )
    print(
        f"  SiLU FFN:     output {out_silu.shape}, params {sum(p.numel() for p in ffn_silu.parameters()):,}"
    )
    print(
        f"  SwiGLU FFN:   output {out_swiglu.shape}, params {sum(p.numel() for p in ffn_swiglu.parameters()):,}"
    )

    # 输出统计
    for name, out in [
        ("ReLU", out_relu),
        ("GeLU", out_gelu),
        ("SiLU", out_silu),
        ("SwiGLU", out_swiglu),
    ]:
        print(
            f"  {name:<8} stats: mean={out.mean():.4f}, std={out.std():.4f}, min={out.min():.4f}, max={out.max():.4f}"
        )


def main() -> None:
    analyze_gradients()
    analyze_swiglu_capacity()
    demo_ffn_forward()


if __name__ == "__main__":
    main()
