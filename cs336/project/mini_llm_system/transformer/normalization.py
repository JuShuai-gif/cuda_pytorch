"""
从零实现的归一化层。

包含 RMSNorm（Root Mean Square Layer Normalization，均方根层归一化），
用于 LLaMA 风格的架构。不调用 nn.RMSNorm 或 nn.LayerNorm。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization（均方根层归一化）。

    RMSNorm 通过均方根对输入进行归一化，然后应用可学习的缩放参数。
    与 LayerNorm 不同，RMSNorm 不减去均值，也不使用偏置项。计算效率更高。

    归一化公式如下：
        output = x * weight / sqrt(mean(x^2) + eps)

    Args:
        hidden_size: 要归一化的隐藏维度大小。
        eps: 用于数值稳定性的小常数。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用 RMS 归一化。

        Args:
            x: 形状为 [..., hidden_size] 的输入张量。

        Returns:
            与输入形状相同的归一化张量。
        """
        # 计算 RMS：sqrt(mean(x^2))
        # 使用 float32 以保证计算过程中的数值稳定性
        input_dtype: torch.dtype = x.dtype
        x_float: torch.Tensor = x.float()
        rms: torch.Tensor = torch.sqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        x_norm: torch.Tensor = x_float / rms
        return (x_norm * self.weight.float()).to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


# 快速测试
if __name__ == "__main__":
    batch, seq, hidden = 2, 16, 768
    rms_norm = RMSNorm(hidden_size=hidden, eps=1e-5)
    x = torch.randn(batch, seq, hidden)
    out = rms_norm(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    # 验证：输出的 RMS 应约等于 1.0（每个元素，允许 eps 的误差）
    rms_out = torch.sqrt(out.float().pow(2).mean(dim=-1) + rms_norm.eps)
    expected = torch.ones_like(rms_out)
    assert torch.allclose(rms_out, expected, atol=0.1), (
        f"RMS output not normalized: {rms_out}"
    )
    print(f"RMSNorm test passed! Shape: {out.shape}")
