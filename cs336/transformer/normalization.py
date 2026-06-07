"""
Normalization layers: RMSNorm (Root Mean Square Layer Normalization).

Used across all transformer variants for pre-norm stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes: output = x * weight / sqrt(mean(x^2) + eps)

    Unlike LayerNorm, RMSNorm does not subtract the mean or use a bias term,
    making it more computationally efficient while achieving comparable results.

    Args:
        hidden_size: The hidden dimension to normalize.
        eps: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., hidden_size].

        Returns:
            Normalized tensor with same shape.
        """
        input_dtype: torch.dtype = x.dtype
        x_float: torch.Tensor = x.float()
        rms: torch.Tensor = torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        x_norm: torch.Tensor = x_float * rms
        return (x_norm * self.weight.float()).to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


class DeepNorm(nn.Module):
    """DeepNorm initialization wrapper for very deep transformers.

    From DeepNet: "Scaling Transformers to 1,000 Layers" (Wang et al., 2022).
    Used in DeepSeek-V3 for stable training of deep architectures.

    DeepNorm modifies the residual connection: output = x + alpha * f(Norm(x))
    where alpha is a learned or fixed scaling factor.

    Args:
        hidden_size: Hidden dimension.
        alpha: DeepNorm scaling factor (default from DeepNet paper).
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha: float = 1.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.norm: RMSNorm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DeepNorm: x + alpha * sublayer(Norm(x)).

        Note: The residual connection itself is NOT handled here; this class
        only provides the normalized input with the scaling factor baked in.
        The caller should do: x = x + alpha * sublayer(self.norm(x))

        Args:
            x: Input tensor of shape [..., hidden_size].

        Returns:
            Normalized and scaled tensor.
        """
        return self.norm(x) * self.alpha


# Quick test
if __name__ == "__main__":
    batch, seq, hidden = 2, 16, 768

    rms_norm = RMSNorm(hidden_size=hidden, eps=1e-5)
    x = torch.randn(batch, seq, hidden)
    out = rms_norm(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    rms_out = torch.sqrt(out.float().pow(2).mean(dim=-1) + 1e-5)
    expected = torch.ones_like(rms_out)
    assert torch.allclose(rms_out, expected, atol=0.1), "RMS not normalized"
    print(f"RMSNorm: OK, shape={out.shape}")

    dn = DeepNorm(hidden_size=hidden, alpha=1.2)
    out = dn(x)
    assert out.shape == x.shape
    print(f"DeepNorm: OK")
    print("\nAll normalization tests passed!")
