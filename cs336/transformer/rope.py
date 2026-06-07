"""
Rotary Position Embedding (RoPE) implementations.

Provides:
- Standard RoPE with configurable theta
- YaRN extension for context length extrapolation
- Dynamic NTK-aware scaling
- QK Normalization (DeepSeek-V3 style)

Based on:
- Su et al., 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Peng et al., 2023 "YaRN: Efficient Context Window Extension"
- LocalLLaMA / NTK-aware scaling
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with precomputed frequency tables.

    Supports standard RoPE and NTK-aware dynamic scaling.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length for precomputation.
        theta: Base frequency (default: 10000.0 for Llama).
        scaling_factor: NTK-aware scaling factor for extended context.
        use_ntk: Whether to use NTK-aware dynamic scaling.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
        use_ntk: bool = False,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.theta: float = theta
        self.scaling_factor: float = scaling_factor
        self.use_ntk: bool = use_ntk

        self._precompute_freqs(max_seq_len, theta)

    def _precompute_freqs(self, max_seq_len: int, theta: float) -> None:
        """Precompute cos and sin tables for all positions."""
        inv_freqs: torch.Tensor = 1.0 / (
            theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        positions: torch.Tensor = torch.arange(max_seq_len, dtype=torch.float32)
        angles: torch.Tensor = torch.outer(positions, inv_freqs)
        emb: torch.Tensor = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _get_ntk_freqs(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin with NTK-aware scaling on the fly."""
        ntk_theta: float = self.theta * self.scaling_factor
        inv_freqs: torch.Tensor = 1.0 / (
            ntk_theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )
        positions: torch.Tensor = torch.arange(
            seq_len, dtype=torch.float32, device=device
        )
        angles: torch.Tensor = torch.outer(positions, inv_freqs)
        emb: torch.Tensor = torch.cat([angles, angles], dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        seq_len: int,
        device: torch.device | str = "cpu",
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors for the requested sequence length.

        Args:
            seq_len: Number of positions needed.
            device: Target device.
            start_pos: Starting position offset (for incremental decoding).

        Returns:
            (cos, sin) tuple, each of shape [1, 1, seq_len, dim].
        """
        if isinstance(device, str):
            device = torch.device(device)

        need_len: int = start_pos + seq_len

        if self.use_ntk and need_len > self.max_seq_len:
            cos, sin = self._get_ntk_freqs(need_len, device)
            cos = cos[start_pos:need_len].unsqueeze(0).unsqueeze(0)
            sin = sin[start_pos:need_len].unsqueeze(0).unsqueeze(0)
            return cos, sin

        if need_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {need_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Enable use_ntk=True for dynamic scaling."
            )

        return (
            self.cos_cached[start_pos:need_len].to(device).unsqueeze(0).unsqueeze(0),
            self.sin_cached[start_pos:need_len].to(device).unsqueeze(0).unsqueeze(0),
        )


class YaRN(nn.Module):
    """YaRN (Yet another RoPE extensioN) for context length extrapolation.

    Implements the NTK-by-parts interpolation with ramp function as described
    in Peng et al., 2023.

    Args:
        dim: Head dimension.
        max_seq_len: Original maximum sequence length.
        extended_max_seq_len: Target extended maximum sequence length.
        theta: Base frequency.
        beta_fast: Fast frequency boundary (default: 32).
        beta_slow: Slow frequency boundary (default: 1).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        extended_max_seq_len: int = 32768,
        theta: float = 10000.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.extended_max_seq_len: int = extended_max_seq_len
        self.theta: float = theta
        self.beta_fast: int = beta_fast
        self.beta_slow: int = beta_slow

        self.scale: float = float(extended_max_seq_len) / float(max_seq_len)
        self.ntk_alpha: float = self._compute_ntk_alpha()

        self._precompute()

    def _compute_ntk_alpha(self) -> float:
        """Compute the NTK-aware interpolation factor."""
        return self.scale  # Equivalent to 1/w in the paper for the "length" method

    def _compute_ramp(self, freq_mask: torch.Tensor) -> torch.Tensor:
        """Compute the interpolation ramp for NTK-by-parts."""
        wavelength: torch.Tensor = 2.0 * math.pi * self.theta**freq_mask
        low_freq_mask: torch.Tensor = wavelength >= (self.beta_fast * self.max_seq_len)
        high_freq_mask: torch.Tensor = wavelength <= (self.beta_slow * self.max_seq_len)
        ramp: torch.Tensor = torch.where(
            low_freq_mask | high_freq_mask,
            torch.ones_like(freq_mask),
            (self.max_seq_len / wavelength - self.beta_slow)
            / (self.beta_fast - self.beta_slow),
        )
        return ramp.clamp(0.0, 1.0)

    def _precompute(self) -> None:
        """Precompute cos and sin tables for extended sequence length."""
        dim_half: int = self.dim // 2
        inv_freqs: torch.Tensor = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

        # NTK-by-parts: blend between original and scaled frequencies
        freq_mask: torch.Tensor = (
            torch.arange(0, dim_half, dtype=torch.float32) / dim_half
        )
        ramp: torch.Tensor = self._compute_ramp(freq_mask)

        # Linear interpolation: 1/scale for high freqs (unchanged), 1 for low freqs
        blend: torch.Tensor = ramp / self.scale + (1.0 - ramp)
        inv_freqs = inv_freqs * blend

        positions: torch.Tensor = torch.arange(
            self.extended_max_seq_len, dtype=torch.float32
        )
        angles: torch.Tensor = torch.outer(positions, inv_freqs)
        emb: torch.Tensor = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device | str = "cpu",
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors for YaRN-scaled RoPE.

        Args:
            seq_len: Number of positions needed.
            device: Target device.
            start_pos: Starting position offset.

        Returns:
            (cos, sin) tuple, each of shape [1, 1, seq_len, dim].
        """
        if isinstance(device, str):
            device = torch.device(device)

        end_pos: int = start_pos + seq_len
        if end_pos > self.extended_max_seq_len:
            raise ValueError(
                f"Position {end_pos} exceeds extended_max_seq_len {self.extended_max_seq_len}"
            )

        return (
            self.cos_cached[start_pos:end_pos].to(device).unsqueeze(0).unsqueeze(0),
            self.sin_cached[start_pos:end_pos].to(device).unsqueeze(0).unsqueeze(0),
        )


class QKNorm(nn.Module):
    """QK Normalization (DeepSeek-V3 style).

    Applies RMSNorm to query and key tensors before the attention computation.
    This stabilizes training for very deep models and improves loss convergence.

    Args:
        head_dim: Per-head dimension.
        eps: Small constant for numerical stability.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.q_norm: nn.Parameter = nn.Parameter(torch.ones(head_dim))
        self.k_norm: nn.Parameter = nn.Parameter(torch.ones(head_dim))
        self.eps: float = eps

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RMSNorm to query and key.

        Args:
            query: [..., head_dim]
            key: [..., head_dim]

        Returns:
            (normalized_query, normalized_key) with same shapes.
        """
        input_dtype = query.dtype

        def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_float = x.float()
            rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (x_float * rms * weight.float()).to(input_dtype)

        return _rms_norm(query, self.q_norm), _rms_norm(key, self.k_norm)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    The rotation is computed as:
        x_rotated = x * cos + rotate_half(x) * sin

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim].
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        cos: Cosine table of shape [1, 1, seq_len, head_dim].
        sin: Sine table of shape [1, 1, seq_len, head_dim].

    Returns:
        (rotated_q, rotated_k) tuple with same shapes as inputs.
    """

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = x[..., : x.shape[-1] // 2]
        x2: torch.Tensor = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_rotated: torch.Tensor = (q * cos) + (_rotate_half(q) * sin)
    k_rotated: torch.Tensor = (k * cos) + (_rotate_half(k) * sin)

    return q_rotated, k_rotated


# Quick test
if __name__ == "__main__":
    dim, max_len = 64, 256
    batch, n_heads, seq = 2, 8, 32

    # Standard RoPE
    rope = RotaryEmbedding(dim=dim, max_seq_len=max_len, theta=10000.0)
    cos, sin = rope.forward(seq)
    q = torch.randn(batch, n_heads, seq, dim)
    k = torch.randn(batch, n_heads, seq, dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert not torch.allclose(q, q_rot)
    print(f"Standard RoPE: OK, shape={q_rot.shape}")

    # Incremental decoding
    cos2, sin2 = rope.forward(1, start_pos=10)
    assert cos2.shape == (1, 1, 1, dim)
    print(f"Incremental RoPE: OK")

    # NTK-aware scaling
    rope_ntk = RotaryEmbedding(
        dim=dim, max_seq_len=128, theta=10000.0, scaling_factor=2.0, use_ntk=True
    )
    cos3, sin3 = rope_ntk.forward(200)
    assert cos3.shape == (1, 1, 200, dim)
    print(f"NTK RoPE: OK, shape={cos3.shape}")

    # YaRN
    yarn = YaRN(dim=dim, max_seq_len=128, extended_max_seq_len=512, theta=10000.0)
    cos4, sin4 = yarn.forward(300)
    assert cos4.shape == (1, 1, 300, dim)
    print(f"YaRN: OK, shape={cos4.shape}")

    # QK Norm
    qk_norm = QKNorm(head_dim=dim)
    q_normed, k_normed = qk_norm(q, k)
    assert q_normed.shape == q.shape
    assert k_normed.shape == k.shape
    print(f"QKNorm: OK, shapes={q_normed.shape}, {k_normed.shape}")

    print("\nAll RoPE tests passed!")
