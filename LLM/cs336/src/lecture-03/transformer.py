"""
第03讲 — Transformer：从零开始的实现。

提供：
  - MultiHeadAttention（支持 GQA 和 causal masking）
  - TransformerBlock（pre-norm / post-norm）
  - TransformerLM（embedding → N 个 block → LM head）
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """多头（自）注意力，支持可选的 Grouped-Query Attention (GQA)。

    当 ``num_kv_heads < num_heads`` 时，key/value 投影输出更少的头；
    query 头被分成若干组，每组共享同一个 K、V 头。

    Parameters
    ----------
    dim : int
        模型维度。
    num_heads : int
        Query 头的数量。
    num_kv_heads : int, optional
        Key/value 头的数量（默认 = num_heads → MHA）。
    dropout : float
        作用于 attention 权重的 dropout 率。
    bias : bool
        线性投影是否包含偏置。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_head_dim = dim // num_heads  # usually same as head_dim
        self.groups = num_heads // num_kv_heads
        self.scale = self.head_dim**-0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.kv_head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.kv_head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """扩展 KV 头以匹配 GQA 的 Q 头数量。

        输入形状: (B, num_kv_heads, S, head_dim)
        输出形状: (B, num_heads,      S, head_dim)
        """
        if self.groups == 1:
            return kv
        B, n_kv, S, d = kv.shape
        # reshape → (B, n_kv, 1, S, d) → expand → (B, n_kv * groups, S, d)
        kv = kv[:, :, None, :, :].expand(B, n_kv, self.groups, S, d)
        return kv.reshape(B, n_kv * self.groups, S, d)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : (B, S, dim)
            输入张量。
        mask : (S, S) or (B, 1, S, S), optional
            Attention mask。值为 ``True`` / ``-inf`` 的位置会被遮蔽。

        Returns
        -------
        out : (B, S, dim)
        """
        B, S, _ = x.shape

        # 投影
        q: torch.Tensor = (
            self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        )
        k: torch.Tensor = (
            self.k_proj(x)
            .view(B, S, self.num_kv_heads, self.kv_head_dim)
            .transpose(1, 2)
        )
        v: torch.Tensor = (
            self.v_proj(x)
            .view(B, S, self.num_kv_heads, self.kv_head_dim)
            .transpose(1, 2)
        )

        # GQA：扩展 KV 头
        k = self._repeat_kv(k)  # (B, n_heads, S, head_dim)
        v = self._repeat_kv(v)

        # 缩放点积注意力
        # scores: (B, n_heads, S, S)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = self.attn_dropout(attn_weights)
        attn_weights = attn_weights.to(q.dtype)

        out = attn_weights @ v  # (B, n_heads, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Causal mask 辅助函数
# ---------------------------------------------------------------------------


def causal_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """创建下三角 causal mask。

    返回一个 (1, 1, seq_len, seq_len) 张量，其中应被遮蔽的位置设为 ``-inf``。
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask.bool(), float("-inf")).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """简单的 MLP，支持可选的门控（SwiGLU 风格）。

    当 ``gate_proj`` 为 None 时使用单个线性投影（标准 FFN），
    当提供 ``gate_proj`` 时使用门控变体。
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        bias: bool = False,
        gated: bool = False,
    ):
        super().__init__()
        self.gated = gated
        if gated:
            self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            return self.down_proj(
                self.dropout(self.activation(self.gate_proj(x)) * self.up_proj(x))
            )
        return self.down_proj(self.dropout(self.activation(self.up_proj(x))))


# ---------------------------------------------------------------------------
# RMS Normalisation（可选；若未使用则回退到 LayerNorm）
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(dtype) * self.weight


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """单个 transformer block。

    支持 pre-norm（默认）和 post-norm 布局。

    Parameters
    ----------
    dim : int
    num_heads : int
    num_kv_heads : int, optional
    mlp_ratio : float
        隐藏层维度 = dim * mlp_ratio。
    dropout : float
    norm_eps : float
    pre_norm : bool
        若为 True（默认），LayerNorm 在 Attention / MLP *之前*应用。
        若为 False，则在*之后*应用。
    rms_norm : bool
        使用 RMSNorm 代替 LayerNorm。
    gated_mlp : bool
        使用 SwiGLU 风格的门控 MLP。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        pre_norm: bool = True,
        rms_norm: bool = False,
        gated_mlp: bool = False,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.pre_norm = pre_norm

        norm_cls = RMSNorm if rms_norm else nn.LayerNorm

        self.norm1 = (
            norm_cls(dim, eps=norm_eps) if rms_norm else nn.LayerNorm(dim, eps=norm_eps)
        )
        self.norm2 = (
            norm_cls(dim, eps=norm_eps) if rms_norm else nn.LayerNorm(dim, eps=norm_eps)
        )

        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        self.mlp = MLP(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            gated=gated_mlp,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.norm1(x), mask=mask)
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, mask=mask))
            x = self.norm2(x + self.mlp(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Language Model
# ---------------------------------------------------------------------------


class TransformerLM(nn.Module):
    """Decoder-only transformer 语言模型。

    Parameters
    ----------
    vocab_size : int
    dim : int
    num_layers : int
    num_heads : int
    num_kv_heads : int, optional
    mlp_ratio : float
    max_seq_len : int
        位置嵌入的最大序列长度。
    dropout : float
    pre_norm : bool
    rms_norm : bool
    gated_mlp : bool
    weight_tying : bool
        共享 embedding 和 LM-head 的权重。
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        pre_norm: bool = True,
        rms_norm: bool = False,
        gated_mlp: bool = False,
        weight_tying: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    rms_norm=rms_norm,
                    gated_mlp=gated_mlp,
                )
                for _ in range(num_layers)
            ]
        )

        norm_cls = RMSNorm if rms_norm else nn.LayerNorm
        self.final_norm = norm_cls(dim) if rms_norm else nn.LayerNorm(dim)

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if weight_tying:
            self.lm_head.weight = self.token_embedding.weight  # type: ignore[assignment]

        # 参数初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        input_ids : (B, S) long 张量。
        mask : 可选的 attention mask。

        Returns
        -------
        logits : (B, S, vocab_size)
        """
        B, S = input_ids.shape

        # Token + 位置嵌入
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        h = self.token_embedding(input_ids) + self.position_embedding(positions)
        h = self.dropout(h)

        # Causal mask（与用户指定的 mask 合并）
        cm = causal_mask(S, device=input_ids.device)
        if mask is not None:
            cm = cm + mask

        for block in self.blocks:
            h = block(h, mask=cm)

        h = self.final_norm(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cpu")
    vocab_size = 1000
    B, S = 2, 32
    dim = 128
    num_heads = 4

    # --- MultiHeadAttention ---
    mha = MultiHeadAttention(dim=dim, num_heads=num_heads, num_kv_heads=2, dropout=0.0)
    x = torch.randn(B, S, dim)
    cm = causal_mask(S)
    out = mha(x, mask=cm)
    assert out.shape == (B, S, dim), f"Expected {(B, S, dim)}, got {out.shape}"
    print(f"MultiHeadAttention output shape: {out.shape}  ✓")

    # --- TransformerBlock ---
    block = TransformerBlock(dim=dim, num_heads=num_heads, num_kv_heads=2)
    out = block(x, mask=cm)
    assert out.shape == (B, S, dim)
    print(f"TransformerBlock output shape: {out.shape}  ✓")

    # Post-norm 变体
    block_post = TransformerBlock(dim=dim, num_heads=num_heads, pre_norm=False)
    out = block_post(x, mask=cm)
    assert out.shape == (B, S, dim)
    print("Post-norm block output shape OK  ✓")

    # --- TransformerLM ---
    model = TransformerLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=2,
        num_heads=num_heads,
        num_kv_heads=2,
        max_seq_len=64,
    )
    input_ids = torch.randint(0, vocab_size, (B, S))
    logits = model(input_ids)
    assert logits.shape == (B, S, vocab_size), (
        f"Expected {(B, S, vocab_size)}, got {logits.shape}"
    )
    print(f"TransformerLM logits shape: {logits.shape}  ✓")

    # 参数数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    print("\nAll shape checks passed.")
