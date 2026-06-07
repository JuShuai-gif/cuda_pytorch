"""
Transformer Efficiency Analysis (Lecture 12)
=============================================
Builds a toy GPT-style transformer and analytically compares:
  - MHA (Multi-Head Attention)
  - MQA (Multi-Query Attention)
  - GQA (Grouped-Query Attention)

Analyses performed:
  1. FLOPs count for forward pass at varying sequence lengths (quadratic growth of attention)
  2. KV-cache memory footprint (FP16) at varying sequence lengths
  3. Parameter count comparison across MHA / MQA / GQA for several model sizes

All computations run on CPU only.  No CUDA required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    """Configuration for a GPT-style transformer."""

    n_layers: int = 12  # number of transformer blocks
    d_model: int = 768  # hidden dimension
    n_heads: int = 12  # number of query heads
    n_kv_heads: int = 12  # number of key/value heads (12=MHA, 1=MQA, 4=GQA)
    head_dim: int = 64  # dimension per head  (d_model = n_heads * head_dim)
    vocab_size: int = 50257  # vocabulary size (GPT-2 default)
    max_seq_len: int = 1024  # maximum sequence length
    ffn_multiplier: int = 4  # FFN hidden dim = d_model * ffn_multiplier

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) "
            f"* head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by "
            f"n_kv_heads ({self.n_kv_heads})"
        )

    @property
    def n_groups(self) -> int:
        """Number of query-head groups sharing one KV head (GQA)."""
        return self.n_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA).

    Each query head has dedicated key and value projections.
    n_kv_heads == n_heads.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = (
            self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B, n_heads, S, head_dim)
        k = self.W_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA).

    All query heads share a single key/value projection.
    n_kv_heads == 1.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = 1
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_k = nn.Linear(cfg.d_model, 1 * cfg.head_dim, bias=False)
        self.W_v = nn.Linear(cfg.d_model, 1 * cfg.head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = (
            self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B, n_heads, S, head_dim)
        k = (
            self.W_k(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, 1, S, head_dim)
        v = self.W_v(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Broadcast K/V across all query heads
        k = k.expand(B, self.n_heads, S, self.head_dim)
        v = v.expand(B, self.n_heads, S, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA).

    Query heads are partitioned into groups; each group shares one KV head.
    1 < n_kv_heads < n_heads.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.n_groups = cfg.n_groups  # n_heads // n_kv_heads

        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = (
            self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B, n_heads, S, head_dim)
        k = (
            self.W_k(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, n_kv_heads, S, head_dim)
        v = self.W_v(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat each KV head n_groups times so that every query head has a
        # corresponding K/V head for batched attention computation.
        k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, S, self.head_dim)
        k = k.reshape(B, self.n_heads, S, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, S, self.head_dim)
        v = v.reshape(B, self.n_heads, S, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """A single transformer block: attention + FFN with pre-norm residuals."""

    def __init__(self, attention: nn.Module, cfg: GPTConfig) -> None:
        super().__init__()
        self.attn = attention
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_multiplier * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ffn_multiplier * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


class GPT(nn.Module):
    """Toy GPT-style transformer model.

    Supports MHA, MQA, or GQA via config.n_kv_heads.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Choose attention variant based on n_kv_heads
        if cfg.n_kv_heads == cfg.n_heads:
            attn_cls = MultiHeadAttention
        elif cfg.n_kv_heads == 1:
            attn_cls = MultiQueryAttention
        else:
            attn_cls = GroupedQueryAttention

        self.blocks = nn.Sequential(
            *[TransformerBlock(attn_cls(cfg), cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie weights (standard practice in GPT models)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0)  # (1, S)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.lm_head(x)  # (B, S, vocab_size)


# ---------------------------------------------------------------------------
# FLOPs Analysis
# ---------------------------------------------------------------------------


def flops_matmul(m: int, n: int, k: int) -> int:
    """Return FLOPs for matrix multiply (m x k) @ (k x n) -> (m x n).

    Each output element requires k multiply-adds = 2*k floating-point operations.
    Total: 2 * m * n * k FLOPs.
    """
    return 2 * m * n * k


def compute_flops(
    seq_len: int,
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    ffn_multiplier: int = 4,
) -> Tuple[int, int]:
    """Compute analytical FLOPs for one forward pass through the transformer.

    Returns (attention_flops, total_flops) as integers.

    FLOPs are counted per the 2* convention: every multiply-add = 2 FLOPs.
    Only matrix-multiply operations are counted (linear layers + attention
    computations). LayerNorm, GELU, softmax, and embedding lookups are
    negligible and excluded for clarity.

    The attention FLOPs exhibit O(seq_len²) growth:
        Q @ K^T   ->  2 * n_layers * n_heads * seq_len² * head_dim
        Attn @ V  ->  2 * n_layers * n_heads * seq_len² * head_dim
        Total     ->  4 * n_layers * d_model * seq_len²
    """
    S = seq_len
    d_kv = n_kv_heads * head_dim  # total dimension for K/V projections

    # --- Per-layer FLOPs ---
    # Q projection:  (S, d_model) x (d_model, d_model)
    flops_q = flops_matmul(S, d_model, d_model)
    # K projection:  (S, d_model) x (d_model, d_kv)
    flops_k = flops_matmul(S, d_kv, d_model)
    # V projection:  same as K
    flops_v = flops_k
    # Q @ K^T:  (n_heads, S, head_dim) @ (n_kv_heads, S, head_dim)^T
    #   per head: (S, head_dim) @ (head_dim, S) with appropriate KV broadcasting
    flops_qkt = flops_matmul(S, S, head_dim) * n_heads
    # Attn @ V: per head: (S, S) @ (S, head_dim)
    flops_attnv = flops_matmul(S, head_dim, S) * n_heads
    # Output projection: (S, d_model) @ (d_model, d_model)
    flops_out = flops_matmul(S, d_model, d_model)
    # FFN up:   (S, d_model) @ (d_model, ffn*d_model)
    flops_ffn_up = flops_matmul(S, ffn_multiplier * d_model, d_model)
    # FFN down: (S, ffn*d_model) @ (ffn*d_model, d_model)
    flops_ffn_down = flops_matmul(S, d_model, ffn_multiplier * d_model)

    flops_per_layer_qkv = flops_q + flops_k + flops_v
    flops_per_layer_attn = flops_qkt + flops_attnv  # the O(S²) term
    flops_per_layer_linear = flops_out + flops_ffn_up + flops_ffn_down

    # --- Totals ---
    attention_flops = n_layers * flops_per_layer_attn
    total_flops = (
        n_layers * (flops_per_layer_qkv + flops_per_layer_attn + flops_per_layer_linear)
        + flops_matmul(S, vocab_size, d_model)  # LM head
    )

    return attention_flops, total_flops


# ---------------------------------------------------------------------------
# KV-Cache Memory Analysis
# ---------------------------------------------------------------------------


def compute_kv_cache_bytes(
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_element: int = 2,  # FP16
) -> int:
    """Compute KV-cache memory in bytes for the given sequence length.

    Each layer stores:
      - Key cache:   seq_len * n_kv_heads * head_dim  elements
      - Value cache: seq_len * n_kv_heads * head_dim  elements

    The key insight: MQA / GQA dramatically reduce KV-cache size because
    n_kv_heads is smaller than n_heads (MHA).

    Returns bytes.
    """
    elements_per_layer = 2 * seq_len * n_kv_heads * head_dim  # K + V
    return n_layers * elements_per_layer * bytes_per_element


# ---------------------------------------------------------------------------
# Parameter Count Analysis
# ---------------------------------------------------------------------------


def count_attention_params(
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> int:
    """Count parameters in the attention sub-layer (no bias).

    Q, K, V, O weight matrices only.
    """
    d_kv = n_kv_heads * head_dim
    params_q = d_model * d_model
    params_k = d_model * d_kv
    params_v = d_model * d_kv
    params_o = d_model * d_model
    return params_q + params_k + params_v + params_o


def count_total_params(
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_seq_len: int = 1024,
    ffn_multiplier: int = 4,
) -> int:
    """Count total model parameters.

    Matches the GPT model implementation: positional embedding is present,
    and the LM head is weight-tied with the token embedding (counted once).
    Biases are present only in LayerNorm layers.
    """
    # Token embedding (shared with LM head via weight tying)
    emb = vocab_size * d_model
    # Positional embedding
    pos_emb = max_seq_len * d_model
    # Per-block
    attn = count_attention_params(d_model, n_heads, n_kv_heads, head_dim)
    ffn = 2 * d_model * ffn_multiplier * d_model  # W1 + W2 in FFN
    ln = 2 * d_model * 2  # two LayerNorms per block (weight + bias)
    per_layer = attn + ffn + ln
    # Final LayerNorm
    final_ln = 2 * d_model
    return emb + pos_emb + n_layers * per_layer + final_ln


# ---------------------------------------------------------------------------
# Printing Helpers
# ---------------------------------------------------------------------------


def _human_readable(num: int) -> str:
    """Format a number into human-readable form (K, M, B, T)."""
    if abs(num) >= 1e12:
        return f"{num / 1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def print_header(title: str) -> None:
    """Print a centered section header."""
    width = 78
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_flops_table(
    seq_lengths: List[int],
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
) -> None:
    """Print a table showing FLOPs breakdown vs sequence length."""
    header = f"{'Seq Len':>8s}  {'Attn FLOPs':>14s}  {'Total FLOPs':>14s}  {'Attn %':>8s}  {'KV Cache (MB)':>14s}"
    sep = "-" * len(header)

    print()
    print(
        f"  Model: n_layers={n_layers}, d_model={d_model}, "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}"
    )
    print(f"  (Q @ K^T + Attn @ V  grows as O(S²), linear projections grow as O(S))")
    print()
    print(sep)
    print(header)
    print(sep)

    for s in seq_lengths:
        attn_flops, total_flops = compute_flops(
            s,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
        )
        kv_bytes = compute_kv_cache_bytes(s, n_layers, n_kv_heads, head_dim)
        kv_mb = kv_bytes / (1024 * 1024)
        attn_pct = attn_flops / total_flops * 100
        print(
            f"{s:>8d}  "
            f"{_human_readable(attn_flops):>14s}  "
            f"{_human_readable(total_flops):>14s}  "
            f"{attn_pct:>7.1f}%  "
            f"{kv_mb:>14.2f}"
        )

    print(sep)
    print()

    # Show the quadratic growth ratio explicitly
    base_s = seq_lengths[0]
    base_attn, _ = compute_flops(
        base_s,
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
    )
    print("  Quadratic scaling verification (attention FLOPs vs seq_len):")
    print(
        f"  {'Seq Len':>8s}  {'Attn FLOPs':>14s}  {'Ratio':>8s}  {'Expected (S/S0)²':>18s}"
    )
    print(f"  {'-' * 8}  {'-' * 14}  {'-' * 8}  {'-' * 18}")
    for s in seq_lengths:
        attn_flops, _ = compute_flops(
            s,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
        )
        ratio = attn_flops / base_attn
        expected = (s / base_s) ** 2
        print(
            f"  {s:>8d}  "
            f"{_human_readable(attn_flops):>14s}  "
            f"{ratio:>7.1f}x  "
            f"{expected:>18.1f}"
        )


def print_comparison_table() -> None:
    """Print comparison table: MHA vs MQA vs GQA parameter counts."""
    configs = [
        # (label, n_layers, d_model, n_heads, n_kv_heads, head_dim, vocab_size)
        ("Small  (d=512, h=8)", 12, 512, 8, 8, 64, 50257),
        ("Medium (d=768, h=12)", 12, 768, 12, 12, 64, 50257),
        ("Large  (d=1024,h=16)", 12, 1024, 16, 16, 64, 50257),
        ("XL     (d=2048,h=32)", 12, 2048, 32, 32, 64, 50257),
    ]

    header = (
        f"{'Config':>22s}  {'MHA Params':>14s}  {'MQA Params':>14s}  "
        f"{'GQA-4 Params':>14s}  {'GQA-2 Params':>14s}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for label, nl, dm, nh, _, hd, vs in configs:
        # MHA:  n_kv_heads = n_heads
        # MQA:  n_kv_heads = 1
        # GQA:  n_kv_heads = 4 (4 groups)
        # GQA2: n_kv_heads = 2 (2 groups)
        params_mha = count_total_params(nl, dm, nh, nh, hd, vs)
        params_mqa = count_total_params(nl, dm, nh, 1, hd, vs)
        params_gqa4 = count_total_params(nl, dm, nh, 4, hd, vs) if nh >= 4 else None
        params_gqa2 = count_total_params(nl, dm, nh, 2, hd, vs) if nh >= 2 else None

        gqa4_str = _human_readable(params_gqa4) if params_gqa4 else "N/A"
        gqa2_str = _human_readable(params_gqa2) if params_gqa2 else "N/A"

        print(
            f"{label:>22s}  "
            f"{_human_readable(params_mha):>14s}  "
            f"{_human_readable(params_mqa):>14s}  "
            f"{gqa4_str:>14s}  "
            f"{gqa2_str:>14s}"
        )

    print(sep)
    print()
    print("  Legend:")
    print("    MHA   = Multi-Head Attention   (n_kv_heads == n_heads)")
    print("    MQA   = Multi-Query Attention  (n_kv_heads == 1)")
    print("    GQA-4 = Grouped-Query Attention (n_kv_heads == 4)")
    print("    GQA-2 = Grouped-Query Attention (n_kv_heads == 2)")
    print("  Parameter savings come from smaller K and V projection matrices.")


def print_kv_cache_comparison_table() -> None:
    """Print KV-cache memory comparison across MHA/MQA/GQA for a fixed model."""
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    n_layers = 12
    n_heads = 12
    head_dim = 64

    header = (
        f"{'Seq Len':>8s}  "
        f"{'MHA (MB)':>12s}  "
        f"{'GQA-4 (MB)':>12s}  "
        f"{'GQA-2 (MB)':>12s}  "
        f"{'MQA (MB)':>12s}  "
        f"{'MQA vs MHA':>12s}"
    )
    sep = "-" * len(header)

    print()
    print(
        f"  KV-Cache Memory (FP16)  -  n_layers={n_layers}, "
        f"d_model={n_heads * head_dim}, n_heads={n_heads}, head_dim={head_dim}"
    )
    print()
    print(sep)
    print(header)
    print(sep)

    for s in seq_lengths:
        mha_mb = compute_kv_cache_bytes(s, n_layers, n_heads, head_dim) / (1024 * 1024)
        gqa4_mb = compute_kv_cache_bytes(s, n_layers, 4, head_dim) / (1024 * 1024)
        gqa2_mb = compute_kv_cache_bytes(s, n_layers, 2, head_dim) / (1024 * 1024)
        mqa_mb = compute_kv_cache_bytes(s, n_layers, 1, head_dim) / (1024 * 1024)
        reduction = (1 - mqa_mb / mha_mb) * 100

        print(
            f"{s:>8d}  "
            f"{mha_mb:>12.2f}  "
            f"{gqa4_mb:>12.2f}  "
            f"{gqa2_mb:>12.2f}  "
            f"{mqa_mb:>12.2f}  "
            f"{reduction:>10.1f}%"
        )

    print(sep)
    print()
    print("  MQA reduces KV-cache by a factor of n_heads (12x in this example).")
    print("  GQA trades off between memory savings and attention quality.")


# ---------------------------------------------------------------------------
# Correctness Check (quick smoke test)
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    """Verify the model can do a forward pass and that our FLOPs framework
    produces self-consistent results."""
    print_header("Smoke Test: Forward Pass & Basic Checks")

    cfg = GPTConfig(
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        vocab_size=1000,
        max_seq_len=64,
    )

    # Instantiate all three attention variants
    cfg_mha = GPTConfig(**{**cfg.__dict__, "n_kv_heads": 4})
    cfg_gqa = GPTConfig(**{**cfg.__dict__, "n_kv_heads": 2})
    cfg_mqa = GPTConfig(**{**cfg.__dict__, "n_kv_heads": 1})

    for name, c in [("MHA", cfg_mha), ("GQA", cfg_gqa), ("MQA", cfg_mqa)]:
        model = GPT(c)
        model.eval()
        tokens = torch.randint(0, c.vocab_size, (1, 32))
        with torch.no_grad():
            logits = model(tokens)
        assert logits.shape == (1, 32, c.vocab_size), f"{name}: bad output shape"
        total = sum(p.numel() for p in model.parameters())
        expected = count_total_params(
            c.n_layers,
            c.d_model,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim,
            c.vocab_size,
            max_seq_len=c.max_seq_len,
        )
        assert total == expected, f"{name}: param count mismatch: {total} vs {expected}"
        print(
            f"  ✓ {name} passed  (params={_human_readable(total)}, "
            f"output={list(logits.shape)})"
        )

    # Verify the quadratic growth holds on a small config
    attn_64, _ = compute_flops(64, 2, 128, 4, 4, 32, 1000)
    attn_128, _ = compute_flops(128, 2, 128, 4, 4, 32, 1000)
    ratio = attn_128 / attn_64
    print(f"\n  ✓ Attention FLOPs ratio  64->128: {ratio:.1f}x  (expected 4.0x)")
    assert abs(ratio - 4.0) < 0.1, f"Quadratic scaling broken: got {ratio}"
    print("  All smoke tests passed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full transformer efficiency analysis."""

    print_header("Transformer Efficiency Analysis (Lecture 12)")
    print("  Demonstrates the quadratic complexity of attention (O(S²)),")
    print("  the impact of MHA/MQA/GQA on parameter count, and")
    print("  KV-cache memory savings from MQA/GQA.")
    print()

    # ------------------------------------------------------------------
    # 1. Smoke test
    # ------------------------------------------------------------------
    run_smoke_test()

    # ------------------------------------------------------------------
    # 2. FLOPs analysis at different sequence lengths
    # ------------------------------------------------------------------
    print_header("1. FLOPs vs Sequence Length  (GPT-2 Small-ish config)")

    seq_lengths = [64, 128, 256, 512, 1024]

    # Default config (MHA)
    print_flops_table(
        seq_lengths=seq_lengths,
        n_layers=12,
        d_model=768,
        n_heads=12,
        n_kv_heads=12,  # MHA
        head_dim=64,
        vocab_size=50257,
    )

    # Also show that MQA has nearly identical FLOPs (only K/V proj differ)
    print("  Note: MQA/GQA have the same attention FLOPs as MHA because each")
    print("        query head still computes attention against all S key tokens.")
    print("        The savings are in K/V projection FLOPs (smaller matrices)")
    print("        and KV-cache memory, not in the O(S²) attention itself.")

    # ------------------------------------------------------------------
    # 3. MHA vs MQA vs GQA parameter comparison
    # ------------------------------------------------------------------
    print_header("2. Parameter Count: MHA vs MQA vs GQA")
    print_comparison_table()

    # ------------------------------------------------------------------
    # 4. KV-cache memory comparison
    # ------------------------------------------------------------------
    print_header("3. KV-Cache Memory: MHA vs MQA vs GQA")
    print_kv_cache_comparison_table()

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print_header("Summary of Key Insights")

    print("""
  a) Attention FLOPs are O(S²) in sequence length.
     - Doubling S quadruples the Q@K^T and Attn@V costs.
     - For long sequences (S > 512), attention dominates total FLOPs.

  b) MQA and GQA do NOT reduce attention FLOPs.
     - The O(S²) Q@K^T + Attn@V work is identical to MHA.
     - Savings are in K/V linear projections and KV-cache memory.

  c) KV-cache memory grows linearly with S, but is slashed by MQA/GQA:
     - MQA:  n_kv_heads = 1    -> 1/n_heads  of MHA cache size
     - GQA:  n_kv_heads = g    -> g/n_heads  of MHA cache size

  d) Parameter reductions from MQA/GQA are real but modest:
     - Attention params shrink from 4*d_model²  to  ~2*d_model² (approx).
     - FFN and embedding params (bulk of the model) are unchanged.
     - The primary motive for MQA/GQA is KV-cache memory, not FLOPs.
""")


if __name__ == "__main__":
    main()
