"""
Lecture 15: Long-Context Attention Optimizations
=================================================
Implements and compares:
  (1) Full scaled dot-product attention       -- O(n^2) memory
  (2) Sliding-window attention               -- O(n * w) memory
  (3) Streaming attention with window cache  -- chunked sliding-window
  (4) RoPE with NTK-aware scaling            -- extend context via frequency scaling
  (5) KV cache eviction strategy             -- keep first + last tokens

All implementations run on CPU.  Dependencies: torch, numpy, math (standard).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _print_memory(title: str, matrix_shape: Tuple[int, ...], elements: int) -> None:
    """Pretty-print a single memory-footprint line."""
    print(f"  {title:40s} shape={str(matrix_shape):24s}  elements={elements:>12,d}")


def _divider(char: str = "=", width: int = 100) -> None:
    print(char * width)


# ===========================================================================
# 1. Full scaled dot-product attention  -  O(n^2) memory
# ===========================================================================


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> Tuple[torch.Tensor, int]:
    """
    Standard scaled dot-product attention.
    Returns (output, num_elements_in_attention_matrix).
    """
    d_k = q.size(-1)
    # scores: (batch, heads, seq_q, seq_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    elements = scores.numel()  # O(seq^2) elements

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, elements


# ===========================================================================
# 2. Sliding-window attention  -  O(n * w) memory
# ===========================================================================


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, int]:
    """
    Each query position i attends only to keys [max(0, i-window_size+1) .. i].

    We implement this with a causal mask that is further restricted to a
    window, so the effective attention matrix has at most n * window_size
    non-masked entries.  Returns (output, num_non_masked_elements).
    """
    seq_len = q.size(2)
    device = q.device

    # Causal mask: rows = queries, cols = keys; 1 = attend, 0 = mask
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # Window mask: only allow positions within `window_size` distance
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    window_mask = (col_idx >= row_idx - window_size + 1).int()

    mask = causal * window_mask  # intersect
    elements = mask.sum().item()  # non-masked (attended) positions

    # Apply mask by setting disallowed positions to -inf
    attn_mask = (
        mask.float().masked_fill(mask == 0, float("-inf")).unsqueeze(0).unsqueeze(0)
    )

    return full_attention(q, k, v, mask=1 - mask.unsqueeze(0).unsqueeze(0))[0], int(
        elements
    )


# ===========================================================================
# 3. Streaming attention  (chunked sliding-window)
# ===========================================================================


def streaming_attention(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    chunk_size: int = 128,
    window_size: int = 256,
) -> Tuple[torch.Tensor, int]:
    """
    Process the sequence in *chunks*.  For every chunk we maintain a
    rolling KV cache whose size is bounded by `window_size` tokens.
    The attention matrix for each chunk is (chunk_size) x (cache_size),
    so the total memory is roughly  n_chunks * chunk_size * window_size.

    This demonstrates constant per-step memory w.r.t. sequence length.
    Returns (concatenated_output, total_elements_in_all_chunks).
    """
    batch, heads, seq_len, d_k = q_full.shape
    device = q_full.device

    # We model a *causal* streaming scenario: position t can only look
    # backward up to `window_size` tokens.
    out_chunks: list[torch.Tensor] = []
    total_elements = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        # Determine the cache window  [cache_start, end)
        cache_start = max(0, end - window_size)

        q_chunk = q_full[:, :, start:end, :]  # (B, H, chunk, d)
        k_cache = k_full[:, :, cache_start:end, :]  # (B, H, cache, d)
        v_cache = v_full[:, :, cache_start:end, :]

        scores = torch.matmul(q_chunk, k_cache.transpose(-2, -1)) / math.sqrt(d_k)
        total_elements += scores.numel()

        attn_weights = F.softmax(scores, dim=-1)
        out_chunks.append(torch.matmul(attn_weights, v_cache))

    output = torch.cat(out_chunks, dim=2)
    return output, total_elements


# ===========================================================================
# 4. Rotary Position Embedding (RoPE) with NTK-aware scaling
# ===========================================================================


def _compute_rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Return the inverse-frequency vector for RoPE (shape: dim//2)."""
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    theta = base ** (-i / dim)
    return theta  # shape (dim//2,)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to tensor x.

    x:  (..., seq_len, dim)   -- typically query or key
    positions: (seq_len,)       -- absolute position indices
    """
    *prefix, seq_len, dim = x.shape
    assert dim % 2 == 0, "RoPE requires an even dimension."

    theta = _compute_rope_frequencies(dim, base)  # (dim//2,)

    # Compute cos/sin for each (position, dimension-pair)
    # Shape: (1, 1, seq_len, dim//2)  -- broadcastable over batch & heads
    pos = positions.float().unsqueeze(-1)  # (seq_len, 1)
    freqs = pos * theta.unsqueeze(0)  # (seq_len, dim//2)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    # Reshape x into pairs for rotation
    x_pairs = x.reshape(*prefix, seq_len, dim // 2, 2)  # last dim = (real, imag)

    x_out = torch.empty_like(x_pairs)
    x_out[..., 0] = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
    x_out[..., 1] = x_pairs[..., 0] * sin + x_pairs[..., 1] * cos

    return x_out.reshape(*prefix, seq_len, dim)


def ntk_aware_rope_base(
    dim: int,
    original_max_seq_len: int = 2048,
    target_max_seq_len: int = 8192,
    original_base: float = 10000.0,
) -> float:
    """
    NTK-aware scaling: adjust the RoPE base frequency so that high-frequency
    (low-dimension) pairs stay nearly unchanged while low-frequency (high-
    dimension) pairs are "stretched" to accommodate the longer context.

    The scaling factor s is derived from the NTK (Neural Tangent Kernel)
    intuition: let  s = (target / original) ^ (dim / (dim-2)).
    Then  new_base = original_base * s.

    Reference: "NTK-Aware Scaled RoPE" (bloc97, 2023)
               https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
    """
    if target_max_seq_len <= original_max_seq_len:
        return original_base

    scale = target_max_seq_len / original_max_seq_len
    # The exponent dim/(dim-2) ensures that the highest-frequency
    # components (small dim indices) are barely scaled, while the lowest
    # frequencies get the full scale factor applied.
    exponent = dim / (dim - 2)
    ntk_factor = scale**exponent
    new_base = original_base * ntk_factor

    return new_base


# ===========================================================================
# 5. KV-cache eviction  (keep first-k + last-m, evict middle)
# ===========================================================================


def kv_cache_eviction(
    k: torch.Tensor,
    v: torch.Tensor,
    keep_first: int,
    keep_last: int,
    cache_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    StreamLLM-style eviction: when the KV cache exceeds `cache_capacity`,
    retain the first `keep_first` tokens (attention sinks) and the last
    `keep_last` tokens (recent context), discarding the middle segment.

    Returns (evicted_k, evicted_v, num_evicted_tokens).
    """
    seq_len = k.size(2)
    if seq_len <= cache_capacity:
        return k, v, 0

    keep = keep_first + keep_last
    if keep >= cache_capacity:
        # Not enough room; keep as much as we can from the tail
        k_evicted = k[:, :, -cache_capacity:, :]
        v_evicted = v[:, :, -cache_capacity:, :]
        evicted = seq_len - cache_capacity
    else:
        middle_keep = cache_capacity - keep
        idx = list(range(keep_first)) + list(
            range(seq_len - keep_last - middle_keep, seq_len)
        )
        k_evicted = k[:, :, idx, :]
        v_evicted = v[:, :, idx, :]
        evicted = seq_len - cache_capacity

    return k_evicted, v_evicted, evicted


# ===========================================================================
# Comparison driver
# ===========================================================================


def compare_methods() -> None:
    """
    Run all five attention variants across sequence lengths
    [256, 512, 1024, 2048, 4096] and print comparison tables.
    """
    seq_lengths = [256, 512, 1024, 2048, 4096]
    batch, heads, dim = 1, 4, 64  # small dim for demonstration
    window_size = 128
    chunk_size = 128
    keep_first, keep_last, kv_capacity = 4, 128, 128

    print("\n" + "=" * 100)
    print(
        "  LECTURE 15: LONG-CONTEXT ATTENTION OPTIMIZATIONS  --  MEMORY FOOTPRINT COMPARISON"
    )
    print("=" * 100)
    print(
        f"  Config: batch={batch}  heads={heads}  d_model={dim}  window={window_size}  chunk={chunk_size}"
    )
    print()

    # ------------------------------------------------------------------
    # Table 1: Full attention vs Sliding-window (theoretical)
    # ------------------------------------------------------------------
    print(
        "  TABLE 1 -- Memory footprint (number of elements in attention score matrix)"
    )
    print(
        f"  {'Seq Len':>8s}  {'Full (n^2)':>15s}  {'Window (n*w)':>15s}  {'Reduction':>12s}"
    )
    print(f"  {'-' * 8}  {'-' * 15}  {'-' * 15}  {'-' * 12}")

    for seq_len in seq_lengths:
        full_el = seq_len * seq_len
        window_el = seq_len * window_size
        ratio = full_el / window_el if window_el > 0 else float("inf")
        print(f"  {seq_len:>8d}  {full_el:>15,d}  {window_el:>15,d}  {ratio:>11.1f}x")

    # ------------------------------------------------------------------
    # Build actual tensors and run methods
    # ------------------------------------------------------------------
    print(f"\n  {'=' * 70}")
    print(
        "  TABLE 2 -- Measured elements (actual tensor runs) + Kv-cache eviction stats"
    )
    print(f"  {'=' * 70}")

    torch.manual_seed(42)

    for seq_len in seq_lengths:
        _divider("-", 90)
        print(f"  Sequence length = {seq_len}")
        _divider("-", 90)

        # Create random Q/K/V
        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)
        v = torch.randn(batch, heads, seq_len, dim)

        # ---- 1. Full attention ----
        _, full_el = full_attention(q, k, v)
        _print_memory("1. Full attention", (seq_len, seq_len), full_el)

        # ---- 2. Sliding-window attention ----
        _, sw_el = sliding_window_attention(q, k, v, window_size)
        _print_memory("2. Sliding-window", (seq_len, window_size), sw_el)

        # ---- 3. Streaming attention ----
        _, stream_el = streaming_attention(q, k, v, chunk_size, window_size)
        _print_memory("3. Streaming (chunked)", (seq_len, window_size), stream_el)

        # ---- 4. NTK-aware RoPE scaling ----
        # Demonstrate base-frequency change
        original_base = 10000.0
        new_base = ntk_aware_rope_base(dim, 2048, 8192, original_base)
        positions = torch.arange(seq_len)
        q_rope = apply_rope(q, positions, base=original_base)
        k_rope = apply_rope(k, positions, base=original_base)
        # Attention with RoPE has same memory but we record the scaling info
        _, rope_el = full_attention(q_rope, k_rope, v)
        print(
            f"  4. RoPE (same memory as full)   base: {original_base:.0f}"
            f"  ->  NTK-scaled base for 4x context: {new_base:.1f}"
        )

        # ---- 5. KV-cache eviction ----
        evicted_k, evicted_v, num_evicted = kv_cache_eviction(
            k, v, keep_first, keep_last, kv_capacity
        )
        orig_tokens = k.size(2)
        remaining = evicted_k.size(2)
        print(f"  5. KV-cache eviction")
        print(
            f"       keep_first={keep_first}  keep_last={keep_last}"
            f"  capacity={kv_capacity}"
        )
        print(
            f"       original tokens={orig_tokens:>5d}"
            f"  remaining={remaining:>5d}"
            f"  evicted={num_evicted:>5d}"
            f"  memory saved={orig_tokens - remaining:>5d} tokens"
        )
        print()

    # ------------------------------------------------------------------
    # Bonus: NTK-aware scaling demonstration across target lengths
    # ------------------------------------------------------------------
    _divider("=", 100)
    print("  BONUS: NTK-aware base scaling for RoPE")
    print(f"  Original base = 10000.0, training context = 2048")
    print(
        f"  {'Target Len':>12s}  {'Extension Ratio':>18s}  {'New Base':>15s}  {'Log2(Base)':>12s}"
    )
    print(f"  {'-' * 12}  {'-' * 18}  {'-' * 15}  {'-' * 12}")

    targets = [4096, 8192, 16384, 32768, 65536, 131072]
    for target in targets:
        nb = ntk_aware_rope_base(dim, 2048, target)
        print(
            f"  {target:>12,d}  {target / 2048:>17.1f}x  {nb:>15.1f}  {math.log2(nb):>11.2f}"
        )

    _divider("=", 100)
    print("\n  Done. All comparisons completed on CPU.\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    compare_methods()
