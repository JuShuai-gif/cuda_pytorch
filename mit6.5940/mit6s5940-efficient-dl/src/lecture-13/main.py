#!/usr/bin/env python3
"""
Lecture 13: LLM Deployment Simulation
======================================

This script simulates key LLM deployment optimisation techniques on CPU:

  - Weight-only quantization: group-wise 4-bit (AWQ-style) quantization of
    linear layer weights, measuring perplexity degradation on synthetic text.
  - Model size comparison: FP32 vs FP16 vs INT4, showing storage reduction
    ratios and metadata overhead.
  - KV cache quantization: FP16 vs INT8 KV cache storage at various context
    lengths, illustrating the 2x memory savings.
  - FlashAttention concept: extensive inline comments explaining the tiling
    algorithm (no actual implementation, conceptual walkthrough only).

All computation is CPU-only.  Dependencies are limited to torch, numpy, and
the Python standard library (math).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Section 0: Configuration
# =============================================================================


@dataclass
class GPTConfig:
    """Hyper-parameters for the small GPT model used in this simulation."""

    vocab_size: int = 1000  # Synthetic vocabulary size
    d_model: int = 256  # Hidden dimension
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 3  # Number of transformer layers
    d_ff: int = 1024  # Feed-forward inner dimension
    max_seq_len: int = 256  # Maximum sequence length
    dropout: float = 0.0  # Dropout rate (0.0 for deterministic eval)


# =============================================================================
# Section 1: Small GPT Model Definition
# =============================================================================


class MultiHeadAttention(nn.Module):
    """Standard multi-head scaled dot-product attention.

    Splits the hidden dimension into ``n_heads`` independent heads, computes
    Q·K^T / sqrt(d_head) attention scores, applies softmax, and aggregates
    the value vectors.  The four projection matrices (Q, K, V, O) are the
    primary targets for weight-only quantization.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, d_model
        # Project to Q, K, V and reshape for multi-head: (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: softmax(Q·K^T / sqrt(d)) · V
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # Merge heads back: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation, typical of GPT-style transformers.

    Dimensions: d_model → d_ff → d_model (d_ff is typically 4× d_model).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    """A single GPT transformer block: pre-norm attention + pre-norm FFN.

    Uses pre-layer-normalisation (norm → sublayer → residual add), which is
    the standard layout in modern GPT architectures.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SmallGPT(nn.Module):
    """A compact 3-layer GPT-like autoregressive language model.

    This model is deliberately small (~3-4 M parameters) so that all
    quantization experiments run comfortably on CPU.  The architecture
    follows the standard GPT-2 layout: token + position embeddings,
    stacked transformer blocks with pre-norm, final layer norm, and an
    output projection (LM head) tied to the token embeddings for
    parameter efficiency.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)

        # LM head: project hidden states back to vocabulary logits.
        # We tie the weight with token_embedding.weight (weight tying).
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie weights

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for each position.

        Args:
            idx: LongTensor of shape (B, T) with token indices.

        Returns:
            FloatTensor of shape (B, T, vocab_size) containing logits.
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, T, d_model)
        x = tok_emb + pos_emb

        # Stacked transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final layer norm + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def count_parameters(self) -> dict[str, int]:
        """Return a breakdown of parameter counts by component."""
        counts: dict[str, int] = {}
        for name, param in self.named_parameters():
            counts[name] = param.numel()
        return counts


# =============================================================================
# Section 2: Group-wise 4-bit Quantization (AWQ-style)
# =============================================================================
#
# Overview of Activation-aware Weight Quantization (AWQ):
#   AWQ observes that not all weight channels contribute equally to the model
#   output.  Channels associated with large activation magnitudes (salient
#   channels) should be preserved with higher fidelity.  AWQ finds per-channel
#   scaling factors s_i that minimise:
#
#       min_s  || W·X - Q(W·diag(s)) · diag(s)^{-1} · X ||
#
#   where Q(·) is a quantizer.  The scaling factors s_i are found via a fast
#   grid search over the calibration set activations.
#
#   In this simulation we use per-group (group_size=128) min-max symmetric
#   quantization as the backbone and note where AWQ would add activation
#   awareness.  Real AWQ also searches for optimal per-channel scales; our
#   group-wise approach captures the locality benefit already.
#
# Quantization parameters:
#   - group_size: 128 elements per independent quantization group
#   - n_bits:     4 bits → 2^4 = 16 levels (values 0…15)
#   - scale:      (max - min) / 15, stored as FP16
#   - zero:       min value, stored as FP16
#
# Packing scheme:
#   Two consecutive 4-bit values are packed into a single uint8 byte:
#       packed_byte = (high_nibble << 4) | (low_nibble & 0x0F)
#   This yields exactly 0.5 bytes per parameter for the quantized payload.

GROUP_SIZE: int = 128
N_BITS: int = 4
Q_MAX: int = (1 << N_BITS) - 1  # 15


def _quantize_tensor(
    w: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, int]:
    """Quantize a float32 tensor to 4-bit per group and pack into uint8.

    Args:
        w:          Weight tensor of arbitrary shape.
        group_size: Number of scalar elements per quantisation group.

    Returns:
        packed:          uint8 tensor of packed 4-bit values (length ≈ N/2).
        scales:          float16 tensor of per-group scales (length = n_groups).
        zeros:           float16 tensor of per-group zero-points (length = n_groups).
        original_shape:  Shape of the input tensor for reconstruction.
        n_elements:      Number of valid (un-padded) elements.
    """
    original_shape = w.shape
    w_flat = w.detach().reshape(-1).float()
    n_elements = w_flat.numel()
    n_groups = (n_elements + group_size - 1) // group_size

    # Pad to a multiple of group_size so every group has the same cardinality.
    padded_size = n_groups * group_size
    w_padded = torch.zeros(padded_size, dtype=torch.float32)
    w_padded[:n_elements] = w_flat

    scales = torch.zeros(n_groups, dtype=torch.float16)
    zeros = torch.zeros(n_groups, dtype=torch.float16)
    q_vals = torch.zeros(padded_size, dtype=torch.uint8)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = w_padded[start:end]

        w_min = group.min().item()
        w_max = group.max().item()

        if abs(w_max - w_min) < 1e-9:
            # Degenerate group: all values identical — skip division.
            scale = 1.0
            zero = float(w_min)
        else:
            scale = (w_max - w_min) / Q_MAX
            zero = float(w_min)

        scales[g] = scale
        zeros[g] = zero

        q = torch.round((group - zero) / scale).clamp(0, Q_MAX)
        q_vals[start:end] = q.to(torch.uint8)

    # Pack two 4-bit nibbles into one uint8 byte.
    packed_size = padded_size // 2
    packed = torch.zeros(packed_size, dtype=torch.uint8)
    for i in range(packed_size):
        high = q_vals[2 * i].item() & 0x0F
        low = q_vals[2 * i + 1].item() & 0x0F
        packed[i] = (high << 4) | low

    return packed, scales, zeros, original_shape, n_elements


def _dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    original_shape: torch.Size,
    n_elements: int,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Reverse the 4-bit group-wise quantisation, returning a float32 tensor.

    Args:
        packed:         uint8 packed tensor from ``_quantize_tensor``.
        scales:         float16 per-group scales.
        zeros:          float16 per-group zero-points.
        original_shape: Shape of the original weight tensor.
        n_elements:     Number of valid elements (before padding).
        group_size:     Must match the value used during quantisation.

    Returns:
        Dequantized float32 tensor with shape ``original_shape``.
    """
    packed_size = packed.numel()
    padded_size = packed_size * 2  # 2 × 4-bit values per byte

    # Unpack nibbles.
    q_vals = torch.zeros(padded_size, dtype=torch.float32)
    for i in range(packed_size):
        byte_val = packed[i].item()
        q_vals[2 * i] = float((byte_val >> 4) & 0x0F)
        q_vals[2 * i + 1] = float(byte_val & 0x0F)

    n_groups = scales.numel()
    w_deq = torch.zeros(padded_size, dtype=torch.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, padded_size)
        q_group = q_vals[start:end]
        w_deq[start:end] = q_group * scales[g].float() + zeros[g].float()

    # Trim padding and reshape.
    w_deq = w_deq[:n_elements].reshape(original_shape)
    return w_deq


def quantize_linear_layers(model: nn.Module) -> dict[str, Any]:
    """Collect all nn.Linear weights, quantize them, return quantised data.

    Embedding and LayerNorm parameters are intentionally left in FP32.

    Returns:
        Dict mapping parameter name →
          {'packed', 'scales', 'zeros', 'original_shape', 'n_elements'}.
    """
    quantized: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            packed, scales, zeros, shape, n_el = _quantize_tensor(module.weight.data)
            quantized[name] = {
                "packed": packed,
                "scales": scales,
                "zeros": zeros,
                "original_shape": shape,
                "n_elements": n_el,
            }
    return quantized


def apply_dequantized_weights(model: nn.Module) -> None:
    """In-place replacement: quantize → dequantize every nn.Linear weight."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            packed, scales, zeros, shape, n_el = _quantize_tensor(module.weight.data)
            w_deq = _dequantize_tensor(packed, scales, zeros, shape, n_el)
            module.weight.data = w_deq


def compute_quantized_model_size(model: nn.Module) -> dict[str, float]:
    """Compute the total storage (bytes) of a model in FP32, FP16, and INT4.

    Classification is done by walking the module tree and checking
    ``isinstance`` against ``nn.Linear`` (quantizable) vs ``nn.Embedding`` /
    ``nn.LayerNorm`` (non-quantized).  Weight-tied parameters are detected
    via ``id()`` so they are never double-counted.  When a parameter is
    shared between a quantized module (nn.Linear) and a non-quantized one
    (e.g., token_embedding), it is classified as quantized because the
    ``apply_dequantized_weights`` routine will quantize it in practice.

    The INT4 size bundles the packed 4-bit payload (0.5 B per weight) plus
    per-group scale + zero metadata (2×2 B FP16 each).

    Returns:
        {'fp32_bytes': ..., 'fp16_bytes': ..., 'int4_bytes': ...,
         'quantizable_params': ..., 'non_quantized_params': ...,
         'total_groups': ...}
    """
    # -- Pass 1: collect parameter IDs owned by nn.Linear modules --------
    quantizable_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            for _, param in module.named_parameters(recurse=False):
                quantizable_ids.add(id(param))

    # -- Pass 2: count totals with deduplication --------------------------
    seen_ids: set[int] = set()
    total_params = 0
    total_quantizable_params = 0
    total_groups = 0

    for module in model.modules():
        for _, param in module.named_parameters(recurse=False):
            pid = id(param)
            if pid in seen_ids:
                continue  # tied weight (e.g., lm_head ↔ token_embedding)
            seen_ids.add(pid)

            n = param.numel()
            total_params += n

            if pid in quantizable_ids:
                total_quantizable_params += n
                total_groups += (n + GROUP_SIZE - 1) // GROUP_SIZE

    fp32_bytes = total_params * 4
    fp16_bytes = total_params * 2

    # INT4: packed weights (0.5 B/param) + per-group metadata.
    packed_weight_bytes = math.ceil(total_quantizable_params / 2)
    metadata_bytes = total_groups * 2 * 2  # scale (2 B) + zero (2 B), each FP16
    int4_bytes_quantizable = packed_weight_bytes + metadata_bytes

    # Non-quantized params stay in FP32 (embedding + layernorm).
    non_quantized_params = total_params - total_quantizable_params
    non_quantized_bytes = non_quantized_params * 4
    int4_bytes = int4_bytes_quantizable + non_quantized_bytes

    return {
        "fp32_bytes": float(fp32_bytes),
        "fp16_bytes": float(fp16_bytes),
        "int4_bytes": float(int4_bytes),
        "quantizable_params": float(total_quantizable_params),
        "non_quantized_params": float(non_quantized_params),
        "total_groups": float(total_groups),
    }


# =============================================================================
# Section 3: Perplexity Measurement
# =============================================================================


@torch.no_grad()
def compute_perplexity(model: nn.Module, input_ids: torch.Tensor) -> float:
    """Compute perplexity of an autoregressive model on a token sequence.

    Perplexity = exp(cross-entropy loss), where the loss is computed by
    predicting each token from its prefix (standard causal LM setup).

    Args:
        model:     SmallGPT model in eval mode.
        input_ids: LongTensor of shape (1, T).

    Returns:
        Perplexity as a Python float.
    """
    model.eval()
    logits = model(input_ids)  # (1, T, vocab_size)

    # Shift: use token 0..T-2 to predict token 1..T-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return math.exp(loss.item())


# =============================================================================
# Section 4: KV Cache Size Computation
# =============================================================================


def compute_kv_cache_size(
    config: GPTConfig,
    seq_len: int,
    kv_dtype_bytes: int,
    batch_size: int = 1,
) -> float:
    """Compute KV cache memory footprint for a given sequence length.

    The KV cache stores one Key and one Value tensor per transformer layer.
    Each has shape (batch_size, n_heads, seq_len, head_dim).

    Args:
        config:         GPT model configuration.
        seq_len:        Current context / sequence length.
        kv_dtype_bytes: Bytes per element (2 for FP16, 1 for INT8).
        batch_size:     Batch size (default 1 for single-sequence inference).

    Returns:
        Total KV cache size in bytes.
    """
    head_dim = config.d_model // config.n_heads
    per_tensor_elements = batch_size * config.n_heads * seq_len * head_dim
    # 2 tensors (K + V) per layer, config.n_layers layers
    total_elements = 2 * config.n_layers * per_tensor_elements
    return float(total_elements * kv_dtype_bytes)


# =============================================================================
# Section 5: FlashAttention Concept Walkthrough
# =============================================================================
#
# The comments below explain the core ideas behind FlashAttention
# (Dao et al., 2022).  No actual implementation is provided; the purpose
# is to build conceptual understanding of the algorithm.
#
# ---------------------------------------------------------------------------
# Problem: Standard attention is memory-bound
# ---------------------------------------------------------------------------
#
#   S = Q @ K^T          # (N, N) attention scores  ← O(N²) memory!
#   P = softmax(S)        # (N, N) attention weights ← O(N²) memory!
#   O = P @ V            # (N, d) output
#
# Both S and P are N×N matrices.  For N = 2048 tokens and FP16, this is
# 2048² × 2 = 8.4 MB per head × 4 heads = 33.6 MB.  For N = 8192 tokens:
# 8192² × 2 = 134 MB per head — it explodes quadratically.  On GPU, these
# intermediate tensors must be written to and read from High Bandwidth
# Memory (HBM), whose bandwidth (~1-2 TB/s) is the bottleneck.  The actual
# matmuls are fast (tensor cores deliver ~312 TFLOPS); the problem is
# that we spend most of the time moving data, not computing.
#
# ---------------------------------------------------------------------------
# Key insight: Compute attention in tiles
# ---------------------------------------------------------------------------
#
# FlashAttention divides Q, K, V into blocks (tiles) that fit into fast
# on-chip SRAM (~100 KB per SM on A100).  Instead of materialising the
# full N×N attention matrix in HBM, we:
#
#   1. LOAD  a block of Q rows into SRAM          [size: Br × d]
#   2. LOAD  a block of K, V rows into SRAM       [size: Bc × d]
#   3. COMPUTE local attention scores S_block = Q_block @ K_block^T  [Br × Bc]
#   4. SOFTMAX rescale with online softmax (see below)
#   5. ACCUMULATE output O_block += P_block @ V_block, rescaling as needed
#   6. WRITE  the final O_block back to HBM
#
# By iterating over K,V blocks for each Q block, we never hold the full
# N×N attention matrix in HBM.  The HBM traffic drops from O(N²) to O(N).
#
# ---------------------------------------------------------------------------
# Online Softmax: computing softmax without the full vector
# ---------------------------------------------------------------------------
#
# Standard softmax requires two passes over the score vector s:
#
#   m = max(s)                    # pass 1: find max for numerical stability
#   p = exp(s - m) / sum(exp(s - m))  # pass 2: exponentiate and normalise
#
# Online softmax maintains a running maximum and sum:
#
#   def online_softmax_update(s_block, m_old, l_old, O_old, V_block):
#       m_new = max(m_old, max(s_block))
#       l_new = exp(m_old - m_new) * l_old + sum(exp(s_block - m_new))
#       # Rescale old output and add new contribution
#       O_new = (l_old / l_new) * exp(m_old - m_new) * O_old
#             + exp(s_block - m_new) / l_new * V_block
#       return m_new, l_new, O_new
#
# This lets us process K,V blocks one at a time, accumulating the final
# softmax-normalised output incrementally.
#
# ---------------------------------------------------------------------------
# SRAM vs HBM: why tiling matters
# ---------------------------------------------------------------------------
#
# On a modern GPU (e.g., NVIDIA A100):
#   - HBM capacity:   40-80 GB, bandwidth ~2 TB/s
#   - SRAM capacity:  ~192 KB per SM (20 MB total), bandwidth ~19 TB/s
#
# Standard attention stores S and P in HBM → limited by HBM bandwidth.
# FlashAttention keeps all intermediate blocks in SRAM → limited by
# compute throughput.  The result is 2-4× wall-clock speedup and 10-20×
# memory reduction on long sequences.
#
# ---------------------------------------------------------------------------
# Casual masking in FlashAttention
# ---------------------------------------------------------------------------
#
# For autoregressive (GPT-style) models, the attention mask is lower-
# triangular.  FlashAttention handles this by only loading K,V blocks
# that are not masked out for the current Q block.  Specifically, for
# Q block i (covering rows i*Br … (i+1)*Br), only K,V blocks with
# column indices ≤ (i+1)*Br - 1 are loaded.  This further reduces
# memory traffic for early tokens.
#
# ---------------------------------------------------------------------------
# Why we are *not* implementing it here
# ---------------------------------------------------------------------------
#
# A correct FlashAttention implementation requires writing custom CUDA
# kernels (or Triton kernels) that carefully manage shared memory,
# thread-block tiling, and warp-level tensor core instructions.  This
# simulation runs on CPU only, so we restrict ourselves to conceptual
# explanation.  Readers interested in the production implementation
# should consult the official FlashAttention repository and the Dao et al.
# (2022) paper "FlashAttention: Fast and Memory-Efficient Exact Attention
# with IO-Awareness".


# =============================================================================
# Section 6: Main Demonstration
# =============================================================================


def _format_bytes(b: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024.0:
            return f"{b:,.1f} {unit}"
        b /= 1024.0
    return f"{b:,.1f} TB"


def _separator(title: str) -> None:
    """Print a formatted section separator."""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    """Run the full LLM deployment simulation.

    Steps:
      1. Instantiate the small GPT model.
      2. Measure FP32 perplexity on synthetic text.
      3. Quantize weights to 4-bit (group-wise), measure INT4 perplexity.
      4. Compare model sizes (FP32 / FP16 / INT4).
      5. Compare KV cache sizes (FP16 vs INT8) at various context lengths.
      6. Print the FlashAttention concept explanation.
    """

    # ------------------------------------------------------------------
    # Seed for reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1.  Build the model
    # ------------------------------------------------------------------
    config = GPTConfig()
    model = SmallGPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model built: {total_params:,} total parameters "
        f"({config.n_layers} layers, d_model={config.d_model}, "
        f"n_heads={config.n_heads})"
    )

    # Print detailed parameter breakdown
    print("\nParameter breakdown:")
    counts = model.count_parameters()
    total = sum(counts.values())
    for name, cnt in sorted(counts.items()):
        pct = 100.0 * cnt / total
        print(f"  {name:<55s} {cnt:>10,}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<55s} {total:>10,}")

    # ------------------------------------------------------------------
    # 2.  Synthetic text generation & baseline perplexity
    # ------------------------------------------------------------------
    _separator("Perplexity Measurement (FP32 Baseline)")

    # Create a synthetic sequence: 64 random tokens from our vocabulary.
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    print(
        f"Synthetic input sequence: shape={tuple(input_ids.shape)}, "
        f"vocab_size={config.vocab_size}"
    )

    ppl_fp32 = compute_perplexity(model, input_ids)
    print(f"FP32 Perplexity: {ppl_fp32:.4f}")

    # ------------------------------------------------------------------
    # 3.  Weight-only quantization → INT4 perplexity
    # ------------------------------------------------------------------
    _separator("Weight-Only Quantization (Group-wise 4-bit)")

    # Deep-copy the model so we can keep the original FP32 weights.
    model_int4 = copy.deepcopy(model)

    # Quantize-then-dequantize all nn.Linear weights in-place.
    apply_dequantized_weights(model_int4)

    ppl_int4 = compute_perplexity(model_int4, input_ids)
    delta_ppl = ppl_int4 - ppl_fp32
    print(f"INT4 Perplexity: {ppl_int4:.4f}")
    print(
        f"Perplexity degradation: {delta_ppl:+.4f} "
        f"({100.0 * delta_ppl / ppl_fp32:+.2f}%)"
    )
    print(
        "\nNote: The absolute perplexity values are extremely high because "
        "the model has\nrandomly-initialised (untrained) weights and is "
        "evaluated on random synthetic tokens.\nOn a properly-trained model "
        "perplexity would be ~10-50 and INT4 degradation\ntypically stays "
        "below 1-5% with group-wise quantisation.  The relative change\n"
        "still illustrates how quantisation noise affects output quality."
    )

    # ------------------------------------------------------------------
    # 4.  Model size comparison
    # ------------------------------------------------------------------
    _separator("Model Size Comparison")

    sizes = compute_quantized_model_size(model)
    fp32_b = sizes["fp32_bytes"]
    fp16_b = sizes["fp16_bytes"]
    int4_b = sizes["int4_bytes"]

    print(f"{'Format':<8s} {'Size':>14s} {'Ratio vs FP32':>15s}")
    print("-" * 40)
    print(f"{'FP32':<8s} {_format_bytes(fp32_b):>14s} {'1.00x (baseline)':>15s}")
    print(f"{'FP16':<8s} {_format_bytes(fp16_b):>14s} {f'{fp32_b / fp16_b:.2f}x':>15s}")
    print(f"{'INT4':<8s} {_format_bytes(int4_b):>14s} {f'{fp32_b / int4_b:.2f}x':>15s}")
    print()
    print(
        f"Quantizable parameters: {sizes['quantizable_params']:,.0f} "
        f"({100.0 * sizes['quantizable_params'] / total_params:.1f}% of total)"
    )
    print(
        f"Non-quantized parameters (embeddings + norms): "
        f"{sizes['non_quantized_params']:,.0f}"
    )
    print(
        f"Quantization groups (group_size={GROUP_SIZE}): {sizes['total_groups']:,.0f}"
    )

    # Detailed INT4 breakdown
    packed_bytes = math.ceil(sizes["quantizable_params"] / 2)
    metadata_bytes = int4_b - sizes["non_quantized_params"] * 4 - packed_bytes
    print(f"\nINT4 storage breakdown:")
    print(f"  Packed 4-bit weights:    {_format_bytes(packed_bytes):>14s}")
    print(
        f"  Scale + zero metadata:   {_format_bytes(metadata_bytes):>14s} "
        f"(FP16, 2×2 B per group)"
    )
    print(
        f"  Non-quantized (FP32):    "
        f"{_format_bytes(sizes['non_quantized_params'] * 4):>14s}"
    )

    # ------------------------------------------------------------------
    # 5.  KV cache size comparison
    # ------------------------------------------------------------------
    _separator("KV Cache Size: FP16 vs INT8")

    context_lengths = [256, 512, 1024, 2048]
    print(
        f"{'Context':>8s}  "
        f"{'FP16 KV Cache':>16s}  "
        f"{'INT8 KV Cache':>16s}  "
        f"{'Reduction':>12s}"
    )
    print("-" * 64)

    for L in context_lengths:
        kv_fp16 = compute_kv_cache_size(config, L, kv_dtype_bytes=2)
        kv_int8 = compute_kv_cache_size(config, L, kv_dtype_bytes=1)
        reduction = 100.0 * (1.0 - kv_int8 / kv_fp16) if kv_fp16 > 0 else 0.0

        print(
            f"{L:>8d}  "
            f"{_format_bytes(kv_fp16):>16s}  "
            f"{_format_bytes(kv_int8):>16s}  "
            f"{reduction:>9.1f}%"
        )

    # Show the per-layer breakdown for the longest context.
    L_max = context_lengths[-1]
    head_dim = config.d_model // config.n_heads
    per_layer_fp16 = 2 * config.n_heads * L_max * head_dim * 2  # K + V per layer
    print(f"\nPer-layer KV cache at L={L_max}:")
    print(
        f"  K tensor: (1, {config.n_heads}, {L_max}, {head_dim}) "
        f"= {_format_bytes(float(config.n_heads * L_max * head_dim * 2))} FP16"
    )
    print(
        f"  V tensor: (1, {config.n_heads}, {L_max}, {head_dim}) "
        f"= {_format_bytes(float(config.n_heads * L_max * head_dim * 2))} FP16"
    )
    print(f"  Per layer (K+V): {_format_bytes(float(per_layer_fp16))} FP16")
    print(
        f"  Total ({config.n_layers} layers): "
        f"{_format_bytes(float(per_layer_fp16 * config.n_layers))} FP16"
    )

    # ------------------------------------------------------------------
    # 6.  Summary & FlashAttention note
    # ------------------------------------------------------------------
    _separator("Summary")

    print("Weight Quantization:")
    print(f"  - Group-wise 4-bit (group_size={GROUP_SIZE})")
    print(
        f"  - Perplexity:  {ppl_fp32:.4f} (FP32) → {ppl_int4:.4f} (INT4) "
        f"({delta_ppl:+.4f})"
    )
    print(
        f"  - Model size:  {_format_bytes(fp32_b)} → {_format_bytes(int4_b)} "
        f"({fp32_b / int4_b:.1f}× smaller)"
    )

    print("\nKV Cache Quantization:")
    for L in context_lengths:
        kv_fp16 = compute_kv_cache_size(config, L, kv_dtype_bytes=2)
        kv_int8 = compute_kv_cache_size(config, L, kv_dtype_bytes=1)
        print(
            f"  - L={L:>4d}: {_format_bytes(kv_fp16):>10s} (FP16) → "
            f"{_format_bytes(kv_int8):>10s} (INT8) "
            f"({100.0 * (kv_fp16 - kv_int8) / kv_fp16:.0f}% reduction)"
        )

    print("\nFlashAttention:")
    print("  See the extensive inline comments in Section 5 above for the")
    print("  complete conceptual walkthrough of the tiling algorithm,")
    print("  online softmax, and the SRAM vs HBM memory hierarchy.")
    print("  Key takeaway: HBM traffic drops from O(N²) to O(N),")
    print("  enabling 2-4× speedup and 10-20× memory savings.")

    print("\n" + "=" * 72)
    print("  Simulation complete.  All computations performed on CPU.")
    print("=" * 72)


if __name__ == "__main__":
    main()
