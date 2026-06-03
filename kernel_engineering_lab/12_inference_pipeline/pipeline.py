"""
End-to-end inference pipeline for a transformer-like model.

Ties together fused kernels, optimized attention, and KV cache management
from previous modules to simulate a production inference framework.

Architecture mirrors GPT/LLaMA:
  x = input [batch, seq_len, hidden_dim]
  For each layer:
    1. Self-attention with KV cache support
    2. Residual + LayerNorm
    3. FFN with gated activation
    4. Residual + LayerNorm
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_operator_fusion"))
sys.path.insert(0, str(_PROJECT_ROOT / "06_attention_flash_like"))
sys.path.insert(0, str(_PROJECT_ROOT / "02_triton_basics"))

from kernel_add_relu import fused_add_relu, sequential_add_relu
from kernel_bias_gelu import fused_bias_gelu, sequential_bias_gelu
from kernel_residual_layernorm import fused_residual_layernorm, sequential_residual_layernorm
from tiled_attention import tiled_attention

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

EPS = 1e-5


def _reshape_for_attention(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]."""
    B, L, H_dim = x.shape
    assert H_dim == num_heads * head_dim, (
        f"hidden_dim {H_dim} != num_heads * head_dim {num_heads * head_dim}"
    )
    return x.view(B, L, num_heads, head_dim).transpose(1, 2).contiguous()


def _reshape_from_attention(x: torch.Tensor) -> torch.Tensor:
    """Reshape [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]."""
    B, H, L, D = x.shape
    return x.transpose(1, 2).contiguous().view(B, L, H * D)


def _scaled_dot_product_attention_ref(q, k, v, causal=False, scale=None):
    """Reference attention using torch.nn.functional.scaled_dot_product_attention."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    is_causal = causal and q.shape[2] == k.shape[2]
    if is_causal:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


class TransformerBlock(nn.Module):
    """A transformer block using optimized kernels when use_fusions=True.

    Architecture (simplified GPT/LLaMA block):
      x = input [batch, seq_len, hidden_dim]

      1. Self-attention:
         qkv = fused_linear_qkv(x)  -> uses torch F.linear
         q, k, v = split(qkv)
         attn_out = attention(q, k, v, causal_mask)  -> uses tiled_attention or torch SDPA
         attn_proj = fused_linear_o(attn_out)  -> uses torch F.linear

      2. Residual + LayerNorm (fused with our Triton kernel)
         x = fused_residual_layernorm(x, attn_proj, weight, bias)

      3. FFN:
         ffn_h = fused_linear_gate(x)  -> torch F.linear
         ffn_h = fused_bias_gelu(ffn_h, bias)  -> fused bias+gelu or sequential
         ffn_out = fused_linear_down(ffn_h)  -> torch F.linear

      4. Residual + LayerNorm (fused)
         x = fused_residual_layernorm(x, ffn_out, weight2, bias2)

    When use_fusions=False: all ops are PyTorch eager (baseline).
    When use_fusions=True: attention uses tiled_attention, residual+layernorm
    uses fused_residual_layernorm, and bias+gelu uses fused_bias_gelu.
    Matmuls always use torch F.linear (cuBLAS) since cuBLAS is already optimal
    for matmuls.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        use_fusions: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_fusions = use_fusions
        self.dtype = dtype

        # QKV projection: single weight for Q, K, V combined
        qkv_dim = 3 * num_heads * head_dim
        self.qkv_weight = nn.Parameter(torch.empty(qkv_dim, hidden_dim, dtype=dtype))
        self.qkv_bias = nn.Parameter(torch.zeros(qkv_dim, dtype=dtype))

        # Output projection
        self.o_weight = nn.Parameter(torch.empty(hidden_dim, num_heads * head_dim, dtype=dtype))
        self.o_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))

        # FFN: gated (like SwiGLU but using GELU for simplicity)
        self.gate_weight = nn.Parameter(torch.empty(ffn_dim, hidden_dim, dtype=dtype))
        self.gate_bias = nn.Parameter(torch.zeros(ffn_dim, dtype=dtype))
        self.down_weight = nn.Parameter(torch.empty(hidden_dim, ffn_dim, dtype=dtype))
        self.down_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))

        # LayerNorm weights (per-element scale/offset for residual+layernorm)
        self.ln1_weight = nn.Parameter(torch.ones(hidden_dim, dtype=dtype))
        self.ln1_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))
        self.ln2_weight = nn.Parameter(torch.ones(hidden_dim, dtype=dtype))
        self.ln2_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))

        self._init_weights()

    def _init_weights(self):
        std_qkv = 0.02
        std_o = 0.02 / math.sqrt(2 * self.num_heads)
        std_gate = 0.02
        std_down = 0.02 / math.sqrt(2)

        nn.init.normal_(self.qkv_weight, std=std_qkv)
        nn.init.normal_(self.o_weight, std=std_o)
        nn.init.normal_(self.gate_weight, std=std_gate)
        nn.init.normal_(self.down_weight, std=std_down)

    def _attention(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """Compute self-attention for the transformer block.

        Args:
            x: [batch, seq_len, hidden_dim]
            causal_mask: Whether to apply causal mask.

        Returns:
            attn_output: [batch, seq_len, hidden_dim]
        """
        B, L, _ = x.shape

        # QKV projection
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        if self.use_fusions and TRITON_AVAILABLE:
            attn_out = tiled_attention(q, k, v, causal_mask=causal_mask)
        else:
            attn_out = _scaled_dot_product_attention_ref(q, k, v, causal=causal_mask)

        # Output projection
        attn_out = _reshape_from_attention(attn_out)
        attn_out = F.linear(attn_out, self.o_weight, self.o_bias)
        return attn_out

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward network with gated activation.

        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            ffn_output: [batch, seq_len, hidden_dim]
        """
        B, L, H = x.shape

        # Gate projection
        gate_out = F.linear(x, self.gate_weight, self.gate_bias)

        # Activation
        if self.use_fusions and TRITON_AVAILABLE:
            gate_out = gate_out.reshape(-1, self.gate_weight.shape[0])
            gate_bias_reshaped = self.gate_bias
            gate_out_flat = fused_bias_gelu(
                gate_out,
                torch.zeros_like(gate_out) + gate_bias_reshaped,
            )
            gate_out = gate_out_flat.reshape(B, L, -1)
        else:
            gate_out = F.gelu(gate_out, approximate="tanh")

        # Down projection
        ffn_out = F.linear(gate_out, self.down_weight, self.down_bias)
        return ffn_out

    def _residual_layernorm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Residual connection followed by LayerNorm.

        Args:
            x: [batch, seq_len, hidden_dim]
            residual: [batch, seq_len, hidden_dim]
            ln_weight: [hidden_dim]
            ln_bias: [hidden_dim]

        Returns:
            [batch, seq_len, hidden_dim]
        """
        B, L, H = x.shape

        if self.use_fusions and TRITON_AVAILABLE:
            x_flat = x.reshape(B * L, H)
            res_flat = residual.reshape(B * L, H)
            fused = fused_residual_layernorm(x_flat, res_flat, block_size=H)
            normalized = fused.reshape(B, L, H)
        else:
            combined = x + residual
            normalized = F.layer_norm(combined, [H], weight=None, bias=None)

        # Apply learnable weight and bias
        result = normalized * ln_weight + ln_bias
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the transformer block.

        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            [batch, seq_len, hidden_dim]
        """
        # Self-attention
        attn_out = self._attention(x)
        x = self._residual_layernorm(x, attn_out, self.ln1_weight, self.ln1_bias)

        # FFN
        ffn_out = self._ffn(x)
        x = self._residual_layernorm(x, ffn_out, self.ln2_weight, self.ln2_bias)

        return x


class OptimizedTransformer(nn.Module):
    """Stack of N TransformerBlocks.

    Configurable model:
      - num_layers: Number of transformer blocks
      - hidden_dim: Hidden dimension
      - num_heads: Number of attention heads
      - head_dim: Dimension per head (hidden_dim = num_heads * head_dim)
      - ffn_dim: Intermediate FFN dimension
      - use_fusions: Whether to use fused Triton kernels
      - dtype: Data type for parameters
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        use_fusions: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert hidden_dim == num_heads * head_dim, (
            f"hidden_dim ({hidden_dim}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
        )

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_fusions = use_fusions
        self.dtype = dtype

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_dim=ffn_dim,
                    use_fusions=use_fusions,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            [batch, seq_len, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class InferencePipeline:
    """Wraps the model with inference-specific optimizations.

    Supports:
      - prefill mode: process full prompt (batch, prompt_len, hidden)
      - decode mode: autoregressive generation (batch, 1, hidden) with KV cache
      - KV cache management using pre-allocated buffers
      - Batching: process multiple sequences
      - Timing: measure latency per token, throughput

    The pipeline uses our tiled_attention for prefill and attention_decode
    for decode steps. KV cache is managed by the KVCache class.
    """

    def __init__(
        self,
        model: OptimizedTransformer,
        kv_cache_cls=None,
        max_seq_len: int = 2048,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

        self.num_layers = model.num_layers
        self.num_heads = model.num_heads
        self.head_dim = model.head_dim
        self.hidden_dim = model.hidden_dim

        # Timing records
        self.prefill_timings: list[float] = []
        self.decode_timings: list[float] = []

        # KV cache will be initialized on first use
        self._kv_cache_class = kv_cache_cls or _SimpleKVCachePipeline

    def _compute_logits(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states.

        Uses the model's first layer's qkv_weight as a simple LM head for
        demonstration purposes. In production, this would be a separate weight.
        """
        # Use the first layer's qkv_weight as proxy for embedding/output projection
        # Reduce to vocab_size dimension using a simple projection
        return F.linear(hidden, weight)

    def prefill(
        self,
        input_embeds: torch.Tensor,
        kv_cache,
    ) -> torch.Tensor:
        """Process full prompt (prefill phase).

        Args:
            input_embeds: [batch, prompt_len, hidden_dim]
            kv_cache: KVCache instance to populate

        Returns:
            hidden_states: [batch, prompt_len, hidden_dim] from last layer
        """
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        x = input_embeds
        for layer_idx, layer in enumerate(self.model.layers):
            B, L, _ = x.shape

            # QKV projection
            qkv = F.linear(x, layer.qkv_weight, layer.qkv_bias)
            qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Store K and V in cache
            kv_cache.update(layer_idx, 0, k, v, positions=torch.arange(L, device=self.device))

            # Attention with all K/V (prefill, causally masked)
            if self.model.use_fusions and TRITON_AVAILABLE:
                attn_out = tiled_attention(q, k, v, causal_mask=True)
            else:
                attn_out = _scaled_dot_product_attention_ref(q, k, v, causal=True)

            attn_out = _reshape_from_attention(attn_out)
            attn_out = F.linear(attn_out, layer.o_weight, layer.o_bias)
            x = layer._residual_layernorm(x, attn_out, layer.ln1_weight, layer.ln1_bias)

            # FFN
            ffn_out = layer._ffn(x)
            x = layer._residual_layernorm(x, ffn_out, layer.ln2_weight, layer.ln2_bias)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        self.prefill_timings.append(elapsed)

        return x

    def decode_step(
        self,
        input_embeds: torch.Tensor,
        kv_cache,
        step: int,
    ) -> torch.Tensor:
        """Process a single token (decode phase).

        Args:
            input_embeds: [batch, 1, hidden_dim]
            kv_cache: KVCache instance with stored K, V
            step: Current generation step (position index)

        Returns:
            hidden_states: [batch, 1, hidden_dim] from last layer
        """
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        x = input_embeds
        for layer_idx, layer in enumerate(self.model.layers):
            B, L, _ = x.shape
            assert L == 1, f"Decode expects seq_len=1, got {L}"

            # Q projection for new token
            qkv = F.linear(x, layer.qkv_weight, layer.qkv_bias)
            qkv = qkv.reshape(B, 1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Store new K, V in cache at current step
            kv_cache.update(layer_idx, 0, k, v, positions=torch.tensor([step], device=self.device))

            # Get cached K, V (including the one we just stored)
            k_cached, v_cached = kv_cache.get(layer_idx, batch_idx=0, up_to=step + 1)

            # Attention: single Q against cached K/V (no causal mask needed for decode)
            if self.model.use_fusions and TRITON_AVAILABLE:
                from flash_attention_kv_cache import attention_decode

                attn_out = attention_decode(q, k_cached, v_cached)
            else:
                attn_out = _scaled_dot_product_attention_ref(q, k_cached, v_cached, causal=False)

            attn_out = _reshape_from_attention(attn_out)
            attn_out = F.linear(attn_out, layer.o_weight, layer.o_bias)
            x = layer._residual_layernorm(x, attn_out, layer.ln1_weight, layer.ln1_bias)

            # FFN
            ffn_out = layer._ffn(x)
            x = layer._residual_layernorm(x, ffn_out, layer.ln2_weight, layer.ln2_bias)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        self.decode_timings.append(elapsed)

        return x

    def generate(
        self,
        input_embeds: torch.Tensor,
        max_new_tokens: int,
        kv_cache,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            input_embeds: [batch, prompt_len, hidden_dim]
            max_new_tokens: Number of tokens to generate
            kv_cache: KVCache instance

        Returns:
            generated_embeds: [batch, prompt_len + max_new_tokens, hidden_dim]
        """
        B, prompt_len, H = input_embeds.shape

        # Prefill
        hidden = self.prefill(input_embeds, kv_cache)
        generated = [input_embeds]

        # Decode loop: generate one token at a time
        for step in range(max_new_tokens):
            # Use last hidden state as next input (simplified - in real models
            # you'd go through an LM head and embedding lookup)
            next_input = hidden[:, -1:, :]
            hidden = self.decode_step(next_input, kv_cache, prompt_len + step)
            generated.append(hidden.clone())

        return torch.cat(generated, dim=1)


class _SimpleKVCachePipeline:
    """Simple contiguous KV cache for the inference pipeline.

    Pre-allocates a contiguous buffer per layer:
      [2, batch, num_heads, max_seq_len, head_dim]
    where 2 stores K and V.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.k_cache = torch.zeros(
            num_layers,
            2,
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.filled = 0

    def update(
        self,
        layer_idx: int,
        batch_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Write K/V at specified positions.

        Args:
            layer_idx: Layer index.
            batch_idx: Batch index.
            k: Key tensor [batch_idx, num_heads, seq_len, head_dim]
            v: Value tensor [batch_idx, num_heads, seq_len, head_dim]
            positions: Positions to write [seq_len].
        """
        self.k_cache[layer_idx, 0, batch_idx, :, positions, :] = k.squeeze(0)
        self.k_cache[layer_idx, 1, batch_idx, :, positions, :] = v.squeeze(0)
        filled_pos = int(positions.max().item()) + 1
        if filled_pos > self.filled:
            self.filled = filled_pos

    def get(
        self,
        layer_idx: int,
        batch_idx: int,
        up_to: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get valid portion of K/V cache.

        Args:
            layer_idx: Layer index.
            batch_idx: Batch index.
            up_to: Get cache up to this position (exclusive).

        Returns:
            (k_cache, v_cache): [batch, num_heads, valid_len, head_dim]
        """
        limit = up_to if up_to is not None else self.filled
        k = self.k_cache[layer_idx, 0, batch_idx : batch_idx + 1, :, :limit, :].clone()
        v = self.k_cache[layer_idx, 1, batch_idx : batch_idx + 1, :, :limit, :].clone()
        return k, v

    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        return self.k_cache.numel() * self.k_cache.element_size()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        B, L, hidden, heads, h_dim, ffn = 2, 16, 256, 4, 64, 512
        x = torch.randn(B, L, hidden, device="cuda", dtype=torch.float32)

        block_fused = TransformerBlock(hidden, heads, h_dim, ffn, use_fusions=True).cuda()
        block_unfused = TransformerBlock(hidden, heads, h_dim, ffn, use_fusions=False).cuda()

        # Verify forward pass works
        y_fused = block_fused(x)
        y_unfused = block_unfused(x)

        assert y_fused.shape == y_unfused.shape == (B, L, hidden)
        print(f"TransformerBlock forward pass OK: shape={y_fused.shape}")
        print(f"Fused vs unfused output shapes match: {y_fused.shape == y_unfused.shape}")

        # Test full model
        model = OptimizedTransformer(
            num_layers=2,
            hidden_dim=hidden,
            num_heads=heads,
            head_dim=h_dim,
            ffn_dim=ffn,
            use_fusions=True,
        ).cuda()
        out = model(x)
        assert out.shape == (B, L, hidden)
        print(f"OptimizedTransformer forward pass OK: shape={out.shape}")

        # Test inference pipeline with prefill
        cache = _SimpleKVCachePipeline(
            num_layers=2,
            batch_size=B,
            num_heads=heads,
            max_seq_len=128,
            head_dim=h_dim,
        )
        pipeline = InferencePipeline(model)
        hidden_out = pipeline.prefill(x, cache)
        assert hidden_out.shape == (B, L, hidden)

        # Test decode step
        x_next = torch.randn(B, 1, hidden, device="cuda", dtype=torch.float32)
        hidden_dec = pipeline.decode_step(x_next, cache, step=L)
        assert hidden_dec.shape == (B, 1, hidden)

        print("Inference pipeline demo passed!")
