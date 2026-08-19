"""Simplified transformer layer with a KV cache, to measure prefill vs decode.

The layer does QKV projection -> scaled-dot-product attention -> output
projection.  Prefill processes a full sequence at once; decode processes one
token against the accumulated KV cache.  This is a single-layer, single-head
probe - enough to expose the compute-bound vs memory-bound split without a
full model.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TransformerLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def _attn(self, q, k, v):
        scale = 1.0 / math.sqrt(q.shape[-1])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a full sequence x (B, S, d); return (out, K, V)."""
        qkv = self.qkv(x)  # (B, S, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        out = self._attn(q, k, v)
        return self.out(out), k, v

    def decode(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """Process one token x (B, 1, d) against accumulated K/V caches."""
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        k_full = torch.cat([k_cache, k], dim=1)  # (B, S+1, d)
        v_full = torch.cat([v_cache, v], dim=1)
        out = self._attn(q, k_full, v_full)
        return self.out(out), k_full, v_full
