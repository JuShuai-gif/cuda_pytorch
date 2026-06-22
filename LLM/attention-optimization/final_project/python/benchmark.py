# Mini TensorRT-LLM Attention Engine - Python Benchmark

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)


class MiniAttentionEngine:
    """
    Simulated mini attention engine demonstrating the architecture.

    In production, this would call compiled CUDA kernels.
    Here we use PyTorch ops to demonstrate the data flow and architecture.
    """

    def __init__(
        self,
        n_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        block_size: int = 16,
        dtype=torch.float16,
        device="cuda",
    ):
        self.n_layers = n_layers
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads  # For GQA
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # KV Cache as block-based storage
        self.num_blocks = max_seq_len // block_size
        self.kv_cache_shape = (
            self.num_blocks,
            self.n_layers,
            block_size,
            self.n_kv_heads,
            head_dim,
        )

        self.k_cache = torch.zeros(self.kv_cache_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(self.kv_cache_shape, dtype=dtype, device=device)
        self.block_allocated = torch.zeros(
            self.num_blocks, dtype=torch.bool, device="cpu"
        )

    def allocate_blocks(self, n_blocks: int):
        """Allocate physical blocks."""
        free_indices = (~self.block_allocated).nonzero(as_tuple=True)[0]
        if len(free_indices) < n_blocks:
            raise RuntimeError(
                f"Not enough free blocks: {len(free_indices)} < {n_blocks}"
            )
        allocated = free_indices[:n_blocks].tolist()
        self.block_allocated[allocated] = True
        return allocated

    def free_blocks(self, block_ids):
        """Free physical blocks."""
        self.block_allocated[block_ids] = False

    def prefill(self, tokens: torch.Tensor, block_table: list):
        """Prefill: process all prompt tokens."""
        # Simplified: use PyTorch SDPA for prefill
        n_tokens = tokens.size(0)
        out = F.scaled_dot_product_attention(
            tokens.unsqueeze(0),
            tokens.unsqueeze(0),
            tokens.unsqueeze(0),
            is_causal=True,
        )
        return out.squeeze(0)

    def decode(self, q: torch.Tensor, block_table: list, context_len: int):
        """Decode: single token with KV Cache."""
        # Simplified: load from block-based cache
        n_blocks = len(block_table)
        k_blocks = []
        v_blocks = []

        for layer in range(self.n_layers):
            for b_idx in block_table:
                k_blocks.append(self.k_cache[b_idx, layer])
                v_blocks.append(self.v_cache[b_idx, layer])

            k_full = torch.cat(k_blocks, dim=0)[:context_len]
            v_full = torch.cat(v_blocks, dim=0)[:context_len]

            # Attention: Q [1, n_q_heads, d] with K,V from cache
            # Repeat K,V for GQA if needed
            n_groups = self.n_q_heads // self.n_kv_heads
            if n_groups > 1:
                k_full = k_full.repeat_interleave(n_groups, dim=1)
                v_full = v_full.repeat_interleave(n_groups, dim=1)

            out = F.scaled_dot_product_attention(
                q.unsqueeze(0), k_full.unsqueeze(0), v_full.unsqueeze(0)
            )
            q = out.squeeze(0)

        return q

    @property
    def cache_memory_mb(self):
        elements = self.k_cache.numel() + self.v_cache.numel()
        return elements * self.k_cache.element_size() / (1024 * 1024)


def benchmark_engine():
    """Benchmark the mini engine against PyTorch baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 70)
    print("Mini Attention Engine Benchmark")
    print(f"Device: {device}, dtype: {dtype}")
    print("=" * 70)

    configs = [
        {"n_layers": 4, "n_q_heads": 8, "n_kv_heads": 8, "head_dim": 64},  # MHA
        {"n_layers": 4, "n_q_heads": 8, "n_kv_heads": 2, "head_dim": 64},  # GQA
        {"n_layers": 4, "n_q_heads": 8, "n_kv_heads": 1, "head_dim": 64},  # MQA
    ]

    for cfg in configs:
        engine = MiniAttentionEngine(
            n_layers=cfg["n_layers"],
            n_q_heads=cfg["n_q_heads"],
            n_kv_heads=cfg["n_kv_heads"],
            head_dim=cfg["head_dim"],
            max_seq_len=4096,
            dtype=dtype,
            device=device,
        )

        gqa_ratio = cfg["n_q_heads"] / cfg["n_kv_heads"]
        label = (
            f"MHA"
            if gqa_ratio == 1
            else f"GQA(g={int(gqa_ratio)})"
            if cfg["n_kv_heads"] > 1
            else "MQA"
        )

        print(f"\n--- {label}: {cfg['n_q_heads']}Q / {cfg['n_kv_heads']}KV heads ---")
        print(f"  KV Cache memory: {engine.cache_memory_mb:.1f} MB")

        # Simulate allocation
        prompt_len = 1024
        n_blocks_needed = (prompt_len + engine.block_size - 1) // engine.block_size
        block_table = engine.allocate_blocks(n_blocks_needed)
        print(f"  Allocated {n_blocks_needed} blocks for {prompt_len} tokens")

        # Decode benchmark
        q = torch.randn(
            1, cfg["n_q_heads"], cfg["head_dim"], device=device, dtype=dtype
        )
        context_len = min(512, prompt_len)

        warmup = 10 if device == "cuda" else 2
        iters = 100 if device == "cuda" else 20

        for _ in range(warmup):
            engine.decode(q, block_table, context_len)

        if device == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                engine.decode(q, block_table, context_len)
            end.record()
            torch.cuda.synchronize()
            ms_per_decode = start.elapsed_time(end) / iters
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                engine.decode(q, block_table, context_len)
            ms_per_decode = (time.perf_counter() - t0) * 1000 / iters

        print(
            f"  Decode latency: {ms_per_decode:.3f} ms/token "
            f"(context_len={context_len})"
        )

        engine.free_blocks(block_table)


if __name__ == "__main__":
    benchmark_engine()
