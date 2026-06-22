"""
KV Cache implementation in Python - Chapter 07.

Demonstrates:
1. KV Cache data structures
2. Prefill vs Decode phases
3. Memory usage analysis
4. Correctness against standard attention
"""

import math
import torch
import torch.nn.functional as F


class KVCache:
    """Simple KV Cache for a single transformer layer."""

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype=torch.float16,
        device="cuda",
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.k_cache = torch.zeros(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        self.cur_len = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new K,V tokens to the cache."""
        n_tokens = k_new.size(0)
        assert self.cur_len + n_tokens <= self.max_seq_len
        self.k_cache[self.cur_len : self.cur_len + n_tokens] = k_new
        self.v_cache[self.cur_len : self.cur_len + n_tokens] = v_new
        self.cur_len += n_tokens

    def clear(self):
        self.cur_len = 0

    @property
    def memory_mb(self):
        """Memory used by KV cache in MB."""
        elements = self.k_cache.numel() + self.v_cache.numel()
        return elements * self.k_cache.element_size() / (1024 * 1024)


def prefill_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, cache: KVCache
) -> torch.Tensor:
    """
    Prefill phase: process all prompt tokens at once.

    Q,K,V: [n_tokens, n_heads, head_dim]
    """
    # Store K,V in cache
    cache.append(K, V)

    # Full attention
    d_k = Q.size(-1)
    S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    P = F.softmax(S, dim=-1)
    O = P @ V
    return O


def decode_attention(Q_single: torch.Tensor, cache: KVCache) -> torch.Tensor:
    """
    Decode phase: process a single new token using cached K,V.

    Q_single: [1, n_heads, head_dim]
    """
    # Extract cached K,V up to current length
    K_cached = cache.k_cache[: cache.cur_len]  # [cur_len, n_heads, head_dim]
    V_cached = cache.v_cache[: cache.cur_len]

    d_k = Q_single.size(-1)
    # Q: [1, n_heads, d] @ K^T: [n_heads, d, cur_len] -> [1, n_heads, cur_len]
    S = torch.einsum("nhd,nhd->nh", Q_single, K_cached.permute(1, 2, 0)).unsqueeze(-1)
    # Actually need full dot product properly:
    # Q: [1, nh, d], K: [L, nh, d]
    # S: [1, nh, L]
    S = (Q_single.unsqueeze(0) @ K_cached.permute(1, 2, 0).unsqueeze(0)).squeeze(0)
    S = S / math.sqrt(d_k)  # [n_heads, cur_len]
    P = F.softmax(S, dim=-1)  # [n_heads, cur_len]
    # O: [1, nh, d]
    O = torch.einsum("hl,lhd->hd", P.squeeze(0), V_cached).unsqueeze(0)
    return O


def test_kv_cache():
    """Verify KV Cache correctness against full attention."""
    print("=== KV Cache Correctness Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use FP32 for exact comparison
    torch.manual_seed(42)

    n_tokens = 16
    n_heads = 4
    head_dim = 64
    max_seq_len = 256

    cache = KVCache(max_seq_len, n_heads, head_dim, dtype=dtype, device=device)

    # Step 1: Prefill with 8 tokens
    prompt_len = 8
    Q_prompt = torch.randn(prompt_len, n_heads, head_dim, device=device, dtype=dtype)
    K_prompt = torch.randn(prompt_len, n_heads, head_dim, device=device, dtype=dtype)
    V_prompt = torch.randn(prompt_len, n_heads, head_dim, device=device, dtype=dtype)

    O_prefill = prefill_attention(Q_prompt, K_prompt, V_prompt, cache)
    assert cache.cur_len == prompt_len, f"Cache length {cache.cur_len} != {prompt_len}"

    # Step 2: Decode one token at a time
    for step in range(4):
        Q_new = torch.randn(1, n_heads, head_dim, device=device, dtype=dtype)
        K_new = torch.randn(1, n_heads, head_dim, device=device, dtype=dtype)
        V_new = torch.randn(1, n_heads, head_dim, device=device, dtype=dtype)

        # Decode with cache
        O_decode = decode_attention(Q_new, cache)

        # Verify: full attention on all tokens so far should match
        cache.append(K_new, V_new)
        all_Q = torch.cat([Q_prompt[:1], Q_new])  # Just check last token
        all_K = cache.k_cache[: cache.cur_len]
        all_V = cache.v_cache[: cache.cur_len]

        # Full attention result for the last token
        d_k = Q_new.size(-1)
        S_full = all_Q[-1:] @ all_K.transpose(-2, -1) / math.sqrt(d_k)
        P_full = F.softmax(S_full, dim=-1)
        O_ref = P_full @ all_V  # [1, nh, d]

        diff = (O_decode - O_ref).abs().max().item()
        print(f"  Step {step}: max diff = {diff:.6e}")
        assert diff < 1e-4, f"Verification failed at step {step}: diff={diff}"

    print(f"  KV Cache memory: {cache.memory_mb:.2f} MB")
    print("  PASS: All decode steps match full attention!")


def benchmark_kv_cache():
    """Benchmark prefill vs decode latency."""
    print("\n=== KV Cache Benchmark ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    n_heads = 32
    head_dim = 128
    max_seq_len = 4096

    cache = KVCache(max_seq_len, n_heads, head_dim, dtype=dtype, device=device)
    torch.cuda.synchronize()

    # Prefill benchmark
    for prompt_len in [256, 512, 1024, 2048]:
        Q = torch.randn(prompt_len, n_heads, head_dim, device=device, dtype=dtype)
        K = torch.randn(prompt_len, n_heads, head_dim, device=device, dtype=dtype)
        V = torch.randn(prompt_len, n_heads, head_dim, device=device, dtype=dtype)
        cache.clear()

        # Warmup
        for _ in range(5):
            _ = prefill_attention(Q, K, V, cache)
            cache.clear()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = prefill_attention(Q, K, V, cache)
        end.record()
        torch.cuda.synchronize()
        prefill_ms = start.elapsed_time(end)

        # Decode benchmark (single token, with full cache)
        Q_single = torch.randn(1, n_heads, head_dim, device=device, dtype=dtype)

        # Warmup
        for _ in range(10):
            _ = decode_attention(Q_single, cache)

        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = decode_attention(Q_single, cache)
        end.record()
        torch.cuda.synchronize()
        decode_ms = start.elapsed_time(end) / 100

        print(
            f"  cache_len={prompt_len:5d} | prefill: {prefill_ms:8.3f}ms | decode: {decode_ms:8.3f}ms "
            f"| cache_mem: {cache.memory_mb:8.1f}MB"
        )


if __name__ == "__main__":
    test_kv_cache()
    benchmark_kv_cache()
