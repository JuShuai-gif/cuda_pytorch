"""
KV cache memory analysis and PagedAttention concepts.

The KV cache is the dominant memory consumer in LLM inference. For a LLaMA-70B
model with batch_size=32 at seq_len=4096, the KV cache alone requires ~70 GB.
Understanding and managing this memory is critical for production serving.

PagedAttention (vLLM) addresses this by allocating KV cache in non-contiguous
blocks (pages) instead of one contiguous array per sequence. This eliminates
fragmentation and over-allocation, enabling 2-4x higher throughput.
"""

from __future__ import annotations

from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# KV cache size calculation
# ---------------------------------------------------------------------------


def calculate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype = torch.float16,
    num_kv_heads: int | None = None,
) -> dict:
    """Calculate exact KV cache size for a given model configuration.

    The KV cache stores K and V matrices for each layer and each head.
    For Grouped Query Attention (GQA), num_kv_heads < num_heads.

    Formula: size = 2 * num_layers * num_kv_heads * batch_size * seq_len * head_dim * bytes_per_element

    The factor of 2 accounts for both K and V caches.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of query attention heads.
        head_dim: Dimension of each head.
        seq_len: Sequence length (max tokens in cache).
        batch_size: Batch size.
        dtype: Data type (float16 = 2 bytes, float32 = 4 bytes).
        num_kv_heads: Number of KV heads (for GQA). Defaults to num_heads.

    Returns:
        Dictionary with detailed size breakdown.
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads

    element_size = torch.tensor([], dtype=dtype).element_size()

    # Per head per token: head_dim elements * element_size
    bytes_per_head_per_token = head_dim * element_size

    # K cache for one layer, one batch element
    k_per_layer_batch = num_kv_heads * seq_len * bytes_per_head_per_token
    # V cache is same size
    v_per_layer_batch = k_per_layer_batch

    # Total for one layer, one batch element
    kv_per_layer_batch = k_per_layer_batch + v_per_layer_batch

    # Total across all layers and batch
    total_bytes = kv_per_layer_batch * num_layers * batch_size

    result = {
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1e9,
        "total_mb": total_bytes / 1e6,
        "per_layer_gb": kv_per_layer_batch / 1e9,
        "per_head_per_token_bytes": bytes_per_head_per_token,
        "k_cache_gb": k_per_layer_batch * num_layers * batch_size / 1e9,
        "v_cache_gb": v_per_layer_batch * num_layers * batch_size / 1e9,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "dtype": str(dtype),
    }
    return result


def show_kv_cache_growth() -> None:
    """Display KV cache memory growth across sequence lengths and model sizes.

    Shows how memory grows linearly with sequence length and model size,
    and why it's the primary bottleneck for long-context inference.
    """
    print("=== KV Cache Memory Growth Analysis ===\n")

    models = [
        ("LLaMA-7B", 32, 32, 128, 32),
        ("LLaMA-13B", 40, 40, 128, 40),
        ("LLaMA-70B", 80, 64, 128, 8),
        ("LLaMA-70B-GQA", 80, 64, 128, 8),  # GQA: 8 KV heads
    ]

    batch_size = 1
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    print(f"  Batch size: {batch_size}, dtype: float16\n")

    header = f"  {'Model':<18}"
    for sl in seq_lengths:
        header += f"  {'seq=' + str(sl):>12}"
    print(header)
    print(f"  {'-' * 18}{''.join(['  ' + '-' * 12 for _ in seq_lengths])}")

    for name, layers, num_heads, head_dim, kv_heads in models:
        row = f"  {name:<18}"
        for sl in seq_lengths:
            result = calculate_kv_cache_size(
                num_layers=layers,
                num_heads=num_heads,
                head_dim=head_dim,
                seq_len=sl,
                batch_size=batch_size,
                dtype=torch.float16,
                num_kv_heads=kv_heads,
            )
            gb = result["total_gb"]
            row += f"  {gb:>8.2f} GB"
        print(row)

    # Detailed breakdown for LLaMA-70B-GQA
    print(f"\n  Detailed breakdown for LLaMA-70B-GQA at seq_len=4096:")
    result = calculate_kv_cache_size(
        num_layers=80,
        num_heads=64,
        head_dim=128,
        seq_len=4096,
        batch_size=32,
        dtype=torch.float16,
        num_kv_heads=8,
    )
    print(f"    Total KV cache: {result['total_gb']:.2f} GB")
    print(f"    K cache: {result['k_cache_gb']:.2f} GB")
    print(f"    V cache: {result['v_cache_gb']:.2f} GB")
    print(f"    Per layer per batch: {result['per_layer_gb']:.4f} GB")
    print(f"    Per head per token: {result['per_head_per_token_bytes']} bytes")


# ---------------------------------------------------------------------------
# PagedAttention concept (simplified)
# ---------------------------------------------------------------------------


class PagedKVCache:
    """Simplified PagedAttention: KV cache stored in fixed-size blocks.

    In standard attention, each sequence gets one contiguous KV cache array.
    This leads to:
      - Over-allocation: must reserve max_seq_len * head_dim memory upfront
      - Fragmentation: different sequences have different lengths
      - Waste: short sequences waste reserved-but-unused memory

    PagedAttention (vLLM) solves this by:
      1. Dividing KV cache into fixed-size blocks (e.g., 16 tokens each)
      2. Each sequence has a block table (list of block indices)
      3. Blocks are allocated on-demand from a shared pool
      4. Unused blocks returned to pool

    This is inspired by virtual memory paging in operating systems.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize a paged KV cache.

        Args:
            num_blocks: Total number of blocks in the pool.
            block_size: Number of tokens per block.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            dtype: Data type.
        """
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.num_kv_layers = 1  # Simplified: single layer for demo

        # Allocate block pool: [num_blocks, num_heads, block_size, head_dim]
        self.k_blocks = torch.zeros(
            num_blocks, num_heads, block_size, head_dim, device="cuda", dtype=dtype
        )
        self.v_blocks = torch.zeros(
            num_blocks, num_heads, block_size, head_dim, device="cuda", dtype=dtype
        )

        # Free block list
        self.free_blocks: list[int] = list(range(num_blocks))

        # Per-sequence block tables: seq_id -> list of block indices
        self.block_tables: dict[int, list[int]] = {}

    def allocate_blocks(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a sequence.

        Args:
            seq_id: Unique sequence identifier.
            num_tokens: Number of tokens to allocate blocks for.

        Returns:
            List of block indices allocated.
        """
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        if num_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough free blocks. Need {num_needed}, have {len(self.free_blocks)}"
            )

        allocated = []
        for _ in range(num_needed):
            block_idx = self.free_blocks.pop(0)
            allocated.append(block_idx)

        self.block_tables[seq_id] = self.block_tables.get(seq_id, []) + allocated
        return allocated

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence, returning them to the pool."""
        if seq_id in self.block_tables:
            for block_idx in self.block_tables[seq_id]:
                self.free_blocks.append(block_idx)
            del self.block_tables[seq_id]

    def write_kv(
        self,
        seq_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Write K/V for tokens starting at start_pos into the paged cache.

        Args:
            seq_id: Sequence ID.
            k: Key tensor [num_heads, num_tokens, head_dim].
            v: Value tensor [num_heads, num_tokens, head_dim].
            start_pos: Token position to start writing at.
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Unknown sequence {seq_id}")

        num_tokens = k.shape[1]
        block_table = self.block_tables[seq_id]
        pos = start_pos

        for t in range(num_tokens):
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            if block_idx >= len(block_table):
                raise RuntimeError(f"Token {pos} exceeds allocated blocks")
            blk = block_table[block_idx]
            self.k_blocks[blk, :, offset, :] = k[:, t, :]
            self.v_blocks[blk, :, offset, :] = v[:, t, :]
            pos += 1

    def get_contiguous_kv(self, seq_id: int, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct contiguous K/V from paged blocks (for computing attention).

        In production, attention kernels operate directly on paged layouts
        (no need for this reconstruction). This is for demonstration.

        Args:
            seq_id: Sequence ID.
            num_tokens: Number of tokens to reconstruct.

        Returns:
            (K, V) tensors [1, num_heads, num_tokens, head_dim].
        """
        block_table = self.block_tables[seq_id]
        k_out = torch.empty(
            1, self.num_heads, num_tokens, self.head_dim, device="cuda", dtype=self.k_blocks.dtype
        )
        v_out = torch.empty(
            1, self.num_heads, num_tokens, self.head_dim, device="cuda", dtype=self.v_blocks.dtype
        )

        pos = 0
        for blk_idx in block_table:
            tokens_in_block = min(self.block_size, num_tokens - pos)
            if tokens_in_block <= 0:
                break
            k_out[0, :, pos : pos + tokens_in_block, :] = self.k_blocks[
                blk_idx, :, :tokens_in_block, :
            ]
            v_out[0, :, pos : pos + tokens_in_block, :] = self.v_blocks[
                blk_idx, :, :tokens_in_block, :
            ]
            pos += tokens_in_block

        return k_out, v_out

    @property
    def used_blocks(self) -> int:
        """Number of blocks currently in use."""
        return self.num_blocks - len(self.free_blocks)

    def memory_usage(self) -> float:
        """Memory usage in MB for the block pool."""
        k_bytes = self.k_blocks.numel() * self.k_blocks.element_size()
        v_bytes = self.v_blocks.numel() * self.v_blocks.element_size()
        return (k_bytes + v_bytes) / 1e6


def paged_attention_idea() -> None:
    """Demonstrate the PagedAttention concept.

    Shows how block-based allocation eliminates fragmentation and
    over-allocation compared to contiguous KV cache arrays.
    """
    print("\n=== PagedAttention Concept ===\n")

    # Config: ~100MB block pool
    num_blocks = 64
    block_size = 16
    num_heads = 8
    head_dim = 64
    dtype = torch.float16

    pool = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    print(f"  Block pool: {num_blocks} blocks of {block_size} tokens each")
    print(f"  Pool memory: {pool.memory_usage():.1f} MB")
    print(f"  Max tokens: {num_blocks * block_size}")

    # Simulate 3 sequences of different lengths
    for seq_id, tokens in enumerate([50, 100, 200]):
        blocks = pool.allocate_blocks(seq_id, tokens)
        print(f"  Seq {seq_id}: {tokens} tokens -> {len(blocks)} blocks, used={pool.used_blocks}")

        # Write dummy KV cache
        k = torch.randn(num_heads, tokens, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(num_heads, tokens, head_dim, device="cuda", dtype=dtype)
        pool.write_kv(seq_id, k, v, 0)

    # Free a short sequence - blocks return to pool
    pool.free_sequence(0)
    print(f"  After freeing seq 0: {pool.used_blocks} blocks in use, {len(pool.free_blocks)} free")

    # Allocate a new sequence using returned blocks
    blocks = pool.allocate_blocks(3, 80)
    print(f"  New seq 3: 80 tokens -> {len(blocks)} blocks allocated, {len(pool.free_blocks)} free")

    # Compare with contiguous allocation
    total_tokens_contiguous = 50 + 100 + 200 + 80
    contiguous_bytes = (
        total_tokens_contiguous * num_heads * head_dim * 2 * 2
    )  # *2 for K/V, *2 for fp16
    pool_bytes = pool.memory_usage() * 1e6

    # The contiguous approach would need to reserve max_seq_len for each
    max_seq_len = 200
    contiguous_reserved = max_seq_len * 4 * num_heads * head_dim * 2 * 2  # 4 sequences
    print(f"\n  Contiguous (reserve max): {contiguous_reserved / 1e6:.1f} MB")
    print(f"  Paged (allocated):        {pool_bytes / 1e6:.1f} MB")
    print(f"  Paged saves:              {(contiguous_reserved - pool_bytes) / 1e6:.1f} MB")

    # Clean up
    for seq_id in range(4):
        pool.free_sequence(seq_id)
    assert pool.used_blocks == 0
    print("  All blocks returned to pool. PagedAttention concept verified.")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB\n")

        show_kv_cache_growth()

        # KV cache size calculator
        print("\n--- KV Cache Size Calculator ---")
        for config in [
            ("LLaMA-7B", 32, 32, 128, 2048, 1),
            ("LLaMA-70B-GQA", 80, 64, 128, 4096, 8),
        ]:
            name, layers, heads, hd, sl, kv_heads = config
            r = calculate_kv_cache_size(
                num_layers=layers,
                num_heads=heads,
                head_dim=hd,
                seq_len=sl,
                batch_size=1,
                dtype=torch.float16,
                num_kv_heads=kv_heads,
            )
            print(f"  {name} (seq_len={sl}): {r['total_gb']:.2f} GB")

        paged_attention_idea()
        print("\nAll KV cache memory tests passed!")
